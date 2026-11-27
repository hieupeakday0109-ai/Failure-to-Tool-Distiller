"""Closed-loop ScarForgeEngine integrating Scar memory, BM25, Sandbox, ASTGuard, and Skills."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scar_forge.benchmark.tasks import BENCHMARK_TASKS
from scar_forge.config import EngineConfig
from scar_forge.distiller import TracebackDistiller
from scar_forge.forge.registry import SkillRegistry
from scar_forge.forge.sanitizer import ASTGuard
from scar_forge.llm.adapter import DualTierLLM
from scar_forge.llm.local_solver import detect_domain
from scar_forge.memory.retriever import BM25Retriever
from scar_forge.memory.scar_store import ScarStore
from scar_forge.models import (
    ActionType,
    DistilledError,
    EpochName,
    ModelTier,
    RunMetrics,
    ScarRecord,
    SkillRecord,
    TaskPurpose,
    TaskSpec,
    TierUsage,
)
from scar_forge.protocol import (
    PatchApplyError,
    ProtocolParser,
    ProtocolViolationError,
    apply_code_patch,
)
from scar_forge.sandbox.runner import SandboxRunner

_DOMAIN_TASKS_BY_ID: dict[str, TaskSpec] = {t.task_id: t for t in BENCHMARK_TASKS}

_SYSTEM_CORE = (
    "Return strict Zero-Prose JSON only: "
    '{"action":"RUN_CODE","code":"..."} | '
    '{"action":"CALL_SKILL","skill_name":"...","skill_args":{...}} | '
    '{"action":"PATCH_CODE","diff_patch":"..."} | '
    '{"action":"FINAL_ANSWER","result":...}.'
)


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return isinstance(actual, float) and math.isnan(actual)
        if math.isinf(expected):
            return actual == expected
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _values_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual.keys()) == set(expected.keys()) and all(
            _values_equal(actual[k], expected[k]) for k in expected
        )
    return actual == expected


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".tmp_{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class ScarForgeEngine:
    """Self-evolving Python agent engine learning Scars and forging hot-loaded Skills."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        data_dir: Path | str | None = None,
        provider: str | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif data_dir is not None:
            self.config = EngineConfig(data_dir=Path(data_dir))
        else:
            self.config = EngineConfig()
        self.config.ensure_dirs()

        self.scar_store = ScarStore(self.config)
        self.retriever = BM25Retriever(self.config)
        self.sandbox = SandboxRunner(self.config)
        self.ast_guard = ASTGuard()
        self.skill_registry = SkillRegistry(
            self.config, ast_guard=self.ast_guard, sandbox=self.sandbox
        )
        self.llm = DualTierLLM(self.config, provider=provider)
        self.protocol = ProtocolParser(allow_fence_fallback=True)
        self.distiller = TracebackDistiller()

    def _resolve_task_spec(
        self,
        task: str | TaskSpec,
        *,
        use_variant: bool = False,
        use_reuse_cases: bool = False,
    ) -> tuple[TaskSpec, list[dict[str, Any]], str]:
        if isinstance(task, TaskSpec):
            prompt = (
                task.variant_prompt
                if use_variant and task.variant_prompt
                else task.prompt
            )
            if use_reuse_cases and task.reuse_test_cases:
                cases = list(task.reuse_test_cases)
            elif use_variant and task.variant_test_cases:
                cases = list(task.variant_test_cases)
            else:
                cases = list(task.test_cases)
            return task, cases, prompt

        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task prompt must be a non-empty string")

        prompt = task.strip()
        domain_id = detect_domain(prompt)
        if domain_id and domain_id in _DOMAIN_TASKS_BY_ID:
            base = _DOMAIN_TASKS_BY_ID[domain_id]
            cases = (
                list(base.variant_test_cases)
                if ("batch 2" in prompt.lower() or "second" in prompt.lower())
                and base.variant_test_cases
                else list(base.test_cases)
            )
            spec = base.model_copy(update={"prompt": prompt})
            return spec, cases, prompt

        spec = TaskSpec(
            task_id=f"adhoc_{uuid.uuid4().hex[:8]}",
            title=prompt[:60],
            prompt=prompt,
            domain="general",
            tags=[w.lower() for w in prompt.split()[:4] if len(w) >= 3],
            entry_fn="solve",
            test_cases=[{"name": "adhoc_smoke", "kwargs": {}}],
        )
        return spec, list(spec.test_cases), prompt

    def _build_messages(
        self,
        prompt: str,
        scars: list[ScarRecord],
        skills: list[SkillRecord],
        *,
        last_error: DistilledError | None = None,
        last_code: str = "",
        protocol_hint: str = "",
    ) -> list[dict[str, str]]:
        sys_parts = [_SYSTEM_CORE]
        if scars:
            scar_lines = "\n".join(f"- {s.axiom}" for s in scars)
            sys_parts.append(f"[ACTIVE SCARS]\n{scar_lines}")
        if skills:
            skill_lines = "\n".join(
                f"- {sk.skill_name} ({sk.signature}): {sk.docstring}"
                for sk in skills
            )
            sys_parts.append(f"[ACTIVE SKILLS]\n{skill_lines}")

        user_parts = [f"[TASK]\n{prompt}"]
        if last_error is not None:
            user_parts.append(last_error.format_for_prompt())
        if last_code:
            user_parts.append(f"[PREVIOUS CODE]\n{last_code}")
        if protocol_hint:
            user_parts.append(f"[PROTOCOL_ERROR] {protocol_hint}")

        return [
            {"role": "system", "content": "\n\n".join(sys_parts)},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def _forge_skill_from_solution(
        self,
        spec: TaskSpec,
        solved_code: str,
        cases: list[dict[str, Any]],
        run_usage_acc: dict[ModelTier, TierUsage],
        schema_warnings: list[str],
    ) -> tuple[str | None, int, int, int, int]:
        """Distill passing solution into an AST-sanitized hot-loaded Skill on FRONTIER."""
        distill_msgs: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Refactor passing code into typed standalone SkillDraft JSON only: "
                    "{skill_name, source, docstring, tags, param_schema, test_cases}. "
                    "source must define typed top-level def <skill_name>(...) with docstring, "
                    "allowlisted imports only (no os/sys/subprocess/open/eval/exec/getattr/Path). "
                    "test_cases items: {\"name\":str,\"kwargs\":dict,\"expected\":val}."
                ),
            },
            {
                "role": "user",
                "content": f"[TASK]\n{spec.prompt}\n\n[SOLVED_CODE]\n{solved_code}",
            },
        ]
        tot_pt = 0
        tot_ct = 0
        tot_tt = 0
        esc_cnt = 0

        for attempt in range(2):
            resp = self.llm.complete(
                distill_msgs,
                purpose=TaskPurpose.SKILL_DISTILLATION,
                consecutive_failures=0,
            )
            self._accumulate_tier(run_usage_acc, resp.tier_used, resp)
            tot_pt += resp.prompt_tokens
            tot_ct += resp.completion_tokens
            tot_tt += resp.total_tokens
            if resp.escalated:
                esc_cnt += 1

            try:
                parsed = self.protocol.parse(resp.content, self._skill_draft_cls())
            except ProtocolViolationError as exc:
                if attempt == 0:
                    distill_msgs.append(
                        {"role": "user", "content": f"[PROTOCOL_ERROR] {exc}"}
                    )
                    continue
                return None, tot_pt, tot_ct, tot_tt, esc_cnt

            if parsed.schema_warning:
                schema_warnings.extend(parsed.warnings)
            draft = parsed.data
            eff_cases = draft.test_cases if draft.test_cases else cases

            try:
                record = self.skill_registry.register_and_reload(
                    source=draft.source,
                    skill_name=draft.skill_name,
                    docstring=draft.docstring,
                    tags=draft.tags or spec.tags,
                    test_cases=eff_cases,
                    param_schema=draft.param_schema or spec.param_schema,
                )
                return record.skill_name, tot_pt, tot_ct, tot_tt, esc_cnt
            except Exception as exc:
                if attempt == 0:
                    distill_msgs.append(
                        {
                            "role": "user",
                            "content": f"[FORGE_ERROR] {exc}. Fix and emit valid SkillDraft JSON.",
                        }
                    )
                    continue
                return None, tot_pt, tot_ct, tot_tt, esc_cnt

        return None, tot_pt, tot_ct, tot_tt, esc_cnt

    @staticmethod
    def _skill_draft_cls():
        from scar_forge.models import SkillDraft

        return SkillDraft

    @staticmethod
    def _accumulate_tier(
        acc: dict[ModelTier, TierUsage], tier: ModelTier, resp: Any
    ) -> None:
        cur = acc[tier]
        acc[tier] = TierUsage(
            tier=tier,
            calls=cur.calls + 1,
            prompt_tokens=cur.prompt_tokens + resp.prompt_tokens,
            completion_tokens=cur.completion_tokens + resp.completion_tokens,
            total_tokens=cur.total_tokens + resp.total_tokens,
            duration_ms=round(cur.duration_ms + resp.duration_ms, 2),
        )

    def _verify_skill_cases(
        self, skill_name: str, cases: list[dict[str, Any]], fallback_args: dict[str, Any]
    ) -> tuple[bool, float, DistilledError | None]:
        """Execute hot-loaded skill in-process across all task test cases."""
        if not cases:
            try:
                self.skill_registry.invoke_active(skill_name, **fallback_args)
                return True, 1.0, None
            except Exception as exc:
                err = self.distiller.distill(f"{type(exc).__name__}: {exc}")
                return False, 0.0, err

        passed = 0
        for case in cases:
            kwargs = case.get("kwargs") or {}
            args = case.get("args") or []
            expected_error = case.get("expected_error")
            has_expected = "expected" in case
            try:
                actual = self.skill_registry.invoke_active(skill_name, *args, **kwargs)
                if expected_error:
                    return (
                        False,
                        passed / len(cases),
                        self.distiller.distill(
                            f"AssertionError: Expected {expected_error}, got {actual!r}"
                        ),
                    )
                if has_expected and not _values_equal(actual, case["expected"]):
                    return (
                        False,
                        passed / len(cases),
                        self.distiller.distill(
                            f"AssertionError: Expected {case['expected']!r}, got {actual!r}"
                        ),
                    )
                passed += 1
            except Exception as exc:
                if expected_error and type(exc).__name__ == expected_error:
                    passed += 1
                    continue
                return (
                    False,
                    passed / len(cases),
                    self.distiller.distill(f"{type(exc).__name__}: {exc}"),
                )

        return True, 1.0, None

    def run_task(
        self,
        task: str | TaskSpec,
        *,
        epoch: EpochName | None = None,
        allow_scars: bool = True,
        allow_skills: bool = True,
        forge_on_success: bool = True,
        use_variant: bool = False,
        use_reuse_cases: bool = False,
    ) -> RunMetrics:
        """Execute a task through the closed Scar->Sandbox->Skill learning loop."""
        t0 = time.perf_counter()
        spec, cases, prompt = self._resolve_task_spec(
            task, use_variant=use_variant, use_reuse_cases=use_reuse_cases
        )
        is_adhoc_smoke = not isinstance(task, TaskSpec) and spec.task_id.startswith(
            "adhoc_"
        )
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        query = f"{prompt} {' '.join(spec.tags)}".strip()

        active_scars = (
            self.retriever.retrieve_scars(
                query, self.scar_store.list_scars(), top_k=self.config.top_k_scars
            )
            if allow_scars
            else []
        )
        active_skills = (
            self.retriever.retrieve_skills(
                query, self.skill_registry.list_skills(), top_k=self.config.top_k_skills
            )
            if allow_skills
            else []
        )

        scars_injected = [s.scar_id for s in active_scars]
        skills_injected = [sk.skill_name for sk in active_skills]
        scars_created: list[str] = []
        scars_prevented: list[str] = []
        schema_warnings: list[str] = []

        run_usage: dict[ModelTier, TierUsage] = {
            ModelTier.FAST_LOCAL: TierUsage(tier=ModelTier.FAST_LOCAL),
            ModelTier.FRONTIER: TierUsage(tier=ModelTier.FRONTIER),
        }
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        errors_count = 0
        repeated_errors_count = 0
        consecutive_failures = 0
        escalation_count = 0
        escalated = False
        success = False
        accuracy = 0.0
        skill_forged: str | None = None
        skill_reused: str | None = None
        attempts = 0

        last_error: DistilledError | None = None
        current_code = ""
        protocol_hint = ""
        skills_for_turn = list(active_skills)

        for attempt in range(1, self.config.max_attempts + 1):
            attempts = attempt
            if attempt == 1:
                purpose = TaskPurpose.INITIAL_DRAFT
            elif last_error is not None and last_error.is_syntax_error:
                purpose = TaskPurpose.SYNTAX_REPAIR
            else:
                purpose = TaskPurpose.TEST_LOOP

            messages = self._build_messages(
                prompt,
                active_scars,
                skills_for_turn,
                last_error=last_error,
                last_code=current_code if attempt > 1 else "",
                protocol_hint=protocol_hint,
            )
            resp = self.llm.complete(
                messages,
                purpose=purpose,
                consecutive_failures=consecutive_failures,
            )
            self._accumulate_tier(run_usage, resp.tier_used, resp)
            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            total_tokens += resp.total_tokens
            if resp.escalated:
                escalated = True
                escalation_count += 1

            try:
                parsed = self.protocol.parse_action(resp.content)
                if parsed.schema_warning:
                    schema_warnings.extend(parsed.warnings)
                action = parsed.data
                protocol_hint = ""
            except ProtocolViolationError as exc:
                errors_count += 1
                consecutive_failures += 1
                protocol_hint = str(exc)
                continue

            if action.action == ActionType.CALL_SKILL:
                sk_name = action.skill_name or ""
                verify_cases = [] if is_adhoc_smoke else cases
                ok, acc, sk_err = self._verify_skill_cases(
                    sk_name, verify_cases, action.skill_args or {}
                )
                if ok:
                    success = True
                    accuracy = acc
                    skill_reused = sk_name
                    if active_scars:
                        self.scar_store.mark_prevented(scars_injected)
                        scars_prevented = list(scars_injected)
                    break

                # Self-healing fallback: record scar & continue via RUN_CODE
                errors_count += 1
                consecutive_failures += 1
                last_error = sk_err
                skills_for_turn = []
                if sk_err is not None:
                    scar, created = self.scar_store.record_failure(
                        sk_err, f"{sk_name}()", spec.tags
                    )
                    if created:
                        scars_created.append(scar.scar_id)
                    else:
                        repeated_errors_count += 1
                continue

            if action.action == ActionType.PATCH_CODE:
                patch_obj = action.patch or action.diff_patch
                if patch_obj is None or not current_code:
                    errors_count += 1
                    consecutive_failures += 1
                    protocol_hint = "PATCH_CODE requires prior code context"
                    continue
                try:
                    current_code = apply_code_patch(
                        current_code, patch_obj, strip_fences=True
                    )
                except PatchApplyError as exc:
                    errors_count += 1
                    consecutive_failures += 1
                    last_error = self.distiller.distill(f"PatchApplyError: {exc}")
                    continue
            elif action.action == ActionType.RUN_CODE:
                current_code = action.code or ""
            elif action.action == ActionType.FINAL_ANSWER:
                if action.code and action.code.strip():
                    current_code = action.code
                else:
                    # Never trust unverified LLM self-evaluation without sandbox execution
                    errors_count += 1
                    consecutive_failures += 1
                    protocol_hint = (
                        "FINAL_ANSWER requires verified 'code' or CALL_SKILL; "
                        "sandbox verification cannot be skipped"
                    )
                    continue

            sb_res = self.sandbox.execute_code(
                current_code, cases, entry_fn=spec.entry_fn
            )
            if sb_res.passed:
                success = True
                accuracy = (
                    sb_res.passed_cases / sb_res.total_cases
                    if sb_res.total_cases > 0
                    else 1.0
                )
                if active_scars and errors_count == 0:
                    self.scar_store.mark_prevented(scars_injected)
                    scars_prevented = list(scars_injected)

                if forge_on_success and skill_reused is None:
                    (
                        forged_name,
                        d_pt,
                        d_ct,
                        d_tt,
                        _,
                    ) = self._forge_skill_from_solution(
                        spec, current_code, cases, run_usage, schema_warnings
                    )
                    skill_forged = forged_name
                    prompt_tokens += d_pt
                    completion_tokens += d_ct
                    total_tokens += d_tt
                break

            # Failure path: distill stderr, record Scar, re-retrieve Top-K Scars for repair
            errors_count += 1
            consecutive_failures += 1
            accuracy = (
                sb_res.passed_cases / sb_res.total_cases
                if sb_res.total_cases > 0
                else 0.0
            )
            last_error = sb_res.distilled_error or self.distiller.distill(
                sb_res.stderr, sb_res.stdout
            )
            scar, created = self.scar_store.record_failure(
                last_error, current_code, spec.tags
            )
            if created:
                scars_created.append(scar.scar_id)
            else:
                repeated_errors_count += 1

            if allow_scars:
                active_scars = self.retriever.retrieve_scars(
                    f"{query} {last_error.exception_type} {last_error.core_message}",
                    self.scar_store.list_scars(),
                    top_k=self.config.top_k_scars,
                )
                new_ids = [s.scar_id for s in active_scars]
                scars_injected = list(dict.fromkeys([*scars_injected, *new_ids]))

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        metrics = RunMetrics(
            run_id=run_id,
            task_id=spec.task_id,
            epoch=epoch,
            success=success,
            attempts=max(1, attempts),
            errors_count=errors_count,
            repeated_errors_count=repeated_errors_count,
            escalated=escalated,
            escalation_count=escalation_count,
            scars_injected=scars_injected,
            scars_created=scars_created,
            scars_prevented=scars_prevented,
            skills_injected=skills_injected,
            skill_forged=skill_forged,
            skill_reused=skill_reused,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tier_usage={tier.value: usage for tier, usage in run_usage.items()},
            duration_ms=duration_ms,
            accuracy=accuracy,
            schema_warnings=schema_warnings,
        )
        _atomic_write_json(
            self.config.runs_dir / f"{run_id}.json",
            metrics.model_dump(mode="json"),
        )
        return metrics
