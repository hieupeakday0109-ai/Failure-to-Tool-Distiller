"""3-Epoch recursive learning benchmark runner, metrics aggregator, and Markdown reporter."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scar_forge.benchmark.tasks import get_benchmark_tasks
from scar_forge.config import EngineConfig
from scar_forge.models import BenchmarkReport, EpochName, EpochSummary, RunMetrics, TaskSpec
from scar_forge.tokens import TOKENIZER_BACKEND

VALID_BENCHMARK_MODES = frozenset({"variant", "standard", "ablation"})


def _atomic_write_text(path: Path, text: str) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _epoch_to_api_dict(summary: EpochSummary) -> dict[str, Any]:
    task_items = []
    for r in summary.runs:
        item = r.model_dump(mode="json")
        item["tokens"] = r.total_tokens
        item["passed"] = r.success
        item["errors"] = r.errors_count
        task_items.append(item)
    return {
        "epoch": summary.epoch.value,
        "total_tasks": summary.total_tasks,
        "passed_tasks": summary.passed_tasks,
        "accuracy": summary.accuracy,
        "total_tokens": summary.total_tokens,
        "tokens": summary.total_tokens,
        "avg_tokens_per_task": summary.avg_tokens_per_task,
        "total_errors": summary.total_errors,
        "errors": summary.total_errors,
        "repeated_errors": summary.repeated_errors,
        "total_duration_ms": summary.total_duration_ms,
        "duration_ms": summary.total_duration_ms,
        "avg_duration_ms": summary.avg_duration_ms,
        "scars_active": summary.scars_active,
        "skills_active": summary.skills_active,
        "skills_reused_count": summary.skills_reused_count,
        "skill_calls": summary.skills_reused_count,
        "tasks": task_items,
        "runs": task_items,
    }


class BenchmarkRunner:
    """Executes the 8-task suite across Epoch 1 (Cold), Epoch 2 (Scar), and Epoch 3 (Skill)."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        provider: str | None = None,
        tasks: list[TaskSpec] | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.ensure_dirs()
        self.provider = provider or self.config.provider
        self.tasks = tasks if tasks is not None else get_benchmark_tasks()

    @property
    def latest_report_path(self) -> Path:
        return self.config.data_dir / "latest_benchmark.json"

    @staticmethod
    def _summarize_epoch(
        epoch: EpochName,
        runs: list[RunMetrics],
        *,
        scars_active: int,
        skills_active: int,
    ) -> EpochSummary:
        n = len(runs)
        passed = sum(1 for r in runs if r.success)
        total_tokens = sum(r.total_tokens for r in runs)
        total_errors = sum(r.errors_count for r in runs)
        repeated_errors = sum(r.repeated_errors_count for r in runs)
        total_dur = round(sum(r.duration_ms for r in runs), 2)
        reused_cnt = sum(1 for r in runs if r.skill_reused is not None)
        return EpochSummary(
            epoch=epoch,
            total_tasks=n,
            passed_tasks=passed,
            accuracy=round(passed / n, 4) if n > 0 else 0.0,
            total_tokens=total_tokens,
            avg_tokens_per_task=round(total_tokens / n, 2) if n > 0 else 0.0,
            total_errors=total_errors,
            repeated_errors=repeated_errors,
            total_duration_ms=total_dur,
            avg_duration_ms=round(total_dur / n, 2) if n > 0 else 0.0,
            scars_active=scars_active,
            skills_active=skills_active,
            skills_reused_count=reused_cnt,
            runs=runs,
        )

    def run(self, *, mode: str = "variant") -> tuple[BenchmarkReport, dict[str, Any]]:
        """Execute all 3 benchmark epochs and return (BenchmarkReport, serialized_dict)."""
        norm_mode = (mode or "variant").strip().lower()
        if norm_mode not in VALID_BENCHMARK_MODES:
            raise ValueError(
                f"Invalid benchmark mode {mode!r}; expected one of {sorted(VALID_BENCHMARK_MODES)}"
            )

        from scar_forge.agent.engine import ScarForgeEngine

        engine = ScarForgeEngine(config=self.config, provider=self.provider)

        # Epoch 1: Cold Start (no pre-existing skill calls; distills Scars and forges Skills)
        e1_runs: list[RunMetrics] = []
        for task in self.tasks:
            e1_runs.append(
                engine.run_task(
                    task,
                    epoch=EpochName.EPOCH_1_COLD_START,
                    allow_scars=True,
                    allow_skills=False,
                    forge_on_success=True,
                    use_variant=False,
                    use_reuse_cases=False,
                )
            )
        e1_summary = self._summarize_epoch(
            EpochName.EPOCH_1_COLD_START,
            e1_runs,
            scars_active=len(engine.scar_store.list_scars()),
            skills_active=len(engine.skill_registry.list_skills()),
        )

        # Epoch 2: Scar-Guided Execution (Scars active, Skills disabled to isolate Scar effect)
        e2_runs: list[RunMetrics] = []
        use_var = norm_mode in {"variant", "ablation"}
        for task in self.tasks:
            e2_runs.append(
                engine.run_task(
                    task,
                    epoch=EpochName.EPOCH_2_SCAR_GUIDED,
                    allow_scars=True,
                    allow_skills=False,
                    forge_on_success=False,
                    use_variant=use_var,
                    use_reuse_cases=False,
                )
            )
        e2_summary = self._summarize_epoch(
            EpochName.EPOCH_2_SCAR_GUIDED,
            e2_runs,
            scars_active=len(engine.scar_store.list_scars()),
            skills_active=len(engine.skill_registry.list_skills()),
        )

        # Epoch 3: Skill-Reuse Execution (AST-verified hot-reloaded Skills invoked in-process)
        e3_runs: list[RunMetrics] = []
        for task in self.tasks:
            e3_runs.append(
                engine.run_task(
                    task,
                    epoch=EpochName.EPOCH_3_SKILL_REUSE,
                    allow_scars=True,
                    allow_skills=True,
                    forge_on_success=False,
                    use_variant=False,
                    use_reuse_cases=True,
                )
            )
        e3_summary = self._summarize_epoch(
            EpochName.EPOCH_3_SKILL_REUSE,
            e3_runs,
            scars_active=len(engine.scar_store.list_scars()),
            skills_active=len(engine.skill_registry.list_skills()),
        )

        t1 = float(e1_summary.total_tokens)
        t3 = float(e3_summary.total_tokens)
        token_reduction = round(1.0 - (t3 / t1), 4) if t1 > 0 else 0.0

        err1 = float(e1_summary.total_errors)
        err2 = float(e2_summary.repeated_errors)
        error_reduction = round(1.0 - (err2 / err1), 4) if err1 > 0 else 0.0

        dur1 = float(e1_summary.total_duration_ms)
        dur3 = float(e3_summary.total_duration_ms)
        speedup = round(dur1 / max(dur3, 0.01), 2) if dur1 > 0 else 1.0

        report = BenchmarkReport(
            report_id=f"bench_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}",
            provider=engine.llm.provider,
            tokenizer_backend=TOKENIZER_BACKEND,
            epochs=[e1_summary, e2_summary, e3_summary],
            token_reduction_epoch3_vs_epoch1=token_reduction,
            error_reduction_epoch2_vs_epoch1=error_reduction,
            speedup_epoch3_vs_epoch1=speedup,
        )

        payload = report.model_dump(mode="json")
        payload["mode"] = norm_mode
        payload["epoch_1"] = _epoch_to_api_dict(e1_summary)
        payload["epoch_2"] = _epoch_to_api_dict(e2_summary)
        payload["epoch_3"] = _epoch_to_api_dict(e3_summary)

        _atomic_write_text(
            self.latest_report_path, json.dumps(payload, indent=2, ensure_ascii=False)
        )
        return report, payload

    @staticmethod
    def render_markdown(payload: dict[str, Any]) -> str:
        """Render the 3-Epoch benchmark payload as a crisp Markdown report table."""
        e1 = payload["epoch_1"]
        e2 = payload["epoch_2"]
        e3 = payload["epoch_3"]
        tok_red = float(payload.get("token_reduction_epoch3_vs_epoch1", 0.0)) * 100.0
        err_red = float(payload.get("error_reduction_epoch2_vs_epoch1", 0.0)) * 100.0
        speedup = float(payload.get("speedup_epoch3_vs_epoch1", 1.0))

        tbl_hdr = (
            "| Epoch | Passed | Accuracy | Total Tokens | Avg Tokens/Task | "
            "Total Errors | Repeated Errors | Duration (ms) | "
            "Active Scars | Active Skills | Skill Reuses |"
        )
        tbl_sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        lines = [
            "# Scar-Forge 3-Epoch Benchmark Results",
            "",
            f"- **Provider:** `{payload.get('provider', 'local')}`",
            f"- **Mode:** `{payload.get('mode', 'variant')}`",
            f"- **Tokenizer:** `{payload.get('tokenizer_backend', TOKENIZER_BACKEND)}`",
            f"- **Epoch 3 vs Epoch 1 Token Reduction:** **{tok_red:.1f}%**",
            f"- **Epoch 2 Repeated Error Reduction:** **{err_red:.1f}%**",
            f"- **Epoch 3 vs Epoch 1 Speedup:** **{speedup:.2f}x**",
            "",
            tbl_hdr,
            tbl_sep,
        ]
        for label, ep in (
            ("Epoch 1 (Cold Start)", e1),
            ("Epoch 2 (Scar-Guided)", e2),
            ("Epoch 3 (Skill-Reuse)", e3),
        ):
            lines.append(
                f"| {label} | {ep['passed_tasks']}/{ep['total_tasks']} "
                f"| {ep['accuracy'] * 100:.1f}% "
                f"| {ep['total_tokens']:,} "
                f"| {ep['avg_tokens_per_task']:.1f} "
                f"| {ep['total_errors']} "
                f"| {ep['repeated_errors']} "
                f"| {ep['total_duration_ms']:.1f} "
                f"| {ep['scars_active']} "
                f"| {ep['skills_active']} "
                f"| {ep['skills_reused_count']} |"
            )

        task_hdr = (
            "| Task ID | E1 Tokens (Errors) | E2 Tokens (Repeated Err) | "
            "E3 Tokens (Skill Reused) | Token Savings (E3 vs E1) |"
        )
        lines.extend(
            [
                "",
                "## Per-Task Breakdown (Epoch 1 -> Epoch 2 -> Epoch 3)",
                "",
                task_hdr,
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for r1, r2, r3 in zip(e1["tasks"], e2["tasks"], e3["tasks"], strict=True):
            t1 = r1["total_tokens"]
            t3 = r3["total_tokens"]
            sav = (1.0 - (t3 / t1)) * 100.0 if t1 > 0 else 0.0
            lines.append(
                f"| `{r1['task_id']}` | {t1} ({r1['errors_count']}) "
                f"| {r2['total_tokens']} ({r2['repeated_errors_count']}) "
                f"| {t3} (`{r3.get('skill_reused') or '-'}`) "
                f"| {sav:.1f}% |"
            )

        return "\n".join(lines) + "\n"
