"""Atomic, AST-sanitized, hot-reloadable Skill registry without .pyc staleness."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import keyword
import os
import re
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scar_forge.config import EngineConfig
from scar_forge.forge.sanitizer import ASTGuard
from scar_forge.models import SkillRecord
from scar_forge.sandbox.runner import SandboxRunner

_SKILL_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".tmp_{target.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _extract_signature_and_schema(
    source: str, skill_name: str
) -> tuple[str, dict[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return f"def {skill_name}(*args, **kwargs)", {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == skill_name:
            parts: list[str] = []
            schema: dict[str, str] = {}
            all_args = list(node.args.posonlyargs) + list(node.args.args)
            for arg in all_args:
                ann = ast.unparse(arg.annotation) if arg.annotation else "Any"
                parts.append(f"{arg.arg}: {ann}")
                schema[arg.arg] = ann
            for arg in node.args.kwonlyargs:
                ann = ast.unparse(arg.annotation) if arg.annotation else "Any"
                parts.append(f"{arg.arg}: {ann}")
                schema[arg.arg] = ann
            ret = ast.unparse(node.returns) if node.returns else "Any"
            return f"def {skill_name}({', '.join(parts)}) -> {ret}", schema
    return f"def {skill_name}(*args, **kwargs)", {}


class SkillRegistry:
    """Persistent, AST-and-Sandbox-gated Python skill registry with hot-reload."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        ast_guard: ASTGuard | None = None,
        sandbox: SandboxRunner | None = None,
        skills_dir: Path | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        if skills_dir is not None:
            self.skills_dir = Path(skills_dir)
        else:
            self.config.ensure_dirs()
            self.skills_dir = self.config.skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.skills_dir / "manifest.json"
        self.ast_guard = ast_guard or ASTGuard()
        self.sandbox = sandbox or SandboxRunner(self.config)
        self._lock = threading.RLock()
        self._records: dict[str, SkillRecord] = {}
        self._callables: dict[str, Callable[..., Any]] = {}
        self._load_persisted()

    def _validate_skill_path(self, skill_name: str) -> Path:
        if (
            not isinstance(skill_name, str)
            or not _SKILL_NAME_RE.match(skill_name)
            or keyword.iskeyword(skill_name)
        ):
            raise ValueError(f"Invalid skill name {skill_name!r}: must be snake_case")
        target = (self.skills_dir / f"{skill_name}.py").resolve()
        if target.parent != self.skills_dir.resolve():
            raise ValueError(f"Path traversal rejected for skill {skill_name!r}")
        return target

    def _purge_stale_pycache(self, skill_name: str) -> None:
        pycache = self.skills_dir / "__pycache__"
        if pycache.is_dir():
            for pyc in pycache.glob(f"{skill_name}.*.pyc"):
                pyc.unlink(missing_ok=True)

    def _load_module_callable(
        self, skill_path: Path, skill_name: str, version: int, source_hash: str
    ) -> Callable[..., Any]:
        self._purge_stale_pycache(skill_name)
        importlib.invalidate_caches()
        unique_name = f"scar_forge_skills.{skill_name}_v{version}_{source_hash[:10]}"
        prev_dont_write = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location(unique_name, str(skill_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load module spec from {skill_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = prev_dont_write

        fn = getattr(module, skill_name, None)
        if not callable(fn):
            raise RuntimeError(f"Skill {skill_name!r} missing callable entry in {skill_path}")
        return fn

    def _save_manifest(self) -> None:
        payload = [
            rec.model_dump(mode="json")
            for rec in sorted(self._records.values(), key=lambda r: r.skill_name)
        ]
        _atomic_write_text(self.manifest_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _load_persisted(self) -> None:
        with self._lock:
            if not self.manifest_path.is_file():
                return
            try:
                raw_items = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return
            if not isinstance(raw_items, list):
                return
            for item in raw_items:
                try:
                    rec = SkillRecord.model_validate(item)
                except Exception:
                    continue
                skill_path = self.skills_dir / f"{rec.skill_name}.py"
                if not skill_path.is_file():
                    continue
                try:
                    fn = self._load_module_callable(
                        skill_path, rec.skill_name, rec.version, rec.source_hash
                    )
                    self._records[rec.skill_name] = rec
                    self._callables[rec.skill_name] = fn
                except Exception:
                    continue

    def register_and_reload(
        self,
        source: str,
        skill_name: str,
        docstring: str,
        tags: Sequence[str] | None = None,
        test_cases: Sequence[dict[str, Any]] | None = None,
        *,
        param_schema: dict[str, str] | None = None,
    ) -> SkillRecord:
        """Validate via ASTGuard + adversarial Sandbox, atomically persist, and hot-reload."""
        skill_path = self._validate_skill_path(skill_name)

        # Gate 1: Static AST security, typing, docstring, zero-stub validation
        ast_res = self.ast_guard.validate(source, expected_name=skill_name)
        if not ast_res.valid:
            raise ValueError(
                f"AST validation failed for skill '{skill_name}': {'; '.join(ast_res.violations)}"
            )

        signature, inferred_schema = _extract_signature_and_schema(source, skill_name)
        eff_schema = dict(param_schema) if param_schema is not None else inferred_schema

        # Gate 2: Isolated adversarial sandbox verification (before mutating disk/state!)
        sb_res = self.sandbox.verify_skill_adversarial(
            source=source,
            skill_name=skill_name,
            explicit_cases=test_cases or [],
            param_schema=eff_schema,
        )
        if not sb_res.passed:
            err_msg = (
                sb_res.distilled_error.format_for_prompt()
                if sb_res.distilled_error
                else sb_res.stderr or "Sandbox verification failed"
            )
            raise RuntimeError(
                f"Adversarial sandbox verification failed for '{skill_name}': {err_msg}"
            )

        clean_source = source if source.endswith("\n") else f"{source}\n"
        source_hash = hashlib.sha256(clean_source.encode("utf-8")).hexdigest()
        clean_tags = sorted(
            {t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()}
        )
        clean_doc = (docstring or "").strip() or f"Skill {skill_name}."
        now = datetime.now(UTC)

        with self._lock:
            existing = self._records.get(skill_name)
            version = (existing.version + 1) if existing is not None else 1
            reuse_count = existing.reuse_count if existing is not None else 0
            created_at = existing.created_at if existing is not None else now

            _atomic_write_text(skill_path, clean_source)
            fn = self._load_module_callable(skill_path, skill_name, version, source_hash)

            record = SkillRecord(
                skill_id=existing.skill_id if existing is not None else f"skill_{skill_name}",
                skill_name=skill_name,
                signature=signature,
                docstring=clean_doc,
                file_path=str(skill_path),
                source_hash=source_hash,
                tags=clean_tags,
                param_schema=eff_schema,
                ast_verified=True,
                sandbox_verified=True,
                hot_reloaded=True,
                version=version,
                reuse_count=reuse_count,
                created_at=created_at,
                updated_at=now,
            )
            self._records[skill_name] = record
            self._callables[skill_name] = fn
            self._save_manifest()
            return record

    def invoke_active(self, skill_name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Execute a hot-loaded skill directly in-process and record reuse telemetry."""
        with self._lock:
            fn = self._callables.get(skill_name)
            if fn is None:
                raise KeyError(f"Unknown active skill: {skill_name!r}")

        result = fn(*args, **kwargs)

        with self._lock:
            rec = self._records.get(skill_name)
            if rec is not None:
                self._records[skill_name] = rec.model_copy(
                    update={
                        "reuse_count": rec.reuse_count + 1,
                        "updated_at": datetime.now(UTC),
                    }
                )
                self._save_manifest()
        return result

    def list_skills(self) -> list[SkillRecord]:
        """Return metadata for all active persisted skills."""
        with self._lock:
            self._load_persisted() if not self._records and self.manifest_path.is_file() else None
            return sorted(self._records.values(), key=lambda r: r.skill_name)

    def get_skill(self, skill_name: str) -> SkillRecord | None:
        """Fetch metadata for a single active skill."""
        with self._lock:
            return self._records.get(skill_name)

    def get_source(self, skill_name: str) -> str | None:
        """Read the persisted Python source file for a registered skill."""
        path = self.skills_dir / f"{skill_name}.py"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None
