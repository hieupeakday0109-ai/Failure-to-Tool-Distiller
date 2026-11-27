"""Unit & edge-case tests for ASTGuard, SkillRegistry, DualTierLLM, and Engine (M3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scar_forge.agent.engine import ScarForgeEngine
from scar_forge.config import EngineConfig
from scar_forge.forge.registry import SkillRegistry
from scar_forge.forge.sanitizer import ASTGuard
from scar_forge.llm.adapter import DualTierLLM
from scar_forge.models import ModelTier, TaskPurpose


class TestASTGuardUnit:
    def test_allows_pure_posix_path_but_blocks_pathlib_path(self) -> None:
        guard = ASTGuard()
        safe = (
            "from pathlib import PurePosixPath\n\n"
            "def norm_rel(p: str) -> str:\n"
            '    """Normalize POSIX path."""\n'
            "    return str(PurePosixPath(p or '.'))\n"
        )
        assert guard.validate(safe, expected_name="norm_rel").valid is True

        unsafe = (
            "from pathlib import Path\n\n"
            "def read_secret(p: str) -> str:\n"
            '    """Read arbitrary file."""\n'
            "    return Path(p).read_text()\n"
        )
        res = guard.validate(unsafe, expected_name="read_secret")
        assert res.valid is False
        assert any("Blocked" in v for v in res.violations)

    def test_blocks_relative_imports_and_top_level_calls(self) -> None:
        guard = ASTGuard()
        rel_imp = (
            "from .other import helper\n"
            "def fn(x: int) -> int:\n"
            '    """Doc."""\n'
            "    return x\n"
        )
        assert guard.validate(rel_imp).valid is False

        top_call = (
            "print('side effect')\n"
            "def fn(x: int) -> int:\n"
            '    """Doc."""\n'
            "    return x\n"
        )
        assert guard.validate(top_call).valid is False


class TestSkillRegistryUnit:
    def test_same_second_hot_reload_and_reuse_persistence(self, tmp_path: Path) -> None:
        cfg = EngineConfig(data_dir=tmp_path)
        reg = SkillRegistry(cfg)
        v1 = (
            "def mul_k(x: int) -> int:\n"
            '    """Multiply by 2."""\n'
            "    return x * 2\n"
        )
        v2 = (
            "def mul_k(x: int) -> int:\n"
            '    """Multiply by 5."""\n'
            "    return x * 5\n"
        )
        r1 = reg.register_and_reload(
            v1, "mul_k", "Multiply by 2.", ["math"], [{"kwargs": {"x": 3}, "expected": 6}]
        )
        assert r1.version == 1
        assert reg.invoke_active("mul_k", x=3) == 6

        r2 = reg.register_and_reload(
            v2, "mul_k", "Multiply by 5.", ["math"], [{"kwargs": {"x": 3}, "expected": 15}]
        )
        assert r2.version == 2
        assert reg.invoke_active("mul_k", x=3) == 15

        reg2 = SkillRegistry(cfg)
        loaded = reg2.get_skill("mul_k")
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.reuse_count == 2
        assert reg2.get_source("mul_k") is not None


class TestDualTierLLMUnit:
    def test_invalid_provider_and_reset_usage(self, tmp_path: Path) -> None:
        cfg = EngineConfig(data_dir=tmp_path)
        with pytest.raises(ValueError):
            DualTierLLM(cfg, provider="invalid_backend")

        llm = DualTierLLM(cfg, provider="local", )
        llm.complete(
            [{"role": "user", "content": "Sort semantic version tags"}],
            purpose=TaskPurpose.INITIAL_DRAFT,
            consecutive_failures=0,
        )
        assert llm.usage[ModelTier.FAST_LOCAL].calls == 1
        llm.reset_usage()
        assert llm.usage[ModelTier.FAST_LOCAL].calls == 0


class TestScarForgeEngineUnit:
    def test_self_healing_when_reused_skill_fails_and_patch_recovery(
        self, tmp_path: Path
    ) -> None:
        cfg = EngineConfig(data_dir=tmp_path)
        engine = ScarForgeEngine(config=cfg, provider="local")

        # Register a broken skill matching semver query that raises RuntimeError on reuse
        bad_semver = (
            "def sort_semver_tags(versions: list[str]) -> list[str]:\n"
            '    """Broken semver skill."""\n'
            "    if versions and '1.10.0' in versions:\n"
            "        raise RuntimeError('corrupted skill version')\n"
            "    return list(versions)\n"
        )
        engine.skill_registry.register_and_reload(
            bad_semver,
            "sort_semver_tags",
            "Sort semantic version tags numerically.",
            ["semver", "version", "sort"],
            [{"kwargs": {"versions": ["1.2.0"]}, "expected": ["1.2.0"]}],
        )

        # Engine invokes broken skill, catches failure, records Scar, and self-heals via RUN_CODE
        res = engine.run_task("Sort semantic version tags numerically with prerelease ordering")
        assert res.success is True
        assert res.errors_count >= 1
        assert len(res.scars_created) >= 1
        assert res.skill_forged == "sort_semver_tags"
        assert engine.skill_registry.invoke_active(
            "sort_semver_tags", versions=["1.10.0", "1.2.0"]
        ) == ["1.2.0", "1.10.0"]
