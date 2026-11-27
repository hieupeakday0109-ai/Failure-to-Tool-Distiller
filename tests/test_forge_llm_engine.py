"""Unit & edge-case tests for ASTGuard, SkillRegistry, DualTierLLM, and Engine (M3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from scar_forge.agent.engine import ScarForgeEngine
from scar_forge.config import EngineConfig
from scar_forge.forge.registry import SkillRegistry
from scar_forge.forge.sanitizer import ASTGuard
from scar_forge.llm.adapter import DualTierLLM
from scar_forge.llm.providers import call_ollama_chat, call_openai_chat
from scar_forge.models import LLMResponse, ModelTier, TaskPurpose


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

    def test_ast_guard_blocks_operator_pickle_and_allows_super_init(self) -> None:
        guard = ASTGuard()
        bypasses = [
            (
                "from operator import attrgetter\n"
                "def f(x: int) -> object:\n"
                '    """Doc."""\n'
                "    return attrgetter('__class__')(x)\n"
            ),
            (
                "import pandas as pd\n"
                "def f(p: str) -> object:\n"
                '    """Doc."""\n'
                "    return pd.read_pickle(p)\n"
            ),
            (
                "import numpy as np\n"
                "def f(p: str) -> object:\n"
                '    """Doc."""\n'
                "    return np.load(p, allow_pickle=True)\n"
            ),
            (
                "def f(x: str) -> type:\n"
                '    """Doc."""\n'
                "    return type('Evil', (), {'a': 1})\n"
            ),
            (
                "from math import *\n"
                "def f(x: float) -> float:\n"
                '    """Doc."""\n'
                "    return sqrt(x)\n"
            ),
        ]
        for src in bypasses:
            assert guard.validate(src).valid is False

        helper_cls = (
            "from html.parser import HTMLParser\n\n"
            "class CustomParser(HTMLParser):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n"
            "        self.items: list[str] = []\n\n"
            "def parse_txt(s: str) -> str:\n"
            '    """Parse text."""\n'
            "    p = CustomParser()\n"
            "    p.feed(s)\n"
            "    return s\n"
        )
        assert guard.validate(helper_cls, expected_name="parse_txt").valid is True

    def test_engine_rejects_unverified_final_answer_and_uses_adhoc_skill_args(
        self, tmp_path: Path
    ) -> None:
        eng = ScarForgeEngine(data_dir=tmp_path / "e1", provider="local")

        def _final_only(
            _msgs: Any,
            purpose: TaskPurpose = TaskPurpose.INITIAL_DRAFT,
            consecutive_failures: int = 0,
        ) -> LLMResponse:
            return LLMResponse(
                content='{"action": "FINAL_ANSWER", "result": "trust me"}',
                tier_used=ModelTier.FAST_LOCAL,
                purpose=purpose,
                prompt_tokens=10,
                completion_tokens=10,
                duration_ms=1.0,
            )

        eng.llm.complete = _final_only  # type: ignore[method-assign]
        res1 = eng.run_task("Merge list of record dicts into DataFrame")
        assert res1.success is False
        assert res1.errors_count == eng.config.max_attempts

        eng2 = ScarForgeEngine(data_dir=tmp_path / "e2", provider="local")
        eng2.skill_registry.register_and_reload(
            'def add_two(a: int, b: int) -> int:\n    """Add two integers."""\n    return a + b\n',
            "add_two",
            "Add two integers.",
            ["math", "add", "integers"],
            [{"kwargs": {"a": 2, "b": 3}, "expected": 5}],
        )

        def _call_skill(
            _msgs: Any,
            purpose: TaskPurpose = TaskPurpose.INITIAL_DRAFT,
            consecutive_failures: int = 0,
        ) -> LLMResponse:
            return LLMResponse(
                content='{"action": "CALL_SKILL", "skill_name": "add_two", '
                '"skill_args": {"a": 10, "b": 20}}',
                tier_used=ModelTier.FAST_LOCAL,
                purpose=purpose,
                prompt_tokens=10,
                completion_tokens=10,
                duration_ms=1.0,
            )

        eng2.llm.complete = _call_skill  # type: ignore[method-assign]
        res2 = eng2.run_task("Add two integers a and b")
        assert res2.success is True
        assert res2.skill_reused == "add_two"

    def test_provider_http_and_ollama_frontier_routing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen_models: list[str] = []

        def _fake_post(url: str, json: dict[str, Any], **_kw: Any) -> httpx.Response:
            seen_models.append(str(json.get("model")))
            if url.endswith("/api/chat"):
                body = {"message": {"content": '{"action":"FINAL_ANSWER","result":"ok"}'}}
            else:
                body = {
                    "choices": [
                        {"message": {"content": '{"action":"FINAL_ANSWER","result":"ok"}'}}
                    ]
                }
            return httpx.Response(200, json=body, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "post", _fake_post)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        cfg = EngineConfig(
            data_dir=tmp_path,
            provider="ollama",
            ollama_fast_model="fast-7b",
            ollama_frontier_model="frontier-32b",
            escalation_threshold=2,
        )
        llm = DualTierLLM(cfg, provider="ollama")
        r_fast = llm.complete(
            [{"role": "user", "content": "hi"}],
            TaskPurpose.INITIAL_DRAFT,
            0,
        )
        r_front = llm.complete(
            [{"role": "user", "content": "hi"}],
            TaskPurpose.SKILL_DISTILLATION,
            0,
        )
        assert r_fast.model_name == "fast-7b"
        assert r_front.model_name == "frontier-32b"
        assert seen_models == ["fast-7b", "frontier-32b"]
        assert "FINAL_ANSWER" in call_ollama_chat(
            [{"role": "user", "content": "hi"}], model="fast-7b", config=cfg
        )
        assert "FINAL_ANSWER" in call_openai_chat(
            [{"role": "user", "content": "hi"}], model="gpt-4o", config=cfg
        )
