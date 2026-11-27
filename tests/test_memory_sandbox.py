"""Unit & edge-case tests for ScarStore, BM25Retriever, and SandboxRunner (M2)."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from scar_forge.config import EngineConfig
from scar_forge.memory.retriever import BM25Retriever, tokenize_for_bm25
from scar_forge.memory.scar_store import ScarStore
from scar_forge.models import DistilledError, ScarRecord, SkillRecord
from scar_forge.sandbox.runner import SandboxRunner
from scar_forge.tokens import count_tokens, validate_axiom_format


class TestScarStoreUnit:
    def test_wal_persistence_stats_and_clear(self, tmp_path: Path) -> None:
        cfg = EngineConfig(data_dir=tmp_path)
        store = ScarStore(cfg)
        e1 = DistilledError(
            exception_type="AttributeError",
            core_message="'DataFrame' object has no attribute 'append'",
            failing_line="df = df.append(row)",
            line_number=10,
        )
        e2 = DistilledError(
            exception_type="ZeroDivisionError",
            core_message="division by zero",
            failing_line="return total / count",
            line_number=4,
        )
        s1, c1 = store.record_failure(e1, "df = df.append(row)", ["pandas"])
        s2, c2 = store.record_failure(e2, "return total / count", ["math"])
        assert c1 is True and c2 is True
        assert count_tokens(s1.axiom) <= 14
        assert count_tokens(s2.axiom) <= 14
        assert validate_axiom_format(s1.axiom)
        assert validate_axiom_format(s2.axiom)

        assert store.get_scar(s1.scar_id) is not None
        assert store.get_scar("missing_id") is None

        assert store.mark_prevented([s1.scar_id, "does_not_exist"]) == 1
        assert store.mark_prevented([]) == 0
        stats = store.stats()
        assert stats["count"] == 2
        assert stats["total_prevented"] == 1

        mirrored = json.loads(cfg.scar_json_path.read_text(encoding="utf-8"))
        assert len(mirrored) == 2

        store.clear()
        assert store.list_scars() == []
        assert store.stats()["count"] == 0

    def test_multithreaded_wal_writes(self, tmp_path: Path) -> None:
        cfg = EngineConfig(data_dir=tmp_path)
        store = ScarStore(cfg)
        errors: list[Exception] = []

        def _worker(idx: int) -> None:
            try:
                err = DistilledError(
                    exception_type=f"WorkerError{idx}",
                    core_message=f"unique thread failure alpha_{idx} omega_{idx * 13}",
                    failing_line=f"worker_op_{idx}(payload)",
                    line_number=idx + 1,
                )
                store.record_failure(err, f"worker_op_{idx}(payload)", [f"t{idx}"])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(store.list_scars()) == 8


class TestBM25RetrieverUnit:
    def test_tokenization_splits_camel_and_snake(self) -> None:
        toks = tokenize_for_bm25("DataFrame_merge_rows PurePosixPath")
        assert "dataframe" in toks
        assert "data" in toks
        assert "frame" in toks
        assert "pure" in toks
        assert "posix" in toks

    def test_clamping_and_tie_breaking(self) -> None:
        retriever = BM25Retriever()
        scars = [
            ScarRecord(
                scar_id=f"s{i}",
                axiom=f"[AVOID] op_{i} -> [USE] safe_op_{i}",
                token_count=12,
                exception_type="ValueError",
                error_signature=f"sig_{i}",
                tags=["common", f"tag{i}"],
                prevented_count=i,
            )
            for i in range(8)
        ]
        top1 = retriever.retrieve_scars("common", scars, top_k=0)
        assert len(top1) == 1
        assert top1[0].scar_id == "s7"

        top5 = retriever.retrieve_scars("common", scars, top_k=50)
        assert len(top5) == 5

        assert retriever.retrieve_scars("", scars) == []
        assert retriever.retrieve_skills("anything", []) == []

        skills = [
            SkillRecord(
                skill_id="sk1",
                skill_name="semver_sort",
                signature="def semver_sort(v: list[str]) -> list[str]",
                docstring="Sort semantic version strings numerically.",
                file_path="semver_sort.py",
                source_hash="a" * 64,
                tags=["semver"],
            )
        ]
        assert len(retriever.retrieve_skills("unrelated xyz", skills)) == 0
        assert len(retriever.retrieve_skills("semver", skills)) == 1


class TestSandboxRunnerUnit:
    def test_rich_types_and_expected_error_cases(self, tmp_path: Path) -> None:
        runner = SandboxRunner(EngineConfig(data_dir=tmp_path))
        code = (
            "from datetime import datetime\n"
            "from decimal import Decimal\n"
            "import json\n"
            "import math\n\n"
            "def solve(payload: dict) -> str:\n"
            "    out = {}\n"
            "    for k, v in payload.items():\n"
            "        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):\n"
            "            raise ValueError('nan disallowed')\n"
            "        if isinstance(v, Decimal):\n"
            "            out[k] = str(v)\n"
            "        elif isinstance(v, datetime):\n"
            "            out[k] = v.isoformat()\n"
            "        elif isinstance(v, set):\n"
            "            out[k] = sorted(v)\n"
            "        else:\n"
            "            out[k] = v\n"
            "    return json.dumps(out, allow_nan=False, sort_keys=True)\n"
        )
        cases = [
            {
                "name": "rich_types",
                "kwargs": {
                    "payload": {
                        "price": Decimal("9.99"),
                        "ts": datetime(2026, 1, 2, tzinfo=UTC),
                        "tags": {"b", "a"},
                    }
                },
                "expected": (
                    '{"price": "9.99", "tags": ["a", "b"], '
                    '"ts": "2026-01-02T00:00:00+00:00"}'
                ),
            },
            {
                "name": "reject_nan",
                "kwargs": {"payload": {"x": float("nan")}},
                "expected_error": "ValueError",
            },
        ]
        res = runner.execute_code(code, cases, entry_fn="solve")
        assert res.passed is True
        assert res.passed_cases == 2

    def test_os_system_and_env_stripped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-999")
        runner = SandboxRunner(EngineConfig(data_dir=tmp_path))
        code_spawn = (
            "import os\n"
            "def run_os() -> int:\n"
            "    return os.system('echo hi')\n"
        )
        res_spawn = runner.execute_code(
            code_spawn, [{"kwargs": {}, "expected": 0}], entry_fn="run_os"
        )
        assert res_spawn.passed is False
        assert res_spawn.distilled_error is not None
        assert res_spawn.distilled_error.exception_type == "PermissionError"

        code_env = (
            "import os\n"
            "def check_env() -> bool:\n"
            "    return os.environ.get('GEMINI_API_KEY') is None\n"
        )
        res_env = runner.execute_code(
            code_env, [{"kwargs": {}, "expected": True}], entry_fn="check_env"
        )
        assert res_env.passed is True
        assert os.environ.get("GEMINI_API_KEY") == "secret-gemini-999"

    def test_import_guard_blocks_name_and_globals_bypass(self, tmp_path: Path) -> None:
        runner = SandboxRunner(EngineConfig(data_dir=tmp_path))
        for mod in ("socket", "ctypes", "shutil"):
            code = f"__name__ = 'innocent'\nimport {mod}\ndef pwn() -> int:\n    return 1\n"
            res = runner.execute_code(code, [{"kwargs": {}, "expected": 1}], entry_fn="pwn")
            assert res.passed is False

        g_bypass = "def pwn() -> str:\n    return __import__('socket', {}).__name__\n"
        assert (
            runner.execute_code(
                g_bypass, [{"kwargs": {}, "expected": "socket"}], entry_fn="pwn"
            ).passed
            is False
        )

        c_bypass = (
            "def pwn() -> str:\n"
            "    return __import__.__closure__[0].cell_contents('socket').__name__\n"
        )
        assert (
            runner.execute_code(
                c_bypass, [{"kwargs": {}, "expected": "socket"}], entry_fn="pwn"
            ).passed
            is False
        )

        pd_code = (
            "import pandas as pd\n"
            "def ok(x: int) -> int:\n"
            "    return int(pd.DataFrame([{'a': x}])['a'].iloc[0])\n"
        )
        assert (
            runner.execute_code(
                pd_code, [{"kwargs": {"x": 7}, "expected": 7}], entry_fn="ok"
            ).passed
            is True
        )

    def test_adversarial_unwraps_optional_dict_hint(self, tmp_path: Path) -> None:
        runner = SandboxRunner(EngineConfig(data_dir=tmp_path))
        src = (
            "from typing import Any, Optional\n\n"
            "def get_val(d: Optional[dict[str, Any]]) -> int:\n"
            '    """Read key v safely."""\n'
            "    if d is None:\n"
            "        return 0\n"
            "    return int(d.get('v', 0))\n"
        )
        res = runner.verify_skill_adversarial(
            src, "get_val", [{"kwargs": {"d": {"v": 4}}, "expected": 4}]
        )
        assert res.passed is True


class TestScarStoreRegressions:
    def test_low_token_budget_and_raise_exception_fallback(self, tmp_path: Path) -> None:
        store6 = ScarStore(EngineConfig(data_dir=tmp_path / "low", max_scar_tokens=6))
        err = DistilledError(
            exception_type="ValueError",
            core_message="bad int literal",
            failing_line="int(raw)",
            line_number=2,
        )
        scar, created = store6.record_failure(err, "int(raw)", ["parse"])
        assert created is True
        assert scar.token_count <= 6
        assert validate_axiom_format(scar.axiom, max_tokens=6)

        store14 = ScarStore(EngineConfig(data_dir=tmp_path / "std"))
        err_raise = DistilledError(
            exception_type="ValueError",
            core_message="negative port number",
            failing_line="raise ValueError('negative port')",
            line_number=5,
        )
        scar_r, _ = store14.record_failure(
            err_raise, "raise ValueError('negative port')", ["config"]
        )
        assert "guard ValueError ValueError" not in scar_r.axiom
        assert validate_axiom_format(scar_r.axiom)
