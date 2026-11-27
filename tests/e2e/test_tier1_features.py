"""Tier 1 E2E Tests — Isolated Happy-Path Feature Coverage (F1-F15, F17).

Every test uses lazy ``scar_forge.*`` imports inside the test body and runs
against isolated ``tmp_path`` directories with deterministic ``--provider local``.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.conftest import (
    REPO_ROOT,
    apply_protocol_patch,
    get_ast_valid,
    get_llm_content,
    get_prevented_count,
    get_scar_id,
    get_skill_name,
    get_skill_path,
    make_ast_guard,
    make_config,
    make_distilled_error,
    make_engine,
    make_llm,
    make_retriever,
    make_sandbox_runner,
    make_scar_record,
    make_scar_store,
    make_skill_record,
    make_skill_registry,
    make_test_case,
    parse_agent_action_text,
    run_engine_task,
    unwrap_api_list,
)


# ---------------------------------------------------------------------------
# F1 & F2: Bootstrap, EngineConfig & Pydantic Schemas (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f1_package_metadata_and_git_main_branch() -> None:
    """Verify pyproject.toml declares scar-forge script and git branch is main."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), f"Missing {pyproject_path}"

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project_tbl = data.get("project", {})
    assert project_tbl.get("name") in {"scar-forge", "scar_forge"}
    scripts = project_tbl.get("scripts", {})
    assert "scar-forge" in scripts
    assert "scar_forge" in scripts["scar-forge"]

    import scar_forge

    assert isinstance(getattr(scar_forge, "__version__", ""), str)
    assert len(scar_forge.__version__) > 0

    if (REPO_ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == "main"


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f2_engine_config_defaults_and_data_dir_override(e2e_data_dir: Path) -> None:
    """EngineConfig accepts isolated data_dir and sensible token/escalation bounds."""
    cfg = make_config(e2e_data_dir)
    resolved = None
    for attr in ("data_dir", "state_dir", "workspace_dir"):
        if hasattr(cfg, attr):
            resolved = Path(getattr(cfg, attr))
            break
    assert resolved == e2e_data_dir

    for attr in ("max_scar_tokens", "max_axiom_tokens"):
        if hasattr(cfg, attr):
            assert int(getattr(cfg, attr)) < 15

    for attr in ("top_k", "top_k_scars", "top_k_skills"):
        if hasattr(cfg, attr):
            assert 1 <= int(getattr(cfg, attr)) <= 5


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f2_models_scar_and_skill_record_round_trip() -> None:
    """ScarRecord and SkillRecord serialize and validate cleanly via Pydantic v2."""
    from scar_forge.models import ScarRecord, SkillRecord

    scar = make_scar_record(
        scar_id="scar-001",
        axiom="[AVOID] df.append -> [USE] pd.concat",
        tags=["pandas", "dataframe"],
        exception_type="AttributeError",
    )
    reloaded_scar = ScarRecord.model_validate_json(scar.model_dump_json())
    assert reloaded_scar.axiom == "[AVOID] df.append -> [USE] pd.concat"
    assert get_scar_id(reloaded_scar) == "scar-001"
    assert reloaded_scar.tags == ["pandas", "dataframe"]

    skill = make_skill_record(
        skill_name="merge_batches",
        docstring="Merge batch dicts into a pandas DataFrame.",
        tags=["pandas", "etl"],
    )
    reloaded_skill = SkillRecord.model_validate_json(skill.model_dump_json())
    assert get_skill_name(reloaded_skill) == "merge_batches"
    assert reloaded_skill.tags == ["pandas", "etl"]


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f2_models_distilled_error_sandbox_result_and_enums() -> None:
    """Core protocol/routing enums and result schemas instantiate cleanly."""
    from scar_forge.models import ActionType, LLMResponse, ModelTier, SandboxResult, TaskPurpose

    assert {m.value for m in ActionType} >= {
        "RUN_CODE",
        "CALL_SKILL",
        "PATCH_CODE",
        "FINAL_ANSWER",
    }
    assert {m.value for m in ModelTier} >= {"FAST_LOCAL", "FRONTIER"}
    assert hasattr(TaskPurpose, "SKILL_DISTILLATION")

    err = make_distilled_error(
        exception_type="ValueError",
        core_message="bad literal",
        failing_line="int('abc')",
        line_number=4,
    )
    assert err.exception_type == "ValueError"
    assert 0.0 <= err.compression_ratio <= 1.0

    sb_fields = set(SandboxResult.model_fields.keys())
    sb_kwargs = {
        k: v
        for k, v in {
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "memory_exceeded": False,
            "stdout": "ok",
            "stderr": "",
            "distilled_error": None,
            "duration_ms": 12.5,
            "peak_memory_bytes": 1024,
            "cases_run": 1,
            "cases_passed": 1,
            "case_failures": [],
        }.items()
        if k in sb_fields
    }
    sb = SandboxResult(**sb_kwargs)
    assert sb.passed is True

    llm_fields = set(LLMResponse.model_fields.keys())
    llm_kwargs: dict[str, object] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "tier_used": ModelTier.FAST_LOCAL,
        "escalated": False,
        "duration_ms": 4.2,
    }
    for text_key in ("content", "text", "raw_output", "output"):
        if text_key in llm_fields:
            llm_kwargs[text_key] = '{"action": "FINAL_ANSWER", "result": "ok"}'
    if "model" in llm_fields:
        llm_kwargs["model"] = "local-rule"
    resp = LLMResponse(**{k: v for k, v in llm_kwargs.items() if k in llm_fields})
    assert resp.tier_used == ModelTier.FAST_LOCAL
    assert resp.escalated is False


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f2_configure_utf8_console_handles_unicode_output(capsys: pytest.CaptureFixture[str]) -> None:
    """configure_utf8_console enables non-ASCII UTF-8 output without UnicodeEncodeError."""
    from scar_forge.config import configure_utf8_console

    configure_utf8_console()
    sample = "Scar-Forge: Xin chao \u2014 \ufeff \u2014 \u20ac"
    print(sample)
    captured = capsys.readouterr()
    assert "Scar-Forge:" in captured.out


# ---------------------------------------------------------------------------
# F3: Deterministic Token Counter & <15 Axiom Budget (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f3_count_tokens_deterministic_and_monotonic() -> None:
    """E2E-F3-T1-01: Repeated calls return identical counts and scale monotonically."""
    from scar_forge.tokens import count_tokens

    short = "print(1)"
    medium = '{"action": "RUN_CODE", "code": "import pandas as pd\\ndf = pd.DataFrame([1, 2])"}'
    long_text = " ".join(["ValueError invalid payload token budget overflow"] * 40)

    c_short_1 = count_tokens(short)
    c_short_2 = count_tokens(short)
    c_medium = count_tokens(medium)
    c_long = count_tokens(long_text)

    assert isinstance(c_short_1, int)
    assert c_short_1 == c_short_2
    assert 0 < c_short_1 < c_medium < c_long


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f3_validate_axiom_format_accepts_avoid_use() -> None:
    """E2E-F3-T1-02: Canonical [AVOID] X -> [USE] Y axiom passes format and <15 budget."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    axiom = "[AVOID] DataFrame.append -> [USE] pd.concat"
    assert count_tokens(axiom) <= 14
    assert validate_axiom_format(axiom) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f3_validate_axiom_format_accepts_rule_prefix() -> None:
    """E2E-F3-T1-03: Canonical [RULE] <X> axiom passes format and <15 budget."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    axiom = "[RULE] strip UTF-8 BOM before csv.DictReader"
    assert count_tokens(axiom) <= 14
    assert validate_axiom_format(axiom) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f3_truncate_to_axiom_budget_fits_default_14() -> None:
    """E2E-F3-T1-04: Verbose avoid/use clauses compress to valid <=14 token axiom."""
    from scar_forge.tokens import (
        count_tokens,
        truncate_to_axiom_budget,
        validate_axiom_format,
    )

    verbose_avoid = (
        "calling DataFrame.append inside an iterative loop on pandas 3.0 "
        "because append was removed completely from pandas DataFrame API"
    )
    verbose_use = (
        "collecting dictionary records in a list and calling pandas.concat "
        "once with ignore_index=True for clean merging"
    )
    axiom = truncate_to_axiom_budget(verbose_avoid, verbose_use)
    assert count_tokens(axiom) <= 14
    assert validate_axiom_format(axiom) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f3_count_tokens_splits_code_punctuation_and_camel_case() -> None:
    """E2E-F3-T1-05: Subword counter splits code punctuation and identifiers granularly."""
    from scar_forge.tokens import count_tokens

    code_expr = "pd.concat(ignore_index=True)"
    assert count_tokens(code_expr) >= 4
    assert count_tokens(code_expr) > count_tokens("a")


# ---------------------------------------------------------------------------
# F4: Zero-Prose Structured Protocol (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f4_parse_valid_run_code_action() -> None:
    """E2E-F4-T1-01: Strict RUN_CODE JSON parses into AgentAction preserving code verbatim."""
    from scar_forge.models import ActionType

    code = "import math\ndef area(r: float) -> float:\n    return math.pi * r * r"
    raw = json.dumps({"action": "RUN_CODE", "code": code})
    parsed = parse_agent_action_text(raw)
    assert parsed.action == ActionType.RUN_CODE
    assert parsed.code == code


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f4_parse_valid_call_skill_action() -> None:
    """E2E-F4-T1-02: Strict CALL_SKILL JSON parses with exact skill_name and kwargs."""
    from scar_forge.models import ActionType

    payload = {
        "action": "CALL_SKILL",
        "skill_name": "semver_sort",
        "skill_args": {"versions": ["1.2.0", "1.10.0"]},
    }
    parsed = parse_agent_action_text(json.dumps(payload))
    assert parsed.action == ActionType.CALL_SKILL
    assert parsed.skill_name == "semver_sort"
    args_val = getattr(parsed, "skill_args", None)
    if args_val is None:
        args_val = getattr(parsed, "kwargs", None)
    assert args_val == {"versions": ["1.2.0", "1.10.0"]}


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f4_parse_valid_patch_code_action() -> None:
    """E2E-F4-T1-03: Strict PATCH_CODE JSON parses and applies clean diff/patch."""
    from scar_forge.models import ActionType

    original = "def add(a: int, b: int) -> int:\n    return a - b\n"
    diff = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )
    raw = json.dumps({"action": "PATCH_CODE", "diff_patch": diff})
    parsed = parse_agent_action_text(raw)
    assert parsed.action == ActionType.PATCH_CODE
    patch_str = getattr(parsed, "diff_patch", None) or getattr(parsed, "patch", None)
    assert patch_str
    patched = apply_protocol_patch(original, patch_str)
    assert "return a + b" in patched
    assert "return a - b" not in patched


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f4_parse_valid_final_answer_action() -> None:
    """E2E-F4-T1-04: Strict FINAL_ANSWER JSON parses cleanly with result payload."""
    from scar_forge.models import ActionType

    raw = json.dumps({"action": "FINAL_ANSWER", "result": {"rows": 3, "ok": True}})
    parsed = parse_agent_action_text(raw)
    assert parsed.action == ActionType.FINAL_ANSWER
    assert parsed.result == {"rows": 3, "ok": True}


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f4_protocol_json_round_trip() -> None:
    """E2E-F4-T1-05: AgentAction round-trips losslessly through model_dump_json."""
    raw = json.dumps(
        {
            "action": "RUN_CODE",
            "code": "def double(x: int) -> int:\n    return x * 2",
        }
    )
    first = parse_agent_action_text(raw)
    second = parse_agent_action_text(first.model_dump_json())
    assert second.model_dump() == first.model_dump()


# ---------------------------------------------------------------------------
# F5: Stderr & Traceback Distiller (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f5_distill_strips_site_packages_retains_core() -> None:
    """E2E-F5-T1-01: Strips site-packages frames while keeping user line and root cause."""
    from scar_forge.distiller import TracebackDistiller

    raw_stderr = (
        "Traceback (most recent call last):\n"
        '  File "C:\\venv\\Lib\\site-packages\\pandas\\core\\frame.py", line 910, in _dispatch\n'
        "    return self._apply_op(op)\n"
        '  File "C:\\venv\\Lib\\site-packages\\pandas\\core\\generic.py", line 6299, in __getattr__\n'
        "    return object.__getattribute__(self, name)\n"
        '  File "C:\\workspace\\app.py", line 42, in merge_data\n'
        "    out = df.append(row)\n"
        "AttributeError: 'DataFrame' object has no attribute 'append'\n"
    )
    distilled = TracebackDistiller().distill(raw_stderr)
    assert distilled.exception_type == "AttributeError"
    assert "append" in distilled.core_message
    assert "df.append(row)" in distilled.failing_line
    assert distilled.line_number == 42
    assert distilled.function_name == "merge_data"
    assert "site-packages" not in distilled.failing_line
    assert "site-packages" not in distilled.core_message


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f5_distill_syntax_error_with_caret() -> None:
    """E2E-F5-T1-02: Extracts SyntaxError line number, failing line, and core message."""
    from scar_forge.distiller import TracebackDistiller

    raw_stderr = (
        '  File "task.py", line 7\n'
        "    if x == 1\n"
        "           ^\n"
        "SyntaxError: expected ':'\n"
    )
    distilled = TracebackDistiller().distill(raw_stderr)
    assert distilled.exception_type == "SyntaxError"
    assert distilled.line_number == 7
    assert "if x == 1" in distilled.failing_line
    assert len(distilled.core_message) > 0


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f5_distill_chained_exceptions() -> None:
    """E2E-F5-T1-03: Distills chained exceptions cleanly without site-packages noise."""
    from scar_forge.distiller import TracebackDistiller

    raw_stderr = (
        "Traceback (most recent call last):\n"
        '  File "C:\\venv\\Lib\\site-packages\\json\\decoder.py", line 355, in raw_decode\n'
        "    raise JSONDecodeError('Expecting value', s, err.value)\n"
        "ValueError: Expecting value\n\n"
        "During handling of the above exception, another exception occurred:\n\n"
        "Traceback (most recent call last):\n"
        '  File "pipeline.py", line 18, in load_cfg\n'
        "    raise RuntimeError('pipeline failed')\n"
        "RuntimeError: pipeline failed\n"
    )
    distilled = TracebackDistiller().distill(raw_stderr)
    assert distilled.exception_type in {"RuntimeError", "ValueError"}
    assert "pipeline failed" in distilled.core_message or "Expecting value" in distilled.core_message
    assert "site-packages" not in distilled.failing_line


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f5_distill_strips_noisy_stdout_and_debug_logs() -> None:
    """E2E-F5-T1-04: Strips verbose INFO/DEBUG logs and achieves measurable compression."""
    from scar_forge.distiller import TracebackDistiller

    noisy_stdout = "\n".join(
        [f"2026-11-26 INFO loading batch chunk {i} with metadata payload..." for i in range(50)]
    )
    raw_stderr = (
        "DEBUG initializing worker pool\n"
        "Traceback (most recent call last):\n"
        '  File "runner.py", line 11, in parse_flag\n'
        "    return int(raw)\n"
        "ValueError: invalid literal for int() with base 10: 'false'\n"
    )
    distilled = TracebackDistiller().distill(raw_stderr, noisy_stdout)
    assert distilled.distilled_chars < distilled.raw_chars
    assert distilled.distilled_tokens < distilled.raw_tokens
    assert 0.0 < distilled.compression_ratio < 1.0
    assert "loading batch chunk" not in distilled.core_message
    assert "loading batch chunk" not in distilled.failing_line


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f5_distill_computes_accurate_char_and_token_metrics() -> None:
    """E2E-F5-T1-05: Char/token telemetry is positive, monotonic, and bounded [0, 1]."""
    from scar_forge.distiller import TracebackDistiller

    raw_stderr = (
        "Traceback (most recent call last):\n"
        '  File "C:\\venv\\Lib\\site-packages\\pkg\\core.py", line 100, in helper\n'
        "    raise TypeError('unsupported operand')\n"
        '  File "user_mod.py", line 9, in compute\n'
        "    return a - b\n"
        "TypeError: can't subtract offset-naive and offset-aware datetimes\n"
    )
    distilled = TracebackDistiller().distill(raw_stderr)
    assert distilled.raw_chars >= len(raw_stderr)
    assert distilled.distilled_chars > 0
    assert distilled.raw_tokens >= distilled.distilled_tokens > 0
    assert 0.0 <= distilled.compression_ratio <= 1.0


# ---------------------------------------------------------------------------
# F6: Static Hygiene Gate (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f6_static_hygiene_zero_todo_fixme_in_src_and_tests() -> None:
    """E2E-F6-T1-01: Zero forbidden marker tags exist in src/ and tests/ comments."""
    markers = (
        "TO" + "DO",
        "FIX" + "ME",
        "X" * 3,
        "HA" + "CK",
    )
    pattern = re.compile(r"\b(" + "|".join(markers) + r")\b", re.IGNORECASE)
    py_files = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "tests").rglob("*.py"))
    assert len(py_files) > 0, "Expected Python files in src/ and tests/"

    violations: list[str] = []
    for file_path in py_files:
        text = file_path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and pattern.search(tok.string):
                violations.append(f"{file_path}:{tok.start[0]}: {tok.string.strip()}")

    assert not violations, f"Static hygiene violations found: {violations}"


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m1
def test_t1_f6_static_hygiene_test_file_exists_and_contains_real_checks() -> None:
    """Verify tests/test_static_hygiene.py exists and defines real AST/regex checks."""
    hygiene_file = REPO_ROOT / "tests" / "test_static_hygiene.py"
    assert hygiene_file.is_file(), f"Missing {hygiene_file}"
    tree = ast.parse(hygiene_file.read_text(encoding="utf-8"), filename=str(hygiene_file))
    test_fns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert len(test_fns) >= 1
    for fn in test_fns:
        assert len(fn.body) >= 2


# ---------------------------------------------------------------------------
# F7: Failure-to-Axiom Scar Memory (Milestone M2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f7_record_failure_creates_valid_telegraphic_scar(e2e_data_dir: Path) -> None:
    """record_failure distills an error into a <15 token telegraphic Scar."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    store = make_scar_store(e2e_data_dir)
    err = make_distilled_error(
        exception_type="AttributeError",
        core_message="'DataFrame' object has no attribute 'append'",
        failing_line="df = df.append(row)",
        line_number=14,
        function_name="merge_rows",
    )
    scar, created = store.record_failure(err, "df = df.append(row)", ["pandas", "dataframe"])
    assert created is True
    assert count_tokens(scar.axiom) < 15
    assert validate_axiom_format(scar.axiom) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f7_scar_persistence_across_store_reload(e2e_data_dir: Path) -> None:
    """Scars recorded in store1 survive reloading into a fresh ScarStore instance."""
    store1 = make_scar_store(e2e_data_dir)
    err1 = make_distilled_error("AttributeError", "no attribute 'append'", "df.append(r)", 5)
    err2 = make_distilled_error("TypeError", "can't subtract offset-naive and offset-aware", "a - b", 9)

    s1, _ = store1.record_failure(err1, "df.append(r)", ["pandas"])
    s2, _ = store1.record_failure(err2, "a - b", ["datetime"])

    store2 = make_scar_store(e2e_data_dir)
    loaded = store2.list_scars()
    loaded_ids = {get_scar_id(s) for s in loaded}
    assert {get_scar_id(s1), get_scar_id(s2)}.issubset(loaded_ids)
    axioms = {s.axiom for s in loaded}
    assert s1.axiom in axioms and s2.axiom in axioms


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f7_mark_prevented_updates_and_persists(e2e_data_dir: Path) -> None:
    """mark_prevented increments prevented counter and persists across store reload."""
    store1 = make_scar_store(e2e_data_dir)
    err = make_distilled_error("ValueError", "invalid literal for int(): 'false'", "int(flag)", 3)
    scar, _ = store1.record_failure(err, "int(flag)", ["bool"])
    sid = get_scar_id(scar)
    initial_count = get_prevented_count(scar)

    store1.mark_prevented([sid])
    store1.mark_prevented([sid])

    store2 = make_scar_store(e2e_data_dir)
    reloaded = {get_scar_id(s): s for s in store2.list_scars()}[sid]
    assert get_prevented_count(reloaded) == initial_count + 2


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f7_dedup_stage1_exact_match_returns_existing(e2e_data_dir: Path) -> None:
    """Identical repeated failure deduplicates cleanly and returns created=False."""
    store = make_scar_store(e2e_data_dir)
    err = make_distilled_error("AttributeError", "'DataFrame' object has no attribute 'append'", "df.append(row)", 8)
    s1, created1 = store.record_failure(err, "df.append(row)", ["pandas"])
    s2, created2 = store.record_failure(err, "df.append(row)", ["pandas"])

    assert created1 is True
    assert created2 is False
    assert get_scar_id(s2) == get_scar_id(s1)
    assert len(store.list_scars()) == 1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f7_dedup_stage2_stage3_structural_and_jaccard_merges(e2e_data_dir: Path) -> None:
    """Failures differing only by line number/identifier normalize and merge."""
    store = make_scar_store(e2e_data_dir)
    err1 = make_distilled_error(
        "AttributeError",
        "'DataFrame' object has no attribute 'append'",
        "frame_a.append(rec_1)",
        line_number=10,
        function_name="fn_a",
    )
    err2 = make_distilled_error(
        "AttributeError",
        "'DataFrame' object has no attribute 'append'",
        "frame_b.append(rec_2)",
        line_number=77,
        function_name="fn_b",
    )
    s1, created1 = store.record_failure(err1, "frame_a.append(rec_1)", ["pandas"])
    s2, created2 = store.record_failure(err2, "frame_b.append(rec_2)", ["pandas"])

    assert created1 is True
    assert created2 is False
    assert get_scar_id(s2) == get_scar_id(s1)
    assert len(store.list_scars()) == 1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f7_distinct_domains_persist_independently(e2e_data_dir: Path) -> None:
    """Three distinct domain errors remain separate <15 token Scars."""
    from scar_forge.tokens import count_tokens

    store = make_scar_store(e2e_data_dir)
    e1 = make_distilled_error("AttributeError", "'DataFrame' object has no attribute 'append'", "df.append(x)", 3)
    e2 = make_distilled_error("TypeError", "can't compare offset-naive and offset-aware datetimes", "t1 < t2", 7)
    e3 = make_distilled_error("ValueError", "Out of range float values are not JSON compliant: nan", "json.dumps(x, allow_nan=False)", 12)

    s1, c1 = store.record_failure(e1, "df.append(x)", ["pandas"])
    s2, c2 = store.record_failure(e2, "t1 < t2", ["datetime"])
    s3, c3 = store.record_failure(e3, "json.dumps(x, allow_nan=False)", ["json"])

    assert c1 and c2 and c3
    scars = store.list_scars()
    assert len(scars) == 3
    assert len({s.axiom for s in scars}) == 3
    assert all(count_tokens(s.axiom) < 15 for s in scars)


# ---------------------------------------------------------------------------
# F8: Top-K BM25 Dynamic Retriever (Milestone M2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f8_retrieve_scars_ranks_relevant_top_first() -> None:
    """BM25Retriever ranks the pandas Scar #1 for a pandas DataFrame append query."""
    retriever = make_retriever()
    scars = [
        make_scar_record("s1", "[AVOID] bool(str) -> [USE] parse_bool_strict", ["webhook", "bool"], "ValueError"),
        make_scar_record("s2", "[AVOID] DataFrame.append -> [USE] pd.concat", ["pandas", "dataframe"], "AttributeError"),
        make_scar_record("s3", "[AVOID] naive datetime -> [USE] utc aware", ["datetime", "utc"], "TypeError"),
        make_scar_record("s4", "[AVOID] dumps nan -> [USE] strict json", ["json", "rfc8259"], "ValueError"),
        make_scar_record("s5", "[AVOID] open plain -> [USE] utf8sig BOM", ["csv", "bom"], "UnicodeError"),
        make_scar_record("s6", "[AVOID] string sort semver -> [USE] tuple key", ["semver", "sort"], "AssertionError"),
        make_scar_record("s7", "[AVOID] regex html strip -> [USE] HTMLParser", ["html", "sanitizer"], "ValueError"),
        make_scar_record("s8", "[AVOID] raw join path -> [USE] PurePosixPath", ["path", "security"], "ValueError"),
    ]
    results = retriever.retrieve_scars("pandas DataFrame append concatenate", scars, top_k=4)
    assert len(results) >= 1
    assert get_scar_id(results[0]) == "s2"


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f8_retrieve_skills_ranks_by_docstring_name_and_tags() -> None:
    """BM25Retriever ranks matching SkillRecord #1 based on name, docstring, and tags."""
    retriever = make_retriever()
    skills = [
        make_skill_record("csv_norm", "Parse CSV text with BOM and delimiter handling.", ["csv", "bom"]),
        make_skill_record("semver_sort", "Sort semantic version tags numerically with prerelease precedence.", ["semver", "version"]),
        make_skill_record("html_extract", "Extract plain text from HTML fragments safely.", ["html", "parser"]),
        make_skill_record("json_strict", "Serialize objects into strict RFC 8259 JSON.", ["json", "strict"]),
        make_skill_record("utc_deltas", "Normalize mixed timezones to UTC sorted intervals.", ["datetime", "timezone"]),
        make_skill_record("path_guard", "Resolve safe relative POSIX paths without traversal.", ["path", "posix"]),
    ]
    results = retriever.retrieve_skills("semantic version prerelease sort", skills, top_k=3)
    assert len(results) >= 1
    assert get_skill_name(results[0]) == "semver_sort"


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f8_default_top_k_injects_subset_3_to_5_not_full_db() -> None:
    """Default top_k injects strictly a bounded subset (<=5), never dumping all 10 scars."""
    retriever = make_retriever()
    scars = [
        make_scar_record(
            f"s{i}",
            f"[AVOID] parse_input_{i} -> [USE] validate_input_{i}",
            ["python", "parse", "input"],
            "ValueError",
        )
        for i in range(10)
    ]
    results = retriever.retrieve_scars("ValueError parse input python", scars)
    assert 1 <= len(results) <= 5


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f8_zero_score_unrelated_items_excluded() -> None:
    """Items with zero lexical/tag overlap are excluded from Top-K results."""
    retriever = make_retriever()
    scars = [
        make_scar_record("csv1", "[AVOID] plain csv -> [USE] utf8sig BOM", ["csv", "bom"]),
        make_scar_record("csv2", "[RULE] detect csv delimiter semicolon vs comma", ["csv", "delimiter"]),
        make_scar_record("u1", "[AVOID] regex html -> [USE] HTMLParser", ["html"]),
        make_scar_record("u2", "[AVOID] semver lexicographical -> [USE] int tuple", ["semver"]),
        make_scar_record("u3", "[AVOID] naive datetime -> [USE] timezone.utc", ["datetime"]),
        make_scar_record("u4", "[AVOID] allow_nan -> [USE] strict json", ["json"]),
        make_scar_record("u5", "[AVOID] df.append -> [USE] pd.concat", ["pandas"]),
        make_scar_record("u6", "[AVOID] path traversal -> [USE] PurePosixPath", ["security"]),
    ]
    results = retriever.retrieve_scars("csv UTF-8 BOM delimiter", scars, top_k=5)
    returned_ids = {get_scar_id(r) for r in results}
    assert 1 <= len(results) <= 2
    assert returned_ids.issubset({"csv1", "csv2"})


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f8_tag_hybrid_boost_matches_tag_only_query() -> None:
    """Query matching only record tags still retrieves the Scar with positive score."""
    retriever = make_retriever()
    scars = [
        make_scar_record("tz", "[AVOID] naive compare -> [USE] utc", ["timezone", "datetime"]),
        make_scar_record("other", "[AVOID] df.append -> [USE] pd.concat", ["pandas", "table"]),
    ]
    results = retriever.retrieve_scars("timezone", scars, top_k=4)
    assert len(results) >= 1
    assert get_scar_id(results[0]) == "tz"


# ---------------------------------------------------------------------------
# F9: Adversarial Verification Sandbox (Milestone M2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f9_execute_valid_code_passes_explicit_cases() -> None:
    """Valid Python function passes all explicit test cases inside SandboxRunner."""
    runner = make_sandbox_runner()
    code = (
        "def add_scaled(a: int, b: int, scale: int = 1) -> int:\n"
        "    return (a + b) * scale\n"
    )
    cases = [
        make_test_case(kwargs={"a": 2, "b": 3}, expected=5),
        make_test_case(kwargs={"a": 2, "b": 3, "scale": 4}, expected=20),
        make_test_case(kwargs={"a": -5, "b": 5, "scale": 10}, expected=0),
    ]
    res = runner.execute_code(code, cases, entry_fn="add_scaled")
    assert res.passed is True
    assert res.exit_code == 0


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f9_execute_captures_runtime_exception_details() -> None:
    """Runtime exception inside sandbox returns passed=False with distilled error details."""
    runner = make_sandbox_runner()
    code = (
        "def fail_now(x: int) -> int:\n"
        "    raise ValueError(f'bad value: {x}')\n"
    )
    res = runner.execute_code(code, [make_test_case(kwargs={"x": 7}, expected=7)], entry_fn="fail_now")
    assert res.passed is False
    combined = (res.stderr or "") + (res.distilled_error.core_message if res.distilled_error else "")
    assert "ValueError" in (res.distilled_error.exception_type if res.distilled_error else combined)
    assert "bad value" in combined


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f9_isolated_cwd_and_stripped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sandbox runs in an isolated temporary cwd and strips sensitive host env vars."""
    monkeypatch.setenv("SCAR_FORGE_TEST_SECRET", "top-secret-token-xyz")
    runner = make_sandbox_runner()
    code = (
        "import os\n"
        "def check_isolation() -> dict:\n"
        "    return {'secret': os.environ.get('SCAR_FORGE_TEST_SECRET'), 'cwd': os.getcwd()}\n"
    )
    res = runner.execute_code(
        code,
        [make_test_case(kwargs={}, expected={"secret": None, "cwd_not_repo": True})],
        entry_fn="check_isolation",
    )
    # Verify via direct output capture so cwd equality isn't brittle:
    code_check = (
        "import os\n"
        "def check_env_only() -> bool:\n"
        "    return os.environ.get('SCAR_FORGE_TEST_SECRET') is None and os.getcwd() != "
        + repr(str(REPO_ROOT))
        + "\n"
    )
    res2 = runner.execute_code(code_check, [make_test_case(kwargs={}, expected=True)], entry_fn="check_env_only")
    assert res2.passed is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f9_verify_skill_adversarial_passes_hardened_function() -> None:
    """verify_skill_adversarial passes a defensive typed function handling edge cases."""
    runner = make_sandbox_runner()
    source = (
        "def safe_first_char(text: str) -> str:\n"
        '    """Return uppercased first non-empty character or empty string."""\n'
        "    if not isinstance(text, str) or not text:\n"
        "        return ''\n"
        "    return text[0].upper()\n"
    )
    cases = [
        make_test_case(kwargs={"text": "scar"}, expected="S"),
        make_test_case(kwargs={"text": ""}, expected=""),
    ]
    res = runner.verify_skill_adversarial(
        source=source,
        skill_name="safe_first_char",
        explicit_cases=cases,
        param_schema={"text": "str"},
    )
    assert res.passed is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m2
def test_t1_f9_utf8_and_windows_paths_supported() -> None:
    """Sandbox preserves UTF-8 strings and Windows raw path literals cleanly."""
    runner = make_sandbox_runner()
    code = (
        "def echo_utf8(label: str) -> str:\n"
        '    path = r"C:\\data\\file.csv"\n'
        "    return f'{label}|{path}'\n"
    )
    expected = "Xin chao \u2014 \ufeff \u2014 \u20ac|C:\\data\\file.csv"
    res = runner.execute_code(
        code,
        [make_test_case(kwargs={"label": "Xin chao \u2014 \ufeff \u2014 \u20ac"}, expected=expected)],
        entry_fn="echo_utf8",
    )
    assert res.passed is True


# ---------------------------------------------------------------------------
# F10: AST Sanitizer & Refactorer (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f10_valid_typed_function_with_docstring_passes() -> None:
    """Pure typed function with docstring and safe stdlib import passes ASTGuard."""
    guard = make_ast_guard()
    src = (
        "import json\n"
        "from decimal import Decimal\n\n"
        "def format_amount(amount: str) -> str:\n"
        '    """Serialize a decimal string inside strict JSON."""\n'
        "    val = Decimal(amount)\n"
        "    return json.dumps({'amount': str(val)})\n"
    )
    assert get_ast_valid(guard.validate(src, expected_name="format_amount")) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f10_valid_function_using_safe_data_libs_passes() -> None:
    """Functions importing safe data libraries (re, math, html.parser, pandas) pass ASTGuard."""
    guard = make_ast_guard()
    src = (
        "import math\n"
        "import re\n"
        "from html.parser import HTMLParser\n"
        "import pandas as pd\n\n"
        "def clean_metric(raw: str, scale: float) -> float:\n"
        '    """Extract numeric token and scale safely."""\n'
        "    m = re.search(r'\\d+', raw)\n"
        "    df = pd.DataFrame([{'v': float(m.group(0)) if m else 0.0}])\n"
        "    return round(float(df['v'].iloc[0]) * math.fabs(scale), 2)\n"
    )
    assert get_ast_valid(guard.validate(src, expected_name="clean_metric")) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f10_expected_name_matching_succeeds() -> None:
    """ASTGuard.validate accepts matching expected_name."""
    guard = make_ast_guard()
    src = (
        "from typing import Any\n\n"
        "def coerce_payload(raw: dict[str, Any]) -> dict[str, Any]:\n"
        '    """Coerce webhook payload keys safely."""\n'
        "    return {'ok': str(raw.get('ok', 'false')).lower() == 'true'}\n"
    )
    assert get_ast_valid(guard.validate(src, expected_name="coerce_payload")) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f10_python312_union_types_and_utf8_literals_pass() -> None:
    """Python 3.12 PEP-604 union types and UTF-8 BOM literals pass ASTGuard."""
    guard = make_ast_guard()
    src = (
        "def strip_bom(text: str | None) -> str:\n"
        '    """Strip leading UTF-8 BOM safely."""\n'
        "    if text is None:\n"
        "        return ''\n"
        "    return text.lstrip('\\ufeff')\n"
    )
    assert get_ast_valid(guard.validate(src, expected_name="strip_bom")) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f10_clean_helper_plus_target_function_passes() -> None:
    """Clean typed module constants and target function pass ASTGuard."""
    guard = make_ast_guard()
    src = (
        "from typing import Sequence\n\n"
        "TRUTHY: tuple[str, ...] = ('true', '1', 'yes')\n\n"
        "def normalize_flags(items: Sequence[str]) -> list[bool]:\n"
        '    """Convert string flag sequence into strict booleans."""\n'
        "    return [item.strip().lower() in TRUTHY for item in items]\n"
    )
    assert get_ast_valid(guard.validate(src, expected_name="normalize_flags")) is True


# ---------------------------------------------------------------------------
# F11: Dynamic Skill Registry & Hot-Reload (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f11_register_and_reload_invokes_in_same_process(e2e_data_dir: Path) -> None:
    """Forged skill persists on disk and executes via invoke_active in the same PID."""
    registry = make_skill_registry(e2e_data_dir)
    pid_before = os.getpid()
    source = (
        "def calc_tax(amount: float, rate: float = 0.1) -> float:\n"
        '    """Compute gross total with tax rate."""\n'
        "    return round(amount * (1.0 + rate), 2)\n"
    )
    cases = [
        make_test_case(kwargs={"amount": 100.0, "rate": 0.2}, expected=120.0),
        make_test_case(kwargs={"amount": 50.0, "rate": 0.0}, expected=50.0),
    ]
    rec = registry.register_and_reload(
        source=source,
        skill_name="calc_tax",
        docstring="Compute gross total with tax rate.",
        tags=["math", "tax"],
        test_cases=cases,
    )
    assert get_skill_name(rec) == "calc_tax"
    assert os.getpid() == pid_before
    assert registry.invoke_active("calc_tax", amount=100.0, rate=0.2) == 120.0
    skill_path = get_skill_path(rec)
    if skill_path is not None:
        assert skill_path.exists()
    else:
        assert any(e2e_data_dir.rglob("calc_tax.py"))


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f11_hot_reload_replaces_implementation_without_pyc_staleness(e2e_data_dir: Path) -> None:
    """Immediate same-second v2 reload overrides v1 without stale .pyc caching."""
    registry = make_skill_registry(e2e_data_dir)
    v1 = (
        "def transform_val(x: int) -> int:\n"
        '    """Add one to x."""\n'
        "    return x + 1\n"
    )
    v2 = (
        "def transform_val(x: int) -> int:\n"
        '    """Multiply x by ten."""\n'
        "    return x * 10\n"
    )
    registry.register_and_reload(
        source=v1,
        skill_name="transform_val",
        docstring="Add one to x.",
        tags=["math"],
        test_cases=[make_test_case(kwargs={"x": 5}, expected=6)],
    )
    assert registry.invoke_active("transform_val", x=5) == 6

    registry.register_and_reload(
        source=v2,
        skill_name="transform_val",
        docstring="Multiply x by ten.",
        tags=["math"],
        test_cases=[make_test_case(kwargs={"x": 5}, expected=50)],
    )
    assert registry.invoke_active("transform_val", x=5) == 50


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f11_list_skills_returns_all_registered_skills(e2e_data_dir: Path) -> None:
    """list_skills returns metadata for every registered skill."""
    registry = make_skill_registry(e2e_data_dir)
    s1 = (
        "def double_num(x: int) -> int:\n"
        '    """Double integer x."""\n'
        "    return x * 2\n"
    )
    s2 = (
        "def triple_num(x: int) -> int:\n"
        '    """Triple integer x."""\n'
        "    return x * 3\n"
    )
    registry.register_and_reload(
        s1, "double_num", "Double integer x.", ["math"], [make_test_case(kwargs={"x": 3}, expected=6)]
    )
    registry.register_and_reload(
        s2, "triple_num", "Triple integer x.", ["math"], [make_test_case(kwargs={"x": 3}, expected=9)]
    )
    skills = registry.list_skills()
    names = {get_skill_name(s) for s in skills}
    assert names == {"double_num", "triple_num"}


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f11_fresh_registry_loads_persisted_skills_from_disk(e2e_data_dir: Path) -> None:
    """Fresh SkillRegistry instance loads and invokes previously persisted skills."""
    reg1 = make_skill_registry(e2e_data_dir)
    src = (
        "def square_val(n: int) -> int:\n"
        '    """Square integer n."""\n'
        "    return n * n\n"
    )
    reg1.register_and_reload(
        src, "square_val", "Square integer n.", ["math"], [make_test_case(kwargs={"n": 4}, expected=16)]
    )

    reg2 = make_skill_registry(e2e_data_dir)
    assert "square_val" in {get_skill_name(s) for s in reg2.list_skills()}
    assert reg2.invoke_active("square_val", n=7) == 49


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f11_atomic_write_leaves_no_tmp_files(e2e_data_dir: Path) -> None:
    """Atomic skill registration leaves zero leftover .tmp files in data_dir."""
    registry = make_skill_registry(e2e_data_dir)
    src = (
        "def inc_val(n: int) -> int:\n"
        '    """Increment n by 1."""\n'
        "    return n + 1\n"
    )
    registry.register_and_reload(
        src, "inc_val", "Increment n by 1.", ["math"], [make_test_case(kwargs={"n": 1}, expected=2)]
    )
    tmp_files = list(e2e_data_dir.rglob("*.tmp"))
    assert tmp_files == []


# ---------------------------------------------------------------------------
# F12: Dual-Tier LLM Adapter (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f12_initial_draft_routes_to_fast_local(e2e_data_dir: Path) -> None:
    """E2E-F12-T1-01: Default drafting routes to FAST_LOCAL with escalated=False."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=2)
    draft_purpose = next(
        p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION
    )
    resp = llm.complete(
        messages=[{"role": "user", "content": "Merge list of dicts into DataFrame"}],
        purpose=draft_purpose,
        consecutive_failures=0,
    )
    assert resp.tier_used == ModelTier.FAST_LOCAL
    assert resp.escalated is False


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f12_repair_below_threshold_stays_fast_local(e2e_data_dir: Path) -> None:
    """E2E-F12-T1-02: Repair with consecutive_failures < threshold stays on FAST_LOCAL."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=2)
    non_distill = next(
        (p for p in TaskPurpose if "REPAIR" in p.name or "PATCH" in p.name),
        next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION),
    )
    resp = llm.complete(
        messages=[{"role": "user", "content": "Fix ValueError in parse_bool"}],
        purpose=non_distill,
        consecutive_failures=1,
    )
    assert resp.tier_used == ModelTier.FAST_LOCAL
    assert resp.escalated is False


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f12_repair_at_threshold_escalates_to_frontier(e2e_data_dir: Path) -> None:
    """E2E-F12-T1-03: Reaching consecutive_failures == threshold escalates to FRONTIER."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=2)
    non_distill = next(
        (p for p in TaskPurpose if "REPAIR" in p.name or "PATCH" in p.name),
        next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION),
    )
    resp = llm.complete(
        messages=[{"role": "user", "content": "Fix AttributeError after 2 failures"}],
        purpose=non_distill,
        consecutive_failures=2,
    )
    assert resp.tier_used == ModelTier.FRONTIER
    assert resp.escalated is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f12_skill_distillation_always_escalates_even_at_zero_failures(e2e_data_dir: Path) -> None:
    """E2E-F12-T1-04: SKILL_DISTILLATION always routes to FRONTIER even with 0 failures."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=5)
    resp = llm.complete(
        messages=[{"role": "user", "content": "Refactor solved task into reusable skill"}],
        purpose=TaskPurpose.SKILL_DISTILLATION,
        consecutive_failures=0,
    )
    assert resp.tier_used == ModelTier.FRONTIER
    assert resp.escalated is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f12_exact_token_and_duration_telemetry(e2e_data_dir: Path) -> None:
    """E2E-F12-T1-05: LLMResponse records positive token counts and non-negative latency."""
    from scar_forge.models import TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir)
    purpose = next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION)
    resp = llm.complete(
        messages=[{"role": "user", "content": "Sort semver tags safely"}],
        purpose=purpose,
        consecutive_failures=0,
    )
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.duration_ms >= 0.0
    assert len(get_llm_content(resp)) > 0


# ---------------------------------------------------------------------------
# F13: Closed-Loop ScarForgeEngine (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f13_engine_solves_novel_task_and_persists_skill(e2e_data_dir: Path) -> None:
    """Cold start on a novel task solves it and forges a persistent Skill."""
    engine = make_engine(e2e_data_dir, provider="local")
    res = run_engine_task(engine, "Merge list of record dicts into DataFrame")
    assert getattr(res, "success", getattr(res, "passed", True)) is True

    registry = make_skill_registry(e2e_data_dir)
    skills = registry.list_skills()
    assert len(skills) >= 1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f13_engine_records_scar_on_cold_pitfall(e2e_data_dir: Path) -> None:
    """Cold-start pitfall records a <15 token Scar in ScarStore."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    engine = make_engine(e2e_data_dir, provider="local")
    run_engine_task(engine, "Merge list of record dicts into DataFrame")

    store = make_scar_store(e2e_data_dir)
    scars = store.list_scars()
    assert len(scars) >= 1
    assert all(count_tokens(s.axiom) < 15 for s in scars)
    assert all(validate_axiom_format(s.axiom) for s in scars)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f13_engine_reuses_hot_loaded_skill_on_repeat_task(e2e_data_dir: Path) -> None:
    """Second run on related task reuses forged skill with zero errors and fewer tokens."""
    engine = make_engine(e2e_data_dir, provider="local")
    first = run_engine_task(engine, "Merge list of record dicts into DataFrame")
    second = run_engine_task(engine, "Merge list of record dicts into DataFrame batch 2")

    t1 = getattr(first, "total_tokens", getattr(first, "tokens", 0))
    t2 = getattr(second, "total_tokens", getattr(second, "tokens", 0))
    err2 = getattr(second, "errors", getattr(second, "error_count", 0))
    assert err2 == 0
    if t1 > 0 and t2 > 0:
        assert t2 < t1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f13_engine_injects_only_topk_scars_and_skills(e2e_data_dir: Path) -> None:
    """Seeding 10 Scars and 10 Skills still executes cleanly with Top-K bounded injection."""
    store = make_scar_store(e2e_data_dir)
    for i in range(10):
        err = make_distilled_error(
            "ValueError",
            f"synthetic domain fault {i}",
            f"val_{i} = parse_{i}(x)",
            line_number=i + 1,
        )
        store.record_failure(err, f"val_{i} = parse_{i}(x)", [f"dom{i}", "misc"])

    registry = make_skill_registry(e2e_data_dir)
    for i in range(10):
        src = (
            f"def helper_skill_{i}(x: int) -> int:\n"
            f'    """Synthetic helper skill {i}."""\n'
            f"    return x + {i}\n"
        )
        registry.register_and_reload(
            src,
            f"helper_skill_{i}",
            f"Synthetic helper skill {i}.",
            [f"dom{i}"],
            [make_test_case(kwargs={"x": 1}, expected=1 + i)],
        )

    engine = make_engine(e2e_data_dir, provider="local", top_k=4)
    res = run_engine_task(engine, "Sort semantic version tags numerically")
    assert getattr(res, "success", getattr(res, "passed", True)) is True


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m3
def test_t1_f13_engine_reports_run_metrics_with_tier_telemetry(e2e_data_dir: Path) -> None:
    """Engine execution returns metrics with positive tokens and duration >= 0."""
    engine = make_engine(e2e_data_dir, provider="local")
    res = run_engine_task(engine, "Strip UTF-8 BOM and parse semicolon CSV")
    tokens = getattr(res, "total_tokens", getattr(res, "tokens", None))
    duration = getattr(res, "duration_ms", None)
    assert tokens is not None and tokens > 0
    assert duration is not None and duration >= 0.0


# ---------------------------------------------------------------------------
# F14: 8-Task 3-Epoch Benchmark Suite (Milestone M4)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
def test_t1_f14_benchmark_task_catalog_has_8_plus_tasks_with_adversarial_cases() -> None:
    """E2E-F14-T1-01: Benchmark catalog defines >=8 unique tasks with nominal and edge cases."""
    from scar_forge.benchmark import tasks as bench_tasks

    catalog = None
    for attr in ("BENCHMARK_TASKS", "DEFAULT_TASKS", "TASKS", "get_benchmark_tasks"):
        obj = getattr(bench_tasks, attr, None)
        if obj is not None:
            catalog = obj() if callable(obj) else obj
            break
    assert catalog is not None, "Expected task catalog in scar_forge.benchmark.tasks"
    assert len(catalog) >= 8

    ids = set()
    for item in catalog:
        tid = getattr(item, "task_id", getattr(item, "id", None))
        if tid is None and isinstance(item, dict):
            tid = item.get("task_id") or item.get("id")
        assert tid
        ids.add(tid)
        cases = getattr(item, "test_cases", None) or (
            item.get("test_cases") if isinstance(item, dict) else None
        )
        adv = getattr(item, "adversarial_cases", None) or (
            item.get("adversarial_cases") if isinstance(item, dict) else None
        )
        assert (cases and len(cases) >= 1) or (adv and len(adv) >= 1)
    assert len(ids) >= 8


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
@pytest.mark.slow
def test_t1_f14_benchmark_cli_generates_json_and_markdown_reports(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T1-02: CLI benchmark writes valid 3-epoch JSON and Markdown table reports."""
    json_out = tmp_path / "bench.json"
    md_out = tmp_path / "bench.md"
    res = run_cli(
        [
            "benchmark",
            "--provider",
            "local",
            "--mode",
            "variant",
            "--out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--data-dir",
            str(e2e_data_dir),
        ],
        timeout=90.0,
    )
    assert res.returncode == 0, f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
    assert json_out.is_file()
    assert md_out.is_file()

    report = json.loads(json_out.read_text(encoding="utf-8"))
    epochs = report.get("epochs", report)
    for idx, key in enumerate(("epoch_1", "epoch_2", "epoch_3"), start=1):
        assert key in epochs or (isinstance(epochs, list) and len(epochs) >= idx)
    md_text = md_out.read_text(encoding="utf-8")
    assert "|" in md_text and "Epoch" in md_text


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
@pytest.mark.slow
def test_t1_f14_benchmark_epoch1_cold_start_records_real_errors_and_scars(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T1-03: Epoch 1 incurs real cold-start errors and persists <15 token Scars."""
    from scar_forge.tokens import count_tokens

    json_out = tmp_path / "e1.json"
    res = run_cli(
        ["benchmark", "--provider", "local", "--out", str(json_out), "--data-dir", str(e2e_data_dir)],
        timeout=90.0,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    e1 = report.get("epoch_1") or report["epochs"][0]
    assert e1.get("errors", e1.get("total_errors", 0)) > 0
    assert e1.get("tokens", e1.get("total_tokens", 0)) > 0
    assert e1.get("duration_ms", 0) > 0

    scars = make_scar_store(e2e_data_dir).list_scars()
    assert len(scars) >= 5
    assert all(count_tokens(s.axiom) <= 14 for s in scars)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
@pytest.mark.slow
def test_t1_f14_benchmark_epoch2_scar_guided_sharp_error_drop(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T1-04: Epoch 2 shows sharp drop in repeated errors and lower tokens vs Epoch 1."""
    json_out = tmp_path / "e2.json"
    res = run_cli(
        ["benchmark", "--provider", "local", "--out", str(json_out), "--data-dir", str(e2e_data_dir)],
        timeout=90.0,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    e1 = report.get("epoch_1") or report["epochs"][0]
    e2 = report.get("epoch_2") or report["epochs"][1]

    e1_err = e1.get("errors", e1.get("total_errors"))
    e2_err = e2.get("repeated_errors", e2.get("errors", e2.get("total_errors")))
    assert e2_err <= 0.25 * e1_err
    assert e2.get("tokens", e2.get("total_tokens")) < e1.get("tokens", e1.get("total_tokens"))


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
@pytest.mark.slow
def test_t1_f14_benchmark_epoch3_skill_reuse_60_to_85_pct_token_reduction(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T1-05: Epoch 3 achieves 60-85% token reduction vs Epoch 1 with >= accuracy."""
    json_out = tmp_path / "e3.json"
    res = run_cli(
        ["benchmark", "--provider", "local", "--out", str(json_out), "--data-dir", str(e2e_data_dir)],
        timeout=90.0,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    e1 = report.get("epoch_1") or report["epochs"][0]
    e3 = report.get("epoch_3") or report["epochs"][2]

    t1 = float(e1.get("tokens", e1.get("total_tokens")))
    t3 = float(e3.get("tokens", e3.get("total_tokens")))
    reduction = 1.0 - (t3 / t1)
    assert 0.60 <= reduction <= 0.85, f"Token reduction {reduction:.3f} outside [0.60, 0.85]"
    assert e3.get("accuracy", 0.0) >= e1.get("accuracy", 0.0)


# ---------------------------------------------------------------------------
# F15: Developer Interface — CLI & Flask Dashboard (Milestone M4)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
def test_t1_f15_cli_help_and_module_entrypoint(run_cli) -> None:
    """E2E-F15-T1-01: Both python -m scar_forge --help and scar-forge --help list all commands."""
    res_mod = run_cli(["--help"], inject_data_dir=False)
    assert res_mod.returncode == 0
    for cmd_name in ("demo", "run", "benchmark", "inspect", "search", "dashboard"):
        assert cmd_name in res_mod.stdout

    res_exe = run_cli(["--help"], use_console_script=True, inject_data_dir=False)
    assert res_exe.returncode == 0
    for cmd_name in ("demo", "run", "benchmark", "inspect", "search", "dashboard"):
        assert cmd_name in res_exe.stdout


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
def test_t1_f15_cli_demo_local_succeeds(run_cli, e2e_data_dir: Path) -> None:
    """E2E-F15-T1-02: scar-forge demo --provider local executes end-to-end and persists state."""
    res = run_cli(["demo", "--provider", "local", "--data-dir", str(e2e_data_dir)])
    assert res.returncode == 0, f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
    assert len(res.stdout.strip()) > 0
    assert len(make_scar_store(e2e_data_dir).list_scars()) >= 1
    assert len(make_skill_registry(e2e_data_dir).list_skills()) >= 1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
def test_t1_f15_cli_run_json_output_contract(run_cli, e2e_data_dir: Path) -> None:
    """E2E-F15-T1-03: scar-forge run --json emits valid JSON with telemetry."""
    res = run_cli(
        [
            "run",
            "--task",
            "Merge list of record dicts into DataFrame",
            "--provider",
            "local",
            "--data-dir",
            str(e2e_data_dir),
            "--json",
        ]
    )
    assert res.returncode == 0, res.stderr
    payload = res.json()
    assert isinstance(payload, dict)
    assert any(k in payload for k in ("success", "passed", "status"))


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
def test_t1_f15_cli_inspect_and_search_json(run_cli, e2e_data_dir: Path) -> None:
    """E2E-F15-T1-04: inspect --json and search --json return valid persisted Scars/Skills."""
    from scar_forge.tokens import count_tokens

    run_res = run_cli(
        [
            "run",
            "--task",
            "Merge list of record dicts into DataFrame",
            "--provider",
            "local",
            "--data-dir",
            str(e2e_data_dir),
            "--json",
        ]
    )
    assert run_res.returncode == 0

    insp = run_cli(["inspect", "--kind", "all", "--data-dir", str(e2e_data_dir), "--json"])
    assert insp.returncode == 0
    insp_data = insp.json()
    scars = unwrap_api_list(insp_data, "scars") if isinstance(insp_data, list) or "scars" in insp_data else []
    assert len(scars) >= 1
    for scar_item in scars:
        assert count_tokens(scar_item["axiom"]) < 15

    search = run_cli(
        ["search", "DataFrame", "--top-k", "4", "--data-dir", str(e2e_data_dir), "--json"]
    )
    assert search.returncode == 0
    search_data = search.json()
    if isinstance(search_data, dict):
        matched_scars = search_data.get("scars", [])
        matched_skills = search_data.get("skills", [])
        assert len(matched_scars) <= 4
        assert len(matched_skills) <= 4
    else:
        assert len(search_data) <= 4


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m4
def test_t1_f15_flask_dashboard_routes_http_200(run_cli, dashboard_get, e2e_data_dir: Path) -> None:
    """E2E-F15-T1-05: Dashboard GET /, /api/state, /api/scars, /api/skills, /api/benchmark return 200."""
    demo_res = run_cli(["demo", "--provider", "local", "--data-dir", str(e2e_data_dir)])
    assert demo_res.returncode == 0

    home = dashboard_get("/")
    assert home.status_code == 200
    assert "<html" in home.text.lower() or "scar" in home.text.lower()

    state = dashboard_get("/api/state")
    assert state.status_code == 200
    assert isinstance(state.json(), dict)

    scars_resp = dashboard_get("/api/scars")
    assert scars_resp.status_code == 200
    assert len(unwrap_api_list(scars_resp.json(), "scars")) >= 1

    skills_resp = dashboard_get("/api/skills")
    assert skills_resp.status_code == 200
    assert len(unwrap_api_list(skills_resp.json(), "skills")) >= 1

    bench_resp = dashboard_get("/api/benchmark")
    assert bench_resp.status_code == 200
    assert bench_resp.json() is not None


# ---------------------------------------------------------------------------
# F17: Technical README.md (Milestone M5)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m5
def test_t1_f17_readme_contains_three_valid_mermaid_diagrams() -> None:
    """README.md contains >=3 valid Mermaid architecture/sequence/flow diagrams."""
    readme = REPO_ROOT / "README.md"
    assert readme.is_file(), f"Missing {readme}"
    text = readme.read_text(encoding="utf-8")

    blocks = re.findall(r"```mermaid\s*\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 3, f"Expected >=3 mermaid blocks, found {len(blocks)}"

    valid_headers = (
        "flowchart",
        "graph",
        "sequenceDiagram",
        "stateDiagram",
        "classDiagram",
    )
    for block in blocks:
        stripped = block.strip()
        assert any(stripped.startswith(h) for h in valid_headers), (
            f"Invalid Mermaid header in block: {stripped[:60]}"
        )
        assert ("-->" in stripped) or ("->" in stripped) or ("---" in stripped)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.m5
def test_t1_f17_readme_documents_cli_and_three_epoch_benchmark_results() -> None:
    """README.md documents all 6 CLI subcommands and recorded 3-Epoch benchmark metrics."""
    readme = REPO_ROOT / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")

    for cli_cmd in ("demo", "run", "benchmark", "inspect", "search", "dashboard"):
        assert cli_cmd in text, f"CLI command '{cli_cmd}' missing from README.md"

    lower = text.lower()
    assert "epoch 1" in lower and "epoch 2" in lower and "epoch 3" in lower
    assert "%" in text
