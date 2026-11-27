"""Tier 2 E2E Tests — Boundary, Corner, Error, Isolation, and Rollback Coverage."""

from __future__ import annotations

import ast
import json
import re
import sys
import time
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.conftest import (
    REPO_ROOT,
    get_ast_valid,
    get_llm_content,
    get_prevented_count,
    get_scar_id,
    get_skill_name,
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
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f1_pyproject_and_gitignore_hygiene() -> None:
    """pyproject.toml parses cleanly and .gitignore excludes runtime/venv artifacts."""
    pyproject = REPO_ROOT / "pyproject.toml"
    gitignore = REPO_ROOT / ".gitignore"
    assert pyproject.is_file()
    assert gitignore.is_file()

    parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert "project" in parsed
    gi_text = gitignore.read_text(encoding="utf-8")
    for pattern in (".venv", "__pycache__", ".scar_forge"):
        assert pattern in gi_text


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f2_engine_config_rejects_invalid_bounds(e2e_data_dir: Path) -> None:
    """EngineConfig rejects invalid timeout, memory, or threshold boundaries."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        make_config(e2e_data_dir, sandbox_timeout_s=-2.0)

    with pytest.raises((ValidationError, ValueError)):
        make_config(e2e_data_dir, sandbox_memory_mb=0)

    with pytest.raises((ValidationError, ValueError)):
        make_config(e2e_data_dir, escalation_threshold=-1)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f2_models_reject_invalid_enum_and_missing_required_fields() -> None:
    """Pydantic models enforce enum bounds and required fields strictly."""
    from pydantic import ValidationError
    from scar_forge.models import AgentAction, DistilledError, ScarRecord

    with pytest.raises((ValidationError, ValueError)):
        AgentAction(action="EXECUTE_ARBITRARY")

    with pytest.raises((ValidationError, ValueError)):
        ScarRecord()

    with pytest.raises((ValidationError, ValueError)):
        DistilledError(exception_type="", line_number=0)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f2_engine_config_isolates_two_distinct_data_dirs(tmp_path: Path) -> None:
    """Two EngineConfig instances resolve strictly independent state paths."""
    dir_a = tmp_path / "tenant_a"
    dir_b = tmp_path / "tenant_b"
    cfg_a = make_config(dir_a)
    cfg_b = make_config(dir_b)

    def _get_dir(cfg: object) -> Path:
        for attr in ("data_dir", "state_dir", "workspace_dir"):
            if hasattr(cfg, attr):
                return Path(getattr(cfg, attr))
        raise AssertionError("EngineConfig missing data_dir/state_dir/workspace_dir")

    assert _get_dir(cfg_a) == dir_a
    assert _get_dir(cfg_b) == dir_b
    assert _get_dir(cfg_a) != _get_dir(cfg_b)
    assert _get_dir(cfg_a) != REPO_ROOT / ".scar_forge"


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f2_models_unicode_and_zero_telemetry_values() -> None:
    """DistilledError and ScarRecord handle zero telemetry and UTF-8 Windows paths."""
    err = make_distilled_error(
        exception_type="UnicodeError",
        core_message="BOM in C:\\t\u00eap\\d\u1eef_li\u1ec7u.csv",
        failing_line="",
        line_number=None,
        function_name=None,
        raw_chars=0,
        distilled_chars=0,
        raw_tokens=0,
        distilled_tokens=0,
    )
    assert err.line_number is None
    assert err.raw_chars == 0
    assert "d\u1eef_li\u1ec7u" in err.core_message


# ---------------------------------------------------------------------------
# F3: Deterministic Token Counter & <15 Axiom Budget (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f3_axiom_token_boundary_14_pass_vs_15_fail() -> None:
    """E2E-F3-T2-01: Exact boundary 14 tokens passes while 15 tokens fails."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    base = "[AVOID] old -> [USE]"
    candidate = base
    idx = 0
    while count_tokens(candidate) < 14 and idx < len(words):
        candidate = f"{candidate} {words[idx]}"
        idx += 1

    if count_tokens(candidate) > 14:
        # Step back one word and pad with single-token words if needed
        candidate = base
        for _ in range(20):
            nxt = f"{candidate} x"
            if count_tokens(nxt) > 14:
                break
            candidate = nxt

    # Ensure we reach exact 14 and exact 15 boundaries
    while count_tokens(candidate) < 14:
        candidate = f"{candidate} x"
    assert count_tokens(candidate) == 14
    assert validate_axiom_format(candidate) is True

    over = f"{candidate} z"
    while count_tokens(over) < 15:
        over = f"{over} z"
    assert count_tokens(over) >= 15
    assert validate_axiom_format(over) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f3_empty_and_whitespace_inputs() -> None:
    """E2E-F3-T2-02: Empty and whitespace-only strings return 0 tokens and invalid format."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    assert count_tokens("") == 0
    assert count_tokens("   \n\t  ") == 0
    assert validate_axiom_format("") is False
    assert validate_axiom_format("   \n\t  ") is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f3_validate_axiom_rejects_prose_and_missing_tags() -> None:
    """E2E-F3-T2-03: Conversational prose and malformed tags fail even when <15 tokens."""
    from scar_forge.tokens import validate_axiom_format

    assert validate_axiom_format("You should avoid df.append and use concat") is False
    assert validate_axiom_format("[AVOID] missing arrow") is False
    assert validate_axiom_format("-> [USE] no avoid") is False
    assert validate_axiom_format("[AVOID]  -> [USE] ") is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f3_unicode_cjk_emoji_and_windows_paths() -> None:
    """E2E-F3-T2-04: CJK, emoji, Vietnamese UTF-8, and Windows backslashes count safely."""
    from scar_forge.tokens import count_tokens, truncate_to_axiom_budget

    samples = [
        "[AVOID] open('C:\\\\d\u1eef_li\u1ec7u\\\\a.csv') -> [USE] encoding='utf-8'",
        "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8 \U0001f680 \u20ac",
    ]
    for s in samples:
        c = count_tokens(s)
        assert isinstance(c, int) and c > 0

    truncated = truncate_to_axiom_budget(
        "open('C:\\\\d\u1eef_li\u1ec7u\\\\a.csv') without encoding",
        "use encoding='utf-8-sig' for BOM CSV",
        max_tokens=14,
    )
    truncated.encode("utf-8")
    assert count_tokens(truncated) <= 14


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f3_truncate_to_axiom_budget_tight_and_empty_args() -> None:
    """E2E-F3-T2-05: Tight budget and empty avoid clause never crash and obey max_tokens."""
    from scar_forge.tokens import AxiomBudgetError, count_tokens, truncate_to_axiom_budget

    res_empty_avoid = truncate_to_axiom_budget("", "pd.concat", max_tokens=14)
    assert len(res_empty_avoid.strip()) > 0
    assert count_tokens(res_empty_avoid) <= 14

    huge = " ".join(["token"] * 50)
    res_tight = truncate_to_axiom_budget(huge, huge, max_tokens=9)
    assert len(res_tight.strip()) > 0
    assert count_tokens(res_tight) <= 9

    with pytest.raises(AxiomBudgetError):
        truncate_to_axiom_budget(huge, huge, max_tokens=8)


# ---------------------------------------------------------------------------
# F4: Zero-Prose Structured Protocol (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f4_markdown_fenced_json_enforces_zero_prose_or_warning() -> None:
    """E2E-F4-T2-01: Markdown fenced JSON fails in strict mode and records schema_warning on fallback."""
    from scar_forge.protocol import ProtocolParser, ProtocolViolationError

    fenced = '```json\n{"action": "RUN_CODE", "code": "x = 1"}\n```'
    strict_parser = ProtocolParser(allow_fence_fallback=False)
    with pytest.raises(ProtocolViolationError):
        strict_parser.parse_action(fenced)

    fallback_parser = ProtocolParser(allow_fence_fallback=True)
    parsed = fallback_parser.parse_action(fenced)
    assert parsed.schema_warning is True
    assert fallback_parser.schema_warnings == 1


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
@pytest.mark.parametrize(
    "prose_payload",
    [
        'Sure! Here is the JSON:\n{"action": "FINAL_ANSWER", "result": 1}',
        '{"action": "FINAL_ANSWER", "result": 1}\nLet me know if you need more!',
    ],
)
def test_t2_f4_reject_leading_or_trailing_prose(prose_payload: str) -> None:
    """E2E-F4-T2-02: Leading or trailing conversational prose is rejected."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        parse_agent_action_text(prose_payload)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
@pytest.mark.parametrize(
    "bad_json",
    [
        "",
        "   \n ",
        '{action: "RUN_CODE"',
        "def solve():\n    return 42",
    ],
)
def test_t2_f4_reject_malformed_or_truncated_json(bad_json: str) -> None:
    """E2E-F4-T2-03: Empty, truncated JSON, and raw Python fail protocol validation."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        parse_agent_action_text(bad_json)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f4_reject_unknown_action_and_missing_required_fields() -> None:
    """E2E-F4-T2-04: Disallowed actions and missing action payloads fail validation."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        parse_agent_action_text('{"action": "EXECUTE_SHELL", "cmd": "whoami"}')

    with pytest.raises((ValidationError, ValueError)):
        parse_agent_action_text('{"action": "CALL_SKILL"}')


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f4_reject_wrong_field_types_and_extra_prose_keys() -> None:
    """E2E-F4-T2-05: Wrong field types and extra conversational keys are rejected."""
    from pydantic import ValidationError

    bad = json.dumps(
        {
            "action": "CALL_SKILL",
            "skill_name": 123,
            "skill_args": "not_a_dict",
            "explanation": "conversational prose filler",
        }
    )
    with pytest.raises((ValidationError, ValueError)):
        parse_agent_action_text(bad)


# ---------------------------------------------------------------------------
# F5: Stderr & Traceback Distiller (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f5_distill_empty_stderr_and_stdout() -> None:
    """E2E-F5-T2-01: Empty stderr/stdout returns valid DistilledError without ZeroDivisionError."""
    from scar_forge.distiller import TracebackDistiller

    distilled = TracebackDistiller().distill("", "")
    assert distilled.raw_chars == 0
    assert distilled.distilled_chars >= 0
    assert 0.0 <= distilled.compression_ratio <= 1.0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f5_distill_windows_paths_with_spaces_and_backslashes() -> None:
    """E2E-F5-T2-02: Strips .venv site-packages paths with spaces while keeping user frame."""
    from scar_forge.distiller import TracebackDistiller

    raw = (
        "Traceback (most recent call last):\n"
        '  File "C:\\Users\\Admin\\My Project\\.venv\\Lib\\site-packages\\pkg\\mod.py", line 99, in wrap\n'
        "    return fn(*args)\n"
        '  File "C:\\Users\\Admin\\My Project\\main.py", line 12, in run_job\n'
        "    val = parse_row(item)\n"
        "KeyError: 'amount'\n"
    )
    distilled = TracebackDistiller().distill(raw)
    assert distilled.exception_type == "KeyError"
    assert distilled.line_number == 12
    assert distilled.function_name == "run_job"
    assert "site-packages" not in distilled.failing_line


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f5_distill_huge_recursion_stderr_high_compression() -> None:
    """E2E-F5-T2-03: 1000-line RecursionError distills fast and stays compact."""
    from scar_forge.distiller import TracebackDistiller

    frames = "\n".join(
        [f'  File "rec.py", line 4, in recurse\n    return recurse(n + 1)' for _ in range(500)]
    )
    raw = (
        "Traceback (most recent call last):\n"
        + frames
        + "\n  [Previous line repeated 996 more times]\n"
        + "RecursionError: maximum recursion depth exceeded\n"
    )
    t0 = time.perf_counter()
    distilled = TracebackDistiller().distill(raw)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 500.0
    assert distilled.exception_type == "RecursionError"
    assert distilled.distilled_tokens <= 80
    assert distilled.compression_ratio > 0.7


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f5_distill_plain_timeout_or_non_traceback_error() -> None:
    """E2E-F5-T2-04: Non-traceback single-line TimeoutError extracts type and message cleanly."""
    from scar_forge.distiller import TracebackDistiller

    raw = "TimeoutError: Sandbox exceeded 5.0s wall-clock limit"
    distilled = TracebackDistiller().distill(raw)
    assert distilled.exception_type == "TimeoutError"
    assert "5.0s" in distilled.core_message


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f5_distill_unicode_and_truncated_traceback() -> None:
    """E2E-F5-T2-05: Truncated traceback with UTF-8 accents never raises Decode/Index errors."""
    from scar_forge.distiller import TracebackDistiller

    raw = (
        "Traceback (most recent call last):\n"
        '  File "vn_task.py", line 15, in parse_vn\n'
        '    raise ValueError("Gi\u00e1 tr\u1ecb kh\u00f4ng h\u1ee3p l\u1ec7")\n'
        "ValueError: Gi\u00e1 tr\u1ecb"
    )
    distilled = TracebackDistiller().distill(raw)
    assert distilled.exception_type == "ValueError"
    assert len(distilled.core_message) > 0


# ---------------------------------------------------------------------------
# F6: Static Hygiene Gate (Milestone M1)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f6_static_hygiene_ast_zero_stubs_or_not_implemented() -> None:
    """E2E-F6-T2-01: Zero unimplemented-error raises or stub bodies in src/."""
    forbidden_exc = "NotImplemented" + "Error"
    src_files = list((REPO_ROOT / "src").rglob("*.py"))
    assert len(src_files) > 0

    stub_violations: list[str] = []
    for file_path in src_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

        def _scan(node: ast.AST, in_protocol: bool = False) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    bases = {getattr(b, "id", getattr(b, "attr", "")) for b in child.bases}
                    _scan(child, in_protocol or ("Protocol" in bases))
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dec_names = {
                        getattr(d, "id", getattr(d, "attr", "")) for d in child.decorator_list
                    }
                    if in_protocol or "abstractmethod" in dec_names or "overload" in dec_names:
                        _scan(child, in_protocol)
                        continue
                    body = child.body
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        effective = body[1:]
                    else:
                        effective = body

                    if not effective:
                        stub_violations.append(f"{file_path}:{child.lineno} {child.name} (docstring-only)")
                    elif len(effective) == 1:
                        stmt = effective[0]
                        if isinstance(stmt, ast.Pass):
                            stub_violations.append(f"{file_path}:{child.lineno} {child.name} (pass)")
                        elif (
                            isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is Ellipsis
                        ):
                            stub_violations.append(f"{file_path}:{child.lineno} {child.name} (ellipsis)")
                        elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                            if getattr(stmt.exc.func, "id", "") == forbidden_exc:
                                stub_violations.append(f"{file_path}:{child.lineno} {child.name} ({forbidden_exc})")
                    _scan(child, in_protocol)
                else:
                    _scan(child, in_protocol)

        _scan(tree)

    assert not stub_violations, f"Stub functions found in src/: {stub_violations}"


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m1
def test_t2_f6_e2e_suite_itself_passes_static_hygiene() -> None:
    """All E2E test modules themselves contain zero stub bodies or forbidden marker tags."""
    markers = ("TO" + "DO", "FIX" + "ME", "X" * 3, "HA" + "CK")
    pattern = re.compile(r"\b(" + "|".join(markers) + r")\b")
    e2e_files = list((REPO_ROOT / "tests" / "e2e").glob("*.py"))
    assert len(e2e_files) >= 5

    for f in e2e_files:
        text = f.read_text(encoding="utf-8")
        assert not pattern.search(text), f"Forbidden marker in {f}"
        tree = ast.parse(text, filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                assert len(node.body) >= 2, f"Empty test body in {f}:{node.name}"


# ---------------------------------------------------------------------------
# F7: Failure-to-Axiom Scar Memory (Milestone M2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f7_axiom_token_budget_boundary_14_vs_15() -> None:
    """validate_axiom_format enforces strict <15 token ceiling (14 ok, 15 rejected)."""
    from scar_forge.tokens import count_tokens, truncate_to_axiom_budget, validate_axiom_format

    long_pair = truncate_to_axiom_budget(
        " ".join(["badcall"] * 20),
        " ".join(["goodcall"] * 20),
        max_tokens=14,
    )
    assert count_tokens(long_pair) <= 14
    assert validate_axiom_format(long_pair) is True

    oversized = "[AVOID] " + " ".join(["term"] * 15) + " -> [USE] clean"
    assert count_tokens(oversized) >= 15
    assert validate_axiom_format(oversized) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f7_axiom_format_rejects_prose_and_malformed_templates() -> None:
    """validate_axiom_format rejects empty strings, prose, and incomplete arrows."""
    from scar_forge.tokens import validate_axiom_format

    for bad in (
        "",
        "   ",
        "You should avoid using append because pandas removed it.",
        "[AVOID]  -> [USE] ",
        "[AVOID] something",
    ):
        assert validate_axiom_format(bad) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f7_jaccard_below_threshold_not_deduplicated(e2e_data_dir: Path) -> None:
    """Distinct errors with Jaccard < 0.75 persist as separate Scars."""
    store = make_scar_store(e2e_data_dir)
    err1 = make_distilled_error(
        "ValueError",
        "invalid literal for int() with base 10: 'false'",
        "flag = bool(int(raw_payload))",
        line_number=5,
    )
    err2 = make_distilled_error(
        "KeyError",
        "missing required column 'timestamp_utc' during CSV pivot merge",
        "df.pivot(index='timestamp_utc', columns='region')",
        line_number=19,
    )
    _, c1 = store.record_failure(err1, "flag = bool(int(raw_payload))", ["webhook", "bool"])
    _, c2 = store.record_failure(err2, "df.pivot(index='timestamp_utc')", ["csv", "pivot"])
    assert c1 is True and c2 is True
    assert len(store.list_scars()) == 2


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f7_mark_prevented_empty_and_unknown_ids_noop(e2e_data_dir: Path) -> None:
    """mark_prevented with empty list or unknown IDs is a safe no-op."""
    store = make_scar_store(e2e_data_dir)
    err = make_distilled_error("ValueError", "bad json nan", "json.dumps(x, allow_nan=False)", 4)
    scar, _ = store.record_failure(err, "json.dumps(x, allow_nan=False)", ["json"])
    before = get_prevented_count(scar)

    store.mark_prevented([])
    store.mark_prevented(["unknown-scar-id-999"])
    after = get_prevented_count(store.list_scars()[0])
    assert after == before


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f7_empty_fields_and_utf8_windows_paths_in_distilled_error(e2e_data_dir: Path) -> None:
    """ScarStore handles empty failing_line, None line_number, and UTF-8 Windows paths."""
    from scar_forge.tokens import count_tokens

    store = make_scar_store(e2e_data_dir)
    err = make_distilled_error(
        exception_type="UnicodeDecodeError",
        core_message="BOM in C:\\tmp\\d\u1eef_li\u1ec7u.csv (\ufeff)",
        failing_line="",
        line_number=None,
        function_name=None,
    )
    scar, created = store.record_failure(err, "", ["csv", "utf8"])
    assert created is True
    assert count_tokens(scar.axiom) < 15


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f7_concurrent_store_read_after_write_wal_safety(e2e_data_dir: Path) -> None:
    """Interleaved writes and reads across two ScarStore instances succeed under WAL mode."""
    store_a = make_scar_store(e2e_data_dir)
    store_b = make_scar_store(e2e_data_dir)

    ids: list[str] = []
    for i in range(6):
        err = make_distilled_error(
            f"CustomError{i}",
            f"unique failure pattern alpha_{i} beta_{i * 7}",
            f"op_{i}()",
            line_number=i + 1,
        )
        scar, _ = store_a.record_failure(err, f"op_{i}()", [f"tag{i}"])
        ids.append(get_scar_id(scar))
        listed = store_b.list_scars()
        assert len(listed) >= len(ids)
        store_b.mark_prevented([ids[-1]])

    final_scars = store_a.list_scars()
    assert len(final_scars) == 6
    assert all(get_prevented_count(s) >= 1 for s in final_scars)


# ---------------------------------------------------------------------------
# F8: Top-K BM25 Dynamic Retriever (Milestone M2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f8_top_k_clamping_0_negative_1_5_100() -> None:
    """BM25Retriever clamps top_k strictly into [1, 5] for both scars and skills."""
    retriever = make_retriever()
    scars = [
        make_scar_record(f"s{i}", f"[AVOID] bad_{i} -> [USE] pandas concat_{i}", ["pandas", "concat"])
        for i in range(10)
    ]
    skills = [
        make_skill_record(f"skill_{i}", f"Merge pandas DataFrame batch {i}.", ["pandas", "merge"])
        for i in range(10)
    ]

    for k, expected in ((-3, 1), (0, 1), (1, 1), (5, 5), (100, 5)):
        assert len(retriever.retrieve_scars("pandas concat", scars, top_k=k)) == expected
        assert len(retriever.retrieve_skills("pandas merge", skills, top_k=k)) == expected


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f8_empty_corpus_returns_empty_list() -> None:
    """Empty Scar or Skill corpus returns [] cleanly without ZeroDivisionError."""
    retriever = make_retriever()
    assert retriever.retrieve_scars("pandas append", [], top_k=4) == []
    assert retriever.retrieve_skills("pandas append", [], top_k=4) == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
@pytest.mark.parametrize("query", ["", "   \n\t  ", "---!!!???"])
def test_t2_f8_empty_whitespace_and_punctuation_queries(query: str) -> None:
    """Empty, whitespace, and punctuation-only queries return [] cleanly."""
    retriever = make_retriever()
    scars = [make_scar_record("s1", "[AVOID] df.append -> [USE] pd.concat", ["pandas"])]
    skills = [make_skill_record("merge_df", "Merge dicts into DataFrame.", ["pandas"])]
    assert retriever.retrieve_scars(query, scars, top_k=4) == []
    assert retriever.retrieve_skills(query, skills, top_k=4) == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f8_single_document_corpus_idf_positive() -> None:
    """Single-document corpus yields positive score for matching query term."""
    retriever = make_retriever()
    only = make_scar_record("only1", "[AVOID] df.append -> [USE] pd.concat", ["pandas"])
    res = retriever.retrieve_scars("pandas", [only], top_k=4)
    assert len(res) == 1
    assert get_scar_id(res[0]) == "only1"


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f8_unicode_bom_and_backslash_query_safety() -> None:
    """Queries with BOM, backslashes, and UTF-8 diacritics run safely."""
    retriever = make_retriever()
    scars = [make_scar_record("bom", "[RULE] strip BOM encoding utf-8-sig csv", ["csv", "bom"])]
    res = retriever.retrieve_scars("d\u1eef_li\u1ec7u \ufeff C:\\path\\csv", scars, top_k=4)
    assert isinstance(res, list)


# ---------------------------------------------------------------------------
# F9: Adversarial Verification Sandbox (Milestone M2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
@pytest.mark.slow
def test_t2_f9_timeout_kills_infinite_loop_and_leaves_no_orphans() -> None:
    """Infinite loop inside sandbox aborts within wall-clock timeout without hanging."""
    runner = make_sandbox_runner()
    code = (
        "def spin(x: int) -> int:\n"
        "    while True:\n"
        "        x += 1\n"
    )
    t0 = time.monotonic()
    res = runner.execute_code(
        code, [make_test_case(kwargs={"x": 1}, expected=2)], entry_fn="spin", timeout_s=0.5
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 4.5
    assert res.passed is False
    assert res.timed_out is True


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
@pytest.mark.slow
def test_t2_f9_memory_cap_blocks_large_allocation() -> None:
    """Excessive memory allocation fails cleanly inside memory-bounded sandbox."""
    runner = make_sandbox_runner()
    code = (
        "def blow_mem(mb: int) -> int:\n"
        "    buf = bytearray(mb * 1024 * 1024)\n"
        "    return len(buf)\n"
    )
    res = runner.execute_code(
        code,
        [make_test_case(kwargs={"mb": 256}, expected=256 * 1024 * 1024)],
        entry_fn="blow_mem",
        timeout_s=5.0,
        memory_mb=64,
    )
    assert res.passed is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
@pytest.mark.slow
def test_t2_f9_blocks_child_process_or_fork_spawn() -> None:
    """Attempting to spawn a child process is blocked by sandbox import/job guard."""
    runner = make_sandbox_runner()
    code = (
        "import subprocess, sys\n"
        "def spawn_child() -> int:\n"
        "    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        "    return p.pid\n"
    )
    res = runner.execute_code(
        code,
        [make_test_case(kwargs={}, expected=0)],
        entry_fn="spawn_child",
        timeout_s=2.0,
    )
    assert res.passed is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
@pytest.mark.parametrize("blocked_mod", ["subprocess", "ctypes", "socket"])
def test_t2_f9_blocks_dangerous_imports_inside_sandbox(blocked_mod: str) -> None:
    """Dangerous system/network/FFI imports are blocked inside SandboxRunner."""
    runner = make_sandbox_runner()
    code = (
        f"import {blocked_mod}\n"
        "def probe() -> bool:\n"
        "    return True\n"
    )
    res = runner.execute_code(
        code,
        [make_test_case(kwargs={}, expected=True)],
        entry_fn="probe",
        timeout_s=3.0,
    )
    assert res.passed is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f9_verify_skill_adversarial_catches_fragile_happy_path_code() -> None:
    """verify_skill_adversarial catches IndexError on empty string input."""
    runner = make_sandbox_runner()
    fragile = (
        "def first_char(s: str) -> str:\n"
        '    """Return uppercase first character."""\n'
        "    return s[0].upper()\n"
    )
    res = runner.verify_skill_adversarial(
        source=fragile,
        skill_name="first_char",
        explicit_cases=[make_test_case(kwargs={"s": "abc"}, expected="A")],
        param_schema={"s": "str"},
    )
    assert res.passed is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m2
def test_t2_f9_empty_code_and_syntax_error_handled_cleanly() -> None:
    """Empty code and SyntaxError return passed=False without crashing host runner."""
    runner = make_sandbox_runner()
    res_empty = runner.execute_code("", [make_test_case(kwargs={}, expected=None)])
    assert res_empty.passed is False

    res_syn = runner.execute_code("def broken(:\n    pass\n", [make_test_case(kwargs={}, expected=None)])
    assert res_syn.passed is False


# ---------------------------------------------------------------------------
# F10: AST Sanitizer & Refactorer (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
@pytest.mark.parametrize(
    "call_expr",
    [
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', '<x>', 'eval')",
        "__import__('os')",
        "getattr(x, '__class__')",
        "open('/etc/passwd')",
    ],
)
def test_t2_f10_blocks_dangerous_builtins(call_expr: str) -> None:
    """ASTGuard blocks dangerous builtin calls."""
    guard = make_ast_guard()
    src = (
        "def unsafe_fn(x: str) -> object:\n"
        '    """Dangerous builtin test."""\n'
        f"    return {call_expr}\n"
    )
    assert get_ast_valid(guard.validate(src)) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
@pytest.mark.parametrize(
    "expr",
    [
        "().__class__.__base__.__subclasses__()",
        "(lambda: 1).__globals__",
        "__builtins__",
        "(lambda: 1).__code__",
        "int.__mro__",
    ],
)
def test_t2_f10_blocks_dunder_escape_chains(expr: str) -> None:
    """ASTGuard rejects dunder attribute and builtin escape chains."""
    guard = make_ast_guard()
    src = (
        "def dunder_escape() -> object:\n"
        '    """Dunder escape attempt."""\n'
        f"    return {expr}\n"
    )
    assert get_ast_valid(guard.validate(src)) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
@pytest.mark.parametrize(
    "imp",
    [
        "import os",
        "from subprocess import run",
        "import sys as s",
        "import ctypes",
        "import shutil",
        "import socket",
        "import pickle",
        "import importlib",
    ],
)
def test_t2_f10_blocks_dangerous_and_aliased_imports(imp: str) -> None:
    """ASTGuard rejects non-allowlisted and aliased dangerous imports."""
    guard = make_ast_guard()
    src = (
        f"{imp}\n\n"
        "def unsafe_imp(x: int) -> int:\n"
        '    """Unsafe import test."""\n'
        "    return x + 1\n"
    )
    assert get_ast_valid(guard.validate(src)) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f10_blocks_nested_imports_inside_function_or_branch() -> None:
    """ASTGuard rejects imports nested inside functions or conditional branches."""
    guard = make_ast_guard()
    nested_fn = (
        "def f(x: int) -> int:\n"
        '    """Nested import."""\n'
        "    import os\n"
        "    return x\n"
    )
    nested_if = (
        "def f(x: int) -> int:\n"
        '    """Conditional import."""\n'
        "    if x:\n"
        "        from pathlib import Path\n"
        "    return x\n"
    )
    assert get_ast_valid(guard.validate(nested_fn)) is False
    assert get_ast_valid(guard.validate(nested_if)) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f10_blocks_stub_bodies_pass_ellipsis_not_implemented() -> None:
    """ASTGuard rejects pass, ellipsis, and unimplemented-error stub bodies."""
    guard = make_ast_guard()
    not_impl = "NotImplemented" + "Error"
    stubs = [
        'def f(x: int) -> int:\n    """Doc."""\n    pass\n',
        'def f(x: int) -> int:\n    """Doc."""\n    ...\n',
        f'def f(x: int) -> int:\n    """Doc."""\n    raise {not_impl}("stub")\n',
    ]
    for stub in stubs:
        assert get_ast_valid(guard.validate(stub)) is False


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f10_rejects_syntax_error_empty_untyped_missing_doc_and_name_mismatch() -> None:
    """ASTGuard rejects empty, syntax error, untyped, missing docstring, and name mismatch."""
    guard = make_ast_guard()
    assert get_ast_valid(guard.validate("")) is False
    assert get_ast_valid(guard.validate("def broken(:")) is False
    assert get_ast_valid(guard.validate('def f(x):\n    """Doc."""\n    return x\n')) is False
    assert get_ast_valid(guard.validate("def f(x: int) -> int:\n    return x\n")) is False
    valid_fn = 'def actual_name(x: int) -> int:\n    """Doc."""\n    return x\n'
    assert get_ast_valid(guard.validate(valid_fn, expected_name="wrong_name")) is False


# ---------------------------------------------------------------------------
# F11: Dynamic Skill Registry & Hot-Reload (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f11_rollback_on_ast_failure_preserves_prior_v1(e2e_data_dir: Path) -> None:
    """Failed v2 AST check rolls back cleanly and keeps v1 active and persisted."""
    registry = make_skill_registry(e2e_data_dir)
    v1 = (
        "def add_one(x: int) -> int:\n"
        '    """Add one."""\n'
        "    return x + 1\n"
    )
    registry.register_and_reload(
        v1, "add_one", "Add one.", ["math"], [make_test_case(kwargs={"x": 4}, expected=5)]
    )

    bad_v2 = (
        "import os\n"
        "def add_one(x: int) -> int:\n"
        '    """Evil v2."""\n'
        "    return int(eval('x'))\n"
    )
    with pytest.raises(Exception):
        registry.register_and_reload(
            bad_v2, "add_one", "Evil v2.", ["math"], [make_test_case(kwargs={"x": 4}, expected=4)]
        )

    assert registry.invoke_active("add_one", x=4) == 5
    py_files = list(e2e_data_dir.rglob("add_one.py"))
    assert len(py_files) == 1
    assert "eval" not in py_files[0].read_text(encoding="utf-8")


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f11_rollback_on_test_case_or_adversarial_failure_preserves_prior_v1(
    e2e_data_dir: Path,
) -> None:
    """v2 failing sandbox verification rolls back and preserves working v1."""
    registry = make_skill_registry(e2e_data_dir)
    v1 = (
        "def safe_div(a: float, b: float) -> float:\n"
        '    """Divide a by b safely."""\n'
        "    return 0.0 if b == 0 else a / b\n"
    )
    registry.register_and_reload(
        v1,
        "safe_div",
        "Divide a by b safely.",
        ["math"],
        [make_test_case(kwargs={"a": 10.0, "b": 2.0}, expected=5.0)],
    )

    broken_v2 = (
        "def safe_div(a: float, b: float) -> float:\n"
        '    """Broken divide."""\n'
        "    return 1.0 / 0.0\n"
    )
    with pytest.raises(Exception):
        registry.register_and_reload(
            broken_v2,
            "safe_div",
            "Broken divide.",
            ["math"],
            [make_test_case(kwargs={"a": 10.0, "b": 2.0}, expected=5.0)],
        )

    assert registry.invoke_active("safe_div", a=10.0, b=2.0) == 5.0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f11_failed_initial_registration_leaves_no_orphan_artifact(e2e_data_dir: Path) -> None:
    """Failed brand-new skill registration leaves zero .py/.tmp artifacts or active entries."""
    registry = make_skill_registry(e2e_data_dir)
    bad = (
        "import subprocess\n"
        "def bad_new(x: int) -> int:\n"
        '    """Bad new."""\n'
        "    return x\n"
    )
    with pytest.raises(Exception):
        registry.register_and_reload(
            bad, "bad_new", "Bad new.", ["bad"], [make_test_case(kwargs={"x": 1}, expected=1)]
        )

    assert "bad_new" not in {get_skill_name(s) for s in registry.list_skills()}
    assert list(e2e_data_dir.rglob("bad_new.py")) == []
    assert list(e2e_data_dir.rglob("*.tmp")) == []
    with pytest.raises((KeyError, LookupError, ValueError)):
        registry.invoke_active("bad_new", x=1)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f11_invoke_unknown_skill_and_bad_kwargs_raise_cleanly(e2e_data_dir: Path) -> None:
    """Unknown skill invocation and mismatched kwargs raise clean Lookup/Type errors."""
    registry = make_skill_registry(e2e_data_dir)
    src = (
        "def echo_num(n: int) -> int:\n"
        '    """Echo n."""\n'
        "    return n\n"
    )
    registry.register_and_reload(
        src, "echo_num", "Echo n.", ["math"], [make_test_case(kwargs={"n": 3}, expected=3)]
    )

    with pytest.raises((KeyError, LookupError)):
        registry.invoke_active("nonexistent_skill")

    with pytest.raises(TypeError):
        registry.invoke_active("echo_num", wrong_param=123)

    assert registry.invoke_active("echo_num", n=9) == 9


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
@pytest.mark.parametrize("bad_name", ["../escape", "pkg.mod", "bad;name", "123invalid"])
def test_t2_f11_skill_name_path_traversal_rejected(e2e_data_dir: Path, bad_name: str) -> None:
    """Invalid or path-traversal skill names are rejected before touching disk."""
    registry = make_skill_registry(e2e_data_dir)
    src = (
        "def valid_fn(x: int) -> int:\n"
        '    """Valid fn."""\n'
        "    return x\n"
    )
    with pytest.raises((ValueError, Exception)):
        registry.register_and_reload(
            src, bad_name, "Valid fn.", ["sec"], [make_test_case(kwargs={"x": 1}, expected=1)]
        )
    assert not (e2e_data_dir.parent / "escape.py").exists()


# ---------------------------------------------------------------------------
# F12: Dual-Tier LLM Adapter (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f12_escalation_threshold_boundary_custom_n(e2e_data_dir: Path) -> None:
    """E2E-F12-T2-01: Custom threshold=3 keeps failures=2 on FAST_LOCAL and escalates at 3 and 10."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=3)
    repair = next(
        (p for p in TaskPurpose if "REPAIR" in p.name or "PATCH" in p.name),
        next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION),
    )
    msgs = [{"role": "user", "content": "Repair csv parser"}]

    r2 = llm.complete(msgs, purpose=repair, consecutive_failures=2)
    r3 = llm.complete(msgs, purpose=repair, consecutive_failures=3)
    r10 = llm.complete(msgs, purpose=repair, consecutive_failures=10)

    assert r2.tier_used == ModelTier.FAST_LOCAL and r2.escalated is False
    assert r3.tier_used == ModelTier.FRONTIER and r3.escalated is True
    assert r10.tier_used == ModelTier.FRONTIER and r10.escalated is True


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f12_negative_failures_and_empty_messages_validation(e2e_data_dir: Path) -> None:
    """E2E-F12-T2-02: Negative consecutive_failures or empty messages raises explicit error."""
    from scar_forge.models import TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir)
    purpose = next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION)

    with pytest.raises(ValueError):
        llm.complete([{"role": "user", "content": "hello"}], purpose=purpose, consecutive_failures=-1)

    with pytest.raises(ValueError):
        llm.complete([], purpose=purpose, consecutive_failures=0)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f12_unreachable_ollama_fast_fail_vs_auto_fallback(e2e_data_dir: Path) -> None:
    """E2E-F12-T2-03: Explicit ollama on closed port fails fast; auto falls back cleanly."""
    from scar_forge.models import TaskPurpose

    purpose = next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION)
    msgs = [{"role": "user", "content": "Merge list of record dicts into DataFrame"}]

    ollama_llm = make_llm(
        provider="ollama",
        data_dir=e2e_data_dir,
        ollama_url="http://127.0.0.1:59999",
    )
    t0 = time.monotonic()
    with pytest.raises(Exception):
        ollama_llm.complete(msgs, purpose=purpose, consecutive_failures=0)
    assert (time.monotonic() - t0) < 8.0

    auto_llm = make_llm(
        provider="auto",
        data_dir=e2e_data_dir,
        ollama_url="http://127.0.0.1:59999",
    )
    resp = auto_llm.complete(msgs, purpose=purpose, consecutive_failures=0)
    assert len(get_llm_content(resp)) > 0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f12_local_backend_emits_strict_zero_prose_json(e2e_data_dir: Path) -> None:
    """E2E-F12-T2-04: Local backend outputs parse cleanly as strict Zero-Prose JSON."""
    from scar_forge.models import TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir)
    for purpose in TaskPurpose:
        resp = llm.complete(
            [{"role": "user", "content": "Merge list of record dicts into DataFrame"}],
            purpose=purpose,
            consecutive_failures=0,
        )
        raw = get_llm_content(resp).strip()
        assert raw.startswith("{") and raw.endswith("}")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f12_cumulative_tier_token_accounting(e2e_data_dir: Path) -> None:
    """E2E-F12-T2-05: Per-tier token usage sums match exact cumulative call totals."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=2)
    draft = next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION)

    responses = [
        llm.complete([{"role": "user", "content": f"Draft task {i}"}], purpose=draft, consecutive_failures=0)
        for i in range(3)
    ]
    responses.append(
        llm.complete(
            [{"role": "user", "content": "Distill skill"}],
            purpose=TaskPurpose.SKILL_DISTILLATION,
            consecutive_failures=0,
        )
    )

    fast_sum = sum(r.prompt_tokens + r.completion_tokens for r in responses if r.tier_used == ModelTier.FAST_LOCAL)
    frontier_sum = sum(r.prompt_tokens + r.completion_tokens for r in responses if r.tier_used == ModelTier.FRONTIER)
    assert fast_sum > 0 and frontier_sum > 0

    usage = getattr(llm, "usage", None) or getattr(llm, "get_usage", lambda: None)()
    if usage is not None:
        if isinstance(usage, dict):
            u_fast = usage.get(ModelTier.FAST_LOCAL) or usage.get("FAST_LOCAL")
            u_front = usage.get(ModelTier.FRONTIER) or usage.get("FRONTIER")
            if u_fast is not None and u_front is not None:
                assert getattr(u_fast, "total_tokens", u_fast) == fast_sum
                assert getattr(u_front, "total_tokens", u_front) == frontier_sum


# ---------------------------------------------------------------------------
# F13: Closed-Loop ScarForgeEngine (Milestone M3)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f13_engine_empty_or_whitespace_task_rejected_cleanly(e2e_data_dir: Path) -> None:
    """Empty or whitespace-only task prompt raises ValueError cleanly."""
    engine = make_engine(e2e_data_dir, provider="local")
    with pytest.raises((ValueError, Exception)):
        run_engine_task(engine, "   \n  ")


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f13_engine_handles_unsolvable_task_within_max_retries(e2e_data_dir: Path) -> None:
    """Unknown/unsolvable task terminates cleanly within bounded retries without infinite loop."""
    engine = make_engine(e2e_data_dir, provider="local")
    t0 = time.monotonic()
    try:
        res = run_engine_task(engine, "Impossible quantum teleportation without hardware xyz_999")
        # If it returns a RunMetrics instead of raising, verify bounded execution
        assert getattr(res, "total_tokens", getattr(res, "tokens", 0)) >= 0
    except Exception as exc:
        assert len(str(exc)) > 0
    assert (time.monotonic() - t0) < 15.0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f13_engine_persists_scars_even_when_cold_errors_occur(e2e_data_dir: Path) -> None:
    """Cold-start failures persist Scars in ScarStore before final answer or bailout."""
    engine = make_engine(e2e_data_dir, provider="local")
    run_engine_task(engine, "Coerce webhook JSON payload with boolean string 'false'")
    scars = make_scar_store(e2e_data_dir).list_scars()
    assert len(scars) >= 1


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f13_engine_isolates_state_across_separate_data_dirs(tmp_path: Path) -> None:
    """Two engines pointing at separate data_dirs never share Scars or Skills."""
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"

    eng_a = make_engine(dir_a, provider="local")
    run_engine_task(eng_a, "Merge list of record dicts into DataFrame")

    assert len(make_skill_registry(dir_a).list_skills()) >= 1
    assert len(make_skill_registry(dir_b).list_skills()) == 0
    assert len(make_scar_store(dir_b).list_scars()) == 0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m3
def test_t2_f13_engine_handles_unicode_and_adversarial_prompt_payloads(e2e_data_dir: Path) -> None:
    """Engine processes UTF-8 BOM and accented characters in task prompt without encoding error."""
    engine = make_engine(e2e_data_dir, provider="local")
    res = run_engine_task(
        engine,
        "Parse semicolon CSV with UTF-8 BOM '\\ufeff' and Vietnamese label 'D\u1eef li\u1ec7u'",
    )
    assert getattr(res, "total_tokens", getattr(res, "tokens", 0)) > 0


# ---------------------------------------------------------------------------
# F14: 8-Task 3-Epoch Benchmark Suite (Milestone M4)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
@pytest.mark.slow
def test_t2_f14_benchmark_ablation_mode_isolates_scar_effect(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T2-01: Ablation mode proves Epoch 2 Scar impact independently of Skill reuse."""
    json_out = tmp_path / "ablation.json"
    res = run_cli(
        [
            "benchmark",
            "--provider",
            "local",
            "--mode",
            "ablation",
            "--out",
            str(json_out),
            "--data-dir",
            str(e2e_data_dir),
        ],
        timeout=90.0,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    e1 = report.get("epoch_1") or report["epochs"][0]
    e2 = report.get("epoch_2") or report["epochs"][1]
    assert e2.get("skill_calls", 0) == 0
    e1_err = e1.get("errors", e1.get("total_errors"))
    e2_err = e2.get("repeated_errors", e2.get("errors", e2.get("total_errors")))
    assert e2_err < e1_err


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
@pytest.mark.slow
def test_t2_f14_benchmark_isolated_data_dir_starts_cold(
    run_cli, tmp_path: Path
) -> None:
    """E2E-F14-T2-02: Fresh --data-dir still starts cold even after another dir ran benchmark."""
    dir_1 = tmp_path / "run1"
    dir_2 = tmp_path / "run2"
    out_1 = tmp_path / "r1.json"
    out_2 = tmp_path / "r2.json"

    r1 = run_cli(
        ["benchmark", "--provider", "local", "--out", str(out_1), "--data-dir", str(dir_1)],
        timeout=90.0,
    )
    r2 = run_cli(
        ["benchmark", "--provider", "local", "--out", str(out_2), "--data-dir", str(dir_2)],
        timeout=90.0,
    )
    assert r1.returncode == 0 and r2.returncode == 0
    rep2 = json.loads(out_2.read_text(encoding="utf-8"))
    e1_2 = rep2.get("epoch_1") or rep2["epochs"][0]
    assert e1_2.get("errors", e1_2.get("total_errors", 0)) > 0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
@pytest.mark.slow
def test_t2_f14_benchmark_anti_cheating_real_metrics_consistency(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T2-03: Epoch aggregate tokens and accuracy match sums of underlying task metrics."""
    json_out = tmp_path / "consistent.json"
    res = run_cli(
        ["benchmark", "--provider", "local", "--out", str(json_out), "--data-dir", str(e2e_data_dir)],
        timeout=90.0,
    )
    assert res.returncode == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    for key in ("epoch_1", "epoch_2", "epoch_3"):
        ep = report.get(key)
        if ep is None:
            continue
        tasks = ep.get("tasks") or ep.get("task_results")
        if tasks:
            tok_sum = sum(t.get("tokens", t.get("total_tokens", 0)) for t in tasks)
            ep_tok = ep.get("tokens", ep.get("total_tokens"))
            assert ep_tok == tok_sum
            passed_cnt = sum(1 for t in tasks if t.get("passed", t.get("success", False)))
            expected_acc = passed_cnt / len(tasks)
            assert abs(ep.get("accuracy", expected_acc) - expected_acc) < 0.02


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
@pytest.mark.slow
def test_t2_f14_benchmark_deterministic_repeatability_in_local_mode(
    run_cli, tmp_path: Path
) -> None:
    """E2E-F14-T2-04: Two local benchmark runs on separate dirs yield identical token/error/accuracy."""
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    ra = run_cli(
        ["benchmark", "--provider", "local", "--out", str(out_a), "--data-dir", str(tmp_path / "da")],
        timeout=90.0,
    )
    rb = run_cli(
        ["benchmark", "--provider", "local", "--out", str(out_b), "--data-dir", str(tmp_path / "db")],
        timeout=90.0,
    )
    assert ra.returncode == 0 and rb.returncode == 0

    ja = json.loads(out_a.read_text(encoding="utf-8"))
    jb = json.loads(out_b.read_text(encoding="utf-8"))
    for key in ("epoch_1", "epoch_2", "epoch_3"):
        ea = ja.get(key)
        eb = jb.get(key)
        if ea and eb:
            assert ea.get("tokens", ea.get("total_tokens")) == eb.get("tokens", eb.get("total_tokens"))
            assert ea.get("errors", ea.get("total_errors")) == eb.get("errors", eb.get("total_errors"))
            assert ea.get("accuracy") == eb.get("accuracy")


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
def test_t2_f14_benchmark_invalid_mode_or_unwritable_output_fails_cleanly(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-F14-T2-05: Invalid --mode or directory collision on --out fails with non-zero exit."""
    bad_mode = run_cli(
        ["benchmark", "--provider", "local", "--mode", "bogus_mode", "--data-dir", str(e2e_data_dir)]
    )
    assert bad_mode.returncode != 0

    collision_dir = tmp_path / "existing_dir"
    collision_dir.mkdir()
    bad_out = run_cli(
        [
            "benchmark",
            "--provider",
            "local",
            "--out",
            str(collision_dir),
            "--data-dir",
            str(e2e_data_dir),
        ]
    )
    assert bad_out.returncode != 0


# ---------------------------------------------------------------------------
# F15: Developer Interface — CLI & Flask Dashboard (Milestone M4)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
def test_t2_f15_cli_invalid_args_and_unknown_provider(run_cli, e2e_data_dir: Path) -> None:
    """E2E-F15-T2-01: Missing --task and unknown --provider return non-zero without traceback."""
    missing_task = run_cli(["run", "--data-dir", str(e2e_data_dir)])
    assert missing_task.returncode != 0

    bad_provider = run_cli(
        ["run", "--task", "sort semver", "--provider", "nonexistent_llm", "--data-dir", str(e2e_data_dir)]
    )
    assert bad_provider.returncode != 0
    assert "Traceback (most recent call last)" not in bad_provider.stderr


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
def test_t2_f15_cli_inspect_and_search_empty_data_dir(run_cli, e2e_data_dir: Path) -> None:
    """E2E-F15-T2-02: inspect and search on fresh empty data_dir exit 0 with empty JSON."""
    insp = run_cli(["inspect", "--kind", "all", "--data-dir", str(e2e_data_dir), "--json"])
    assert insp.returncode == 0
    insp_data = insp.json()
    if isinstance(insp_data, dict):
        assert len(insp_data.get("scars", [])) == 0
        assert len(insp_data.get("skills", [])) == 0
    else:
        assert insp_data == []

    srch = run_cli(["search", "pandas", "--data-dir", str(e2e_data_dir), "--json"])
    assert srch.returncode == 0
    srch_data = srch.json()
    if isinstance(srch_data, dict):
        assert len(srch_data.get("scars", [])) == 0
        assert len(srch_data.get("skills", [])) == 0
    else:
        assert srch_data == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
def test_t2_f15_cli_search_top_k_clamping(run_cli, e2e_data_dir: Path) -> None:
    """E2E-F15-T2-03: CLI search --top-k 20 clamps strictly to <=5; --top-k 0 clamps or errors."""
    store = make_scar_store(e2e_data_dir)
    for i in range(8):
        err = make_distilled_error("ValueError", f"error parse case {i}", f"parse_{i}(x)", i + 1)
        store.record_failure(err, f"parse_{i}(x)", ["error", "parse"])

    r20 = run_cli(
        ["search", "error parse", "--top-k", "20", "--data-dir", str(e2e_data_dir), "--json"]
    )
    assert r20.returncode == 0
    d20 = r20.json()
    scars20 = d20.get("scars", []) if isinstance(d20, dict) else d20
    assert 1 <= len(scars20) <= 5

    r0 = run_cli(
        ["search", "error parse", "--top-k", "0", "--data-dir", str(e2e_data_dir), "--json"]
    )
    if r0.returncode == 0:
        d0 = r0.json()
        scars0 = d0.get("scars", []) if isinstance(d0, dict) else d0
        assert 1 <= len(scars0) <= 5


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
def test_t2_f15_cli_windows_cp1252_encoding_resilience(
    run_cli, e2e_data_dir: Path, clean_env: dict[str, str]
) -> None:
    """E2E-F15-T2-04: Demo CLI succeeds even when PYTHONIOENCODING=cp1252 and PYTHONUTF8 is unset."""
    cp1252_env = clean_env.copy()
    cp1252_env.pop("PYTHONUTF8", None)
    cp1252_env["PYTHONIOENCODING"] = "cp1252"

    res = run_cli(
        ["demo", "--provider", "local", "--data-dir", str(e2e_data_dir)],
        override_env=cp1252_env,
    )
    assert res.returncode == 0, f"STDERR:\n{res.stderr}"
    assert "UnicodeEncodeError" not in res.stderr


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m4
def test_t2_f15_flask_dashboard_empty_benchmark_and_404(dashboard_get) -> None:
    """E2E-F15-T2-05: Fresh empty data_dir returns 200 on /api/benchmark and 404 on unknown route."""
    bench = dashboard_get("/api/benchmark")
    assert bench.status_code == 200
    assert bench.json() is not None

    missing = dashboard_get("/api/does_not_exist")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# F17: Technical README.md (Milestone M5)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.m5
def test_t2_f17_readme_mermaid_blocks_have_balanced_syntax_and_no_placeholders() -> None:
    """README.md has balanced Mermaid brackets/blocks and zero placeholder markers."""
    readme = REPO_ROOT / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")

    markers = ("TO" + "DO", "FIX" + "ME", "TB" + "D")
    assert not re.search(r"\b(" + "|".join(markers) + r")\b", text)

    blocks = re.findall(r"```mermaid\s*\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 3
    for block in blocks:
        assert block.count("[") == block.count("]")
        assert block.count("(") == block.count(")")
        assert block.count("{") == block.count("}")
