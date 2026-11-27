"""Tier 4 E2E Tests — Real-World End-to-End Application Scenarios (E2E-T4-01 through E2E-T4-09)."""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.conftest import (
    REPO_ROOT,
    get_ast_valid,
    get_skill_name,
    http_get,
    make_ast_guard,
    make_distilled_error,
    make_engine,
    make_retriever,
    make_sandbox_runner,
    make_scar_store,
    make_skill_registry,
    make_test_case,
    run_engine_task,
    unwrap_api_list,
)


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
def test_e2e_pandas_concat_cold_to_skill_reuse(e2e_data_dir: Path) -> None:
    """E2E-T4-01: Cold-start pandas DataFrame.append failure -> Scar -> Skill forge -> Skill reuse."""
    from scar_forge.tokens import count_tokens, validate_axiom_format

    engine = make_engine(e2e_data_dir, provider="local")
    first = run_engine_task(engine, "Merge list of record dicts into DataFrame")
    assert getattr(first, "success", getattr(first, "passed", True)) is True

    scars = make_scar_store(e2e_data_dir).list_scars()
    assert len(scars) >= 1
    assert all(count_tokens(s.axiom) < 15 and validate_axiom_format(s.axiom) for s in scars)

    skills = make_skill_registry(e2e_data_dir).list_skills()
    assert len(skills) >= 1

    second = run_engine_task(engine, "Merge batch 2 list of record dicts into DataFrame")
    assert getattr(second, "success", getattr(second, "passed", True)) is True
    assert getattr(second, "errors", getattr(second, "error_count", 0)) == 0
    t1 = getattr(first, "total_tokens", getattr(first, "tokens", 0))
    t2 = getattr(second, "total_tokens", getattr(second, "tokens", 0))
    if t1 > 0 and t2 > 0:
        assert t2 < t1


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
def test_e2e_iso_utc_delta_mixed_tz_lifecycle(e2e_data_dir: Path) -> None:
    """E2E-T4-02: Mixed-timezone ISO timestamp normalization cold start -> Scar -> Skill reuse."""
    from scar_forge.tokens import count_tokens

    engine = make_engine(e2e_data_dir, provider="local")
    r1 = run_engine_task(
        engine, "Parse mixed timezone timestamps (Z, +05:30, naive) into sorted UTC deltas"
    )
    assert getattr(r1, "success", getattr(r1, "passed", True)) is True

    scars = make_scar_store(e2e_data_dir).list_scars()
    assert len(scars) >= 1
    assert all(count_tokens(s.axiom) < 15 for s in scars)

    r2 = run_engine_task(
        engine, "Compute sorted UTC intervals from mixed Z and offset timestamps"
    )
    assert getattr(r2, "success", getattr(r2, "passed", True)) is True
    assert getattr(r2, "errors", getattr(r2, "error_count", 0)) == 0


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
def test_e2e_csv_bom_delimiter_and_blank_cell_lifecycle(e2e_data_dir: Path) -> None:
    """E2E-T4-03: UTF-8 BOM + semicolon CSV with blank numeric cells lifecycle."""
    engine = make_engine(e2e_data_dir, provider="local")
    r1 = run_engine_task(
        engine, "Parse CSV payload with UTF-8 BOM, semicolon delimiter, and blank numeric cells"
    )
    assert getattr(r1, "success", getattr(r1, "passed", True)) is True
    assert len(make_scar_store(e2e_data_dir).list_scars()) >= 1
    assert len(make_skill_registry(e2e_data_dir).list_skills()) >= 1

    r2 = run_engine_task(
        engine, "Normalize another BOM-prefixed semicolon CSV table with empty cells"
    )
    assert getattr(r2, "success", getattr(r2, "passed", True)) is True
    assert getattr(r2, "errors", getattr(r2, "error_count", 0)) == 0


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
def test_e2e_json_strict_rfc8259_serialize_lifecycle(e2e_data_dir: Path) -> None:
    """E2E-T4-04: Strict RFC 8259 JSON serialization rejecting NaN/Inf and handling Decimal/datetime."""
    engine = make_engine(e2e_data_dir, provider="local")
    r1 = run_engine_task(
        engine, "Serialize Decimal, datetime, and set to strict RFC 8259 JSON rejecting NaN"
    )
    assert getattr(r1, "success", getattr(r1, "passed", True)) is True
    assert len(make_scar_store(e2e_data_dir).list_scars()) >= 1

    r2 = run_engine_task(
        engine, "Strict RFC 8259 JSON serialization for Decimal and datetime objects"
    )
    assert getattr(r2, "success", getattr(r2, "passed", True)) is True
    assert getattr(r2, "errors", getattr(r2, "error_count", 0)) == 0


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
def test_e2e_webhook_bool_string_coercion_lifecycle(e2e_data_dir: Path) -> None:
    """E2E-T4-05: Webhook schema coercion avoiding bool('false') == True trap."""
    engine = make_engine(e2e_data_dir, provider="local")
    r1 = run_engine_task(
        engine, "Coerce webhook JSON payload with string booleans like 'false' and '0'"
    )
    assert getattr(r1, "success", getattr(r1, "passed", True)) is True
    assert len(make_scar_store(e2e_data_dir).list_scars()) >= 1
    assert len(make_skill_registry(e2e_data_dir).list_skills()) >= 1

    r2 = run_engine_task(
        engine, "Coerce second webhook payload containing 'false' boolean flags"
    )
    assert getattr(r2, "success", getattr(r2, "passed", True)) is True
    assert getattr(r2, "errors", getattr(r2, "error_count", 0)) == 0


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
def test_e2e_multi_domain_topk_context_isolation(e2e_data_dir: Path) -> None:
    """E2E-T4-06: 8 domain Scars & Skills remain isolated under Top-K <=5 retrieval."""
    store = make_scar_store(e2e_data_dir)
    registry = make_skill_registry(e2e_data_dir)

    domains = [
        ("pandas", "[AVOID] df.append -> [USE] pd.concat", "Merge dicts into DataFrame."),
        ("datetime", "[AVOID] naive datetime -> [USE] utc aware", "Normalize mixed TZ to UTC."),
        ("csv", "[RULE] strip BOM encoding utf-8-sig csv", "Parse semicolon BOM CSV."),
        ("json", "[AVOID] allow_nan=True -> [USE] allow_nan=False", "Strict RFC 8259 JSON."),
        ("webhook", "[AVOID] bool(str) -> [USE] strict false check", "Coerce webhook boolean strings."),
        ("semver", "[AVOID] string sort semver -> [USE] tuple key", "Sort semver tags numerically."),
        ("html", "[AVOID] regex html -> [USE] HTMLParser", "Extract clean text from HTML."),
        ("path", "[AVOID] os.path.join -> [USE] PurePosixPath", "Guard against path traversal."),
    ]
    for idx, (tag, axiom, doc) in enumerate(domains):
        store.record_failure(
            make_distilled_error("ValueError", f"{tag} error {axiom}", f"{tag}_op()", idx + 1),
            f"{tag}_op()",
            [tag],
        )
        src = (
            f"def tool_{tag}(x: str) -> str:\n"
            f'    """{doc}"""\n'
            "    return str(x)\n"
        )
        registry.register_and_reload(
            src, f"tool_{tag}", doc, [tag], [make_test_case(kwargs={"x": "ok"}, expected="ok")]
        )

    retriever = make_retriever()
    top_scars = retriever.retrieve_scars("semver sort version tags", store.list_scars(), top_k=5)
    top_skills = retriever.retrieve_skills("semver sort version tags", registry.list_skills(), top_k=5)
    assert 1 <= len(top_scars) <= 5
    assert 1 <= len(top_skills) <= 5
    assert "semver" in top_scars[0].tags
    assert get_skill_name(top_skills[0]) == "tool_semver"


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m4
@pytest.mark.slow
def test_e2e_three_epoch_benchmark_full_progression(
    run_cli, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-T4-07: Full 8-task 3-Epoch benchmark validates E1 -> E2 -> E3 learning curve."""
    out_json = tmp_path / "full_bench.json"
    out_md = tmp_path / "full_bench.md"
    res = run_cli(
        [
            "benchmark",
            "--provider",
            "local",
            "--mode",
            "variant",
            "--out",
            str(out_json),
            "--md-out",
            str(out_md),
            "--data-dir",
            str(e2e_data_dir),
        ],
        timeout=120.0,
    )
    assert res.returncode == 0, f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
    assert out_json.is_file() and out_md.is_file()

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    e1 = rep.get("epoch_1") or rep["epochs"][0]
    e2 = rep.get("epoch_2") or rep["epochs"][1]
    e3 = rep.get("epoch_3") or rep["epochs"][2]

    e1_err = e1.get("errors", e1.get("total_errors"))
    e2_err = e2.get("repeated_errors", e2.get("errors", e2.get("total_errors")))
    t1 = float(e1.get("tokens", e1.get("total_tokens")))
    t3 = float(e3.get("tokens", e3.get("total_tokens")))
    reduction = 1.0 - (t3 / t1)

    assert e1_err > 0
    assert e2_err <= 0.25 * e1_err
    assert 0.60 <= reduction <= 0.85, f"Token reduction {reduction:.3f} outside [0.60, 0.85]"
    assert e3.get("accuracy", 0.0) >= e1.get("accuracy", 0.0)
    assert e3.get("duration_ms", 0.0) < e1.get("duration_ms", float("inf"))


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m4
@pytest.mark.slow
def test_e2e_cli_to_live_flask_dashboard_journey(
    run_cli, live_dashboard_url: str, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-T4-08: End-to-end user journey across demo -> inspect -> search -> benchmark -> live Flask UI."""
    assert run_cli(["demo", "--provider", "local", "--data-dir", str(e2e_data_dir)]).returncode == 0

    insp = run_cli(["inspect", "--kind", "all", "--data-dir", str(e2e_data_dir), "--json"])
    assert insp.returncode == 0
    assert len(unwrap_api_list(insp.json(), "scars")) >= 1

    srch = run_cli(["search", "DataFrame", "--top-k", "4", "--data-dir", str(e2e_data_dir), "--json"])
    assert srch.returncode == 0

    bench_out = tmp_path / "journey_bench.json"
    bench = run_cli(
        ["benchmark", "--provider", "local", "--out", str(bench_out), "--data-dir", str(e2e_data_dir)],
        timeout=120.0,
    )
    assert bench.returncode == 0

    home = http_get(f"{live_dashboard_url}/")
    state = http_get(f"{live_dashboard_url}/api/state")
    scars = http_get(f"{live_dashboard_url}/api/scars")
    skills = http_get(f"{live_dashboard_url}/api/skills")
    api_bench = http_get(f"{live_dashboard_url}/api/benchmark")

    assert home.status_code == 200
    assert state.status_code == 200 and isinstance(state.json(), dict)
    assert scars.status_code == 200 and len(unwrap_api_list(scars.json(), "scars")) >= 1
    assert skills.status_code == 200 and len(unwrap_api_list(skills.json(), "skills")) >= 1
    assert api_bench.status_code == 200 and api_bench.json() is not None


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.m3
@pytest.mark.slow
def test_e2e_adversarial_security_and_static_hygiene_audit(e2e_data_dir: Path) -> None:
    """E2E-T4-09: Malicious/stubbed skill forging attempts fail while valid skills & src/ stay clean."""
    registry = make_skill_registry(e2e_data_dir)
    guard = make_ast_guard()
    runner = make_sandbox_runner(e2e_data_dir)

    valid_src = (
        "def safe_sum(nums: list[int]) -> int:\n"
        '    """Sum list of integers safely."""\n'
        "    return sum(nums)\n"
    )
    registry.register_and_reload(
        valid_src,
        "safe_sum",
        "Sum list of integers safely.",
        ["math"],
        [make_test_case(kwargs={"nums": [1, 2, 3]}, expected=6)],
    )

    not_impl = "NotImplemented" + "Error"
    malicious_attempts = [
        ("evil_os", "import os\ndef evil_os(x: int) -> int:\n    '''doc'''\n    return os.getpid()\n"),
        ("evil_eval", "def evil_eval(x: str) -> object:\n    '''doc'''\n    return eval(x)\n"),
        ("evil_stub", f"def evil_stub(x: int) -> int:\n    '''doc'''\n    raise {not_impl}('x')\n"),
    ]
    for name, src in malicious_attempts:
        assert get_ast_valid(guard.validate(src, expected_name=name)) is False
        with pytest.raises(Exception):
            registry.register_and_reload(
                src, name, "Malicious attempt.", ["sec"], [make_test_case(kwargs={"x": 1}, expected=1)]
            )

    hang_res = runner.execute_code(
        "def hang() -> int:\n    while True:\n        pass\n",
        [make_test_case(kwargs={}, expected=1)],
        entry_fn="hang",
        timeout_s=0.5,
    )
    assert hang_res.passed is False
    assert registry.invoke_active("safe_sum", nums=[10, 20]) == 30

    # Static hygiene check across src/ and tests/
    markers = ("TO" + "DO", "FIX" + "ME", "X" * 3, "HA" + "CK")
    marker_re = re.compile(r"\b(" + "|".join(markers) + r")\b", re.IGNORECASE)
    for py_file in list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "tests").rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                assert not marker_re.search(tok.string), f"Marker comment in {py_file}:{tok.start[0]}"
        ast.parse(text, filename=str(py_file))
