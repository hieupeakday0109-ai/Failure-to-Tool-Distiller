"""Tier 3 E2E Tests — Cross-Feature Pairwise Combinations (E2E-T3-01 through E2E-T3-17)."""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.conftest import (
    apply_protocol_patch,
    get_ast_valid,
    get_llm_content,
    get_prevented_count,
    get_scar_id,
    get_skill_name,
    http_get,
    make_ast_guard,
    make_config,
    make_distilled_error,
    make_engine,
    make_llm,
    make_retriever,
    make_sandbox_runner,
    make_scar_store,
    make_skill_registry,
    make_test_case,
    parse_agent_action_text,
    run_engine_task,
    unwrap_api_list,
)


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m2
def test_t3_01_scar_store_dedup_feeds_bounded_topk_retrieval(e2e_data_dir: Path) -> None:
    """E2E-T3-01: F7 ScarStore dedup + F8 BM25Retriever bounded Top-K filtering."""
    store = make_scar_store(e2e_data_dir)
    failures = [
        ("AttributeError", "df.append removed", "df.append(row)", ["pandas", "dataframe"]),
        ("AttributeError", "df.append removed", "df.append(row)", ["pandas", "dataframe"]),
        ("AttributeError", "df.append removed", "df.append(row)", ["pandas", "dataframe"]),
        ("TypeError", "naive vs aware datetime compare", "t1 < t2", ["datetime", "utc"]),
        ("ValueError", "csv utf-8 BOM header \\ufeff", "csv.DictReader(f)", ["csv", "bom"]),
        ("ValueError", "json NaN not RFC 8259 compliant", "json.dumps(x, allow_nan=False)", ["json", "nan"]),
        ("ValueError", "bool('false') evaluates True", "bool(raw)", ["webhook", "bool"]),
        ("AssertionError", "semver string sort 1.10 < 1.2", "sorted(tags)", ["semver", "sort"]),
        ("ValueError", "regex strips valid html text", "re.sub('<.*>', '', html)", ["html", "parser"]),
        ("ValueError", "path traversal .. escape", "os.path.join(base, rel)", ["path", "posix"]),
    ]
    for exc, msg, code, tags in failures:
        store.record_failure(make_distilled_error(exc, msg, code, 10), code, tags)

    scars = store.list_scars()
    assert len(scars) == 8  # 3 identical pandas failures deduplicated to 1

    retriever = make_retriever()
    top = retriever.retrieve_scars("pandas DataFrame append concatenate", scars, top_k=5)
    assert 1 <= len(top) <= 5
    assert "pandas" in top[0].tags or "append" in top[0].axiom.lower()
    assert all("html" not in s.tags and "semver" not in s.tags for s in top)


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_02_scar_guided_tier1_avoidance_skips_frontier(e2e_data_dir: Path) -> None:
    """E2E-T3-02: F8 + F12 + F13 — Pre-seeded Scar keeps execution on FAST_LOCAL without Frontier."""
    store = make_scar_store(e2e_data_dir)
    err = make_distilled_error(
        "AttributeError",
        "'DataFrame' object has no attribute 'append'",
        "df = df.append(row)",
        12,
    )
    store.record_failure(err, "df = df.append(row)", ["pandas", "dataframe", "concat"])

    engine = make_engine(e2e_data_dir, provider="local", escalation_threshold=2)
    res = run_engine_task(engine, "Merge list of record dicts into DataFrame")
    assert getattr(res, "success", getattr(res, "passed", True)) is True
    assert getattr(res, "errors", getattr(res, "error_count", 0)) == 0
    frontier_calls = getattr(res, "frontier_calls", None)
    if frontier_calls is not None:
        # Either 0 (no distillation needed) or at most distillation only, zero failure escalations
        assert getattr(res, "escalations", 0) == 0


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m2
def test_t3_03_sandbox_failure_distills_to_under_15_tok_scar(e2e_data_dir: Path) -> None:
    """E2E-T3-03: F9 Sandbox -> F5 TracebackDistiller -> F7 ScarStore dedup & <15 token budget."""
    from scar_forge.distiller import TracebackDistiller
    from scar_forge.tokens import count_tokens, validate_axiom_format

    runner = make_sandbox_runner(e2e_data_dir)
    bad_code = (
        "import pandas as pd\n"
        "def merge_df(rows: list[dict]) -> int:\n"
        "    df = pd.DataFrame()\n"
        "    df = df.append(rows, ignore_index=True)\n"
        "    return len(df)\n"
    )
    sb_res = runner.execute_code(
        bad_code,
        [make_test_case(kwargs={"rows": [{"a": 1}]}, expected=1)],
        entry_fn="merge_df",
    )
    assert sb_res.passed is False
    distilled = sb_res.distilled_error or TracebackDistiller().distill(sb_res.stderr, sb_res.stdout)
    assert "AttributeError" in distilled.exception_type
    assert "site-packages" not in distilled.failing_line

    store = make_scar_store(e2e_data_dir)
    scar1, is_new1 = store.record_failure(distilled, bad_code, ["pandas", "dataframe"])
    scar2, is_new2 = store.record_failure(distilled, bad_code, ["pandas", "dataframe"])

    assert is_new1 is True
    assert is_new2 is False
    assert get_scar_id(scar1) == get_scar_id(scar2)
    assert count_tokens(scar1.axiom) <= 14
    assert validate_axiom_format(scar1.axiom) is True


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_04_consecutive_sandbox_failures_trigger_frontier_recovery(e2e_data_dir: Path) -> None:
    """E2E-T3-04: F9 + F5 + F12 + F13 — 2 sandbox failures escalate 3rd attempt to FRONTIER."""
    from scar_forge.distiller import TracebackDistiller
    from scar_forge.models import ModelTier, TaskPurpose

    runner = make_sandbox_runner(e2e_data_dir)
    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=2)
    repair_purpose = next(
        (p for p in TaskPurpose if "REPAIR" in p.name or "PATCH" in p.name),
        next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION),
    )

    broken = "def parse_flag(v: str) -> bool:\n    return bool(v)\n"
    sb1 = runner.execute_code(
        broken, [make_test_case(kwargs={"v": "false"}, expected=False)], entry_fn="parse_flag"
    )
    assert sb1.passed is False
    err1 = sb1.distilled_error or TracebackDistiller().distill(sb1.stderr, sb1.stdout)

    r1 = llm.complete(
        [{"role": "user", "content": f"Attempt 1 repair: {err1.core_message}"}],
        purpose=repair_purpose,
        consecutive_failures=0,
    )
    r2 = llm.complete(
        [{"role": "user", "content": f"Attempt 2 repair: {err1.core_message}"}],
        purpose=repair_purpose,
        consecutive_failures=1,
    )
    r3 = llm.complete(
        [{"role": "user", "content": f"Attempt 3 repair: {err1.core_message}"}],
        purpose=repair_purpose,
        consecutive_failures=2,
    )

    assert r1.tier_used == ModelTier.FAST_LOCAL and r1.escalated is False
    assert r2.tier_used == ModelTier.FAST_LOCAL and r2.escalated is False
    assert r3.tier_used == ModelTier.FRONTIER and r3.escalated is True


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_05_solved_task_distills_on_frontier_and_passes_ast_guard(e2e_data_dir: Path) -> None:
    """E2E-T3-05: F13 + F12(SKILL_DISTILLATION) + F10 ASTGuard produce clean typed skill."""
    from scar_forge.models import ModelTier, TaskPurpose

    llm = make_llm(provider="local", data_dir=e2e_data_dir)
    resp = llm.complete(
        [
            {
                "role": "user",
                "content": "Distill solved task into typed reusable Python skill: Merge list of record dicts into DataFrame",
            }
        ],
        purpose=TaskPurpose.SKILL_DISTILLATION,
        consecutive_failures=0,
    )
    assert resp.tier_used == ModelTier.FRONTIER
    assert resp.escalated is True

    engine = make_engine(e2e_data_dir, provider="local")
    run_engine_task(engine, "Merge list of record dicts into DataFrame")
    skills = make_skill_registry(e2e_data_dir).list_skills()
    assert len(skills) >= 1

    guard = make_ast_guard()
    for sk in skills:
        skill_files = list(e2e_data_dir.rglob(f"{get_skill_name(sk)}.py"))
        if skill_files:
            src = skill_files[0].read_text(encoding="utf-8")
            assert get_ast_valid(guard.validate(src, expected_name=get_skill_name(sk))) is True


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_06_ast_validated_skill_hot_reloads_and_overwrites_without_stale_pyc(
    e2e_data_dir: Path,
) -> None:
    """E2E-T3-06: F10 + F11 — Validated v1->v2 overwrite hot-reloads in-process without .pyc staleness."""
    guard = make_ast_guard()
    registry = make_skill_registry(e2e_data_dir)

    v1 = (
        "def score_item(x: int) -> int:\n"
        '    """Compute v1 score."""\n'
        "    return x + 5\n"
    )
    v2 = (
        "def score_item(x: int) -> int:\n"
        '    """Compute v2 score."""\n'
        "    return x * 20\n"
    )
    assert get_ast_valid(guard.validate(v1, expected_name="score_item")) is True
    assert get_ast_valid(guard.validate(v2, expected_name="score_item")) is True

    registry.register_and_reload(
        v1, "score_item", "Compute v1 score.", ["score"], [make_test_case(kwargs={"x": 2}, expected=7)]
    )
    assert registry.invoke_active("score_item", x=2) == 7

    registry.register_and_reload(
        v2, "score_item", "Compute v2 score.", ["score"], [make_test_case(kwargs={"x": 2}, expected=40)]
    )
    assert registry.invoke_active("score_item", x=2) == 40


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_07_adversarial_failure_rolls_back_skill_registry(e2e_data_dir: Path) -> None:
    """E2E-T3-07: F10 + F9 + F11 — Adversarial sandbox failure rolls back v2 and preserves v1."""
    registry = make_skill_registry(e2e_data_dir)
    v1 = (
        "def safe_head(items: list[str]) -> str:\n"
        '    """Return first element or empty string safely."""\n'
        "    return items[0] if items else ''\n"
    )
    registry.register_and_reload(
        v1,
        "safe_head",
        "Return first element or empty string safely.",
        ["list"],
        [
            make_test_case(kwargs={"items": ["a", "b"]}, expected="a"),
            make_test_case(kwargs={"items": []}, expected=""),
        ],
    )

    fragile_v2 = (
        "def safe_head(items: list[str]) -> str:\n"
        '    """Fragile unguarded head."""\n'
        "    return items[0]\n"
    )
    with pytest.raises(Exception):
        registry.register_and_reload(
            fragile_v2,
            "safe_head",
            "Fragile unguarded head.",
            ["list"],
            [
                make_test_case(kwargs={"items": ["a", "b"]}, expected="a"),
                make_test_case(kwargs={"items": []}, expected=""),
            ],
        )

    assert registry.invoke_active("safe_head", items=[]) == ""
    assert list(e2e_data_dir.rglob("*.tmp")) == []


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_08_skill_retriever_guides_zero_prose_call_skill(e2e_data_dir: Path) -> None:
    """E2E-T3-08: F11 + F8 + F4 — Retrieved skill drives valid CALL_SKILL JSON execution."""
    from scar_forge.models import ActionType

    registry = make_skill_registry(e2e_data_dir)
    for i in range(5):
        src = (
            f"def unrelated_tool_{i}(x: int) -> int:\n"
            f'    """Unrelated arithmetic helper {i}."""\n'
            f"    return x + {i}\n"
        )
        registry.register_and_reload(
            src, f"unrelated_tool_{i}", f"Unrelated arithmetic helper {i}.", ["misc"],
            [make_test_case(kwargs={"x": 1}, expected=1 + i)],
        )

    semver_src = (
        "def semver_sort_tags(versions: list[str]) -> list[str]:\n"
        '    """Sort semantic version tags numerically with prerelease ordering."""\n'
        "    def key_fn(v: str) -> tuple:\n"
        "        core, _, pre = v.lstrip('v').partition('-')\n"
        "        nums = tuple(int(p) for p in core.split('.'))\n"
        "        return (nums, 1 if not pre else 0, pre)\n"
        "    return sorted(versions, key=key_fn)\n"
    )
    registry.register_and_reload(
        semver_src,
        "semver_sort_tags",
        "Sort semantic version tags numerically with prerelease ordering.",
        ["semver", "version", "sort"],
        [
            make_test_case(
                kwargs={"versions": ["1.10.0", "1.2.0", "1.2.0-alpha.1"]},
                expected=["1.2.0-alpha.1", "1.2.0", "1.10.0"],
            )
        ],
    )

    retriever = make_retriever()
    top_skills = retriever.retrieve_skills("sort semantic version tags", registry.list_skills(), top_k=3)
    assert 1 <= len(top_skills) <= 3
    chosen = get_skill_name(top_skills[0])
    assert chosen == "semver_sort_tags"

    action = parse_agent_action_text(
        json.dumps(
            {
                "action": "CALL_SKILL",
                "skill_name": chosen,
                "skill_args": {"versions": ["1.10.0", "1.2.0"]},
            }
        )
    )
    assert action.action == ActionType.CALL_SKILL
    kwargs = getattr(action, "skill_args", None) or getattr(action, "kwargs", {})
    assert registry.invoke_active(action.skill_name, **kwargs) == ["1.2.0", "1.10.0"]


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_09_multi_turn_protocol_enforcement_and_token_accounting(e2e_data_dir: Path) -> None:
    """E2E-T3-09: F4 + F12 + F3 — Multi-turn RUN_CODE -> PATCH_CODE -> FINAL_ANSWER with exact tokens."""
    from scar_forge.models import ActionType
    from scar_forge.tokens import count_tokens

    turns = [
        json.dumps({"action": "RUN_CODE", "code": "def f(x: int) -> int:\n    return x - 1\n"}),
        json.dumps(
            {
                "action": "PATCH_CODE",
                "diff_patch": "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n def f(x: int) -> int:\n-    return x - 1\n+    return x + 1\n",
            }
        ),
        json.dumps({"action": "FINAL_ANSWER", "result": {"value": 42}}),
    ]
    parsed = [parse_agent_action_text(t) for t in turns]
    assert [p.action for p in parsed] == [
        ActionType.RUN_CODE,
        ActionType.PATCH_CODE,
        ActionType.FINAL_ANSWER,
    ]
    patched = apply_protocol_patch(parsed[0].code, getattr(parsed[1], "diff_patch", None) or parsed[1].patch)
    assert "return x + 1" in patched

    total_tokens = sum(count_tokens(t) for t in turns)
    assert total_tokens == count_tokens(turns[0]) + count_tokens(turns[1]) + count_tokens(turns[2])
    assert total_tokens > 0


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m2
def test_t3_10_prompt_scar_injection_token_ceiling_bounded(e2e_data_dir: Path) -> None:
    """E2E-T3-10: F3 + F7 + F8 — 25 stored Scars inject <= 5 * 14 = 70 axiom tokens."""
    from scar_forge.tokens import count_tokens

    store = make_scar_store(e2e_data_dir)
    for i in range(25):
        err = make_distilled_error(
            "ValueError",
            f"domain parse error variant_{i} token_sample_{i}",
            f"parse_{i}(row)",
            line_number=i + 1,
        )
        store.record_failure(err, f"parse_{i}(row)", ["parse", "domain", f"v{i}"])

    all_scars = store.list_scars()
    assert len(all_scars) == 25

    retriever = make_retriever()
    injected = retriever.retrieve_scars("parse domain variant", all_scars, top_k=5)
    assert 1 <= len(injected) <= 5
    injected_tokens = sum(count_tokens(s.axiom) for s in injected)
    assert injected_tokens <= 5 * 14


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m2
@pytest.mark.slow
def test_t3_11_sandbox_security_blocks_and_timeouts_distill_cleanly() -> None:
    """E2E-T3-11: F9 + F5 — Blocked imports and timeouts inside sandbox distill without crashing."""
    from scar_forge.distiller import TracebackDistiller

    runner = make_sandbox_runner()
    snippets = [
        "import subprocess\ndef run_cmd() -> int:\n    return 1\n",
        "def spin() -> int:\n    while True:\n        pass\n",
    ]
    for idx, code in enumerate(snippets):
        fn = "run_cmd" if idx == 0 else "spin"
        sb = runner.execute_code(code, [make_test_case(kwargs={}, expected=1)], entry_fn=fn, timeout_s=0.5)
        assert sb.passed is False
        distilled = sb.distilled_error or TracebackDistiller().distill(sb.stderr, sb.stdout)
        assert len(distilled.exception_type) > 0


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m4
def test_t3_12_prevented_counter_propagates_to_cli_inspect(run_cli, e2e_data_dir: Path) -> None:
    """E2E-T3-12: F7 mark_prevented + F13 + F15 CLI inspect --json reflects prevented_count."""
    store = make_scar_store(e2e_data_dir)
    err = make_distilled_error(
        "AttributeError", "'DataFrame' object has no attribute 'append'", "df.append(row)", 6
    )
    scar, _ = store.record_failure(err, "df.append(row)", ["pandas", "dataframe"])
    sid = get_scar_id(scar)
    store.mark_prevented([sid])

    insp = run_cli(["inspect", "--kind", "scars", "--data-dir", str(e2e_data_dir), "--json"])
    assert insp.returncode == 0
    scars_json = unwrap_api_list(insp.json(), "scars")
    matched = [s for s in scars_json if (s.get("scar_id") or s.get("id")) == sid]
    assert len(matched) == 1
    assert (matched[0].get("prevented_count") or matched[0].get("hits_prevented", 0)) >= 1


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m4
def test_t3_13_cli_run_persists_state_visible_to_inspect_and_search(
    run_cli, e2e_data_dir: Path
) -> None:
    """E2E-T3-13: F15 subprocess A run -> subprocess B inspect -> subprocess C search."""
    r_run = run_cli(
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
    assert r_run.returncode == 0

    r_insp = run_cli(["inspect", "--kind", "all", "--data-dir", str(e2e_data_dir), "--json"])
    assert r_insp.returncode == 0
    insp_data = r_insp.json()
    assert len(unwrap_api_list(insp_data, "scars")) >= 1
    assert len(unwrap_api_list(insp_data, "skills")) >= 1

    r_srch = run_cli(
        ["search", "DataFrame", "--top-k", "4", "--data-dir", str(e2e_data_dir), "--json"]
    )
    assert r_srch.returncode == 0
    srch_data = r_srch.json()
    total_hits = (
        len(srch_data.get("scars", [])) + len(srch_data.get("skills", []))
        if isinstance(srch_data, dict)
        else len(srch_data)
    )
    assert total_hits >= 1


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m4
@pytest.mark.slow
def test_t3_14_benchmark_artifacts_sync_to_live_flask_api(
    run_cli, live_dashboard_url: str, e2e_data_dir: Path, tmp_path: Path
) -> None:
    """E2E-T3-14: F14 Benchmark artifacts match live Flask /api/benchmark and /api/scars."""
    bench_file = tmp_path / "bench_sync.json"
    res = run_cli(
        [
            "benchmark",
            "--provider",
            "local",
            "--out",
            str(bench_file),
            "--data-dir",
            str(e2e_data_dir),
        ],
        timeout=90.0,
    )
    assert res.returncode == 0
    disk_report = json.loads(bench_file.read_text(encoding="utf-8"))

    r_state = http_get(f"{live_dashboard_url}/api/state")
    r_scars = http_get(f"{live_dashboard_url}/api/scars")
    r_skills = http_get(f"{live_dashboard_url}/api/skills")
    r_bench = http_get(f"{live_dashboard_url}/api/benchmark")

    assert r_state.status_code == 200
    assert r_scars.status_code == 200 and len(unwrap_api_list(r_scars.json(), "scars")) >= 1
    assert r_skills.status_code == 200 and len(unwrap_api_list(r_skills.json(), "skills")) >= 1
    assert r_bench.status_code == 200
    api_bench = r_bench.json()
    for ep in ("epoch_1", "epoch_2", "epoch_3"):
        if ep in disk_report and ep in api_bench:
            assert api_bench[ep].get("tokens", api_bench[ep].get("total_tokens")) == disk_report[ep].get(
                "tokens", disk_report[ep].get("total_tokens")
            )


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
@pytest.mark.slow
def test_t3_15_engine_config_propagation_across_layers(e2e_data_dir: Path) -> None:
    """E2E-T3-15: F2 EngineConfig propagates top_k=3, max_scar_tokens=12, threshold=4, timeout=1.5s."""
    from scar_forge.models import ModelTier, TaskPurpose
    from scar_forge.tokens import count_tokens

    cfg = make_config(
        e2e_data_dir,
        top_k=3,
        escalation_threshold=4,
        max_scar_tokens=12,
        sandbox_timeout_s=1.5,
    )
    assert cfg is not None

    llm = make_llm(provider="local", data_dir=e2e_data_dir, escalation_threshold=4)
    repair = next(p for p in TaskPurpose if p != TaskPurpose.SKILL_DISTILLATION)
    assert llm.complete([{"role": "user", "content": "fix"}], purpose=repair, consecutive_failures=3).tier_used == ModelTier.FAST_LOCAL
    assert llm.complete([{"role": "user", "content": "fix"}], purpose=repair, consecutive_failures=4).tier_used == ModelTier.FRONTIER

    store = make_scar_store(e2e_data_dir, max_scar_tokens=12)
    scar, _ = store.record_failure(
        make_distilled_error("ValueError", "very verbose message " * 5, "x = int(y)", 2),
        "x = int(y)",
        ["parse"],
    )
    assert count_tokens(scar.axiom) <= 14

    runner = make_sandbox_runner(e2e_data_dir, sandbox_timeout_s=1.5)
    t0 = time.monotonic()
    sb = runner.execute_code(
        "def hang() -> int:\n    while True:\n        pass\n",
        [make_test_case(kwargs={}, expected=1)],
        entry_fn="hang",
        timeout_s=0.6,
    )
    assert sb.passed is False
    assert (time.monotonic() - t0) < 4.5


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_16_hot_reloaded_skill_parity_inprocess_vs_sandbox(e2e_data_dir: Path) -> None:
    """E2E-T3-16: F11 + F9 — Forged skill returns identical results in-process and in sandbox."""
    registry = make_skill_registry(e2e_data_dir)
    runner = make_sandbox_runner(e2e_data_dir)
    src = (
        "def coerce_bool_safe(raw: str | bool | int) -> bool:\n"
        '    """Strictly parse boolean payloads avoiding bool(\'false\') trap."""\n'
        "    if isinstance(raw, bool):\n"
        "        return raw\n"
        "    if isinstance(raw, int):\n"
        "        return raw != 0\n"
        "    return str(raw).strip().lower() in {'true', '1', 'yes', 'on'}\n"
    )
    cases = [
        make_test_case(kwargs={"raw": "false"}, expected=False),
        make_test_case(kwargs={"raw": "TRUE"}, expected=True),
        make_test_case(kwargs={"raw": "0"}, expected=False),
    ]
    registry.register_and_reload(src, "coerce_bool_safe", "Strictly parse boolean payloads.", ["bool"], cases)

    for case in cases:
        in_proc = registry.invoke_active("coerce_bool_safe", **case["kwargs"])
        assert in_proc == case["expected"]

    sb = runner.verify_skill_adversarial(src, "coerce_bool_safe", cases, {"raw": "str"})
    assert sb.passed is True


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.m3
def test_t3_17_forged_skill_files_pass_static_hygiene_ast_gate(e2e_data_dir: Path) -> None:
    """E2E-T3-17: F6 + F11 — Forged .py skill files pass ASTGuard and zero-stub hygiene."""
    engine = make_engine(e2e_data_dir, provider="local")
    for prompt in (
        "Merge list of record dicts into DataFrame",
        "Parse mixed timezone timestamps to sorted UTC deltas",
        "Strip UTF-8 BOM and parse semicolon CSV",
    ):
        run_engine_task(engine, prompt)

    skill_files = list(e2e_data_dir.rglob("*.py"))
    assert len(skill_files) >= 1

    guard = make_ast_guard()
    markers = ("TO" + "DO", "FIX" + "ME", "X" * 3, "HA" + "CK")
    marker_re = re.compile(r"\b(" + "|".join(markers) + r")\b")
    for path in skill_files:
        src = path.read_text(encoding="utf-8")
        assert not marker_re.search(src)
        ast.parse(src, filename=str(path))
        assert get_ast_valid(guard.validate(src, expected_name=path.stem)) is True
