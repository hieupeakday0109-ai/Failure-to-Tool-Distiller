"""Comprehensive unit and subprocess integration tests for Scar-Forge M1 (F1-F5)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import scar_forge
from scar_forge.config import EngineConfig, configure_utf8_console
from scar_forge.distiller import TracebackDistiller
from scar_forge.models import (
    ActionType,
    AgentAction,
    ASTValidationResult,
    AxiomDraft,
    BenchmarkReport,
    DistilledError,
    EpochName,
    EpochSummary,
    LineRangeEdit,
    LLMResponse,
    ModelTier,
    PatchResponse,
    RunMetrics,
    SandboxResult,
    ScarRecord,
    SearchReplaceEdit,
    SkillDraft,
    SkillRecord,
    TaskPurpose,
    TaskSpec,
    TestCaseResult,
    TierUsage,
)
from scar_forge.protocol import (
    PatchApplyError,
    ProtocolParser,
    ProtocolViolationError,
    apply_code_patch,
    apply_line_range_edits,
    apply_search_replace_edits,
    apply_unified_diff,
    parse_agent_action,
    parse_axiom_draft,
    parse_patch_response,
    parse_skill_draft,
)
from scar_forge.tokens import (
    TOKENIZER_BACKEND,
    AxiomBudgetError,
    TokenCounter,
    count_messages_tokens,
    count_tokens,
    get_tokenizer_backend,
    truncate_rule_to_budget,
    truncate_to_axiom_budget,
    validate_axiom_format,
)


class TestBootstrapAndLazyInit:
    def test_layer0_exports_and_version(self) -> None:
        assert scar_forge.__version__ == "0.1.0"
        assert scar_forge.EngineConfig is EngineConfig
        assert scar_forge.TracebackDistiller is TracebackDistiller

    def test_lazy_engine_resolves_and_handles_missing_layer2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scar_forge.agent.engine import ScarForgeEngine

        assert scar_forge.ScarForgeEngine is ScarForgeEngine
        monkeypatch.setattr(
            scar_forge,
            "import_module",
            lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("not installed")),
        )
        with pytest.raises(AttributeError):
            _ = scar_forge.__getattr__("ScarForgeEngine")


class TestEngineConfig:
    def test_defaults_and_derived_paths(self, tmp_path: Path) -> None:
        cfg = EngineConfig(data_dir=tmp_path)
        assert cfg.max_scar_tokens == 14
        assert 3 <= cfg.top_k_scars <= 5
        assert 3 <= cfg.top_k_skills <= 5
        assert cfg.escalation_threshold >= 1
        assert cfg.sandbox_timeout_s > 0
        assert cfg.sandbox_memory_mb > 0
        assert cfg.scar_db_path == tmp_path / "scars.db"
        assert cfg.scar_json_path == tmp_path / "scars.json"
        assert cfg.skills_dir == tmp_path / "skills"
        assert cfg.runs_dir == tmp_path / "runs"

        cfg.ensure_dirs()
        assert cfg.data_dir.is_dir()
        assert cfg.skills_dir.is_dir()
        assert cfg.runs_dir.is_dir()

    def test_bounds_validation(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig(max_scar_tokens=15)
        with pytest.raises(ValidationError):
            EngineConfig(top_k_scars=0)
        with pytest.raises(ValidationError):
            EngineConfig(top_k_scars=6)
        with pytest.raises(ValidationError):
            EngineConfig(top_k_skills=6)

    def test_secret_keys_never_serialized_in_model_dump(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-secret-xyz")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret-xyz")
        cfg = EngineConfig()
        assert cfg.get_gemini_api_key() == "test-gemini-secret-xyz"
        assert cfg.get_openai_api_key() == "test-openai-secret-xyz"
        dumped = cfg.model_dump()
        assert "GEMINI_API_KEY" not in dumped
        assert "OPENAI_API_KEY" not in dumped
        assert "test-gemini-secret-xyz" not in cfg.model_dump_json()

    def test_from_env_and_utf8_console(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SCAR_FORGE_DATA_DIR", str(tmp_path / "env_data"))
        monkeypatch.setenv("SCAR_FORGE_PROVIDER", "local")
        monkeypatch.setenv("SCAR_FORGE_ESCALATION_THRESHOLD", "3")
        cfg = EngineConfig.from_env()
        assert cfg.data_dir == tmp_path / "env_data"
        assert cfg.provider == "local"
        assert cfg.escalation_threshold == 3

        configure_utf8_console()
        assert os.environ.get("PYTHONUTF8") == "1"


class TestModels:
    def test_all_schemas_roundtrip_json(self) -> None:
        distilled = DistilledError(
            exception_type="AttributeError",
            core_message="'DataFrame' object has no attribute 'append'",
            failing_line="df = df.append(row)",
            line_number=4,
            function_name="merge_rows",
            raw_chars=400,
            distilled_chars=90,
            raw_tokens=95,
            distilled_tokens=20,
            compression_ratio=0.7895,
        )
        scar = ScarRecord(
            scar_id="scar_001",
            axiom="[AVOID] pandas.append -> [USE] pd.concat",
            token_count=13,
            exception_type="AttributeError",
            error_signature="AttributeError:DataFrame.append",
            tags=["pandas", "dataframe"],
            context_cue="dataframe append",
        )
        skill = SkillRecord(
            skill_id="skill_001",
            skill_name="merge_dataframes",
            signature="def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame",
            docstring="Concatenate two DataFrames safely.",
            file_path="skills/merge_dataframes.py",
            source_hash="abc123",
            ast_verified=True,
            sandbox_verified=True,
            hot_reloaded=True,
        )
        ast_res = ASTValidationResult(
            valid=True,
            function_name="merge_dataframes",
            parameters=["df1", "df2"],
            has_type_annotations=True,
            has_docstring=True,
            imports_used=["pandas"],
        )
        tc_res = TestCaseResult(name="case_1", passed=True, expected=42, actual=42)
        sb_res = SandboxResult(
            passed=True,
            exit_code=0,
            stdout="42",
            total_cases=1,
            passed_cases=1,
            case_results=[tc_res],
            return_value=42,
        )
        tier_usage = TierUsage(
            tier=ModelTier.FAST_LOCAL,
            calls=2,
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            duration_ms=12.5,
        )
        action_run = AgentAction(action="run_code", code="def solve(): return 1")
        action_skill = AgentAction(
            action="CALL_SKILL", skill_name="merge_dataframes", kwargs={"x": 1}
        )
        assert action_skill.skill_args == {"x": 1}
        assert action_skill.kwargs == {"x": 1}

        patch_resp = PatchResponse(
            line_edits=[LineRangeEdit(start_line=1, end_line=1, replacement="x = 2")]
        )
        action_patch = AgentAction(action="patch_code", patch=patch_resp)
        action_final = AgentAction(action="FINAL_ANSWER", result={"ok": True})

        axiom_draft = AxiomDraft(avoid="df.append", use="pd.concat", tags=["pandas"])
        skill_draft = SkillDraft(
            skill_name="clean_csv",
            source="def clean_csv(raw: str) -> str:\n    return raw.strip()\n",
            docstring="Strip CSV input.",
        )
        llm_resp = LLMResponse(
            content='{"action": "RUN_CODE", "code": "x = 1"}',
            parsed_action=action_run,
            model_name="qwen2.5-coder:7b",
            provider="local",
            tier_used=ModelTier.FAST_LOCAL,
            purpose=TaskPurpose.INITIAL_DRAFT,
            prompt_tokens=50,
            completion_tokens=15,
            total_tokens=65,
            duration_ms=5.0,
        )
        task_spec = TaskSpec(task_id="t1", title="Task 1", prompt="Solve it")
        run_metrics = RunMetrics(
            run_id="r1",
            task_id="t1",
            epoch=EpochName.EPOCH_1_COLD_START,
            success=True,
            attempts=1,
            tier_usage={"fast_local": tier_usage},
        )
        epoch_summary = EpochSummary(
            epoch=EpochName.EPOCH_1_COLD_START,
            total_tasks=1,
            passed_tasks=1,
            accuracy=1.0,
            total_tokens=140,
            avg_tokens_per_task=140.0,
            total_errors=0,
            repeated_errors=0,
            total_duration_ms=12.5,
            avg_duration_ms=12.5,
            runs=[run_metrics],
        )
        report = BenchmarkReport(
            report_id="rep_1",
            provider="local",
            epochs=[epoch_summary],
        )

        for model in (
            distilled,
            scar,
            skill,
            ast_res,
            tc_res,
            sb_res,
            tier_usage,
            action_run,
            action_skill,
            patch_resp,
            action_patch,
            action_final,
            axiom_draft,
            skill_draft,
            llm_resp,
            task_spec,
            run_metrics,
            epoch_summary,
            report,
        ):
            reloaded = type(model).model_validate_json(model.model_dump_json())
            assert reloaded == model

        assert action_run.action == ActionType.RUN_CODE

    def test_invalid_models_raise_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ScarRecord(
                scar_id="s1",
                axiom="[AVOID] too long -> [USE] way too many tokens here",
                token_count=15,
                exception_type="ValueError",
                error_signature="sig",
            )
        with pytest.raises(ValidationError):
            AgentAction(action="run_code", code="x = 1", prose="extra filler")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            AgentAction(action="call_skill")
        with pytest.raises(ValidationError):
            AgentAction(action="run_code", code="   ")
        with pytest.raises(ValidationError):
            AgentAction(action="patch_code")
        with pytest.raises(ValidationError):
            AgentAction(action="final_answer")
        with pytest.raises(ValidationError):
            PatchResponse()


class TestTokenCounterAndAxiomBudget:
    def test_backend_metadata_and_calibration(self) -> None:
        tc = TokenCounter()
        assert tc.tokenizer_backend == TOKENIZER_BACKEND == "deterministic_subword_regex_v1"
        assert get_tokenizer_backend() == "deterministic_subword_regex_v1"
        assert tc.get_metadata() == {
            "tokenizer_backend": "deterministic_subword_regex_v1",
            "max_axiom_tokens": 14,
        }
        assert count_tokens("") == 0
        assert count_tokens("[AVOID] pandas.append -> [USE] pd.concat") == 13
        assert count_tokens("[AVOID] DataFrame.append -> [USE] pd.concat") == 14
        assert count_tokens("[AVOID] unguarded / 0 -> [USE] check denom != 0") == 14
        assert count_tokens("[RULE] Use pd.concat instead of df.append") == 12
        assert tc.count("a_b") == len(tc.tokenize("a_b"))

    def test_count_messages_tokens(self) -> None:
        assert count_messages_tokens([]) == 0
        msgs = [
            {"role": "system", "content": "[AVOID] df.append -> [USE] pd.concat"},
            {"role": "user", "name": "dev", "content": {"task": "merge"}},
        ]
        c1 = count_messages_tokens(msgs)
        c2 = TokenCounter().count_messages(msgs)
        assert c1 > 10
        assert c1 == c2

    def test_validate_axiom_format(self) -> None:
        assert validate_axiom_format("[AVOID] pandas.append -> [USE] pd.concat") is True
        assert validate_axiom_format("[RULE] strip UTF-8 BOM before csv parse") is True
        assert validate_axiom_format("Always use pd.concat instead of append.") is False
        assert validate_axiom_format("[AVOID] x\n-> [USE] y") is False
        assert (
            validate_axiom_format(
                "[AVOID] pandas.DataFrame.append(other, ignore_index=True) -> [USE] pandas.concat([a, b])"
            )
            is False
        )
        assert (
            validate_axiom_format("[AVOID] a -> [USE] b -> [USE] c") is False
        )

    def test_truncate_to_axiom_budget_on_verbose_and_benchmark_axioms(self) -> None:
        samples = [
            (
                "Avoid calling pandas.DataFrame.append(row, ignore_index=True) directly",
                "Use pandas.concat([df, row_df], ignore_index=True) instead",
            ),
            (
                "unguarded division by zero when denominator is empty",
                "check denominator != 0 before dividing",
            ),
            (
                "naive datetime.utcnow() without timezone awareness",
                "datetime.now(timezone.utc) with tzinfo=utc",
            ),
            (
                "parsing complex nested HTML with regular expressions",
                "use html.parser HTMLParser subclass",
            ),
            (
                "raw csv parsing when UTF-8 BOM prefix exists",
                "strip UTF-8 BOM encoding utf-8-sig first",
            ),
            (
                "raw os.path.normpath across POSIX and Windows",
                "PurePosixPath for canonical POSIX paths",
            ),
            (
                "lexicographical string comparison on version tags",
                "parse numeric semver tuple before sorting",
            ),
            (
                "raw json.dumps without allow_nan=False check",
                "json.dumps with allow_nan=False strict mode",
            ),
        ]
        for avoid, use in samples:
            axiom = truncate_to_axiom_budget(avoid, use, max_tokens=14)
            assert count_tokens(axiom) <= 14
            assert validate_axiom_format(axiom, max_tokens=14) is True
            assert "[AVOID]" in axiom and "-> [USE]" in axiom

        short = truncate_to_axiom_budget("pandas.append", "pd.concat")
        assert short == "[AVOID] pandas.append -> [USE] pd.concat"

        rule = truncate_rule_to_budget(
            "Always remember to validate and strip UTF-8 BOM headers before parsing CSV files"
        )
        assert count_tokens(rule) <= 14
        assert validate_axiom_format(rule, max_tokens=14) is True

        with pytest.raises(AxiomBudgetError):
            truncate_to_axiom_budget("   ", "   ")
        with pytest.raises(AxiomBudgetError):
            truncate_to_axiom_budget("a", "b", max_tokens=8)


class TestZeroProseProtocol:
    def test_all_four_actions_parse_strictly(self) -> None:
        parser = ProtocolParser()
        r1 = parser.parse_action('{"action": "RUN_CODE", "code": "def solve(): return 42"}')
        assert r1.data.action == ActionType.RUN_CODE
        assert r1.schema_warning is False

        r2 = parser.parse_action(
            '{"action": "call_skill", "skill_name": "add_nums", "skill_args": {"a": 1}}'
        )
        assert r2.data.action == ActionType.CALL_SKILL
        assert r2.data.kwargs == {"a": 1}

        r3 = parser.parse_action(
            '{"action": "PATCH_CODE", "diff_patch": "@@ -1 +1 @@\\n-x = 1\\n+x = 2"}'
        )
        assert r3.data.action == ActionType.PATCH_CODE
        assert r3.data.patch is not None

        r4 = parser.parse_action('{"action": "FINAL_ANSWER", "result": 42}')
        assert r4.data.action == ActionType.FINAL_ANSWER
        assert parser.strict_passes == 4
        assert parser.schema_warnings == 0

    def test_fence_stripping_logs_schema_warning_and_handles_inner_backticks(self) -> None:
        parser = ProtocolParser()
        fenced = (
            "```json\n"
            '{"action": "RUN_CODE", "code": "doc = \'```python\\nprint(1)\\n```\'\\n"}\n'
            "```"
        )
        res = parser.parse(fenced, AgentAction)
        assert res.data.action == ActionType.RUN_CODE
        assert res.schema_warning is True
        assert res.fence_stripped is True
        assert res.warnings == ["markdown_fence_stripped"]
        assert parser.schema_warnings == 1

    def test_conversational_prose_is_strictly_rejected(self) -> None:
        parser = ProtocolParser()
        bad_payloads = [
            'Sure! Here is the JSON:\n{"action": "FINAL_ANSWER", "result": 1}',
            '{"action": "FINAL_ANSWER", "result": 1}\nLet me know if you need more!',
            '```json\n{"action": "FINAL_ANSWER", "result": 1}\n```\nHope this helps!',
            '```json\n{"action": "FINAL_ANSWER", "result": 1}\n```\nprose\n```json\n{"action": "FINAL_ANSWER", "result": 2}\n```',
            "I think we should fix line 3 first.",
            "",
        ]
        for payload in bad_payloads:
            with pytest.raises(ProtocolViolationError):
                parser.parse_action(payload)

        metrics = parser.metrics()
        assert metrics["protocol_violations"] == len(bad_payloads)
        assert metrics["prose_rejections"] >= 5

    def test_other_schema_parsers(self) -> None:
        act = parse_agent_action('{"action": "FINAL_ANSWER", "result": "done"}')
        assert act.result == "done"

        pr = parse_patch_response(
            '{"search_replace": [{"search": "a = 1", "replace": "a = 2"}]}'
        )
        assert len(pr.search_replace) == 1

        ax = parse_axiom_draft('{"avoid": "df.append", "use": "pd.concat", "tags": ["pd"]}')
        assert ax.avoid == "df.append"

        sk = parse_skill_draft(
            '{"skill_name": "norm_text", "source": "def norm_text(s: str) -> str:\\n    return s.strip()", "docstring": "Normalize text."}'
        )
        assert sk.skill_name == "norm_text"

    def test_unified_diff_line_and_search_replace_patches(self) -> None:
        original = "def add(a, b):\r\n    return a - b\r\n\r\ndef mul(a, b):\r\n    return a + b\r\n"
        diff = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
            "@@ -4,2 +4,2 @@\n"
            " def mul(a, b):\n"
            "-    return a + b\n"
            "+    return a * b\n"
        )
        patched = apply_unified_diff(original, diff)
        assert patched == "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n"

        # Empty file creation and no-newline handling
        created = apply_unified_diff(
            "",
            "@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n\\ No newline at end of file\n",
        )
        assert created == "x = 1\ny = 2"

        # Deletion to empty
        deleted = apply_unified_diff("a\nb\n", "@@ -1,2 +0,0 @@\n-a\n-b\n")
        assert deleted == ""

        # Context mismatch raises PatchApplyError
        with pytest.raises(PatchApplyError):
            apply_unified_diff("x = 1\n", "@@ -1,1 +1,1 @@\n-x = 999\n+x = 2\n")

        # Line-range edits (replacement + pure insertion)
        lr_out = apply_line_range_edits(
            "a = 1\nb = 2\n",
            [
                LineRangeEdit(
                    start_line=1, end_line=0, replacement="# header", expected_text=None
                ),
                LineRangeEdit(
                    start_line=2, end_line=2, replacement="b = 20", expected_text="b = 2"
                ),
            ],
        )
        assert lr_out == "# header\na = 1\nb = 20\n"

        with pytest.raises(PatchApplyError):
            apply_line_range_edits(
                "a = 1\n",
                [LineRangeEdit(start_line=1, end_line=1, replacement="x", expected_text="wrong")],
            )

        # Search/replace & conflict markers via apply_code_patch
        sr_out = apply_search_replace_edits(
            "x = 1\ny = 2\n",
            [SearchReplaceEdit(search="x = 1", replace="x = 10", expected_count=1)],
        )
        assert sr_out == "x = 10\ny = 2\n"

        with pytest.raises(PatchApplyError):
            apply_search_replace_edits(
                "x = 1\nx = 1\n",
                [SearchReplaceEdit(search="x = 1", replace="x = 2", expected_count=1)],
            )

        block_patch = (
            "<<<<<<< SEARCH\n"
            "y = 2\n"
            "=======\n"
            "y = 200\n"
            ">>>>>>> REPLACE"
        )
        assert apply_code_patch("x = 1\ny = 2\n", block_patch) == "x = 1\ny = 200\n"


class TestTracebackDistiller:
    def test_strips_noise_frames_logs_and_computes_metrics(self) -> None:
        noisy_stderr = (
            "\x1b[32m2026-11-26 12:00:00 INFO Starting sandbox runner\x1b[0m\n"
            "DEBUG: loading environment variables\n"
            "Traceback (most recent call last):\n"
            '  File "C:\\Users\\Admin\\scar_forge\\src\\scar_forge\\sandbox\\_sandbox_runner.py", line 42, in _exec\n'
            "    exec(compiled, ns)\n"
            '  File "C:\\Users\\Admin\\My App\\solution.py", line 14, in transform\n'
            "    return df.append(extra)\n"
            "           ^^^^^^^^^\n"
            '  File "C:\\Users\\Admin\\.venv\\Lib\\site-packages\\pandas\\core\\generic.py", line 6299, in __getattr__\n'
            "    return object.__getattribute__(self, name)\n"
            "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
            '  File "C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\json\\decoder.py", line 338, in decode\n'
            "    obj, end = self.raw_decode(s)\n"
            "               ^^^^^^^^^^^^^^^^^^\n"
            "AttributeError: 'DataFrame' object has no attribute 'append'\n"
            "Process finished with exit code 1\n"
        )
        d_cls = TracebackDistiller.distill(noisy_stderr)
        d_inst = TracebackDistiller().distill(noisy_stderr)
        assert d_cls == d_inst

        assert d_cls.exception_type == "AttributeError"
        assert "no attribute 'append'" in d_cls.core_message
        assert d_cls.failing_line == "return df.append(extra)"
        assert d_cls.line_number == 14
        assert d_cls.function_name == "transform"
        assert d_cls.file_path == "My App/solution.py"
        assert d_cls.frames_dropped == 3
        assert d_cls.frames_kept == 1
        assert d_cls.raw_chars == len(noisy_stderr)
        assert d_cls.raw_tokens == count_tokens(noisy_stderr)
        prompt_text = d_cls.format_for_prompt()
        assert d_cls.distilled_chars == len(prompt_text)
        assert d_cls.distilled_tokens == count_tokens(prompt_text)
        assert d_cls.compression_ratio > 0.5
        assert "^^^^^^^^^" not in d_cls.failing_line

    def test_syntax_and_indentation_errors_preserve_aligned_caret(self) -> None:
        syn_tb = (
            '  File "C:\\Users\\Admin\\My App\\main.py", line 3\n'
            "    def broken(:\n"
            "               ^\n"
            "SyntaxError: invalid syntax\n"
        )
        d = TracebackDistiller.distill(syn_tb)
        assert d.exception_type == "SyntaxError"
        assert d.is_syntax_error is True
        assert d.line_number == 3
        assert d.function_name == ""
        assert d.failing_line == "def broken(:"
        assert d.caret_line == "           ^"
        rendered = d.format_for_prompt()
        assert "Line: def broken(:" in rendered
        assert "                 ^" in rendered

        ind_tb = (
            '  File "/tmp/scar_123/task.py", line 2\n'
            "    pass\n"
            "    ^^^^\n"
            "IndentationError: expected an indented block\n"
        )
        d_ind = TracebackDistiller.distill(ind_tb)
        assert d_ind.exception_type == "IndentationError"
        assert d_ind.caret_line == "^^^^"
        assert d_ind.file_path == "task.py"

    def test_chained_exceptions_and_stdout_fallback(self) -> None:
        chained = (
            "Traceback (most recent call last):\n"
            '  File "app/calc.py", line 9, in divide\n'
            "    return a / b\n"
            "ZeroDivisionError: division by zero\n\n"
            "During handling of the above exception, another exception occurred:\n\n"
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.12/runpy.py", line 88, in _run_code\n'
            "    exec(code, run_globals)\n"
            "RuntimeError: pipeline aborted\n"
        )
        d = TracebackDistiller.distill(chained)
        assert d.exception_type == "RuntimeError"
        assert d.core_message == "pipeline aborted"
        assert "ZeroDivisionError: division by zero" in d.root_cause
        assert d.line_number == 9
        assert d.function_name == "divide"

        # Stdout fallback when stderr is empty
        d_out = TracebackDistiller.distill("", "ValueError: negative count")
        assert d_out.exception_type == "ValueError"
        assert d_out.core_message == "negative count"

        # Non-traceback timeout & empty fallback
        d_to = TracebackDistiller.distill("Sandbox timed out after 5.0s")
        assert d_to.exception_type == "TimeoutError"

        d_empty = TracebackDistiller.distill("", "")
        assert d_empty.exception_type == "UnknownError"
        assert d_empty.compression_ratio == 0.0

    def test_real_python_312_subprocess_traceback(self, tmp_path: Path) -> None:
        code = (
            "import json\n"
            "def parse_payload():\n"
            "    return json.loads('{bad')\n"
            "parse_payload()\n"
        )
        script = tmp_path / "parse_task.py"
        script.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        distilled = TracebackDistiller.distill(proc.stderr, proc.stdout)
        assert "JSONDecodeError" in distilled.exception_type
        assert distilled.function_name == "parse_payload"
        assert distilled.line_number == 3
        assert "json.loads" in distilled.failing_line
        assert distilled.frames_dropped >= 2
        assert distilled.compression_ratio > 0.3
