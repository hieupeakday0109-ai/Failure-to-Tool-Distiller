"""Core Pydantic v2 schemas and enums for Scar-Forge Layer 0 contracts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scar_forge.tokens import TOKENIZER_BACKEND

_SKILL_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ModelTier(StrEnum):
    """Dual-tier model routing destinations."""

    FAST_LOCAL = "FAST_LOCAL"
    FRONTIER = "FRONTIER"


class TaskPurpose(StrEnum):
    """Agent loop purposes driving tier selection."""

    INITIAL_DRAFT = "initial_draft"
    SYNTAX_REPAIR = "syntax_repair"
    TEST_LOOP = "test_loop"
    REASONING_RECOVERY = "reasoning_recovery"
    SKILL_DISTILLATION = "skill_distillation"
    SCAR_DISTILLATION = "scar_distillation"


class ActionType(StrEnum):
    """Zero-Prose agent action identifiers."""

    RUN_CODE = "RUN_CODE"
    CALL_SKILL = "CALL_SKILL"
    PATCH_CODE = "PATCH_CODE"
    FINAL_ANSWER = "FINAL_ANSWER"


class EpochName(StrEnum):
    """3-Epoch learning benchmark stages."""

    EPOCH_1_COLD_START = "epoch_1_cold_start"
    EPOCH_2_SCAR_GUIDED = "epoch_2_scar_guided"
    EPOCH_3_SKILL_REUSE = "epoch_3_skill_reuse"


class DistilledError(BaseModel):
    """Compact distilled representation of stderr/stdout runtime failures."""

    exception_type: str = Field(min_length=1)
    core_message: str = ""
    failing_line: str = ""
    line_number: int | None = Field(default=None, ge=1)
    function_name: str = ""
    raw_chars: int = Field(default=0, ge=0)
    distilled_chars: int = Field(default=0, ge=0)
    raw_tokens: int = Field(default=0, ge=0)
    distilled_tokens: int = Field(default=0, ge=0)
    compression_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    file_path: str = ""
    caret_line: str = ""
    root_cause: str = ""
    frames_dropped: int = Field(default=0, ge=0)
    frames_kept: int = Field(default=0, ge=0)
    is_syntax_error: bool = False

    @field_validator(
        "function_name", "file_path", "failing_line", "caret_line", "root_cause", mode="before"
    )
    @classmethod
    def _none_to_empty_str(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def format_for_prompt(self) -> str:
        """Render distilled error in compact, zero-prose prompt format."""
        header = (
            f"[ERROR] {self.exception_type}: {self.core_message}"
            if self.core_message
            else f"[ERROR] {self.exception_type}"
        )
        lines = [header]
        if self.root_cause:
            lines.append(f"Cause: {self.root_cause}")

        loc_parts: list[str] = []
        if self.file_path and self.line_number is not None:
            loc_parts.append(f"{self.file_path}:{self.line_number}")
        elif self.file_path:
            loc_parts.append(self.file_path)
        elif self.line_number is not None:
            loc_parts.append(f"line {self.line_number}")
        if self.function_name:
            loc_parts.append(f"in {self.function_name}")
        if loc_parts:
            lines.append(f"At: {' '.join(loc_parts)}")

        if self.failing_line:
            lines.append(f"Line: {self.failing_line}")
            if self.caret_line:
                lines.append(f"      {self.caret_line}")

        return "\n".join(lines)


class AxiomDraft(BaseModel):
    """Structured telegraphic axiom draft emitted by scar distillation."""

    model_config = ConfigDict(extra="forbid")

    avoid: str = Field(min_length=1)
    use: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    axiom: str = ""


class ScarRecord(BaseModel):
    """Persisted deduplicated Negative Axiom record (strictly <= 14 tokens)."""

    scar_id: str = Field(min_length=1)
    axiom: str = Field(min_length=1)
    token_count: int = Field(ge=1, le=14)
    exception_type: str = Field(min_length=1)
    error_signature: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    context_cue: str = ""
    occurrence_count: int = Field(default=1, ge=1)
    prevented_count: int = Field(default=0, ge=0)
    source_code_snippet: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    last_seen_at: datetime = Field(default_factory=_utc_now)


class LineRangeEdit(BaseModel):
    """Line-range replacement edit with optional expected-text guard."""

    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=0)
    replacement: str
    expected_text: str | None = None


class SearchReplaceEdit(BaseModel):
    """Exact search-and-replace code edit block."""

    model_config = ConfigDict(extra="forbid")

    search: str = Field(min_length=1)
    replace: str
    expected_count: int | None = Field(default=1, ge=1)


CodeEdit = SearchReplaceEdit


def _coerce_action_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().upper()
    return value


class PatchResponse(BaseModel):
    """Structured code patch supporting unified diffs, line edits, or search-replace."""

    model_config = ConfigDict(extra="forbid")

    action: ActionType = ActionType.PATCH_CODE
    diff_patch: str | None = None
    unified_diff: str | None = None
    line_edits: list[LineRangeEdit] = Field(default_factory=list)
    search_replace: list[SearchReplaceEdit] = Field(default_factory=list)
    edits: list[SearchReplaceEdit] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: Any) -> Any:
        return _coerce_action_value(value)

    @model_validator(mode="after")
    def _validate_and_sync_patch_fields(self) -> PatchResponse:
        if self.action != ActionType.PATCH_CODE:
            raise ValueError(f"PatchResponse action must be PATCH_CODE, got {self.action}")

        if self.diff_patch is not None and self.unified_diff is not None:
            if self.diff_patch != self.unified_diff:
                raise ValueError("diff_patch and unified_diff cannot hold different values")
        elif self.diff_patch is not None:
            self.unified_diff = self.diff_patch
        elif self.unified_diff is not None:
            self.diff_patch = self.unified_diff

        if self.search_replace and self.edits:
            if self.search_replace != self.edits:
                raise ValueError("search_replace and edits cannot hold different lists")
        elif self.search_replace:
            self.edits = list(self.search_replace)
        elif self.edits:
            self.search_replace = list(self.edits)

        has_diff = isinstance(self.diff_patch, str) and self.diff_patch.strip() != ""
        has_lines = len(self.line_edits) > 0
        has_sr = len(self.search_replace) > 0
        if not (has_diff or has_lines or has_sr):
            raise ValueError(
                "PatchResponse requires non-empty diff_patch/unified_diff, "
                "line_edits, or search_replace"
            )
        return self


class AgentAction(BaseModel):
    """Zero-Prose structured action emitted by the agent loop."""

    model_config = ConfigDict(extra="forbid")

    action: ActionType
    code: str | None = None
    entry_fn: str | None = None
    diff_patch: str | None = None
    unified_diff: str | None = None
    patch: PatchResponse | None = None
    line_edits: list[LineRangeEdit] = Field(default_factory=list)
    search_replace: list[SearchReplaceEdit] = Field(default_factory=list)
    edits: list[SearchReplaceEdit] = Field(default_factory=list)
    skill_name: str | None = None
    skill_args: dict[str, Any] | None = None
    kwargs: dict[str, Any] | None = None
    result: Any = None

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: Any) -> Any:
        return _coerce_action_value(value)

    @model_validator(mode="after")
    def _validate_action_payload(self) -> AgentAction:
        # Sync diff_patch / unified_diff
        if self.diff_patch is not None and self.unified_diff is not None:
            if self.diff_patch != self.unified_diff:
                raise ValueError("diff_patch and unified_diff cannot differ")
        elif self.diff_patch is not None:
            self.unified_diff = self.diff_patch
        elif self.unified_diff is not None:
            self.diff_patch = self.unified_diff

        # Sync search_replace / edits
        if self.search_replace and self.edits:
            if self.search_replace != self.edits:
                raise ValueError("search_replace and edits cannot differ")
        elif self.search_replace:
            self.edits = list(self.search_replace)
        elif self.edits:
            self.search_replace = list(self.edits)

        # Sync skill_args / kwargs
        if self.skill_args is not None and self.kwargs is not None:
            if self.skill_args != self.kwargs:
                raise ValueError("skill_args and kwargs cannot differ")
        elif self.skill_args is not None:
            self.kwargs = dict(self.skill_args)
        elif self.kwargs is not None:
            self.skill_args = dict(self.kwargs)

        if self.action == ActionType.RUN_CODE:
            if not isinstance(self.code, str) or self.code.strip() == "":
                raise ValueError("RUN_CODE action requires a non-empty 'code' string")

        elif self.action == ActionType.CALL_SKILL:
            if not isinstance(self.skill_name, str) or not _SKILL_NAME_RE.match(
                self.skill_name
            ):
                raise ValueError(
                    "CALL_SKILL action requires a valid snake_case 'skill_name'"
                )
            if self.skill_args is None:
                self.skill_args = {}
            if self.kwargs is None:
                self.kwargs = dict(self.skill_args)

        elif self.action == ActionType.PATCH_CODE:
            has_top_level_patch = (
                (isinstance(self.diff_patch, str) and self.diff_patch.strip() != "")
                or len(self.line_edits) > 0
                or len(self.search_replace) > 0
            )
            if self.patch is None:
                if not has_top_level_patch:
                    raise ValueError(
                        "PATCH_CODE requires 'patch' or non-empty "
                        "diff_patch/line_edits/search_replace"
                    )
                self.patch = PatchResponse(
                    diff_patch=self.diff_patch,
                    line_edits=self.line_edits,
                    search_replace=self.search_replace,
                )
            elif not self.diff_patch and self.patch.diff_patch:
                self.diff_patch = self.patch.diff_patch
                self.unified_diff = self.patch.diff_patch

        elif self.action == ActionType.FINAL_ANSWER:
            has_code = isinstance(self.code, str) and self.code.strip() != ""
            if self.result is None and not has_code:
                raise ValueError(
                    "FINAL_ANSWER action requires 'result' or non-empty 'code'"
                )

        return self


class SkillDraft(BaseModel):
    """Structured draft emitted during skill forging/distillation."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    source: str = Field(min_length=1)
    docstring: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    param_schema: dict[str, str] = Field(default_factory=dict)
    test_cases: list[dict[str, Any]] = Field(default_factory=list)


class SkillRecord(BaseModel):
    """Persisted AST-verified and hot-reloaded Python skill metadata."""

    skill_id: str = Field(min_length=1)
    skill_name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    signature: str = Field(min_length=1)
    docstring: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    source_hash: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    param_schema: dict[str, str] = Field(default_factory=dict)
    ast_verified: bool = False
    sandbox_verified: bool = False
    hot_reloaded: bool = False
    version: int = Field(default=1, ge=1)
    reuse_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class ASTValidationResult(BaseModel):
    """Static AST security and structural validation summary for forged skills."""

    valid: bool
    violations: list[str] = Field(default_factory=list)
    function_name: str | None = None
    parameters: list[str] = Field(default_factory=list)
    has_type_annotations: bool = False
    has_docstring: bool = False
    imports_used: list[str] = Field(default_factory=list)


class TestCaseResult(BaseModel):
    """Outcome of a single sandbox test case (standard or adversarial)."""

    __test__: ClassVar[bool] = False

    name: str = Field(min_length=1)
    passed: bool
    is_adversarial: bool = False
    expected: Any = None
    actual: Any = None
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)


class SandboxResult(BaseModel):
    """Aggregate execution outcome from the isolated verification sandbox."""

    passed: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    distilled_error: DistilledError | None = None
    timed_out: bool = False
    memory_exceeded: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)
    total_cases: int = Field(default=0, ge=0)
    passed_cases: int = Field(default=0, ge=0)
    case_results: list[TestCaseResult] = Field(default_factory=list)
    return_value: Any = None


class TierUsage(BaseModel):
    """Per-tier LLM invocation and token consumption metrics."""

    tier: ModelTier
    calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)


class LLMResponse(BaseModel):
    """Normalized response returned by DualTierLLM."""

    content: str
    parsed_action: AgentAction | None = None
    model_name: str = Field(default="local-rule", min_length=1)
    provider: str = Field(default="local", min_length=1)
    tier_used: ModelTier
    purpose: TaskPurpose = TaskPurpose.INITIAL_DRAFT
    escalated: bool = False
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    duration_ms: float = Field(ge=0.0)
    schema_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _populate_total_tokens(self) -> LLMResponse:
        if self.total_tokens is None:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class TaskSpec(BaseModel):
    """Benchmark or agent task specification with verification test cases."""

    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    domain: str = "general"
    tags: list[str] = Field(default_factory=list)
    entry_fn: str = "solve"
    param_schema: dict[str, str] = Field(default_factory=dict)
    test_cases: list[dict[str, Any]] = Field(default_factory=list)
    variant_prompt: str | None = None
    variant_test_cases: list[dict[str, Any]] = Field(default_factory=list)
    reuse_test_cases: list[dict[str, Any]] = Field(default_factory=list)


class RunMetrics(BaseModel):
    """Telemetry recorded for a single agent task execution."""

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    epoch: EpochName | None = None
    success: bool
    attempts: int = Field(ge=1)
    errors_count: int = Field(default=0, ge=0)
    repeated_errors_count: int = Field(default=0, ge=0)
    escalated: bool = False
    escalation_count: int = Field(default=0, ge=0)
    scars_injected: list[str] = Field(default_factory=list)
    scars_created: list[str] = Field(default_factory=list)
    scars_prevented: list[str] = Field(default_factory=list)
    skills_injected: list[str] = Field(default_factory=list)
    skill_forged: str | None = None
    skill_reused: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    tier_usage: dict[str, TierUsage] = Field(default_factory=dict)
    duration_ms: float = Field(default=0.0, ge=0.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    tokenizer_backend: str = TOKENIZER_BACKEND
    schema_warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)

    @property
    def errors(self) -> int:
        return self.errors_count

    @property
    def escalations(self) -> int:
        return self.escalation_count

    @property
    def tokens(self) -> int:
        return self.total_tokens


class EpochSummary(BaseModel):
    """Aggregated metrics for one epoch of the 3-Epoch Benchmark."""

    epoch: EpochName
    total_tasks: int = Field(ge=0)
    passed_tasks: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    total_tokens: int = Field(ge=0)
    avg_tokens_per_task: float = Field(ge=0.0)
    total_errors: int = Field(ge=0)
    repeated_errors: int = Field(ge=0)
    total_duration_ms: float = Field(ge=0.0)
    avg_duration_ms: float = Field(ge=0.0)
    scars_active: int = Field(default=0, ge=0)
    skills_active: int = Field(default=0, ge=0)
    skills_reused_count: int = Field(default=0, ge=0)
    runs: list[RunMetrics] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    """Complete 3-Epoch Benchmark output comparing learning & token savings."""

    report_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    tokenizer_backend: str = TOKENIZER_BACKEND
    epochs: list[EpochSummary] = Field(default_factory=list)
    token_reduction_epoch3_vs_epoch1: float = Field(default=0.0)
    error_reduction_epoch2_vs_epoch1: float = Field(default=0.0)
    speedup_epoch3_vs_epoch1: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=_utc_now)
