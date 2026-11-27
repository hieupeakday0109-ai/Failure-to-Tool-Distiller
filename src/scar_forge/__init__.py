"""Scar-Forge: self-evolving AI agent engine with Scar memory and AST-forged Skills."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from scar_forge.config import EngineConfig, configure_utf8_console
from scar_forge.distiller import TracebackDistiller
from scar_forge.models import (
    ActionType,
    AgentAction,
    ASTValidationResult,
    AxiomDraft,
    BenchmarkReport,
    CodeEdit,
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
    ProtocolParseResult,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ActionType",
    "AgentAction",
    "ASTValidationResult",
    "AxiomBudgetError",
    "AxiomDraft",
    "BenchmarkReport",
    "CodeEdit",
    "DistilledError",
    "EngineConfig",
    "EpochName",
    "EpochSummary",
    "LineRangeEdit",
    "LLMResponse",
    "ModelTier",
    "PatchApplyError",
    "PatchResponse",
    "ProtocolParseResult",
    "ProtocolParser",
    "ProtocolViolationError",
    "RunMetrics",
    "SandboxResult",
    "ScarForgeEngine",
    "ScarRecord",
    "SearchReplaceEdit",
    "SkillDraft",
    "SkillRecord",
    "TOKENIZER_BACKEND",
    "TaskPurpose",
    "TaskSpec",
    "TestCaseResult",
    "TierUsage",
    "TokenCounter",
    "TracebackDistiller",
    "apply_code_patch",
    "apply_line_range_edits",
    "apply_search_replace_edits",
    "apply_unified_diff",
    "configure_utf8_console",
    "count_messages_tokens",
    "count_tokens",
    "get_tokenizer_backend",
    "parse_agent_action",
    "parse_axiom_draft",
    "parse_patch_response",
    "parse_skill_draft",
    "truncate_rule_to_budget",
    "truncate_to_axiom_budget",
    "validate_axiom_format",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve Layer 2 symbols without eager cross-layer imports."""
    if name == "ScarForgeEngine":
        try:
            module = import_module("scar_forge.agent.engine")
        except ModuleNotFoundError as exc:
            raise AttributeError(
                "ScarForgeEngine is not yet available (Layer 2 agent engine not installed)"
            ) from exc
        return module.ScarForgeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
