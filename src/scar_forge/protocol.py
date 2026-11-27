"""Zero-Prose structured JSON protocol parser and CRLF-safe code patcher."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from scar_forge.models import (
    AgentAction,
    AxiomDraft,
    LineRangeEdit,
    PatchResponse,
    SearchReplaceEdit,
    SkillDraft,
)

_OPEN_FENCE_RE = re.compile(r"^```(?:[A-Za-z0-9_-]+)?[ \t]*\r?\n")
_CLOSE_FENCE_RE = re.compile(r"\r?\n?```[ \t]*$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_SR_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE",
    re.DOTALL,
)


class ProtocolViolationError(ValueError):
    """Raised when an LLM output violates the Zero-Prose JSON/schema contract."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        raw_text: str = "",
        warnings: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.raw_text = raw_text
        self.warnings = list(warnings) if warnings else []


class PatchApplyError(ValueError):
    """Raised when a unified diff, line-range, or search-replace patch cannot apply cleanly."""

    def __init__(
        self,
        message: str,
        *,
        hunk_index: int | None = None,
        line_number: int | None = None,
    ) -> None:
        super().__init__(message)
        self.hunk_index = hunk_index
        self.line_number = line_number


@dataclass(slots=True)
class ProtocolParseResult[T: BaseModel]:
    """Parsed and validated protocol message with schema-warning telemetry."""

    data: T
    schema_warning: bool = False
    fence_stripped: bool = False
    warnings: list[str] = field(default_factory=list)
    raw_chars: int = 0


class ProtocolParser:
    """Zero-Prose JSON/schema parser tracking strict passes, warnings, and violations."""

    def __init__(self, *, allow_fence_fallback: bool = True) -> None:
        self.allow_fence_fallback = allow_fence_fallback
        self.total_attempts: int = 0
        self.strict_passes: int = 0
        self.schema_warnings: int = 0
        self.protocol_violations: int = 0
        self.prose_rejections: int = 0

    def metrics(self) -> dict[str, int]:
        """Return cumulative parser telemetry counters."""
        return {
            "total_attempts": self.total_attempts,
            "strict_passes": self.strict_passes,
            "schema_warnings": self.schema_warnings,
            "protocol_violations": self.protocol_violations,
            "prose_rejections": self.prose_rejections,
        }

    def _record_failure(self, *, prose: bool = False) -> None:
        self.protocol_violations += 1
        if prose:
            self.prose_rejections += 1

    def extract_json_payload(self, raw_text: str) -> tuple[str, list[str], bool]:
        """Extract a strict JSON object payload, rejecting prose outside fences."""
        if not isinstance(raw_text, str):
            self._record_failure()
            raise ProtocolViolationError(
                f"Expected str payload, got {type(raw_text).__name__}",
                reason="invalid_type",
                raw_text="",
            )

        stripped = raw_text.strip()
        if not stripped:
            self._record_failure()
            raise ProtocolViolationError(
                "Empty response; expected strict JSON object",
                reason="empty_response",
                raw_text=raw_text,
            )

        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped, [], False

        if self.allow_fence_fallback and stripped.startswith("```"):
            open_match = _OPEN_FENCE_RE.match(stripped)
            close_match = _CLOSE_FENCE_RE.search(stripped)
            if open_match and close_match and close_match.start() >= open_match.end():
                inner = stripped[open_match.end() : close_match.start()].strip()
                if inner.startswith("{") and inner.endswith("}"):
                    try:
                        json.loads(inner)
                    except json.JSONDecodeError as exc:
                        self._record_failure(prose=True)
                        raise ProtocolViolationError(
                            f"Invalid JSON or trailing prose inside markdown fence: {exc}",
                            reason="conversational_prose_detected",
                            raw_text=raw_text,
                            warnings=["markdown_fence_stripped"],
                        ) from exc
                    return inner, ["markdown_fence_stripped"], True

        self._record_failure(prose=True)
        raise ProtocolViolationError(
            "Conversational prose or non-JSON content detected outside JSON schema",
            reason="conversational_prose_detected",
            raw_text=raw_text,
        )

    def parse[T: BaseModel](self, raw_text: str, schema: type[T]) -> ProtocolParseResult[T]:
        """Parse raw_text into a validated Pydantic schema without conversational prose."""
        self.total_attempts += 1
        payload, warnings, fence_stripped = self.extract_json_payload(raw_text)

        try:
            model = schema.model_validate_json(payload)
        except ValidationError as exc:
            self._record_failure(prose=False)
            raise ProtocolViolationError(
                f"Schema validation failed for {schema.__name__}: {exc}",
                reason="schema_validation_failed",
                raw_text=raw_text,
                warnings=warnings,
            ) from exc

        schema_warning = bool(warnings)
        if schema_warning:
            self.schema_warnings += 1
        else:
            self.strict_passes += 1

        return ProtocolParseResult(
            data=model,
            schema_warning=schema_warning,
            fence_stripped=fence_stripped,
            warnings=warnings,
            raw_chars=len(raw_text),
        )

    def parse_action(self, raw_text: str) -> ProtocolParseResult[AgentAction]:
        """Parse and validate an AgentAction JSON payload."""
        return self.parse(raw_text, AgentAction)


def parse_agent_action(
    raw: str, parser: ProtocolParser | None = None
) -> AgentAction:
    """Parse an AgentAction using the provided or a fresh ProtocolParser."""
    active_parser = parser if parser is not None else ProtocolParser()
    return active_parser.parse(raw, AgentAction).data


def parse_patch_response(
    raw: str, parser: ProtocolParser | None = None
) -> PatchResponse:
    """Parse a PatchResponse using the provided or a fresh ProtocolParser."""
    active_parser = parser if parser is not None else ProtocolParser()
    return active_parser.parse(raw, PatchResponse).data


def parse_axiom_draft(
    raw: str, parser: ProtocolParser | None = None
) -> AxiomDraft:
    """Parse an AxiomDraft using the provided or a fresh ProtocolParser."""
    active_parser = parser if parser is not None else ProtocolParser()
    return active_parser.parse(raw, AxiomDraft).data


def parse_skill_draft(
    raw: str, parser: ProtocolParser | None = None
) -> SkillDraft:
    """Parse a SkillDraft using the provided or a fresh ProtocolParser."""
    active_parser = parser if parser is not None else ProtocolParser()
    return active_parser.parse(raw, SkillDraft).data


def _normalize_eol(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_lines_lossless(text: str) -> tuple[list[str], bool]:
    normalized = _normalize_eol(text)
    if normalized == "":
        return [], False
    had_trailing_nl = normalized.endswith("\n")
    lines = normalized[:-1].split("\n") if had_trailing_nl else normalized.split("\n")
    return lines, had_trailing_nl


def _join_lines(lines: list[str], *, trailing_nl: bool) -> str:
    if not lines:
        return ""
    joined = "\n".join(lines)
    return f"{joined}\n" if trailing_nl else joined


def _strip_diff_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        open_match = _OPEN_FENCE_RE.match(stripped)
        close_match = _CLOSE_FENCE_RE.search(stripped)
        if open_match and close_match and close_match.start() >= open_match.end():
            return stripped[open_match.end() : close_match.start()]
    return text


def apply_unified_diff(
    original_code: str,
    diff_text: str,
    *,
    strip_fences: bool = False,
) -> str:
    """Apply a strict unified diff to original_code with CRLF normalization."""
    if not isinstance(diff_text, str) or not diff_text.strip():
        raise PatchApplyError("Empty unified diff")

    cleaned_diff = _strip_diff_fence(diff_text) if strip_fences else diff_text
    diff_lines, _ = _split_lines_lossless(cleaned_diff)
    orig_lines, orig_trailing_nl = _split_lines_lossless(original_code)

    idx = 0
    while idx < len(diff_lines) and not diff_lines[idx].startswith("@@ "):
        idx += 1

    if idx >= len(diff_lines):
        raise PatchApplyError("No @@ unified diff hunk headers found")

    result_lines: list[str] = []
    src_cursor = 0
    hunk_index = 0
    final_no_newline = False
    touched_eof = False

    while idx < len(diff_lines):
        header_line = diff_lines[idx]
        if not header_line.strip():
            idx += 1
            continue
        match = _HUNK_HEADER_RE.match(header_line)
        if not match:
            raise PatchApplyError(
                f"Unexpected line outside hunk header: {header_line!r}",
                hunk_index=hunk_index,
                line_number=idx + 1,
            )

        hunk_index += 1
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        idx += 1

        old_lines: list[str] = []
        new_lines: list[str] = []
        new_no_newline = False

        while idx < len(diff_lines) and (
            len(old_lines) < old_count or len(new_lines) < new_count
        ):
            line = diff_lines[idx]
            if line.startswith("@@ "):
                break
            if line.startswith(r"\ No newline at end of file"):
                # Applies to preceding '-' or '+' side; track if it followed '+'
                if len(new_lines) == new_count and (
                    len(old_lines) < old_count or new_count > 0
                ):
                    new_no_newline = True
                idx += 1
                continue
            if line.startswith(" "):
                content = line[1:]
                old_lines.append(content)
                new_lines.append(content)
            elif line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            else:
                raise PatchApplyError(
                    f"Malformed diff hunk line at diff line {idx + 1}: {line!r}",
                    hunk_index=hunk_index,
                    line_number=idx + 1,
                )
            idx += 1

        # Consume trailing '\ No newline at end of file' right after hunk counts complete
        while idx < len(diff_lines) and diff_lines[idx].startswith(
            r"\ No newline at end of file"
        ):
            new_no_newline = True
            idx += 1

        if len(old_lines) != old_count or len(new_lines) != new_count:
            raise PatchApplyError(
                f"Hunk {hunk_index} line count mismatch: expected -{old_count},+{new_count}, "
                f"got -{len(old_lines)},+{len(new_lines)}",
                hunk_index=hunk_index,
                line_number=old_start,
            )

        target_idx = 0 if old_start == 0 else old_start - 1
        if target_idx < src_cursor:
            raise PatchApplyError(
                f"Hunk {hunk_index} overlaps or is out of order "
                f"(target {target_idx + 1} < {src_cursor + 1})",
                hunk_index=hunk_index,
                line_number=old_start,
            )
        if target_idx + len(old_lines) > len(orig_lines):
            raise PatchApplyError(
                f"Hunk {hunk_index} exceeds file bounds "
                f"({target_idx + len(old_lines)} > {len(orig_lines)})",
                hunk_index=hunk_index,
                line_number=old_start,
            )

        for offset, expected_line in enumerate(old_lines):
            actual_line = orig_lines[target_idx + offset]
            if actual_line != expected_line:
                raise PatchApplyError(
                    f"Context mismatch in hunk {hunk_index} at line {target_idx + offset + 1}: "
                    f"expected {expected_line!r}, found {actual_line!r}",
                    hunk_index=hunk_index,
                    line_number=target_idx + offset + 1,
                )

        result_lines.extend(orig_lines[src_cursor:target_idx])
        result_lines.extend(new_lines)
        src_cursor = target_idx + len(old_lines)
        if src_cursor == len(orig_lines):
            touched_eof = True
            final_no_newline = new_no_newline

    result_lines.extend(orig_lines[src_cursor:])

    if not result_lines:
        return ""
    if touched_eof and final_no_newline:
        trailing_nl = False
    elif orig_lines == [] and not final_no_newline:
        trailing_nl = True
    else:
        trailing_nl = orig_trailing_nl or (
            touched_eof and not final_no_newline
        )
    return _join_lines(result_lines, trailing_nl=trailing_nl)


def apply_line_range_edits(
    original_code: str, edits: Sequence[LineRangeEdit]
) -> str:
    """Apply non-overlapping 1-based line-range edits with optional expected_text check."""
    if not edits:
        raise PatchApplyError("No line-range edits provided")

    orig_lines, orig_trailing_nl = _split_lines_lossless(original_code)
    sorted_edits = sorted(edits, key=lambda e: (e.start_line, e.end_line))

    prev_end = 0
    for idx, edit in enumerate(sorted_edits, start=1):
        if edit.start_line < 1:
            raise PatchApplyError(
                f"start_line must be >= 1, got {edit.start_line}",
                hunk_index=idx,
                line_number=edit.start_line,
            )
        if edit.end_line < edit.start_line - 1:
            raise PatchApplyError(
                f"end_line ({edit.end_line}) cannot be less than "
                f"start_line - 1 ({edit.start_line - 1})",
                hunk_index=idx,
                line_number=edit.start_line,
            )
        if edit.end_line > len(orig_lines) or (
            edit.end_line == edit.start_line - 1 and edit.start_line > len(orig_lines) + 1
        ):
            raise PatchApplyError(
                f"Line range {edit.start_line}..{edit.end_line} "
                f"out of bounds for {len(orig_lines)} lines",
                hunk_index=idx,
                line_number=edit.start_line,
            )
        if idx > 1 and edit.start_line <= prev_end:
            raise PatchApplyError(
                f"Overlapping line edits near line {edit.start_line}",
                hunk_index=idx,
                line_number=edit.start_line,
            )
        if edit.expected_text is not None:
            actual_slice = "\n".join(orig_lines[edit.start_line - 1 : edit.end_line])
            expected_norm = _normalize_eol(edit.expected_text).rstrip("\n")
            if actual_slice != expected_norm:
                raise PatchApplyError(
                    f"Expected text mismatch at lines {edit.start_line}..{edit.end_line}: "
                    f"expected {expected_norm!r}, found {actual_slice!r}",
                    hunk_index=idx,
                    line_number=edit.start_line,
                )
        prev_end = max(edit.end_line, edit.start_line - 1)

    result_lines: list[str] = []
    cursor = 0
    for edit in sorted_edits:
        start_idx = edit.start_line - 1
        end_idx = edit.end_line
        result_lines.extend(orig_lines[cursor:start_idx])
        if edit.replacement != "":
            rep_lines, _ = _split_lines_lossless(edit.replacement)
            result_lines.extend(rep_lines)
        cursor = end_idx
    result_lines.extend(orig_lines[cursor:])

    if not result_lines:
        return ""
    return _join_lines(
        result_lines,
        trailing_nl=orig_trailing_nl or original_code == "",
    )


def apply_search_replace_edits(
    original_code: str, edits: Sequence[SearchReplaceEdit]
) -> str:
    """Apply exact search-and-replace edits, rejecting missing or ambiguous matches."""
    if not edits:
        raise PatchApplyError("No search-replace edits provided")

    current = _normalize_eol(original_code)
    for idx, edit in enumerate(edits, start=1):
        search_norm = _normalize_eol(edit.search)
        replace_norm = _normalize_eol(edit.replace)
        found = current.count(search_norm)
        if found == 0:
            raise PatchApplyError(
                f"Search block #{idx} not found in target code",
                hunk_index=idx,
            )
        if edit.expected_count is not None and found != edit.expected_count:
            raise PatchApplyError(
                f"Search block #{idx} ambiguity: expected {edit.expected_count} "
                f"match(es), found {found}",
                hunk_index=idx,
            )
        current = current.replace(search_norm, replace_norm)
    return current


def apply_code_patch(
    original_code: str,
    patch: str | PatchResponse,
    *,
    strip_fences: bool = False,
) -> str:
    """Apply a PatchResponse or diff/search-replace string to original_code."""
    if isinstance(patch, PatchResponse):
        current = original_code
        if patch.diff_patch:
            current = apply_unified_diff(
                current, patch.diff_patch, strip_fences=strip_fences
            )
        if patch.line_edits:
            current = apply_line_range_edits(current, patch.line_edits)
        if patch.search_replace:
            current = apply_search_replace_edits(current, patch.search_replace)
        return current

    if not isinstance(patch, str) or not patch.strip():
        raise PatchApplyError("Empty patch payload")

    candidate = _strip_diff_fence(patch) if strip_fences else patch
    if "@@ " in candidate:
        return apply_unified_diff(original_code, candidate, strip_fences=False)

    sr_matches = _SR_BLOCK_RE.findall(candidate)
    if sr_matches:
        edits = [
            SearchReplaceEdit(search=search, replace=replace, expected_count=1)
            for search, replace in sr_matches
        ]
        return apply_search_replace_edits(original_code, edits)

    raise PatchApplyError(
        "Patch string is neither a unified diff (@@) nor a SEARCH/REPLACE block"
    )
