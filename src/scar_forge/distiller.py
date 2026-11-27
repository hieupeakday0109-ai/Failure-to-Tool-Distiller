"""Stderr and traceback distiller stripping boilerplate frames and verbose logs."""

from __future__ import annotations

import inspect
import re
import sysconfig
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from scar_forge.models import DistilledError
from scar_forge.tokens import count_tokens

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TRACEBACK_HDR_RE = re.compile(r"^Traceback \(most recent call last\):\s*$")
_FRAME_RE = re.compile(
    r'^\s*File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>.+))?$'
)
_CARET_RE = re.compile(r"^\s*[~^]+\s*$")
_REPEAT_RE = re.compile(r"^\s*\[Previous line repeated \d+ more times\]\s*$")
_EXC_IN_TB_RE = re.compile(
    r"^(?P<etype>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*)"
    r"(?::\s*(?P<msg>.*))?$"
)
_FALLBACK_EXC_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*"
    r"(?:Error|Exception|Interrupt|Exit)(?::\s*(.*))?$"
)
_CHAIN_SEPARATORS = frozenset(
    {
        "During handling of the above exception, another exception occurred:",
        "The above exception was the direct cause of the following exception:",
    }
)
_NOISE_PREFIX_RE = re.compile(
    r"^(?:\[(?:INFO|DEBUG|WARN|WARNING|TRACE)\]|(?:INFO|DEBUG|WARNING):|\d{4}-\d{2}-\d{2}[T ])",
    re.IGNORECASE,
)
_POSTAMBLE_RE = re.compile(
    r"^(?:\[(?:INFO|DEBUG|WARN|WARNING)\]|(?:INFO|DEBUG|WARNING):|\d{4}-\d{2}-\d{2}[T ]|"
    r"Process finished|Command exited|Exit code\b)",
    re.IGNORECASE,
)

_WINDOWS_STDLIB_RE = re.compile(
    r"(?:^|/)(?:python\d+(?:\.\d+)?|pypy[^/]*)/lib/(?!site-packages)"
)
_POSIX_STDLIB_RE = re.compile(r"(?:^|/)lib/python\d+\.\d+/(?!site-packages)")

_THIRD_PARTY_MARKERS = (
    "/site-packages/",
    "/dist-packages/",
    "/.venv/",
    "/venv/lib/",
    "/venv/lib64/",
)
_RELATIVE_LIB_PREFIXES = (
    "pandas/core/",
    "numpy/",
    "pydantic/",
    "httpx/",
    "urllib3/",
    "requests/",
    "flask/",
    "werkzeug/",
    "click/",
    "_pytest/",
    "pytest/",
    "pluggy/",
)
_KNOWN_STDLIB_PREFIXES = (
    "importlib/",
    "runpy.py",
    "json/decoder.py",
    "json/__init__.py",
    "traceback.py",
    "subprocess.py",
    "threading.py",
    "asyncio/",
    "unittest/",
    "encodings/",
)
_SANDBOX_WRAPPER_MARKERS = (
    "_sandbox_runner.py",
    "sandbox_runner.py",
    "sandbox/runner.py",
    "scar_forge/sandbox/",
    "_harness.py",
    "<sandbox>",
)
_SYNTAX_ERROR_TYPES = frozenset({"SyntaxError", "IndentationError", "TabError"})


def _stdlib_roots() -> tuple[str, ...]:
    roots: list[str] = []
    for key in ("stdlib", "platstdlib"):
        raw = sysconfig.get_path(key)
        if raw:
            norm = raw.replace("\\", "/").rstrip("/").lower()
            if norm not in roots:
                roots.append(norm)
    return tuple(roots)


_STDLIB_ROOTS = _stdlib_roots()


@dataclass(slots=True)
class _Frame:
    path: str
    line_number: int
    function_name: str
    raw_code_line: str = ""
    raw_caret_line: str = ""
    is_noise: bool = False


@dataclass(slots=True)
class _TracebackBlock:
    frames: list[_Frame] = field(default_factory=list)
    user_frames: list[_Frame] = field(default_factory=list)
    exception_type: str = ""
    core_message: str = ""


class _ClassOrInstanceMethod:
    """Descriptor allowing TracebackDistiller.distill(...) and instance.distill(...)."""

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name

    def __get__(
        self, instance: Any, owner: type[Any] | None = None
    ) -> Callable[..., DistilledError]:
        target = instance if instance is not None else owner()
        bound = getattr(target, self.method_name)

        def _wrapper(raw_stderr: str, raw_stdout: str = "") -> DistilledError:
            return bound(raw_stderr, raw_stdout)

        _wrapper.__name__ = "distill"
        _wrapper.__doc__ = getattr(bound, "__doc__", None)
        _wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=[
                inspect.Parameter(
                    "raw_stderr",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation="str",
                ),
                inspect.Parameter(
                    "raw_stdout",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default="",
                    annotation="str",
                ),
            ],
            return_annotation="DistilledError",
        )
        return _wrapper


class TracebackDistiller:
    """Strips boilerplate stack frames and logs into compact DistilledError records."""

    def __init__(
        self,
        *,
        max_message_chars: int = 300,
        extra_noise_patterns: tuple[str, ...] = (),
    ) -> None:
        self.max_message_chars = max(40, max_message_chars)
        self.extra_noise_patterns = tuple(p.lower() for p in extra_noise_patterns)

    distill = _ClassOrInstanceMethod("_distill")

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return _ANSI_CSI_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))

    @staticmethod
    def _normalize_exception_group(text: str) -> str:
        if "Exception Group Traceback" not in text and "+--" not in text:
            return text
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("+-") or stripped.startswith("+ Exception Group"):
                continue
            if stripped.startswith("| "):
                cleaned_lines.append(stripped[2:])
            elif stripped == "|":
                cleaned_lines.append("")
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _has_traceback_or_exception(text: str) -> bool:
        for line in text.splitlines():
            s = line.strip()
            if _TRACEBACK_HDR_RE.match(line) or _FRAME_RE.match(line):
                return True
            if _FALLBACK_EXC_RE.match(s):
                return True
        return False

    def _is_sandbox_wrapper(self, path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        if any(marker in norm for marker in _SANDBOX_WRAPPER_MARKERS):
            return True
        return any(pattern in norm for pattern in self.extra_noise_patterns)

    def _is_noise_frame(self, path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        if self._is_sandbox_wrapper(path):
            return True
        if any(marker in norm for marker in _THIRD_PARTY_MARKERS):
            return True
        if norm.startswith(_RELATIVE_LIB_PREFIXES):
            return True
        if norm.startswith("<frozen "):
            return True
        if _WINDOWS_STDLIB_RE.search(norm) or _POSIX_STDLIB_RE.search(norm):
            return True
        for root in _STDLIB_ROOTS:
            if (norm == root or norm.startswith(f"{root}/")) and "/site-packages/" not in norm:
                return True
        return any(
            norm == prefix or norm.startswith(prefix) for prefix in _KNOWN_STDLIB_PREFIXES
        )

    @staticmethod
    def _shorten_path(path: str) -> str:
        if path.startswith("<") and path.endswith(">"):
            return path
        if "\\" in path or re.match(r"^[A-Za-z]:", path):
            parts = PureWindowsPath(path).parts
        else:
            parts = PurePosixPath(path).parts
        clean_parts = [p for p in parts if p not in ("/", "\\") and not p.endswith(":\\")]
        if not clean_parts:
            return path
        if len(clean_parts) >= 2:
            parent = clean_parts[-2]
            if parent.lower().startswith(("tmp", "scar_")):
                return clean_parts[-1]
            return f"{parent}/{clean_parts[-1]}"
        return clean_parts[-1]

    def _truncate_message(self, msg: str) -> str:
        collapsed = re.sub(r"\s+", " ", msg).strip()
        if len(collapsed) <= self.max_message_chars:
            return collapsed
        return f"{collapsed[: self.max_message_chars - 3].rstrip()}..."

    def _parse_blocks(self, lines: list[str]) -> list[_TracebackBlock]:
        blocks: list[_TracebackBlock] = []
        current_block = _TracebackBlock()
        current_frame: _Frame | None = None
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped in _CHAIN_SEPARATORS:
                if current_block.frames or current_block.exception_type:
                    blocks.append(current_block)
                current_block = _TracebackBlock()
                current_frame = None
                i += 1
                continue

            if _TRACEBACK_HDR_RE.match(line):
                if current_block.exception_type:
                    blocks.append(current_block)
                    current_block = _TracebackBlock()
                current_frame = None
                i += 1
                continue

            frame_match = _FRAME_RE.match(line)
            if frame_match:
                path = frame_match.group("path")
                line_no = int(frame_match.group("line"))
                func = (frame_match.group("func") or "").strip()
                noise = self._is_noise_frame(path)
                frame = _Frame(
                    path=path,
                    line_number=line_no,
                    function_name=func,
                    is_noise=noise,
                )
                current_block.frames.append(frame)
                if not noise:
                    current_block.user_frames.append(frame)
                current_frame = frame
                i += 1
                continue

            if current_frame is not None and (
                line.startswith("  ") or line.startswith("\t")
            ):
                if _REPEAT_RE.match(line):
                    i += 1
                    continue
                if _CARET_RE.match(line) and current_frame.raw_code_line:
                    current_frame.raw_caret_line = line
                elif not current_frame.raw_code_line and stripped:
                    current_frame.raw_code_line = line
                i += 1
                continue

            if stripped and not line.startswith((" ", "\t")):
                if _POSTAMBLE_RE.match(stripped):
                    i += 1
                    continue
                exc_match = _EXC_IN_TB_RE.match(stripped)
                if exc_match:
                    etype = exc_match.group("etype")
                    msg_parts: list[str] = []
                    first_msg = (exc_match.group("msg") or "").strip()
                    if first_msg:
                        msg_parts.append(first_msg)

                    # Collect up to 2 concise continuation lines for multi-line errors
                    j = i + 1
                    cont_count = 0
                    while j < len(lines) and cont_count < 2:
                        next_line = lines[j]
                        next_stripped = next_line.strip()
                        if not next_stripped:
                            break
                        if (
                            next_stripped in _CHAIN_SEPARATORS
                            or _TRACEBACK_HDR_RE.match(next_line)
                            or _FRAME_RE.match(next_line)
                            or _POSTAMBLE_RE.match(next_stripped)
                        ):
                            break
                        msg_parts.append(next_stripped)
                        cont_count += 1
                        j += 1

                    current_block.exception_type = etype
                    current_block.core_message = self._truncate_message(
                        " ".join(msg_parts)
                    )
                    current_frame = None
                    i = j
                    continue

            i += 1

        if current_block.frames or current_block.exception_type:
            blocks.append(current_block)
        return blocks

    @staticmethod
    def _extract_aligned_code_and_caret(
        frame: _Frame, *, is_syntax: bool
    ) -> tuple[str, str]:
        raw_code = frame.raw_code_line
        if not raw_code:
            return "", ""
        leading_ws = len(raw_code) - len(raw_code.lstrip(" "))
        failing_line = raw_code.strip()

        caret_line = ""
        if is_syntax and frame.raw_caret_line:
            raw_caret = frame.raw_caret_line
            if len(raw_caret) > leading_ws and raw_caret[:leading_ws].strip() == "":
                caret_line = raw_caret[leading_ws:].rstrip()
            else:
                caret_line = raw_caret.strip()
        return failing_line, caret_line

    def _fallback_distill(
        self, clean_text: str, raw_chars: int, raw_tokens: int
    ) -> DistilledError:
        lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
        filtered = [ln for ln in lines if not _NOISE_PREFIX_RE.match(ln)]
        scan_lines = filtered if filtered else lines

        exception_type = ""
        core_message = ""

        for ln in reversed(scan_lines):
            m = _FALLBACK_EXC_RE.match(ln)
            if m:
                if ":" in ln:
                    etype, msg = ln.split(":", 1)
                    exception_type = etype.strip()
                    core_message = self._truncate_message(msg.strip())
                else:
                    exception_type = ln
                    core_message = ""
                break

        if not exception_type:
            lowered = clean_text.lower()
            if "timed out" in lowered or "timeout" in lowered:
                exception_type = "TimeoutError"
                match_ln = next(
                    (
                        ln
                        for ln in reversed(scan_lines)
                        if "timed out" in ln.lower() or "timeout" in ln.lower()
                    ),
                    "Execution timed out",
                )
                core_message = self._truncate_message(match_ln)
            elif any(k in lowered for k in ("memoryerror", "out of memory", "oom", "killed")):
                exception_type = "MemoryError"
                match_ln = next(
                    (
                        ln
                        for ln in reversed(scan_lines)
                        if any(
                            k in ln.lower()
                            for k in ("memory", "oom", "killed")
                        )
                    ),
                    "Memory limit exceeded",
                )
                core_message = self._truncate_message(match_ln)
            elif scan_lines:
                exception_type = "ExecutionError"
                core_message = self._truncate_message(" | ".join(scan_lines[-3:]))
            else:
                exception_type = "UnknownError"
                core_message = "No error output captured"

        draft = DistilledError(
            exception_type=exception_type,
            core_message=core_message,
            raw_chars=raw_chars,
            raw_tokens=raw_tokens,
        )
        formatted = draft.format_for_prompt()
        distilled_chars = len(formatted)
        distilled_tokens = count_tokens(formatted)
        ratio = (
            max(0.0, round(1.0 - (distilled_tokens / raw_tokens), 4))
            if raw_tokens > 0
            else 0.0
        )
        return draft.model_copy(
            update={
                "distilled_chars": distilled_chars,
                "distilled_tokens": distilled_tokens,
                "compression_ratio": ratio,
            }
        )

    def _distill(self, raw_stderr: str, raw_stdout: str = "") -> DistilledError:
        """Distill raw stderr/stdout into a compact DistilledError with token metrics."""
        stderr_str = raw_stderr or ""
        stdout_str = raw_stdout or ""
        if stderr_str and stdout_str:
            combined_raw = f"{stderr_str}\n{stdout_str}"
        else:
            combined_raw = stderr_str or stdout_str

        raw_chars = len(combined_raw)
        raw_tokens = count_tokens(combined_raw)

        clean_stderr = self._normalize_exception_group(self._strip_ansi(stderr_str))
        clean_stdout = self._normalize_exception_group(self._strip_ansi(stdout_str))

        if not clean_stderr.strip() or (
            not self._has_traceback_or_exception(clean_stderr)
            and self._has_traceback_or_exception(clean_stdout)
        ):
            primary_text = clean_stdout if clean_stdout.strip() else clean_stderr
        else:
            primary_text = clean_stderr

        lines = primary_text.splitlines()
        tb_start = None
        for idx, line in enumerate(lines):
            if _TRACEBACK_HDR_RE.match(line) or _FRAME_RE.match(line):
                tb_start = idx
                break

        if tb_start is None:
            return self._fallback_distill(primary_text, raw_chars, raw_tokens)

        blocks = self._parse_blocks(lines[tb_start:])
        if not blocks:
            return self._fallback_distill(primary_text, raw_chars, raw_tokens)

        final_block = blocks[-1]
        exception_type = final_block.exception_type or "ExecutionError"
        core_message = final_block.core_message

        total_frames = sum(len(b.frames) for b in blocks)
        total_user_frames = sum(len(b.user_frames) for b in blocks)
        frames_dropped = total_frames - total_user_frames

        selected_frame: _Frame | None = None
        for block in reversed(blocks):
            if block.user_frames:
                selected_frame = block.user_frames[-1]
                break

        if selected_frame is None:
            non_wrapper = [
                f for f in final_block.frames if not self._is_sandbox_wrapper(f.path)
            ]
            if non_wrapper:
                selected_frame = non_wrapper[-1]
            elif final_block.frames:
                selected_frame = final_block.frames[-1]

        root_cause = ""
        if len(blocks) > 1:
            root_block = blocks[0]
            if root_block.exception_type:
                rc = (
                    f"{root_block.exception_type}: {root_block.core_message}"
                    if root_block.core_message
                    else root_block.exception_type
                )
                if root_block.user_frames:
                    rf = root_block.user_frames[-1]
                    loc = f" (at {self._shorten_path(rf.path)}:{rf.line_number})"
                    if len(rc) + len(loc) <= 80:
                        rc = f"{rc}{loc}"
                root_cause = rc

        short_etype = exception_type.rsplit(".", 1)[-1]
        is_syntax = short_etype in _SYNTAX_ERROR_TYPES or (
            selected_frame is not None and selected_frame.function_name == ""
        )

        failing_line = ""
        caret_line = ""
        line_number: int | None = None
        function_name = ""
        file_path = ""
        frames_kept = 0

        if selected_frame is not None:
            frames_kept = 1
            if total_user_frames == 0 and frames_dropped > 0:
                frames_dropped -= 1
            line_number = selected_frame.line_number
            function_name = selected_frame.function_name
            file_path = self._shorten_path(selected_frame.path)
            failing_line, caret_line = self._extract_aligned_code_and_caret(
                selected_frame, is_syntax=is_syntax
            )

        draft = DistilledError(
            exception_type=exception_type,
            core_message=core_message,
            failing_line=failing_line,
            line_number=line_number,
            function_name=function_name,
            file_path=file_path,
            caret_line=caret_line,
            root_cause=root_cause,
            frames_dropped=max(0, frames_dropped),
            frames_kept=frames_kept,
            is_syntax_error=short_etype in _SYNTAX_ERROR_TYPES,
            raw_chars=raw_chars,
            raw_tokens=raw_tokens,
        )

        formatted = draft.format_for_prompt()
        distilled_chars = len(formatted)
        distilled_tokens = count_tokens(formatted)
        compression_ratio = (
            max(0.0, round(1.0 - (distilled_tokens / raw_tokens), 4))
            if raw_tokens > 0
            else 0.0
        )

        return draft.model_copy(
            update={
                "distilled_chars": distilled_chars,
                "distilled_tokens": distilled_tokens,
                "compression_ratio": compression_ratio,
            }
        )
