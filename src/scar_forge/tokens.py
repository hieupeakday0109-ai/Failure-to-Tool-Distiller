"""Deterministic subword token counter and telegraphic Scar axiom budget gate."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

TOKENIZER_BACKEND = "deterministic_subword_regex_v1"

_TOKEN_RE = re.compile(
    r"->|=>|!=|==|<=|>=|//|\*\*|<<|>>|::"
    r"|_[A-Za-z0-9]+"
    r"|[A-Za-z]+"
    r"|[0-9]+"
    r"|[^\sA-Za-z0-9]"
)
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
_ALPHA_NUM_RE = re.compile(r"[A-Za-z]+|[0-9]+")

_AVOID_USE_RE = re.compile(r"^\[AVOID\]\s+(\S.*?)\s+->\s+\[USE\]\s+(\S.*?)$")
_RULE_RE = re.compile(r"^\[RULE\]\s+(\S.*?)$")

_AVOID_PREFIX_RE = re.compile(
    r"^(?:\[AVOID\]\s*|AVOID\s*:\s*|never\s+use\s+|do\s+not\s+use\s+|avoid\s+)",
    re.IGNORECASE,
)
_USE_PREFIX_RE = re.compile(
    r"^(?:\[USE\]\s*|USE\s*:\s*|instead\s+use\s+|prefer\s+|use\s+)",
    re.IGNORECASE,
)
_RULE_PREFIX_RE = re.compile(r"^(?:\[RULE\]\s*|RULE\s*:\s*)", re.IGNORECASE)
_CALL_ARGS_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\(([^()]+)\)")

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "for",
        "of",
        "with",
        "when",
        "that",
        "this",
        "is",
        "are",
        "be",
        "instead",
        "please",
        "always",
        "never",
        "directly",
        "raw",
        "calling",
        "using",
        "method",
        "function",
        "object",
        "value",
    }
)

_TRAILING_DANGLING_TOKENS = frozenset(
    {".", "(", "[", "{", ",", ":", "/", "+", "-", "=", "_", "\\"}
)


class AxiomBudgetError(ValueError):
    """Raised when an axiom cannot fit within max_tokens or violates format."""


class TokenCounter:
    """Deterministic subword-splitting token counter with <= 14 token axiom enforcement."""

    def __init__(self, max_axiom_tokens: int = 14) -> None:
        if max_axiom_tokens < 1:
            raise ValueError(f"max_axiom_tokens must be >= 1, got {max_axiom_tokens}")
        self.max_axiom_tokens = max_axiom_tokens

    @property
    def tokenizer_backend(self) -> str:
        """Return the identifier of the deterministic tokenization backend."""
        return TOKENIZER_BACKEND

    def get_metadata(self) -> dict[str, str | int]:
        """Return tokenizer backend metadata for benchmark and telemetry reporting."""
        return {
            "tokenizer_backend": self.tokenizer_backend,
            "max_axiom_tokens": self.max_axiom_tokens,
        }

    def _emit_alpha_chunks(
        self, piece: str, base_offset: int
    ) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for cm in _CAMEL_RE.finditer(piece):
            sub = cm.group(0)
            cs = base_offset + cm.start()
            if len(sub) > 10:
                for idx in range(0, len(sub), 6):
                    end = min(idx + 6, len(sub))
                    out.append((sub[idx:end], cs + idx, cs + end))
            else:
                out.append((sub, cs, base_offset + cm.end()))
        return out

    def _emit_digit_chunks(
        self, piece: str, base_offset: int
    ) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for idx in range(0, len(piece), 4):
            end = min(idx + 4, len(piece))
            out.append((piece[idx:end], base_offset + idx, base_offset + end))
        return out

    def tokenize_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Tokenize text into (token, start_char, end_char) subword spans."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text:
            return []

        spans: list[tuple[str, int, int]] = []
        for match in _TOKEN_RE.finditer(text):
            raw = match.group(0)
            base_start = match.start()
            base_end = match.end()

            if raw.startswith("_") and len(raw) > 1:
                sub_items: list[tuple[str, int, int]] = []
                for part in _ALPHA_NUM_RE.finditer(raw[1:]):
                    p_str = part.group(0)
                    p_off = base_start + 1 + part.start()
                    if p_str.isalpha():
                        sub_items.extend(self._emit_alpha_chunks(p_str, p_off))
                    else:
                        sub_items.extend(self._emit_digit_chunks(p_str, p_off))
                if sub_items:
                    first_tok, _, first_end = sub_items[0]
                    sub_items[0] = (f"_{first_tok}", base_start, first_end)
                    spans.extend(sub_items)
                else:
                    spans.append((raw, base_start, base_end))
            elif raw.isalpha():
                spans.extend(self._emit_alpha_chunks(raw, base_start))
            elif raw.isdigit():
                spans.extend(self._emit_digit_chunks(raw, base_start))
            else:
                spans.append((raw, base_start, base_end))
        return spans

    def tokenize(self, text: str) -> list[str]:
        """Return the list of subword tokens for text."""
        return [tok for tok, _, _ in self.tokenize_with_spans(text)]

    def count_tokens(self, text: str) -> int:
        """Count subword tokens in text deterministically."""
        return len(self.tokenize_with_spans(text))

    def count(self, text: str) -> int:
        """Alias for count_tokens."""
        return self.count_tokens(text)

    def count_messages_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        overhead_per_message: int = 4,
        reply_primer_tokens: int = 3,
    ) -> int:
        """Estimate chat-message token count deterministically."""
        if not messages:
            return 0
        total = reply_primer_tokens
        for msg in messages:
            total += overhead_per_message
            total += self.count_tokens(str(msg.get("role", "")))
            if "name" in msg and msg["name"] is not None:
                total += self.count_tokens(str(msg["name"]))
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif content is not None:
                serialized = json.dumps(
                    content, ensure_ascii=False, separators=(",", ":")
                )
                total += self.count_tokens(serialized)
        return total

    def count_messages(self, messages: Sequence[Mapping[str, Any]]) -> int:
        """Alias for count_messages_tokens."""
        return self.count_messages_tokens(messages)

    def validate_axiom_format(
        self, axiom: str, max_tokens: int | None = None
    ) -> bool:
        """Verify axiom matches telegraphic format and fits strictly within max_tokens."""
        limit = self.max_axiom_tokens if max_tokens is None else max_tokens
        if not isinstance(axiom, str) or limit <= 0:
            return False
        stripped = axiom.strip()
        if not stripped or "\n" in stripped or "\r" in stripped:
            return False

        avoid_match = _AVOID_USE_RE.match(stripped)
        if avoid_match:
            avoid_part, use_part = avoid_match.group(1), avoid_match.group(2)
            for forbidden in ("[AVOID]", "[USE]", "[RULE]", "->"):
                if forbidden in avoid_part or forbidden in use_part:
                    return False
        else:
            rule_match = _RULE_RE.match(stripped)
            if not rule_match:
                return False
            rule_part = rule_match.group(1)
            for forbidden in ("[AVOID]", "[USE]", "[RULE]"):
                if forbidden in rule_part:
                    return False

        tok_count = self.count_tokens(stripped)
        return 1 <= tok_count <= limit

    def _clean_clause(self, raw: str, *, is_avoid: bool) -> str:
        text = re.sub(r"\s+", " ", raw).strip()
        text = text.replace("->", " ")
        for marker in ("[AVOID]", "[USE]", "[RULE]"):
            text = text.replace(marker, " ")
        text = re.sub(r"\s+", " ", text).strip()
        prefix_re = _AVOID_PREFIX_RE if is_avoid else _USE_PREFIX_RE
        for _ in range(3):
            updated = prefix_re.sub("", text).strip()
            if updated == text:
                break
            text = updated
        text = text.rstrip(".;,:")
        return re.sub(r"\s+", " ", text).strip()

    def _drop_stopwords(self, clause: str) -> str:
        spans = self.tokenize_with_spans(clause)
        if len(spans) <= 1:
            return clause
        kept_pieces: list[str] = []
        last_end = 0
        dropped = 0
        for tok, start, end in spans:
            is_standalone = (
                (start == 0 or not clause[start - 1].isalnum())
                and (end == len(clause) or not clause[end].isalnum())
            )
            if is_standalone and tok.lower() in _STOPWORDS and (len(spans) - dropped) > 1:
                dropped += 1
                last_end = end
                continue
            gap = clause[last_end:start]
            if not kept_pieces:
                kept_pieces.append(clause[start:end])
            else:
                sep = " " if (gap and any(c.isspace() for c in gap)) else gap
                kept_pieces.append(sep + clause[start:end])
            last_end = end
        result = "".join(kept_pieces).strip()
        return result if result else clause

    def _compact_call_args(self, clause: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            fn_name = match.group(1)
            args = match.group(2)
            if self.count_tokens(args) > 1:
                return f"{fn_name}()"
            return match.group(0)

        compacted = _CALL_ARGS_RE.sub(_replace, clause)
        return re.sub(r"\s+", " ", compacted).strip()

    def _slice_clause_to_tokens(self, clause: str, max_toks: int) -> str:
        spans = self.tokenize_with_spans(clause)
        if not spans or max_toks <= 0:
            return clause
        k = min(max_toks, len(spans))
        while k > 1 and spans[k - 1][0] in _TRAILING_DANGLING_TOKENS:
            k -= 1
        end_idx = spans[k - 1][2]
        sliced = clause[:end_idx].strip().rstrip(".;,:")
        return sliced if sliced else clause[: spans[0][2]].strip()

    def truncate_to_axiom_budget(
        self, avoid: str, use: str, max_tokens: int | None = None
    ) -> str:
        """Compress avoid/use pair into a valid <= max_tokens [AVOID] X -> [USE] Y axiom."""
        limit = self.max_axiom_tokens if max_tokens is None else max_tokens
        if limit < 9:
            raise AxiomBudgetError(
                f"max_tokens={limit} is below minimum 9 required for [AVOID] X -> [USE] Y"
            )
        if not isinstance(avoid, str) or not isinstance(use, str):
            raise AxiomBudgetError("avoid and use must be strings")
        if not avoid.strip() and not use.strip():
            raise AxiomBudgetError("Cannot build axiom from empty avoid and use inputs")

        a = self._clean_clause(avoid, is_avoid=True) or "unsafe pattern"
        u = self._clean_clause(use, is_avoid=False) or "safe pattern"

        candidate = f"[AVOID] {a} -> [USE] {u}"
        if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
            candidate, limit
        ):
            return candidate

        # Pass 1: strip filler stopwords from longer side first.
        if self.count_tokens(a) >= self.count_tokens(u):
            a = self._drop_stopwords(a)
            candidate = f"[AVOID] {a} -> [USE] {u}"
            if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
                candidate, limit
            ):
                return candidate
            u = self._drop_stopwords(u)
        else:
            u = self._drop_stopwords(u)
            candidate = f"[AVOID] {a} -> [USE] {u}"
            if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
                candidate, limit
            ):
                return candidate
            a = self._drop_stopwords(a)

        candidate = f"[AVOID] {a} -> [USE] {u}"
        if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
            candidate, limit
        ):
            return candidate

        # Pass 2: compact verbose call arguments fn(long_args...) -> fn()
        a_comp = self._compact_call_args(a)
        u_comp = self._compact_call_args(u)
        if a_comp != a or u_comp != u:
            a, u = a_comp, u_comp
            candidate = f"[AVOID] {a} -> [USE] {u}"
            if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
                candidate, limit
            ):
                return candidate
            # If still over budget, drop empty () on call expressions to recover 2 tokens
            a = re.sub(r"([A-Za-z_][A-Za-z0-9_.]*)\(\)", r"\1", a)
            u = re.sub(r"([A-Za-z_][A-Za-z0-9_.]*)\(\)", r"\1", u)
            candidate = f"[AVOID] {a} -> [USE] {u}"
            if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
                candidate, limit
            ):
                return candidate

        # Pass 3: balanced span-preserving truncation across content budget
        budget = limit - 7
        use_cap = max(1, (budget + 1) // 2)
        avoid_cap = max(1, budget - use_cap)

        a_toks = self.count_tokens(a)
        u_toks = self.count_tokens(u)

        if a_toks < avoid_cap:
            use_cap = max(use_cap, budget - a_toks)
            avoid_cap = a_toks
        elif u_toks < use_cap:
            avoid_cap = max(avoid_cap, budget - u_toks)
            use_cap = u_toks

        if a_toks > avoid_cap:
            a = self._slice_clause_to_tokens(a, avoid_cap)
        if u_toks > use_cap:
            u = self._slice_clause_to_tokens(u, use_cap)

        candidate = f"[AVOID] {a} -> [USE] {u}"

        # Safety loop guaranteeing strict <= limit compliance
        while self.count_tokens(candidate) > limit:
            cur_a = self.count_tokens(a)
            cur_u = self.count_tokens(u)
            if cur_a <= 1 and cur_u <= 1:
                break
            if cur_a >= cur_u and cur_a > 1:
                a = self._slice_clause_to_tokens(a, cur_a - 1)
            elif cur_u > 1:
                u = self._slice_clause_to_tokens(u, cur_u - 1)
            else:
                a = self._slice_clause_to_tokens(a, cur_a - 1)
            candidate = f"[AVOID] {a} -> [USE] {u}"

        if not self.validate_axiom_format(candidate, limit):
            raise AxiomBudgetError(
                f"Failed to fit axiom within {limit} tokens: {candidate!r}"
            )
        return candidate

    def truncate_rule_to_budget(
        self, rule: str, max_tokens: int | None = None
    ) -> str:
        """Compress a rule string into a valid <= max_tokens [RULE] X axiom."""
        limit = self.max_axiom_tokens if max_tokens is None else max_tokens
        if limit < 4:
            raise AxiomBudgetError(
                f"max_tokens={limit} is below minimum 4 required for [RULE] X"
            )
        if not isinstance(rule, str) or not rule.strip():
            raise AxiomBudgetError("Cannot build [RULE] axiom from empty rule")

        cleaned = re.sub(r"\s+", " ", rule).strip()
        for marker in ("[AVOID]", "[USE]", "[RULE]"):
            cleaned = cleaned.replace(marker, " ")
        cleaned = _RULE_PREFIX_RE.sub("", cleaned).strip().rstrip(".;,:")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            raise AxiomBudgetError("Rule text is empty after marker stripping")

        candidate = f"[RULE] {cleaned}"
        if self.count_tokens(candidate) <= limit and self.validate_axiom_format(
            candidate, limit
        ):
            return candidate

        cleaned = self._drop_stopwords(cleaned)
        cleaned = self._compact_call_args(cleaned)
        content_budget = limit - 3
        if self.count_tokens(cleaned) > content_budget:
            cleaned = self._slice_clause_to_tokens(cleaned, content_budget)
        candidate = f"[RULE] {cleaned}"
        while self.count_tokens(candidate) > limit:
            cur = self.count_tokens(cleaned)
            if cur <= 1:
                break
            cleaned = self._slice_clause_to_tokens(cleaned, cur - 1)
            candidate = f"[RULE] {cleaned}"

        if not self.validate_axiom_format(candidate, limit):
            raise AxiomBudgetError(
                f"Failed to fit rule axiom within {limit} tokens: {candidate!r}"
            )
        return candidate


DEFAULT_TOKEN_COUNTER = TokenCounter(max_axiom_tokens=14)


def count_tokens(text: str) -> int:
    """Count tokens using the default deterministic subword counter."""
    return DEFAULT_TOKEN_COUNTER.count_tokens(text)


def count_messages_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Count chat message tokens using the default deterministic counter."""
    return DEFAULT_TOKEN_COUNTER.count_messages_tokens(messages)


def validate_axiom_format(axiom: str, max_tokens: int = 14) -> bool:
    """Validate telegraphic axiom format and <= max_tokens budget."""
    return DEFAULT_TOKEN_COUNTER.validate_axiom_format(axiom, max_tokens=max_tokens)


def truncate_to_axiom_budget(avoid: str, use: str, max_tokens: int = 14) -> str:
    """Truncate avoid/use pair into a valid <= max_tokens [AVOID] X -> [USE] Y axiom."""
    return DEFAULT_TOKEN_COUNTER.truncate_to_axiom_budget(
        avoid, use, max_tokens=max_tokens
    )


def truncate_rule_to_budget(rule: str, max_tokens: int = 14) -> str:
    """Truncate a rule string into a valid <= max_tokens [RULE] X axiom."""
    return DEFAULT_TOKEN_COUNTER.truncate_rule_to_budget(rule, max_tokens=max_tokens)


def get_tokenizer_backend() -> str:
    """Return the active tokenizer backend identifier."""
    return DEFAULT_TOKEN_COUNTER.tokenizer_backend
