"""Lightweight BM25 + tag-hybrid Top-K retriever for Scars and Skills."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

from scar_forge.config import EngineConfig
from scar_forge.models import ScarRecord, SkillRecord

T = TypeVar("T")

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

_STOPWORDS = frozenset(
    {
        "avoid",
        "use",
        "rule",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "from",
        "by",
        "into",
        "is",
        "are",
        "be",
        "that",
        "this",
        "it",
        "as",
        "at",
    }
)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into normalized subwords, splitting snake_case and CamelCase."""
    if not text:
        return []
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        lower = raw.lower()
        if lower not in _STOPWORDS:
            tokens.append(lower)
        # Split CamelCase sub-parts (e.g. DataFrame -> data, frame)
        parts = [p.lower() for p in _CAMEL_SPLIT_RE.findall(raw)]
        for part in parts:
            if part != lower and len(part) >= 2 and part not in _STOPWORDS:
                tokens.append(part)
    return tokens


class BM25Retriever:
    """Zero-external-ML BM25 + tag/name hybrid Top-K context selector (clamped 1..5)."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tag_boost: float = 1.6,
        name_boost: float = 1.4,
    ) -> None:
        self.config = config or EngineConfig()
        self.k1 = k1
        self.b = b
        self.tag_boost = tag_boost
        self.name_boost = name_boost

    @staticmethod
    def clamp_top_k(top_k: int | None, default: int) -> int:
        val = default if top_k is None else top_k
        return max(1, min(5, int(val)))

    def _score_corpus(
        self,
        query: str,
        docs_tokens: list[list[str]],
        tag_sets: list[set[str]],
        name_sets: list[set[str]],
    ) -> list[float]:
        n_docs = len(docs_tokens)
        if n_docs == 0:
            return []

        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return [0.0] * n_docs

        q_set = set(q_tokens)
        doc_freqs: Counter[str] = Counter()
        term_freqs: list[Counter[str]] = []
        lengths: list[int] = []

        for toks in docs_tokens:
            tf = Counter(toks)
            term_freqs.append(tf)
            lengths.append(len(toks))
            for term in tf:
                doc_freqs[term] += 1

        avg_dl = sum(lengths) / n_docs if n_docs > 0 else 0.0
        if avg_dl <= 0.0:
            avg_dl = 1.0

        scores: list[float] = []
        for idx in range(n_docs):
            tf = term_freqs[idx]
            dl = lengths[idx]
            tag_hits = q_set & tag_sets[idx]
            name_hits = q_set & name_sets[idx]
            lexical_hits = q_set & set(tf.keys())

            if not lexical_hits and not tag_hits and not name_hits:
                scores.append(0.0)
                continue

            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = doc_freqs[term]
                # BM25+ non-negative IDF guarantees positive signal even for N=1 or all-doc terms
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
                score += idf * ((freq * (self.k1 + 1.0)) / max(denom, 1e-9))

            score += self.tag_boost * len(tag_hits)
            score += self.name_boost * len(name_hits)
            scores.append(score)

        return scores

    def score_scars(
        self, query: str, scars: Sequence[ScarRecord]
    ) -> list[tuple[ScarRecord, float]]:
        """Score all Scars against query using BM25 + tag/exception hybrid scoring."""
        if not scars:
            return []
        docs_tokens: list[list[str]] = []
        tag_sets: list[set[str]] = []
        name_sets: list[set[str]] = []

        for scar in scars:
            tags_text = " ".join(scar.tags)
            doc_text = " ".join(
                [scar.axiom, scar.exception_type, scar.context_cue, tags_text]
            )
            docs_tokens.append(tokenize_for_bm25(doc_text))
            tag_sets.append(set(tokenize_for_bm25(tags_text)))
            name_sets.append(set(tokenize_for_bm25(scar.exception_type)))

        scores = self._score_corpus(query, docs_tokens, tag_sets, name_sets)
        ranked = sorted(
            zip(scars, scores, strict=True),
            key=lambda item: (-item[1], -item[0].prevented_count, item[0].scar_id),
        )
        return [(s, sc) for s, sc in ranked if sc > 0.0]

    def retrieve_scars(
        self,
        query: str,
        scars: Sequence[ScarRecord],
        top_k: int | None = None,
    ) -> list[ScarRecord]:
        """Return Top-K (clamped 1..5) strictly relevant Scars for the current query."""
        limit = self.clamp_top_k(top_k, self.config.top_k_scars)
        scored = self.score_scars(query, scars)
        return [scar for scar, _ in scored[:limit]]

    def score_skills(
        self, query: str, skills: Sequence[SkillRecord]
    ) -> list[tuple[SkillRecord, float]]:
        """Score all Skills against query using BM25 + skill_name + tag hybrid scoring."""
        if not skills:
            return []
        docs_tokens: list[list[str]] = []
        tag_sets: list[set[str]] = []
        name_sets: list[set[str]] = []

        for skill in skills:
            tags_text = " ".join(skill.tags)
            doc_text = " ".join(
                [skill.skill_name, skill.docstring, skill.signature, tags_text]
            )
            docs_tokens.append(tokenize_for_bm25(doc_text))
            tag_sets.append(set(tokenize_for_bm25(tags_text)))
            name_sets.append(set(tokenize_for_bm25(skill.skill_name)))

        scores = self._score_corpus(query, docs_tokens, tag_sets, name_sets)
        ranked = sorted(
            zip(skills, scores, strict=True),
            key=lambda item: (-item[1], -item[0].reuse_count, item[0].skill_name),
        )
        return [(sk, sc) for sk, sc in ranked if sc > 0.0]

    def retrieve_skills(
        self,
        query: str,
        skills: Sequence[SkillRecord],
        top_k: int | None = None,
    ) -> list[SkillRecord]:
        """Return Top-K (clamped 1..5) strictly relevant Skills for the current query."""
        limit = self.clamp_top_k(top_k, self.config.top_k_skills)
        scored = self.score_skills(query, skills)
        return [skill for skill, _ in scored[:limit]]
