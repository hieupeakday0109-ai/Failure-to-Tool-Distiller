"""SQLite WAL-backed Failure-to-Axiom Scar memory with 3-stage deduplication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scar_forge.config import EngineConfig
from scar_forge.models import DistilledError, ScarRecord
from scar_forge.tokens import (
    count_tokens,
    truncate_rule_to_budget,
    truncate_to_axiom_budget,
    validate_axiom_format,
)

_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
_LINE_NUM_RE = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s'\"]+")
_POSIX_PATH_RE = re.compile(r"/(?:tmp|var|usr|home|Users|mnt)/[^\s'\"]+")
_IDENT_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "for",
        "of",
        "to",
        "in",
        "on",
        "with",
        "object",
        "has",
        "no",
        "error",
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_text_for_sig(text: str) -> str:
    cleaned = _HEX_ADDR_RE.sub("0xADDR", text or "")
    cleaned = _LINE_NUM_RE.sub("line N", cleaned)
    cleaned = _WIN_PATH_RE.sub("<PATH>", cleaned)
    cleaned = _POSIX_PATH_RE.sub("<PATH>", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _whole_token_set(text: str) -> set[str]:
    return {tok.lower() for tok in _IDENT_TOKEN_RE.findall(text or "") if tok}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


class ScarStore:
    """Thread-safe SQLite WAL Scar store with atomic JSON mirror and 3-stage dedup."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        db_path: Path | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.ensure_dirs()
        self.db_path = Path(db_path) if db_path is not None else self.config.scar_db_path
        self.json_path = (
            self.db_path.with_suffix(".json")
            if db_path is not None
            else self.config.scar_json_path
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scars (
                        scar_id TEXT PRIMARY KEY,
                        axiom TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        exception_type TEXT NOT NULL,
                        error_signature TEXT NOT NULL,
                        tags_json TEXT NOT NULL,
                        context_cue TEXT NOT NULL,
                        occurrence_count INTEGER NOT NULL DEFAULT 1,
                        prevented_count INTEGER NOT NULL DEFAULT 0,
                        source_code_snippet TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_scars_sig ON scars(error_signature)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_scars_exc ON scars(exception_type)"
                )
            finally:
                conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ScarRecord:
        return ScarRecord(
            scar_id=row["scar_id"],
            axiom=row["axiom"],
            token_count=int(row["token_count"]),
            exception_type=row["exception_type"],
            error_signature=row["error_signature"],
            tags=json.loads(row["tags_json"]),
            context_cue=row["context_cue"],
            occurrence_count=int(row["occurrence_count"]),
            prevented_count=int(row["prevented_count"]),
            source_code_snippet=row["source_code_snippet"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
        )

    def _sync_json_mirror(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT * FROM scars ORDER BY created_at ASC, scar_id ASC"
        ).fetchall()
        payload = [self._row_to_record(r).model_dump(mode="json") for r in rows]
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.json_path.parent),
                prefix=".tmp_scars_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.json_path)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _synthesize_axiom(
        self,
        error: DistilledError,
        code: str,
        tags: Sequence[str],
        *,
        avoid_hint: str = "",
        use_hint: str = "",
        rule_hint: str = "",
    ) -> str:
        budget = min(14, self.config.max_scar_tokens)

        if rule_hint.strip():
            axiom = truncate_rule_to_budget(rule_hint, max_tokens=budget)
            if validate_axiom_format(axiom, max_tokens=budget):
                return axiom

        if avoid_hint.strip() or use_hint.strip():
            return truncate_to_axiom_budget(
                avoid_hint or error.exception_type,
                use_hint or "handle safely",
                max_tokens=budget,
            )

        combined = " ".join(
            [
                error.exception_type,
                error.core_message,
                error.failing_line,
                code,
                " ".join(tags),
            ]
        ).lower()

        # Domain-aware telegraphic compression
        if ("dataframe" in combined or "pandas" in combined or "pd." in combined) and (
            "append" in combined
        ):
            return truncate_to_axiom_budget("DataFrame.append", "pd.concat", max_tokens=budget)

        if (
            "offset-naive" in combined
            or "offset-aware" in combined
            or ("naive" in combined and "aware" in combined)
            or ("naive" in combined and "datetime" in combined)
        ):
            return truncate_to_axiom_budget(
                "naive datetime", "timezone.utc aware", max_tokens=budget
            )

        if (
            "bom" in combined
            or "\ufeff" in combined
            or "\\ufeff" in combined
            or ("utf-8-sig" in combined)
            or ("unicodedecodeerror" in combined and "csv" in combined)
        ):
            return truncate_to_axiom_budget(
                "plain csv", "utf8sig strip BOM", max_tokens=budget
            )

        if (
            "allow_nan" in combined
            or "not json compliant" in combined
            or ("nan" in combined and "json" in combined)
            or ("decimal" in combined and "json" in combined)
        ):
            return truncate_to_axiom_budget(
                "json nan", "strict allow_nan", max_tokens=budget
            )

        if (
            "bool(" in combined
            and ("'false'" in combined or "false" in combined or "webhook" in combined)
        ) or ("invalid literal for int()" in combined and "'false'" in combined):
            return truncate_to_axiom_budget(
                "bool(str)", "parse_bool_strict", max_tokens=budget
            )

        if "semver" in combined or ("1.10" in combined and "1.2" in combined):
            return truncate_to_axiom_budget(
                "string sort semver", "int tuple key", max_tokens=budget
            )

        if "html" in combined and (
            "re.sub" in combined or "regex" in combined or "htmlparser" in combined
        ):
            return truncate_to_axiom_budget(
                "regex strip html", "HTMLParser", max_tokens=budget
            )

        if "traversal" in combined or "pureposixpath" in combined or (
            "os.path.join" in combined and "path" in combined
        ):
            return truncate_to_axiom_budget(
                "raw path.join", "PurePosixPath", max_tokens=budget
            )

        if "zerodivisionerror" in combined or "division by zero" in combined:
            return truncate_to_axiom_budget(
                "unguarded / 0", "check denom != 0", max_tokens=budget
            )

        # Generic telegraphic extraction from failing line / code / message
        snippet = (error.failing_line or code or "").strip()
        call_match = _CALL_RE.search(snippet)
        if call_match:
            call_sym = call_match.group(1)
            short_sym = call_sym.split(".")[-1]
            avoid_clause = call_sym
            use_clause = f"guard {error.exception_type} {short_sym}"
            candidate = truncate_to_axiom_budget(avoid_clause, use_clause, max_tokens=budget)
            if count_tokens(candidate) <= budget and validate_axiom_format(
                candidate, max_tokens=budget
            ):
                return candidate

        msg_words = [
            w
            for w in _IDENT_TOKEN_RE.findall(error.core_message)
            if w.lower() not in _STOP_WORDS
        ]
        cue_phrase = " ".join(msg_words[:3]) if msg_words else error.exception_type
        avoid_clause = f"{error.exception_type} {cue_phrase}".strip()
        use_clause = f"validate {cue_phrase}" if cue_phrase else "validate input"
        return truncate_to_axiom_budget(avoid_clause, use_clause, max_tokens=budget)

    @staticmethod
    def _compute_exact_signature(
        error: DistilledError, code: str
    ) -> str:
        norm_exc = error.exception_type.strip().lower()
        norm_msg = _normalize_text_for_sig(error.core_message)
        norm_line = _normalize_text_for_sig(error.failing_line or code)
        raw = f"{norm_exc}|{norm_msg}|{norm_line}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _jaccard_tokens(
        exc_type: str, core_msg: str, failing_line: str, axiom: str
    ) -> set[str]:
        combined = " ".join([exc_type, core_msg, failing_line, axiom])
        return _whole_token_set(combined) - {"avoid", "use", "rule"}

    def record_failure(
        self,
        error: DistilledError,
        code: str = "",
        tags: Sequence[str] | None = None,
        *,
        avoid_hint: str = "",
        use_hint: str = "",
        rule_hint: str = "",
        context_cue: str = "",
    ) -> tuple[ScarRecord, bool]:
        """Distill a failure into a <=14 token Scar with 3-stage deduplication."""
        clean_tags = sorted(
            {t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()}
        )
        axiom = self._synthesize_axiom(
            error,
            code,
            clean_tags,
            avoid_hint=avoid_hint,
            use_hint=use_hint,
            rule_hint=rule_hint,
        )
        tok_count = count_tokens(axiom)
        if tok_count > self.config.max_scar_tokens or not validate_axiom_format(
            axiom, max_tokens=self.config.max_scar_tokens
        ):
            axiom = truncate_to_axiom_budget(
                error.exception_type, "validate input", max_tokens=self.config.max_scar_tokens
            )
            tok_count = count_tokens(axiom)

        exact_sig = self._compute_exact_signature(error, code)
        norm_core = _normalize_text_for_sig(error.core_message)
        new_tokens = self._jaccard_tokens(
            error.exception_type, error.core_message, error.failing_line or code, axiom
        )
        cue = context_cue.strip() or " ".join(
            part
            for part in (error.exception_type, error.core_message[:80], " ".join(clean_tags))
            if part
        ).strip()
        now = _utc_now()
        now_iso = now.isoformat()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")

                # Stage 1: Exact signature match
                existing_row = conn.execute(
                    "SELECT * FROM scars WHERE error_signature = ?", (exact_sig,)
                ).fetchone()

                # Stage 2: Structural match (exception_type + core_message + identical axiom)
                if existing_row is None:
                    candidates = conn.execute(
                        "SELECT * FROM scars WHERE exception_type = ?",
                        (error.exception_type,),
                    ).fetchall()
                    for cand in candidates:
                        cand_rec = self._row_to_record(cand)
                        if cand_rec.axiom == axiom:
                            existing_row = cand
                            break
                        if norm_core and _normalize_text_for_sig(cand_rec.context_cue).startswith(
                            f"{error.exception_type.lower()} {norm_core}"
                        ) and cand_rec.axiom == axiom:
                            existing_row = cand
                            break

                # Stage 3: Token Jaccard similarity >= threshold
                if existing_row is None:
                    all_rows = conn.execute("SELECT * FROM scars").fetchall()
                    best_score = 0.0
                    best_row: sqlite3.Row | None = None
                    for cand in all_rows:
                        cand_rec = self._row_to_record(cand)
                        cand_tokens = self._jaccard_tokens(
                            cand_rec.exception_type,
                            cand_rec.context_cue,
                            cand_rec.source_code_snippet,
                            cand_rec.axiom,
                        )
                        sim = _jaccard_similarity(new_tokens, cand_tokens)
                        if sim >= self.config.dedup_jaccard_threshold and sim > best_score:
                            best_score = sim
                            best_row = cand
                    existing_row = best_row

                if existing_row is not None:
                    rec = self._row_to_record(existing_row)
                    merged_tags = sorted(set(rec.tags) | set(clean_tags))
                    conn.execute(
                        """
                        UPDATE scars
                        SET occurrence_count = occurrence_count + 1,
                            tags_json = ?,
                            last_seen_at = ?
                        WHERE scar_id = ?
                        """,
                        (json.dumps(merged_tags), now_iso, rec.scar_id),
                    )
                    updated_row = conn.execute(
                        "SELECT * FROM scars WHERE scar_id = ?", (rec.scar_id,)
                    ).fetchone()
                    conn.execute("COMMIT")
                    self._sync_json_mirror(conn)
                    return self._row_to_record(updated_row), False

                scar_id = f"scar_{exact_sig[:12]}"
                # Guard against rare hash prefix collision with distinct axioms
                collision = conn.execute(
                    "SELECT 1 FROM scars WHERE scar_id = ?", (scar_id,)
                ).fetchone()
                if collision is not None:
                    suffix = hashlib.sha256(
                        f"{exact_sig}|{axiom}|{now_iso}".encode()
                    ).hexdigest()[:6]
                    scar_id = f"scar_{exact_sig[:8]}_{suffix}"

                snippet = (error.failing_line or code or "")[:300]
                conn.execute(
                    """
                    INSERT INTO scars (
                        scar_id, axiom, token_count, exception_type, error_signature,
                        tags_json, context_cue, occurrence_count, prevented_count,
                        source_code_snippet, created_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?)
                    """,
                    (
                        scar_id,
                        axiom,
                        tok_count,
                        error.exception_type,
                        exact_sig,
                        json.dumps(clean_tags),
                        cue,
                        snippet,
                        now_iso,
                        now_iso,
                    ),
                )
                inserted_row = conn.execute(
                    "SELECT * FROM scars WHERE scar_id = ?", (scar_id,)
                ).fetchone()
                conn.execute("COMMIT")
                self._sync_json_mirror(conn)
                return self._row_to_record(inserted_row), True
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def list_scars(self) -> list[ScarRecord]:
        """Return all persisted Scars ordered deterministically."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM scars ORDER BY created_at ASC, scar_id ASC"
                ).fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                conn.close()

    def get_scar(self, scar_id: str) -> ScarRecord | None:
        """Fetch a single Scar by ID if present."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM scars WHERE scar_id = ?", (scar_id,)
                ).fetchone()
                return self._row_to_record(row) if row is not None else None
            finally:
                conn.close()

    def mark_prevented(self, scar_ids: Sequence[str]) -> int:
        """Increment prevented_count for each valid scar_id in scar_ids."""
        ids = [sid for sid in scar_ids if isinstance(sid, str) and sid.strip()]
        if not ids:
            return 0
        now_iso = _utc_now().isoformat()
        updated = 0
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for sid in ids:
                    cur = conn.execute(
                        """
                        UPDATE scars
                        SET prevented_count = prevented_count + 1,
                            last_seen_at = ?
                        WHERE scar_id = ?
                        """,
                        (now_iso, sid),
                    )
                    updated += cur.rowcount
                conn.execute("COMMIT")
                if updated > 0:
                    self._sync_json_mirror(conn)
                return updated
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def clear(self) -> None:
        """Delete all stored Scars (used in test isolation or resets)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM scars")
                conn.execute("COMMIT")
                self._sync_json_mirror(conn)
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def stats(self) -> dict[str, Any]:
        """Compute aggregate Scar memory statistics."""
        scars = self.list_scars()
        total = len(scars)
        avg_tokens = round(sum(s.token_count for s in scars) / total, 2) if total else 0.0
        return {
            "count": total,
            "total_occurrences": sum(s.occurrence_count for s in scars),
            "total_prevented": sum(s.prevented_count for s in scars),
            "avg_tokens": avg_tokens,
        }
