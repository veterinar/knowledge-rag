"""FTS5 lexical index encapsulation for the knowledge-rag fast-path feature.

Isolated module — instantiated only when ``config.fts5_enabled`` is true and
wired into ``KnowledgeOrchestrator`` (Task 03). Storage layout, tokenizer,
and PRAGMAs are pinned by ADR-001 and ADR-005.

References:
- ADR-001: ``<data_dir>/fts5_index.db`` with WAL + busy_timeout=5000ms
- ADR-004: exception hierarchy — each error subclasses ``RuntimeError`` directly
- ADR-005: tokenizer ``unicode61 remove_diacritics 2 tokenchars '-_.'``
- TechSpec §Impl.Design.Core Interfaces #2, #3
- TechSpec §Security Surface — prepared statements + FTS5 metacharacter escape
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import tempfile
import threading
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

ChunkRow = Tuple[str, str, str, str]

_FTS5_TOKENIZER = "unicode61 remove_diacritics 2 tokenchars '-_.'"

_FTS5_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS fts5_documents USING fts5(
    chunk_id UNINDEXED,
    content,
    filename,
    category,
    tokenize = "{_FTS5_TOKENIZER}"
);
"""

# Metacharacters that carry syntactic meaning in an FTS5 MATCH expression.
# Wrapping each user-provided token in double quotes turns it into a literal
# phrase token, which neutralises operators (AND/OR/NOT/NEAR), prefix wildcard
# (``*``), column filters (``:``), grouping (``(``, ``)``) and injected
# quotes. See TechSpec §Security Surface.
_FTS5_TOKEN_SPLIT = re.compile(r"\s+")


def _escape_fts5_query(query: str) -> str:
    """Return an FTS5 MATCH expression that treats ``query`` as literal tokens.

    Doubles embedded quotes (FTS5 escape convention) and wraps every
    whitespace-separated fragment in double quotes so operators cannot leak.
    Empty input becomes an empty string — callers must skip the search.
    """
    stripped = query.strip()
    if not stripped:
        return ""
    parts = [p for p in _FTS5_TOKEN_SPLIT.split(stripped) if p]
    quoted = ['"' + p.replace('"', '""') + '"' for p in parts]
    return " ".join(quoted)


class Fts5NotReadyError(RuntimeError):
    """Raised when the fast-path is invoked before the FTS5 index is ready.

    Message includes a suggestion to fall back to ``search_method='auto'``
    (PRD OQ-4) so debug users can recover without editing config.
    """


class Fts5CorruptError(RuntimeError):
    """Raised when the FTS5 database file cannot be opened or is malformed."""


class Fts5MigrationError(RuntimeError):
    """Raised when the initial FTS5 rebuild fails. See marker file for cause."""


_SQL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _quote_sql_identifier(identifier: str) -> str:
    """Quote one internally generated SQLite identifier after strict validation."""
    if _SQL_IDENTIFIER.fullmatch(identifier) is None:
        raise Fts5MigrationError("unsafe internal SQLite identifier")
    return f'"{identifier}"'


class Fts5MigrationState:
    """Atomic JSON marker for the FTS5 rebuild lifecycle (PRD OQ-5, Q3).

    Schema::

        {
            "status": "complete" | "in_progress" | "failed",
            "docs_total": int,
            "docs_indexed": int,
            "started_at": ISO8601 str,
            "completed_at": ISO8601 str | None,
            "error": str | None,
        }

    Writes are cross-platform atomic: a NamedTemporaryFile in the same
    directory is fsynced and swapped in via ``os.replace``.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Optional[dict]:
        """Return the persisted payload, or ``None`` if the file is missing.

        Silently returns ``None`` on JSON decode failure — callers treat a
        corrupt marker the same as a missing one and rebuild from scratch.
        """
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def write(self, payload: dict) -> None:
        """Persist ``payload`` atomically (tempfile + fsync + os.replace)."""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        tmp_dir = self._path.parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(prefix=".fts5_state.", suffix=".tmp", dir=str(tmp_dir))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    json.dump(payload, tmp)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                os.replace(tmp_name, self._path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def is_complete(self) -> bool:
        data = self.read()
        return bool(data) and data.get("status") == "complete"


_FTS5_ROWS_DOMAIN = b"knowledge-rag.fts5.rows.v1\x00"
MARKER_SCHEMA_VERSION = 2  # marker schema v2 (task: v4.8.3 Gate 0 Package B)
_HEX64 = re.compile(r"[0-9a-f]{64}")  # digest fields must be exact 64-hex (P1-5)


def _is_uint(value: Any, minimum: int = 0) -> bool:  # strict schema ints: bool is rejected
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def compute_rows_digest(
    rows: Iterable[ChunkRow],
) -> Tuple[str, int]:  # canonical digest: str(value or "") UTF-8 fields, raw-byte sort, dup-reject
    encoded, seen = [], set()
    for row in rows:
        fields = tuple(str(value or "").encode("utf-8") for value in row)
        if fields[0] in seen:
            raise Fts5MigrationError(f"duplicate chunk_id in row set: {fields[0]!r}")
        seen.add(fields[0])
        encoded.append(fields)
    encoded.sort(key=lambda fields: fields[0])
    digest = hashlib.sha256(_FTS5_ROWS_DOMAIN + struct.pack(">Q", len(encoded)))
    for fields in encoded:
        for field in fields:
            digest.update(struct.pack(">Q", len(field)) + field)
    return digest.hexdigest(), len(encoded)


# Chroma COMPLETE row universe: id + document + packed embedding bytes +
# normalized full retrieval metadata. Domain-separated from the common
# (id, document, filename, category) universe the FTS parity uses.
_CHROMA_FULL_ROWS_DOMAIN = b"knowledge-rag.chroma.full-rows.v1\x00"
FullChunkRow = Tuple[str, str, bytes, str]  # (chunk_id, document, packed_embeddings, normalized_metadata_json)


def _pack_embedding(value: Any) -> bytes:
    """Deterministic big-endian IEEE-754 packing of one embedding vector.

    Non-finite floats (NaN/inf) are rejected: they are not canonicalizable
    across stores and would silently corrupt the full-row digest.
    """
    if value is None:
        return b""
    packed = bytearray()
    for component in value:
        if isinstance(component, (int, float)) and not isinstance(component, bool):
            as_float = float(component)
            if math.isnan(as_float) or math.isinf(as_float):
                raise Fts5MigrationError("non-finite embedding component (NaN/inf) is not canonicalizable")
            packed += struct.pack(">d", as_float)
        else:
            token = str(component).encode("utf-8")
            packed += struct.pack(">Q", len(token)) + token
    return bytes(packed)


# JSON-compatible canonical metadata value types. Types OUTSIDE this set fail
# closed: stringifying them (the old behavior) made 1, "1", True and "True"
# collide in the canonical form, silently masking real drift.
_METADATA_VALUE_TYPES = (str, int, float, bool, type(None), list, dict)


def _canonical_metadata_value(value: Any) -> Any:
    """Type-preserving canonical form of one metadata value.

    Nested dicts keep deterministic key order (sorted); nested lists keep
    element order (order is behavioral); unsupported types raise — never a
    lossy stringification.
    """
    if isinstance(value, dict):
        return {str(k): _canonical_metadata_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        return [_canonical_metadata_value(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise Fts5MigrationError("non-finite metadata value (NaN/inf) is not canonicalizable")
    if isinstance(value, _METADATA_VALUE_TYPES):
        return value
    raise Fts5MigrationError(
        f"unsupported metadata value type {type(value).__name__!r} — refusing lossy canonicalization"
    )


def normalize_retrieval_metadata(metadata: Any) -> str:
    """Canonical key-sorted, TYPE-PRESERVING JSON of the full metadata mapping.

    Values keep their JSON types (str/int/float/bool/None/list/dict); dicts
    are key-sorted, lists keep order. Unsupported or non-finite values raise
    ``Fts5MigrationError`` (fail closed) instead of colliding silently.
    """
    if not isinstance(metadata, dict):
        metadata = {}
    canonical = {str(k): _canonical_metadata_value(v) for k, v in sorted(metadata.items(), key=lambda kv: str(kv[0]))}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)


def capture_full_chunk_rows(collection: Any, batch_size: int = 500) -> List[FullChunkRow]:
    """Complete Chroma row universe: ids, documents, embeddings, full metadata.

    Same exact-id listing discipline as :func:`capture_chunk_rows` — the
    listing count must equal ``collection.count()`` and be duplicate-free —
    but hydrates embeddings and FULL metadata so the canonical digest covers
    everything retrieval-significant Chroma stores.
    """
    ids = [str(chunk_id) for chunk_id in (collection.get(include=[]) or {}).get("ids") or []]
    if len(ids) != int(collection.count()) or len(set(ids)) != len(ids):
        raise Fts5MigrationError("full-row id listing count/duplicate mismatch")
    ids.sort(key=lambda chunk_id: chunk_id.encode("utf-8"))
    rows: List[FullChunkRow] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        fetched = collection.get(ids=list(batch), include=["documents", "metadatas", "embeddings"])
        fetched = fetched if fetched is not None else {}
        # NEVER truth-test ndarray-backed values (``or []`` mis-evaluates a
        # 2-D array and can raise): branch on None explicitly and coerce
        # length via len(), which works for lists AND arrays.
        raw_ids = fetched.get("ids")
        raw_docs = fetched.get("documents")
        raw_metas = fetched.get("metadatas")
        raw_embs = fetched.get("embeddings")
        if raw_ids is None or raw_docs is None or raw_metas is None or raw_embs is None:
            raise Fts5MigrationError("full-row hydration missing a required field")
        got_ids = [str(rid) for rid in raw_ids]
        docs, metas, embs = raw_docs, raw_metas, raw_embs
        position_by_id = {rid: i for i, rid in enumerate(got_ids)}
        if not (len(got_ids) == len(docs) == len(metas) == len(embs) == len(position_by_id)) or not set(
            got_ids
        ).issubset(batch):
            raise Fts5MigrationError("full-row response cardinality/identity mismatch")
        for chunk_id in batch:
            i = position_by_id.get(chunk_id)
            if i is None:
                raise Fts5MigrationError(f"full-row read missing chunk_id {chunk_id!r}")
            rows.append(
                (
                    str(chunk_id or ""),
                    str(docs[i] or ""),
                    _pack_embedding(embs[i]),
                    normalize_retrieval_metadata(metas[i]),
                )
            )
    return rows


def compute_full_rows_digest(rows: Iterable[FullChunkRow]) -> Tuple[str, int]:
    """Canonical digest over the COMPLETE Chroma row universe (dup-reject)."""
    encoded, seen = [], set()
    for row in rows:
        fields = (
            str(row[0] or "").encode("utf-8"),
            str(row[1] or "").encode("utf-8"),
            row[2] if isinstance(row[2], (bytes, bytearray)) else _pack_embedding(row[2]),
            str(row[3] or "").encode("utf-8"),
        )
        if fields[0] in seen:
            raise Fts5MigrationError(f"duplicate chunk_id in full row set: {fields[0]!r}")
        seen.add(fields[0])
        encoded.append(fields)
    encoded.sort(key=lambda fields: fields[0])
    digest = hashlib.sha256(_CHROMA_FULL_ROWS_DOMAIN + struct.pack(">Q", len(encoded)))
    for fields in encoded:
        for field in fields:
            digest.update(struct.pack(">Q", len(field)) + field)
    return digest.hexdigest(), len(encoded)


def capture_chunk_rows(
    collection: Any, batch_size: int = 500
) -> List[ChunkRow]:  # exact id listing + explicit-ID hydration mapped by returned id (B03/D4)
    ids = [str(chunk_id) for chunk_id in (collection.get(include=[]) or {}).get("ids") or []]
    if len(ids) != int(collection.count()) or len(set(ids)) != len(ids):
        raise Fts5MigrationError("snapshot id listing count/duplicate mismatch")
    ids.sort(key=lambda chunk_id: chunk_id.encode("utf-8"))  # canonical global population order (P1-6)
    rows: List[ChunkRow] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        fetched = collection.get(ids=list(batch), include=["documents", "metadatas"]) or {}
        got_ids = [str(rid) for rid in fetched.get("ids") or []]
        docs, metas = fetched.get("documents") or [], fetched.get("metadatas") or []
        position_by_id = {rid: i for i, rid in enumerate(got_ids)}
        if not (len(got_ids) == len(docs) == len(metas) == len(position_by_id)) or not set(got_ids).issubset(batch):
            raise Fts5MigrationError("snapshot response cardinality/identity mismatch")
        for chunk_id in batch:
            i = position_by_id.get(chunk_id)
            if i is None:
                raise Fts5MigrationError(f"snapshot read missing chunk_id {chunk_id!r}")
            rows.append(
                (
                    str(chunk_id or ""),
                    str(docs[i] or ""),
                    str((metas[i] or {}).get("filename") or ""),
                    str((metas[i] or {}).get("category") or ""),
                )
            )
    return rows


def is_credible_v2_marker(
    payload: Optional[dict], live_row_count: int
) -> bool:  # complete schema-v2 marker: strict int generation/counts, matching 64-hex digests
    if not isinstance(payload, dict):
        return False
    source = payload.get("source_rows_sha256")
    return (
        payload.get("schema_version") == MARKER_SCHEMA_VERSION
        and payload.get("status") == "complete"
        and _is_uint(payload.get("generation"), 1)
        and isinstance(source, str)
        and bool(_HEX64.fullmatch(source))
        and source == payload.get("verified_fts_rows_sha256")
        and _is_uint(payload.get("docs_total"))
        and _is_uint(payload.get("docs_indexed"))
        and payload.get("docs_total") == payload.get("docs_indexed") == live_row_count
    )


def read_sealed_fts_row_universe(db_path: Path) -> Tuple[str, int]:
    """Independent read-only recomputation of a sealed FTS5 artifact.

    Opens the database through a SEPARATE read-only SQLite URI connection
    (never a writer, never WAL) and recomputes the canonical row digest and
    row count from the ``fts5_documents`` table. Used by the builder (P1-3:
    sealed-artifact parity check after the write handle is closed) and by
    serving verification — it shares zero state with any live handle.

    Raises ``Fts5CorruptError`` when the file is missing, unopenable, lacks
    the table, or contains duplicate chunk_ids.
    """
    path = Path(db_path)
    if not path.is_file():
        raise Fts5CorruptError(f"sealed FTS5 index missing: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, check_same_thread=False, timeout=5.0, uri=True)
    except sqlite3.DatabaseError as exc:
        raise Fts5CorruptError(f"sealed FTS5 index unopenable: {path}: {exc}") from exc
    try:
        try:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts5_documents'"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise Fts5CorruptError(f"sealed FTS5 index unreadable: {path}: {exc}") from exc
        if table is None:
            raise Fts5CorruptError(f"sealed FTS5 index lacks the fts5_documents table: {path}")
        rows: List[ChunkRow] = []
        last_rowid = 0
        while True:
            page = conn.execute(
                "SELECT rowid, chunk_id, content, filename, category FROM fts5_documents "
                "WHERE rowid > ? ORDER BY rowid LIMIT 400",
                (last_rowid,),
            ).fetchall()
            if not page:
                break
            last_rowid = page[-1][0]
            rows.extend(tuple(row[1:]) for row in page)
        try:
            digest, count = compute_rows_digest(rows)
        except Fts5MigrationError as exc:
            raise Fts5CorruptError(f"sealed FTS5 index has duplicate chunk_ids: {path}: {exc}") from exc
        return digest, count
    finally:
        conn.close()


class Fts5LexicalIndex:
    """SQLite FTS5 wrapper — search, CRUD sync, and the Package-B content-bound
    generation rebuild (schema-v2 markers; the positional resume path is retired)."""

    def __init__(self, db_path: Path, state_path: Path, *, read_only: bool = False) -> None:
        self._db_path = Path(db_path)
        self._state_path = Path(state_path)
        # Phase B (spec C): read_only=True admits a SEALED artifact for serving.
        self._read_only = bool(read_only)
        # Q2 (TechSpec): dedicated RLock, independent of BM25 build lock.
        self._fts5_lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ready: bool = False
        self._generation = 1  # in-place generation coordination (reset never swaps objects)
        self._invalidated, self._active_rebuild_gen, self._rollback_snapshot, self._mutation_epoch = (
            False,
            None,
            None,
            0,
        )
        self._connect_and_configure()
        self._migration_state = Fts5MigrationState(self._state_path)
        payload = self._migration_state.read()
        if isinstance(payload, dict) and isinstance(payload.get("generation"), int):
            self._generation = max(1, payload["generation"])  # D2: reopen restores the counter
        self._ready = False  # promoted only by verified rebuild/publication/restore (P1-1)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def state(self) -> Fts5MigrationState:
        return self._migration_state

    def count(self) -> int:
        """Return the total number of indexed FTS5 rows.

        Used by ``_fts5_marker_matches_reality`` (v4.8.3, GH-issue) to
        cross-check the migration marker against actual on-disk state so a
        stale ``complete`` marker on an empty index doesn't silence the
        fast-path forever.
        """
        if self._conn is None:
            return 0
        with self._fts5_lock:
            try:
                return int(self._conn.execute("SELECT count(*) FROM fts5_documents").fetchone()[0])
            except sqlite3.OperationalError:
                return 0

    def is_ready(self) -> bool:
        with self._fts5_lock:  # flag only; promotion solely via verified rebuild/publication/restore (P1-1/P1-2)
            return self._ready

    def _require_writable(self, op: str) -> None:
        """Refuse every mutation path on a read-only (sealed) artifact.

        The read-only SQLite URI already makes on-disk mutation impossible at
        the connection level; this guard makes the refusal explicit and
        immediate for every writer, marker writer, and rebuild entry point.
        """
        if self._read_only:
            raise Fts5MigrationError(f"FTS5 index is open read-only (sealed generation); {op} is forbidden")

    def _live_digest(
        self,
    ) -> Optional[Tuple[str, int, int]]:  # (digest, distinct, epoch); short lock holds (Locks §4/P1-B)
        rows, last_rowid, epoch = [], 0, None
        while True:
            with self._fts5_lock:
                if self._conn is None or epoch not in (None, self._mutation_epoch):
                    return None  # live mutated between pages (P1-B)
                epoch = self._mutation_epoch
                page = self._conn.execute(
                    "SELECT rowid, chunk_id, content, filename, category FROM fts5_documents WHERE rowid > ? ORDER BY rowid LIMIT 400",
                    (last_rowid,),
                ).fetchall()
            if not page:
                try:
                    digest, distinct = compute_rows_digest(rows)
                except Fts5MigrationError:
                    return None  # duplicate chunk_ids in the live table
                return digest, distinct, epoch
            last_rowid = page[-1][0]
            rows.extend(tuple(row[1:]) for row in page)

    def _live_matches(self, payload: Optional[dict]) -> bool:  # exact live read-back vs marker (P1-5/D3)
        if self._conn is None or not is_credible_v2_marker(payload, self.count()):
            return False
        live = self._live_digest()
        return live is not None and live[:2] == (payload["verified_fts_rows_sha256"], payload["docs_total"])

    def verify_and_publish(
        self, source_digest: str, total: int, generation: int, started_at: Optional[str] = None
    ) -> (
        bool
    ):  # BC-04 exact publisher: live count/distinct/digest must equal the source identity under the current generation
        try:
            live = self._live_digest()  # paged short lock holds — searches stay responsive (Locks §4)
        except Exception as exc:  # F5: read failure fails the candidate generation closed
            self.publish_rebuild_failure(generation, exc)
            raise
        if live is None or live[:2] != (source_digest, total):
            return False
        with self._fts5_lock:
            if (
                self._generation != generation
                or self._conn is None
                or self.count() != total
                or self._mutation_epoch != live[2]
            ):  # P1-B: reject any mutation since the scan
                return False
            now = datetime.now(timezone.utc).isoformat()
            if not self._write_v2_marker(
                generation, "complete", total, total, source_digest, source_digest, started_at or now, now, None
            ):
                return False
            self._ready = True
            self._invalidated = False
            self._drop_table_quiet(f"fts5_documents_backup_g{generation}")  # P1-A: reconcile crash leftovers
            return True

    def search_if_ready(self, query: str, top_k: int = 20) -> Optional[List[Tuple[str, float]]]:
        with self._fts5_lock:  # atomic serving path: readiness flag + search under one lock hold (BC-03)
            return self.search(query, top_k=top_k) if self._ready else None

    def invalidate_generation(self) -> int:
        self._require_writable("invalidate_generation")
        with self._fts5_lock:  # retire the generation (reset); a blocked worker's late writes go stale
            self._rollback_snapshot = (
                prior if is_credible_v2_marker((prior := self._migration_state.read()), self.count()) else None
            )
            self._generation += 1
            self._ready = False
            self._invalidated = True
            with suppress(
                Exception
            ):  # D2 durable; restart stays fail-closed via source compare even if this write fails
                self._write_v2_marker(
                    self._generation,
                    "invalidated",
                    0,
                    0,
                    None,
                    None,
                    datetime.now(timezone.utc).isoformat(),
                    None,
                    None,
                )
            return self._generation

    def begin_rebuild(self) -> Optional[int]:
        self._require_writable("begin_rebuild")
        with self._fts5_lock:  # single-flight admission: None while this generation has a worker
            if self._active_rebuild_gen == self._generation:
                return None
            self._active_rebuild_gen = self._generation
            return self._generation

    def end_rebuild(self, generation: int) -> None:
        with self._fts5_lock:
            if self._active_rebuild_gen == generation:
                self._active_rebuild_gen = None

    def _connect_and_configure(self) -> None:
        """Open the SQLite connection and apply ADR-001 PRAGMAs + schema."""
        try:
            in_memory = str(self._db_path) == ":memory:"
            if self._read_only:
                # Sealed-artifact admission (spec C): open the EXISTING database
                # via a SQLite URI with mode=ro. No mkdir, no WAL pragma, no
                # schema or probe creation, no commit — the connection cannot
                # mutate the file, and every writer is refused besides.
                if in_memory:
                    raise Fts5CorruptError("read_only=True cannot be used with an in-memory database")
                if not self._db_path.is_file():
                    raise Fts5CorruptError(f"FTS5 index missing for read-only open: {self._db_path}")
                uri = self._db_path.resolve().as_uri() + "?mode=ro"
                self._conn = sqlite3.connect(
                    uri,
                    check_same_thread=False,
                    timeout=5.0,
                    uri=True,
                )
                cur = self._conn.cursor()
                cur.execute("PRAGMA busy_timeout=5000")  # connection-local; writes nothing
                cur.close()
                return
            if not in_memory:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=5.0,
            )
            self._apply_pragmas(in_memory=in_memory)
            self._verify_fts5_available()
            # _FTS5_SCHEMA is a hardcoded module constant with the tokenizer
            # literal interpolated at import time from another module constant
            # — zero user input reaches this execute() call, zero injection
            # surface. Semgrep rules flag the f-string source pattern
            # regardless of taint origin, so inline suppression is justified.
            self._conn.execute(_FTS5_SCHEMA)  # nosem
            self._conn.commit()
        except sqlite3.DatabaseError as exc:
            raise Fts5CorruptError(f"Failed to open FTS5 index at {self._db_path}: {exc}") from exc

    def _apply_pragmas(self, *, in_memory: bool) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        if not in_memory:
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    def _verify_fts5_available(self) -> None:
        """Confirm the SQLite build exposes FTS5. Raises ``Fts5CorruptError``."""
        assert self._conn is not None
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
            self._conn.execute("DROP TABLE IF EXISTS _fts5_probe")
        except sqlite3.OperationalError as exc:
            raise Fts5CorruptError(f"SQLite build lacks FTS5 support or schema drift detected: {exc}") from exc

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return ``[(chunk_id, score)]`` sorted by best rank.

        Mirrors ``BM25Index.search`` signature (score is positive; higher is
        better — we invert FTS5's negative bm25 rank). Never raises for
        malformed input: metacharacter tokens are quoted so a well-formed
        MATCH is always issued or an empty list is returned.
        """
        if self._conn is None:
            return []
        if top_k <= 0:
            return []
        escaped = _escape_fts5_query(query)
        if not escaped:
            return []
        sql = (
            "SELECT chunk_id, bm25(fts5_documents) AS rank "
            "FROM fts5_documents WHERE fts5_documents MATCH ? "
            "ORDER BY rank LIMIT ?"
        )
        try:
            with self._fts5_lock:
                cur = self._conn.execute(sql, (escaped, int(top_k)))
                rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Malformed MATCH survived escaping (defensive) — treat as no hits.
            return []
        # FTS5 bm25() returns lower-is-better; invert so callers see the same
        # "higher-is-better" contract BM25Index exposes.
        return [(str(chunk_id), -float(rank)) for chunk_id, rank in rows]

    def close(self) -> None:
        """Release the SQLite connection. Safe to call multiple times."""
        with self._fts5_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def admit_existing(self, expected_digest: str, total: int) -> None:
        """Admit a SEALED FTS5 artifact for read-only serving (spec C).

        Fail-closed validation that writes NOTHING: the ``fts5_documents``
        table exists; the persisted schema-v2 state marker is credible and
        ``complete`` with source == verified == ``expected_digest`` and
        docs_total == ``total`` == live row count; and the canonical live row
        digest recomputed from the database equals ``expected_digest``. Only
        then is the in-memory ready flag promoted — sealed bytes are never
        touched, and no sidecar or marker is created.
        """
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            table = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts5_documents'"
            ).fetchone()
            if table is None:
                raise Fts5CorruptError(f"sealed FTS5 index lacks the fts5_documents table: {self._db_path}")
            payload = self._migration_state.read()
            if not is_credible_v2_marker(payload, self.count()):
                raise Fts5CorruptError(
                    f"sealed FTS5 state marker is not a credible complete schema-v2 marker: {self._state_path}"
                )
            if (
                payload["source_rows_sha256"] != expected_digest
                or payload["verified_fts_rows_sha256"] != expected_digest
            ):
                raise Fts5CorruptError(
                    "sealed FTS5 digests do not match the generation receipt: marker "
                    f"{str(payload['source_rows_sha256'])[:12]}... != expected {expected_digest[:12]}..."
                )
            if payload["docs_total"] != int(total):
                raise Fts5CorruptError(
                    f"sealed FTS5 row count does not match the generation receipt: marker {payload['docs_total']} != expected {total}"
                )
        live = self._live_digest()  # paged read, short lock holds
        if live is None or live[0] != expected_digest or live[1] != int(total):
            got = "unavailable" if live is None else f"{live[0][:12]}.../{live[1]}"
            raise Fts5CorruptError(
                f"sealed FTS5 live row universe does not match the generation receipt: {got} != {expected_digest[:12]}.../{total}"
            )
        with self._fts5_lock:
            self._ready = True  # in-memory promotion only

    def seal_for_publication(self) -> None:
        """Builder-side sealing (spec C): quiesce WAL, drop to DELETE journal, close.

        After this returns the database is a standalone regular file with NO
        ``-wal``/``-shm``/``-journal`` sidecars — the generation store's
        exact-entry sealing rejects them. If a sidecar still exists the seal
        FAILS loudly; sidecars are never silently deleted. The caller must
        have already durably written the schema-v2 ``complete`` state marker
        (``rebuild_content_bound`` does this via ``verify_and_publish``).
        """
        self._require_writable("seal_for_publication")
        if self._conn is None:
            raise Fts5MigrationError("FTS5 connection is closed")
        with self._fts5_lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.execute("PRAGMA journal_mode=DELETE")
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                raise Fts5MigrationError(f"FTS5 seal checkpoint failed: {exc}") from exc
            finally:
                if self._conn is not None:
                    try:
                        self._conn.close()
                    finally:
                        self._conn = None
        leftovers = [suffix for suffix in ("-wal", "-shm", "-journal") if Path(str(self._db_path) + suffix).exists()]
        if leftovers:
            raise Fts5MigrationError(
                f"FTS5 seal failed: sqlite sidecars still present (they are never deleted): {leftovers}"
            )

    # -----------------------------------------------------------------
    # CRUD sync (Task 05, ADR-008). SQL nativo incremental — diverge do
    # BM25 full-rebuild pattern porque FTS5 tem INSERT/DELETE O(1) e o
    # full rebuild custaria segundos por mutation em corpus 3865 docs.
    # Todos os writes acquire ``_fts5_lock`` (RLock, Q2 do TechSpec) e
    # sao serializados pelo WAL SQLite (ADR-001).
    # -----------------------------------------------------------------

    def add_document(self, chunk_id: str, content: str, filename: str, category: str) -> None:
        """Insert one chunk row via ``INSERT`` (ADR-008)."""
        self._require_writable("add_document")
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            self._conn.execute(
                "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                (chunk_id, content, filename, category),
            )
            self._commit_live()

    def remove_document(self, chunk_id: str) -> None:
        """Delete every row matching ``chunk_id`` (ADR-008)."""
        self._require_writable("remove_document")
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            self._conn.execute(
                "DELETE FROM fts5_documents WHERE chunk_id = ?",
                (chunk_id,),
            )
            self._commit_live()

    def update_document(self, chunk_id: str, content: str, filename: str, category: str) -> None:
        """DELETE + INSERT atomico — FTS5 nao tem UPDATE efficient em virtual table."""
        self._require_writable("update_document")
        if self._conn is None:
            raise Fts5CorruptError("FTS5 connection is closed")
        with self._fts5_lock:
            self._conn.execute("DELETE FROM fts5_documents WHERE chunk_id = ?", (chunk_id,))
            self._conn.execute(
                "INSERT INTO fts5_documents (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                (chunk_id, content, filename, category),
            )
            self._commit_live()

    def rebuild_content_bound(
        self, rows: Sequence[ChunkRow], *, generation: Optional[int] = None
    ) -> dict:  # digest -> stage -> read-back -> guarded swap -> publish (T1-T9)
        self._require_writable("rebuild_content_bound")
        if self._conn is None:
            raise Fts5MigrationError("FTS5 connection is closed")
        with self._fts5_lock:
            gen = self._generation if generation is None else int(generation)
            prior = self._rollback_snapshot if self._invalidated else self._migration_state.read()
            if self._generation == gen:
                self._ready = False  # observable quickly (T1); population runs outside the lock
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            source_digest, total = compute_rows_digest(rows)
        except Exception as exc:
            self.publish_rebuild_failure(gen, exc)
            raise
        try:
            if not self._write_v2_marker(gen, "in_progress", total, 0, source_digest, None, started_at, None, None):
                return {"status": "stale", "generation": gen}  # P1-2: delayed generation retreats pre-marker
        except Exception as exc:  # F2/T2: prior credible live marker stays untouched and serving
            self._restore_prior_or_fail(prior, gen, exc, source_digest)
            return {"status": "failed", "generation": gen}
        staging = f"fts5_documents_staging_g{gen}"
        try:
            verified_digest = self._populate_staging(staging, rows, source_digest, total)
        except Exception as exc:
            self.publish_rebuild_failure(gen, exc), self._drop_table_quiet(staging)
            raise
        return self._finalize_rebuild(gen, staging, prior, source_digest, verified_digest, total, started_at)

    def _populate_staging(self, staging: str, rows: Sequence[ChunkRow], source_digest: str, total: int) -> str:
        staging_sql = _quote_sql_identifier(staging)
        with self._fts5_lock:  # stage in short lock-held batches; counts alone never suffice (§9)
            # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
            self._conn.execute(f"DROP TABLE IF EXISTS {staging_sql}")
            self._conn.execute(_FTS5_SCHEMA.replace("fts5_documents", staging_sql, 1))
            self._conn.commit()
        ordered = sorted(
            (tuple(str(value or "") for value in row) for row in rows), key=lambda row: row[0].encode("utf-8")
        )  # canonical order (D4)
        for start in range(0, len(ordered), 100):
            with self._fts5_lock:
                self._conn.executemany(
                    f"INSERT INTO {staging_sql} (chunk_id, content, filename, category) VALUES (?, ?, ?, ?)",
                    ordered[start : start + 100],
                )
                self._conn.commit()
        with self._fts5_lock:
            # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
            read_back = self._conn.execute(
                f"SELECT chunk_id, content, filename, category FROM {staging_sql}"
            ).fetchall()
        verified_digest, verified_count = compute_rows_digest(read_back)
        if verified_digest != source_digest or verified_count != total:
            raise Fts5MigrationError(
                f"staging verification failed: rows {verified_count}/{total}, digest {verified_digest[:12]} != source {source_digest[:12]}"
            )
        return verified_digest

    def _finalize_rebuild(
        self,
        gen: int,
        staging: str,
        prior: Optional[dict],
        source_digest: str,
        verified_digest: str,
        total: int,
        started_at: str,
    ) -> dict:
        backup = f"fts5_documents_backup_g{gen}"
        staging_sql = _quote_sql_identifier(staging)
        backup_sql = _quote_sql_identifier(backup)
        with self._fts5_lock:  # guarded final transition (T6-T8); a stale generation only cleans its staging
            if self._generation != gen or self._conn is None:
                self._drop_table_quiet(staging)
                return {"status": "stale", "generation": gen}
            self._drop_table_quiet(backup)  # P1-A: a stale crash backup must never collide
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
                self._conn.execute(f"ALTER TABLE fts5_documents RENAME TO {backup_sql}")
                # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
                self._conn.execute(f"ALTER TABLE {staging_sql} RENAME TO fts5_documents")
                self._commit_live()
            except sqlite3.DatabaseError as exc:
                with suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                self._restore_prior_or_fail(prior, gen, exc, source_digest)  # B09: rollback left the prior table live
                self._drop_table_quiet(staging)
                raise Fts5MigrationError(f"guarded swap failed: {exc.__class__.__name__}") from exc
        # F1: the post-swap digest/publication runs OUTSIDE the outer lock hold —
        # ready=false searches return promptly; publish/rollback stays generation-guarded.
        try:
            published = self.verify_and_publish(source_digest, total, gen, started_at)
        except Exception as exc:  # P1-1/D3: backup survives publication; restore only after a good reverse swap
            if self._reverse_swap(staging, backup):
                self._restore_prior_or_fail(prior, gen, exc, source_digest)
            else:
                self.publish_rebuild_failure(gen, exc)
            raise Fts5MigrationError(f"complete-marker publication failed: {exc.__class__.__name__}") from exc
        if not published:
            rejected = Fts5MigrationError("post-swap exact verification rejected")
            if self._reverse_swap(staging, backup):
                self._restore_prior_or_fail(prior, gen, rejected, source_digest)
            else:
                self.publish_rebuild_failure(gen, rejected)
            raise rejected
        self._drop_table_quiet(backup)
        return {
            "status": "complete",
            "generation": gen,
            "docs_indexed": total,
            "source_rows_sha256": source_digest,
            "verified_fts_rows_sha256": verified_digest,
        }

    def _reverse_swap(self, staging: str, backup: str) -> bool:  # atomic prior-table restore, reports outcome (D3)
        with self._fts5_lock:
            return self._reverse_swap_locked(staging, backup)

    def _reverse_swap_locked(self, staging: str, backup: str) -> bool:
        staging_sql = _quote_sql_identifier(staging)
        backup_sql = _quote_sql_identifier(backup)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
            self._conn.execute(f"ALTER TABLE fts5_documents RENAME TO {staging_sql}")
            # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
            self._conn.execute(f"ALTER TABLE {backup_sql} RENAME TO fts5_documents")
            self._commit_live()
            self._drop_table_quiet(staging)
            return True
        except sqlite3.DatabaseError:
            with suppress(sqlite3.DatabaseError):
                self._conn.rollback()
            return False

    def _restore_prior_or_fail(
        self, prior: Optional[dict], gen: int, exc: BaseException, current_source: Optional[str]
    ) -> None:
        if self._live_matches(prior) and prior.get("source_rows_sha256") == current_source:  # D3/P1-2
            with self._fts5_lock:
                restored = self._generation == gen  # F1: generation-guarded restore
                if restored:
                    self._ready = True
            if restored:
                with suppress(Exception):  # disk usually already holds prior; rewrite is best-effort
                    self._migration_state.write(prior)
                return
        self.publish_rebuild_failure(gen, exc)

    def publish_rebuild_failure(self, generation: int, exc: BaseException) -> None:
        if self._read_only:  # sealed artifacts are never marked, even on failure
            return
        with self._fts5_lock:  # generation-guarded failed/non-ready publication; stale handlers dropped (T9/B10)
            if self._generation != generation:
                return
            self._ready = False
            with suppress(
                Exception
            ):  # P1-10: sanitized error (class only); a failing writer cannot resurrect readiness
                self._write_v2_marker(
                    generation,
                    "failed",
                    0,
                    0,
                    None,
                    None,
                    datetime.now(timezone.utc).isoformat(),
                    None,
                    exc.__class__.__name__,
                )

    def _write_v2_marker(
        self,
        generation: int,
        status: str,
        docs_total: int,
        docs_indexed: int,
        source_digest: Optional[str],
        verified_digest: Optional[str],
        started_at: Optional[str],
        completed_at: Optional[str],
        error: Optional[str],
    ) -> bool:
        if self._read_only:  # sealed artifacts are never marked
            return False
        with self._fts5_lock:  # P1-2: shared marker writes are generation-checked under the lock
            if self._generation != int(generation):
                return False
            self._migration_state.write(
                {
                    "schema_version": MARKER_SCHEMA_VERSION,
                    "generation": int(generation),
                    "status": status,
                    "docs_total": int(docs_total),
                    "docs_indexed": int(docs_indexed),
                    "started_at": started_at,
                    "source_rows_sha256": source_digest,
                    "verified_fts_rows_sha256": verified_digest,
                    "completed_at": completed_at,
                    "error": error,
                }
            )
            return True

    def start_migration_background(
        self, chunk_iter_factory: Any, docs_total: int, *, resume_from: int = 0, on_progress: Any = None
    ) -> (
        threading.Thread
    ):  # legacy API shim: content-bound rebuild-from-zero; resume_from is call-compat only, never a cursor
        self._require_writable("start_migration_background")

        def _runner() -> None:
            generation = self.begin_rebuild()
            if generation is None:
                return
            try:
                rows = list(chunk_iter_factory())
                if len(rows) != int(docs_total):  # F4: caller-declared total must match, fail closed
                    raise Fts5MigrationError(f"docs_total mismatch: declared {docs_total}, captured {len(rows)}")
                result = self.rebuild_content_bound(rows, generation=generation)
                if on_progress is not None and result.get("status") == "complete":
                    on_progress(result["docs_indexed"], docs_total)
            except Exception as exc:  # noqa: BLE001 — fail closed, generation-guarded (F4/TQ-1)
                self.publish_rebuild_failure(generation, exc)
                print(f"[FTS5] migration failed: {exc.__class__.__name__}")
            finally:
                self.end_rebuild(generation)

        thread = threading.Thread(target=_runner, name="fts5-migration", daemon=True)
        thread.start()
        return thread

    def _commit_live(self) -> None:  # commit + mutation-epoch bump for live-table changes (P1-B)
        self._conn.commit()
        self._mutation_epoch += 1

    def _drop_table_quiet(self, table: str) -> None:  # owner-only staging cleanup (T10)
        table_sql = _quote_sql_identifier(table)
        with suppress(sqlite3.DatabaseError), self._fts5_lock:
            if self._conn is not None:
                # nosemgrep: identifiers are internal and validated by _quote_sql_identifier.
                self._conn.execute(f"DROP TABLE IF EXISTS {table_sql}")
                self._conn.commit()
