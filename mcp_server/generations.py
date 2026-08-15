"""Versioned, crash-safe generation store for the Knowledge RAG system (Phase A, rev 4).

On-disk layout
--------------
    <root>/                          store root (typically ``config.data_dir``)
      .building-<safe-id>/           staging area for a generation being built
                                      (at the ROOT, deliberately outside generations/)
      generations/                   immutable published generations
        <safe-id>/                   one directory per generation
          corpus/                    source corpus snapshot (tree artifact)
          chroma_db/                 ChromaDB persistent directory (tree artifact)
          fts5_index.db              FTS5 index database (file artifact)
          fts5_migration.state       FTS5 migration state (file artifact)
          index_metadata.json        index metadata (file artifact)
          generation.json            receipt — written LAST, hash-pinned by the pointer
      current                        JSON pointer: {"generation_id", "receipt_sha256"}
      generations.lock               stable lock path, deliberately OUTSIDE generations/

Opening
-------
``GenerationStore(root)`` (``create=False``, the production read-only open)
validates the anchors and NEVER creates root/, generations/, current, or the
lock file; missing or invalid anchors are rejected. ``create=True`` is the
explicit builder initialization (creates root/ and generations/). The lock
file is created lazily by the first EXCLUSIVE (builder) acquisition — never
during ``__init__`` and never by a production read: read paths take a SHARED
lock on the EXISTING lock file, or proceed with in-process safety alone when
no lock file exists yet (see ``GenerationStore._read_lock``).

Receipt contract (schema_version 2, exact key sets, no extra fields)
--------------------------------------------------------------------
    status                      must be "complete"
    created_at                  canonical UTC ISO-8601: YYYY-MM-DDTHH:MM:SSZ
    identity                    content-bound freshness receipt: the corpus
                                content manifest SHA (canonical sorted
                                relative-path + file-SHA256 entries of the
                                veterinary subtree — the CONTROLLING corpus
                                identity), plus secret-free retrieval-config
                                digest, code digest, installed-distribution
                                RECORD digest (nullable), dependency-lock
                                digest (nullable), exact local model artifact
                                and model-config digests, and chunking
                                identity. ``config_sha256`` is retained as
                                legacy provenance and never controls
                                readiness.
    compatibility               exact binding object — collection_name,
                                embedding_model, embedding_dimension, query_prefix,
                                passage_prefix, model_artifact_sha256, runtime_version,
                                pooling, chunk_size, chunk_overlap
    artifacts                   one entry per REQUIRED artifact (corpus, chroma_db,
                                fts5_index.db, fts5_migration.state, index_metadata.json)
                                with path / kind / count / sha256
    backends.chroma             collection_name, row_count, unique_id_count,
                                hydrated_id_count, row_digest — ALL THREE counts equal
    backends.fts5               schema_version (exactly 2), status ("complete"),
                                row_count, row_digest, source_digest, verified_digest

Cross-binding: ``identity.model_artifact_sha256`` MUST equal
``compatibility.model_artifact_sha256``, and ``identity.chunking_sha256``
MUST equal the chunking identity derived from
``compatibility.chunk_size``/``chunk_overlap`` — a receipt whose identity
block and compatibility object disagree about the bytes that built it is
rejected.

Backend parity: Chroma and FTS5 must attest the same COMMON logical row
universe — the (chunk_id, document, filename, category) fields BOTH backends
store. Equal row counts AND ``fts5.row_digest == fts5.source_digest ==
fts5.verified_digest == chroma.common_row_digest``. Chroma's complete
``row_digest`` additionally covers embeddings and normalized full retrieval
metadata; each backend's complete digest is independently bound and never
forced equal to the other's. Chroma's ``collection_name`` must equal
``compatibility.collection_name``. Parity is enforced at publish (input
validation AND assembled-receipt validation) and at verification.

Exact-entry policy: a staged building directory must contain EXACTLY the five
required artifact entries; a published generation directory must contain
exactly those five plus ``generation.json`` (present only after sealing). Any
extra entry — sqlite WAL/SHM sidecars (``*-wal``, ``*-shm``, ``*-journal``) or
anything else — is rejected, both at the top level and inside tree artifacts.

Guarantees
----------
* Safe IDs — ``^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`` plus realpath containment;
  collision-resistant IDs via :func:`new_generation_id` (UUID-based). Staging
  names (``.building-*``) can never collide with published IDs.
* Fail-closed symlink policy — symlinked root, generations/, current, lock,
  generation dir, receipt, or artifact (including inside tree artifacts) is
  rejected; non-regular files inside artifact trees are rejected.
* Fail-closed pointer — ``resolve_current()`` runs FULL ``verify_generation``
  under the store lock and returns ``None`` on any failure; it only ever
  returns verified state.
* Durability — the ASSEMBLED receipt (schema, compatibility, backend parity,
  canonical UTC ``created_at``) is validated and its encoded size admitted
  (``<= _MAX_RECEIPT_BYTES``) BEFORE any fsync/rename/CAS step; then every
  artifact file and every containing directory is fsynced, the
  receipt is written LAST, the building dir is atomically renamed
  (same filesystem), and the pointer is compare-and-swap replaced.
  Durability errors raise :class:`DurabilityError` and are never swallowed.
* Uncertain commits — if the pointer was atomically replaced but the follow-up
  fsync of the root fails, :class:`CommitStateUncertainError` is raised with
  ``may_have_committed=True`` and the intended full pointer identity; the
  store never falsely claims the pointer is unchanged.
* Single stable lock — one ``generations.lock`` (flock): builders
  (``begin_build``/``abort_build``/``publish``/``activate``) hold LOCK_EX
  across verification + final rename + CAS; production reads
  (``verify_generation``, ``resolve_current``, ``current_identity``) take
  LOCK_SH on the existing lock file WITHOUT creating it — when the lock file
  does not exist, reads run under in-process serialization only, and
  fail-closed full verification bounds any concurrently starting builder to
  a torn read that resolves to ``None``. Anchors are re-checked under the
  lock. Non-POSIX platforms raise :class:`LockUnsupportedError` — there is
  deliberately no silent in-process fallback for builders.
* ABA/lost-update safety — CAS expected identity binds BOTH generation_id and
  the receipt SHA-256.
* No GC, no symlink creation. ``abort_build`` only ever removes staging.

Security note
-------------
Receipt, pointer, and lock opens use ``O_NOFOLLOW`` plus ``fstat`` regular-file
checks, and generation paths are realpath-contained. Full resistance against a
same-UID attacker swapping directories between ``stat`` and ``open`` is an OS
boundary deliberately deferred to a later hardening phase; this module does
NOT claim that guarantee.

Phase B integration: results carry ``restart_required=True`` (index switches
require a server restart under the current single-reader design) via
``ActivationResult`` / ``ActivationResult.to_dict()`` / ``.summary()``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import shutil
import stat
import struct
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import packaging.markers
import packaging.requirements
import packaging.specifiers
import packaging.version

try:  # POSIX only — no silent in-process fallback when unavailable
    import fcntl
except ImportError:  # pragma: no cover - exotic platforms
    fcntl = None  # type: ignore[assignment]


def _fcntl_or_raise() -> Any:
    """The ``fcntl`` module, or LockUnsupportedError — never ``None``.

    Every call site dereferences the module immediately after this guard,
    which also gives static analysis a non-Optional binding.
    """
    if fcntl is None:
        raise LockUnsupportedError(
            "POSIX file locking (fcntl.flock) is unavailable on this platform; "
            "the generation store refuses to degrade to an in-process lock"
        )
    return fcntl


# ============================================================================
# CONSTANTS
# ============================================================================

GENERATIONS_DIRNAME = "generations"
CURRENT_FILENAME = "current"
LOCK_FILENAME = "generations.lock"
RECEIPT_FILENAME = "generation.json"
BUILDING_PREFIX = ".building-"  # staging lives at the ROOT, outside generations/
POINTER_TMP_PREFIX = ".current."
POINTER_TMP_SUFFIX = ".tmp"

# v3 (this revision): adds the mandatory ``provenance`` block and the full
# content-bound identity set; installed RECORD + dependency lock are REQUIRED
# (never nullable, never fail-open). v2 receipts remain INSPECTABLE (status /
# stats tooling) but can never serve, activate, or roll back — rebuild.
RECEIPT_SCHEMA_VERSION = 3
RECEIPT_SCHEMA_VERSION_INSPECTABLE = (2, 3)
FTS_SCHEMA_VERSION = 2
RECEIPT_STATUS_COMPLETE = "complete"
# Stable reason code for missing/unverifiable dependency evidence (P0 #2).
DEPENDENCY_UNVERIFIABLE = "dependency_unverifiable"

CORPUS_ARTIFACT = "corpus"  # directory (tree) artifact
CHROMA_ARTIFACT = "chroma_db"  # directory (tree) artifact
FTS_ARTIFACT = "fts5_index.db"  # regular-file artifact
FTS_STATE_ARTIFACT = "fts5_migration.state"  # regular-file artifact
METADATA_ARTIFACT = "index_metadata.json"  # regular-file artifact

# name -> (path, kind); drives staging checks, receipt schema, and verification
REQUIRED_ARTIFACTS: Dict[str, Tuple[str, str]] = {
    CORPUS_ARTIFACT: (CORPUS_ARTIFACT, "tree"),
    CHROMA_ARTIFACT: (CHROMA_ARTIFACT, "tree"),
    FTS_ARTIFACT: (FTS_ARTIFACT, "file"),
    FTS_STATE_ARTIFACT: (FTS_STATE_ARTIFACT, "file"),
    METADATA_ARTIFACT: (METADATA_ARTIFACT, "file"),
}
_TREE_KINDS = frozenset({"tree"})
_FILE_KINDS = frozenset({"file"})

_MAX_RECEIPT_BYTES = 1 << 20  # 1 MiB
_MAX_POINTER_BYTES = 64 << 10
_READ_CHUNK = 1 << 20

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
# Wheel RECORD ``sha256=`` entries are URL-safe base64 WITHOUT padding:
# sha256 = 32 bytes -> ceil(32/3)*4 - padding = 43 chars of [A-Za-z0-9_-].
_RECORD_HASH_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
# Repository object names: SHA-1 (40 hex) is the minimum; SHA-256 (64 hex)
# objects are also accepted. Provenance only — never identity.
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_VAULT_HEAD_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_CREATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# sqlite sidecar files are never acceptable inside a sealed generation
_SQLITE_SIDECAR_SUFFIXES = (
    "-wal",
    "-shm",
    "-journal",
    ".wal",
    ".shm",
    ".journal",
)

_POINTER_KEYS = frozenset({"generation_id", "receipt_sha256"})
# Explicit "the pointer must be absent" sentinel for CAS-conflict recovery:
# distinguishable from None ("caller did not capture an expectation").
EXPECTED_CURRENT_ABSENT: str = "__expected_current_absent__"
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "generation_id",
        "created_at",
        "status",
        "provenance",
        "identity",
        "compatibility",
        "artifacts",
        "backends",
    }
)
# provenance: vault Git HEAD (nullable ONLY when the source is provably not a
# Git worktree) — informational, never participates in readiness/identity.
_PROVENANCE_KEYS = frozenset({"vault_head"})
_PROVENANCE_NULLABLE_KEYS = frozenset({"vault_head"})
_IDENTITY_KEYS = frozenset(
    {
        "corpus_manifest_sha256",
        "config_sha256",
        "code_sha256",
        "retrieval_config_sha256",
        "installed_record_sha256",
        "dependency_lock_sha256",
        "model_artifact_sha256",
        "model_config_sha256",
        "chunking_sha256",
    }
)
# identity fields that may legitimately be None: NONE in v3. Installed RECORD
# and dependency-lock digests are REQUIRED evidence for a publishable/
# servable generation (P0 #2) — never nullable, never fail-open.
_NULLABLE_IDENTITY_KEYS: frozenset = frozenset()
_COMPATIBILITY_KEYS = frozenset(
    {
        "collection_name",
        "embedding_model",
        "embedding_dimension",
        "query_prefix",
        "passage_prefix",
        "model_artifact_sha256",
        "runtime_version",
        "pooling",
        "chunk_size",
        "chunk_overlap",
        # Reranker binding (P0 #9/A): the enabled/disabled switch is identity;
        # when enabled, the logical model name and the EXACT artifact tree
        # digest are bound too. A disabled reranker is the explicit state
        # {False, None, None} — never absent keys.
        "reranker_enabled",
        "reranker_model",
        "reranker_artifact_sha256",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "kind", "count", "sha256"})
_CHROMA_EVIDENCE_KEYS = frozenset(
    {
        "collection_name",
        "row_count",
        "unique_id_count",
        "hydrated_id_count",
        "row_digest",
        "common_row_digest",
        "backend_generation_id",
    }
)
_FTS_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "row_count",
        "row_digest",
        "source_digest",
        "verified_digest",
        "backend_generation_id",
    }
)
_TREE_DIGEST_VERSION = b"knowledge-rag-generation-tree-v2\n"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class GenerationError(RuntimeError):
    """Base class for all generation-store failures."""


class UnsafePathError(GenerationError):
    """A path/ID attempted escape, was a symlink, or was not a regular file/dir."""


class PointerError(GenerationError):
    """The ``current`` pointer is missing-but-required, malformed, or unreadable."""


class VerificationError(GenerationError):
    """A generation failed receipt-schema, artifact, digest, parity, or identity checks."""


class PublishError(GenerationError):
    """A publish/build flow problem (missing building dir, already published, ...)."""


class BuildExistsError(PublishError):
    """A building directory or published generation already exists for the ID."""


class CurrentConflictError(PublishError):
    """Compare-and-swap lost: ``current`` did not match the expected identity."""


class DurabilityError(GenerationError):
    """An fsync/open durability operation failed; never swallowed silently."""


class DependencyUnverifiableError(GenerationError):
    """Installed RECORD or dependency-lock evidence is missing/unverifiable.

    Carries the stable reason code :data:`DEPENDENCY_UNVERIFIABLE`; builders
    and versioned startup MUST fail closed on it — never fabricate a lock,
    never bless a ranged ``requirements.txt``.
    """

    reason_code = DEPENDENCY_UNVERIFIABLE


class CommitStateUncertainError(PublishError):
    """The pointer was atomically replaced, but its durability fsync failed.

    ``may_have_committed`` is always ``True``: once the atomic replace has
    happened, the on-disk pointer MAY name the new generation. Callers must
    re-read ``current`` (e.g. via ``resolve_current``) instead of assuming the
    old pointer survived. Carries the intended full pointer identity.
    """

    def __init__(self, message: str, *, generation_id: str, receipt_sha256: str) -> None:
        super().__init__(message)
        self.generation_id = generation_id
        self.receipt_sha256 = receipt_sha256
        self.may_have_committed = True

    def pointer_identity(self) -> Dict[str, str]:
        """The full pointer identity this commit intended to install."""
        return {"generation_id": self.generation_id, "receipt_sha256": self.receipt_sha256}


class LockUnsupportedError(GenerationError):
    """POSIX file locking is unavailable; the store refuses to degrade silently."""


# ============================================================================
# RESULT OBJECTS
# ============================================================================


@dataclass
class ActivationResult:
    """Outcome of publish/activate/rollback, shaped for Phase B CLI reporting."""

    generation_id: str
    receipt_sha256: str
    receipt: Dict[str, Any]
    restart_required: bool = True  # index switches require a server restart today

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "receipt_sha256": self.receipt_sha256,
            "restart_required": self.restart_required,
            "receipt": self.receipt,
        }

    def summary(self) -> str:
        return (
            f"generation {self.generation_id} activated "
            f"(receipt_sha256={self.receipt_sha256[:12]}..., restart_required={str(self.restart_required).lower()})"
        )


class CurrentGeneration:
    """A fully verified view of the generation named by ``current``."""

    __slots__ = ("generation_id", "receipt_sha256", "generation_dir", "receipt")

    def __init__(self, generation_id: str, receipt_sha256: str, generation_dir: Path, receipt: Dict[str, Any]):
        self.generation_id = generation_id
        self.receipt_sha256 = receipt_sha256
        self.generation_dir = generation_dir
        self.receipt = receipt

    def identity(self) -> Dict[str, str]:
        """The CAS identity of this generation (id + receipt SHA)."""
        return {"generation_id": self.generation_id, "receipt_sha256": self.receipt_sha256}

    def compatibility(self) -> Dict[str, Any]:
        """The exact compatibility binding this generation was published with."""
        return self.receipt["compatibility"]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CurrentGeneration({self.generation_id!r}, receipt_sha256={self.receipt_sha256[:12]}...)"


# ============================================================================
# LOW-LEVEL HELPERS (module level so tests can observe/patch the choke points)
# ============================================================================


def new_generation_id() -> str:
    """Collision-resistant generation ID (UUID4-based; matches the safe-ID grammar)."""
    return "gen-" + uuid.uuid4().hex


def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomically replace ``dst`` with ``src`` (same filesystem requirement).

    Single choke point for crash-injection tests: both the building->final
    rename and the pointer swap go through here.
    """
    os.replace(src, dst)


def _fsync_file(path: Path) -> None:
    """fsync a regular file; durability errors raise, never pass silently."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise DurabilityError(f"fsync: cannot open {path}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise DurabilityError(f"fsync failed for file {path}: {exc}") from exc
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """fsync a directory so renames/creates inside it survive a crash.

    Durability errors (including platforms that cannot open directories) raise
    :class:`DurabilityError` — they are never silently swallowed.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise DurabilityError(f"fsync: cannot open directory {path}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise DurabilityError(f"fsync failed for directory {path}: {exc}") from exc
    finally:
        os.close(fd)


def _fsync_tree(base: Path) -> None:
    """fsync every regular file and every directory in ``base`` (bottom-up).

    Symlinks and non-regular entries (FIFOs, sockets, devices) are rejected —
    only real bytes this process wrote are acceptable in a generation.
    """
    for root, dirs, names in os.walk(base, topdown=False, followlinks=False):
        root_path = Path(root)
        for d in dirs:
            p = root_path / d
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise UnsafePathError(f"symlinked directory inside artifact tree: {p}")
            if not stat.S_ISDIR(st.st_mode):
                raise UnsafePathError(f"non-directory entry inside artifact tree: {p}")
        for n in names:
            p = root_path / n
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise UnsafePathError(f"symlinked file inside artifact tree: {p}")
            if not stat.S_ISREG(st.st_mode):
                raise UnsafePathError(f"non-regular file inside artifact tree: {p}")
            _fsync_file(p)
        _fsync_dir(root_path)


def _write_file_exclusive(path: Path, payload: bytes) -> None:
    """Create ``path`` with ``payload`` (O_EXCL), flush, and fsync the file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:
            raise DurabilityError(f"fsync failed for {path}: {exc}") from exc


def _read_regular_file(path: Path, max_bytes: int) -> bytes:
    """O_NOFOLLOW + fstat hardened read of a regular file (see security note).

    ``OSError`` propagates to the caller (``ELOOP`` for a symlink swapped in
    racily, ``ENOENT`` for missing); non-regular files raise
    :class:`UnsafePathError`. Reads at most ``max_bytes + 1`` bytes so callers
    can detect oversize payloads without trusting ``st_size`` alone.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsafePathError(f"not a regular file: {path}")
        chunks = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(_READ_CHUNK, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a regular file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_digest(base: Path) -> Tuple[str, int]:
    """Deterministic SHA-256 over a directory tree plus its regular-file count.

    Symlinks, non-regular files, and sqlite sidecar files (WAL/SHM/journal)
    anywhere inside the tree raise :class:`UnsafePathError`. Digest input:
    version tag, then per file (sorted by relative POSIX path):
    ``<rel-path>\\0<file-sha256-hex>\\0<size>\\n``.
    """
    entries: list[Tuple[str, Path]] = []
    for root, dirs, names in os.walk(base, followlinks=False):
        root_path = Path(root)
        for d in dirs:
            p = root_path / d
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise UnsafePathError(f"symlinked directory inside artifact tree: {p}")
            if not stat.S_ISDIR(st.st_mode):
                raise UnsafePathError(f"non-directory entry inside artifact tree: {p}")
        for n in names:
            p = root_path / n
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise UnsafePathError(f"symlinked file inside artifact tree: {p}")
            if not stat.S_ISREG(st.st_mode):
                raise UnsafePathError(f"non-regular file inside artifact tree: {p}")
            if n.lower().endswith(_SQLITE_SIDECAR_SUFFIXES):
                raise UnsafePathError(f"sqlite sidecar file rejected inside tree artifact: {p}")
            entries.append((p.relative_to(base).as_posix(), p))
    entries.sort(key=lambda item: item[0])

    h = hashlib.sha256()
    h.update(_TREE_DIGEST_VERSION)
    for rel, p in entries:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(p).encode("ascii"))
        h.update(b"\0")
        h.update(str(p.stat().st_size).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest(), len(entries)


def digest_tree(path: Path) -> Tuple[str, int]:
    """Public wrapper over the canonical tree digest (see :func:`_tree_digest`).

    Sole public entry point so Phase-B callers (model-artifact digest, corpus
    manifests) and the store share ONE tree algorithm; do not duplicate it.
    Symlinks, non-regular files, and sqlite sidecars anywhere inside the tree
    fail closed exactly as the store's own verification does.
    """
    return _tree_digest(Path(path))


# ============================================================================
# Content-bound identity material (P0). ONE canonical corpus-manifest
# algorithm, shared by the builder, the receipt, and serving verification:
# SHA-256 over the domain tag, the entry count, then (sorted by relative
# POSIX path) `<rel-path> \\0 <file-sha256-hex> \\n` per file. ONLY the
# veterinary-subtree entries of each generation's corpus artifact tree.
# ============================================================================

CORPUS_MANIFEST_DOMAIN = b"knowledge-rag.generation.corpus-manifest.v1\x00"

# Never admitted into a corpus manifest regardless of configured formats:
# VCS internals and OS junk are not veterinary content.
_CORPUS_EXCLUDED_DIRNAMES = frozenset({".git"})
_CORPUS_EXCLUDED_FILENAMES = frozenset({".DS_Store", "Thumbs.db", ".gitignore", ".gitattributes"})


def _ingestion_exclusion_matcher(exclude_patterns: Optional[List[str]]) -> Any:
    """Reuse ingestion's OWN exclusion matcher so builder and ingestion admit
    the SAME file set (P0 #3: "one selector shared with actual ingestion")."""
    from fnmatch import fnmatch

    patterns = [p for p in (exclude_patterns or []) if isinstance(p, str)]

    def _match(rel_posix: str, parts: Tuple[str, ...]) -> bool:
        for pattern in patterns:
            if fnmatch(rel_posix, pattern):
                return True
            for part in parts:
                if fnmatch(part, pattern):
                    return True
        return False

    return _match


def corpus_file_entries(
    root: Path,
    *,
    supported_suffixes: Optional[Set[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, Path]]:
    """The ONE shared corpus file selector (builder, receipt, freshness).

    Selection rules — identical to ingestion's ``parse_directory`` admission
    plus the strict snapshot policy:

    * directories named ``.git`` are never descended into;
    * a file is admitted when its lowercase suffix is in
      ``supported_suffixes`` AND it is NOT matched by the configured
      ``exclude_patterns`` (fnmatch on the relative POSIX path and on each
      individual component — the exact semantics of
      ``ingestion.DocumentParser._should_exclude``) AND its name is not an
      excluded OS/VCS junk name;
    * symlinks and special files ANYWHERE under the root fail closed
      (:class:`UnsafePathError`) — reject, never skip: a silently-skipped
      symlink could hide live drift.

    Returns sorted ``(rel_posix_path, absolute_path)`` pairs.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise UnsafePathError(f"corpus root is not a real directory: {root}")
    matcher = _ingestion_exclusion_matcher(exclude_patterns)
    suffixes = {s.lower() for s in (supported_suffixes or set())}
    out: List[Tuple[str, Path]] = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part in _CORPUS_EXCLUDED_DIRNAMES for part in rel.parts[:-1]) or (
            rel.parts and rel.parts[-1] in _CORPUS_EXCLUDED_DIRNAMES
        ):
            continue
        st = p.lstat()
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode):
            raise UnsafePathError(f"non-regular entry in corpus tree (symlinks/special files are rejected): {p}")
        name = rel.parts[-1]
        if name in _CORPUS_EXCLUDED_FILENAMES:
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        rel_posix = rel.as_posix()
        if matcher(rel_posix, rel.parts):
            continue
        out.append((rel_posix, p))
    out.sort(key=lambda item: item[0])
    return out


def corpus_manifest_entries(
    root: Path,
    *,
    supported_suffixes: Optional[Set[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Sorted (rel_posix_path, sha256) manifest entries for a corpus tree.

    Same admission as :func:`corpus_file_entries`; each admitted file's
    content SHA-256 is computed over its bytes (no size/mtime/inode, no
    absolute path, no Git HEAD — content identity only, P0 #3).
    """
    return [
        (rel, _sha256_file(p))
        for rel, p in corpus_file_entries(
            root, supported_suffixes=supported_suffixes, exclude_patterns=exclude_patterns
        )
    ]


def corpus_manifest_digest(entries: List[Tuple[str, str]]) -> str:
    """Canonical corpus-manifest SHA-256 over sorted ``corpus_manifest_entries``."""
    ordered = sorted(entries, key=lambda item: item[0])
    seen: Set[str] = set()
    h = hashlib.sha256(CORPUS_MANIFEST_DOMAIN + struct.pack(">Q", len(ordered)))
    for rel, sha in ordered:
        if rel in seen:
            raise VerificationError(f"duplicate corpus manifest entry: {rel}")
        seen.add(rel)
        if not _HEX64_RE.match(sha):
            raise VerificationError(f"corpus manifest entry {rel} has a malformed sha256: {sha!r}")
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def corpus_manifest(
    root: Path,
    *,
    supported_suffixes: Optional[Set[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """(digest, entry_count) of a corpus tree under the canonical algorithm."""
    entries = corpus_manifest_entries(root, supported_suffixes=supported_suffixes, exclude_patterns=exclude_patterns)
    return corpus_manifest_digest(entries), len(entries)


def _stable_json_digest(payload: Any) -> str:
    """Deterministic SHA-256 of a secret-free JSON payload (sort_keys)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def vault_head_or_none(source_dir: Optional[Path] = None) -> Optional[str]:
    """Vault Git HEAD of the live source tree — PROVENANCE ONLY.

    Returns the 40-hex commit id when ``source_dir`` (default: the configured
    source documents dir) sits inside a Git worktree, else ``None`` (provably
    not a worktree). Never raises for ordinary absence; never follows a
    ``.git`` symlink (a symlinked .git is treated as non-worktree).
    """
    import subprocess  # stdlib; local to keep module import surface minimal

    base = Path(source_dir) if source_dir is not None else None
    if base is None:
        try:
            from mcp_server.config import config as _cfg

            base = Path(_cfg.source_documents_dir or _cfg.documents_dir)
        except Exception:
            return None
    if not base.is_dir():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None  # not a worktree (or no commits) — provably no HEAD
    head = (proc.stdout or "").strip().lower()
    return head if _VAULT_HEAD_RE.match(head) else None


# ============================================================================
# Retrieval-config identity (P0 #4): FRESH read + EXPLICIT canonical allowlist.
#
# One canonical, exhaustive projection of every effective retrieval/index
# field the running code actually uses. Anything NOT in this projection never
# enters the digest — auth/server/paths/logging/metrics drift stays GREEN —
# and every retrieval-changing field listed here drifts RED.
# ============================================================================

# YAML key paths (dot-separated) that ARE retrieval config. Exact, explicit,
# reviewed against config.py's _get/_get_nested readers and server.py usage.
_RETRIEVAL_YAML_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("documents", "supported_formats"),
    ("documents", "exclude_patterns"),
    ("documents", "chunking", "chunk_size"),
    ("documents", "chunking", "chunk_overlap"),
    ("models", "embedding", "model"),
    ("models", "embedding", "dimensions"),
    ("models", "embedding", "profile"),
    ("models", "embedding", "artifact_path"),
    ("models", "embedding", "runtime_version"),
    ("models", "embedding", "pooling"),
    ("models", "embedding", "gpu"),
    ("models", "reranker", "enabled"),
    ("models", "reranker", "model"),
    ("models", "reranker", "top_k_multiplier"),
    ("models", "reranker", "local_artifact"),
    ("search", "default_results"),
    ("search", "max_results"),
    ("search", "collection_name"),
    ("search", "hybrid_alpha"),
    ("search", "similarity_threshold"),
    ("search", "lexical_fast_path", "enabled"),
    ("search", "lexical_fast_path", "min_hits"),
    ("search", "lexical_fast_path", "rerank_enabled"),
    ("search", "lexical_fast_path", "patterns"),
    ("search", "min_score"),
    ("search", "top_k"),
    ("search", "fts5_min_hits"),
    ("search", "fts5_rerank_enabled"),
    ("search", "fts5_patterns"),
    # TOP-LEVEL YAML maps (config.py reads them via _get_top, NOT search.*):
    # routing/category/expansion semantics live at the document root.
    ("keyword_routes",),
    ("category_mappings",),
    ("query_expansions",),
    ("query_expansion_groups",),
)

# Effective config attributes (resolved values the running code uses). Path
# values contribute ONLY their final name component — never the absolute
# private prefix. ``models_cache_dir`` enters as the pinned artifact digest,
# not as a path.
RETRIEVAL_CONFIG_FIELDS: Tuple[str, ...] = (
    "index_mode",
    "collection_name",
    "embedding_model",
    "embedding_dim",
    "embedding_profile",
    "query_prefix",
    "passage_prefix",
    "chunk_size",
    "chunk_overlap",
    "supported_formats",
    "exclude_patterns",
    "default_results",
    "max_results",
    "reranker_enabled",
    "reranker_model",
    "reranker_top_k_multiplier",
    "reranker_local_artifact",
    "fts5_enabled",
    "fts5_min_hits",
    "fts5_patterns",
    "fts5_rerank_enabled",
    "keyword_routes",
    "category_mappings",
    "query_expansions",
    "query_expansion_groups",
    "embedding_runtime_version",
    "embedding_pooling",
    "hybrid_alpha",
    "similarity_threshold",
    "min_score",
    "top_k",
)


def _normalize_retrieval_value(value: Any) -> Any:
    """Deterministic normalization for one projected value.

    * paths -> final name component (secrets/private prefixes never enter)
    * lists -> ORDER-PRESERVED element-wise normalization: routing tables,
      first-match pattern lists and expansion groups are behaviorally
      ordered, so a reorder is drift and must change the digest. Only
      explicit dict keys are sorted (JSON-canonical).
    * dicts -> recursively key-sorted with normalized values
    * str/int/float/bool/None pass through
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_retrieval_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_retrieval_value(v) for v in value]
    return str(type(value).__name__)


# YAML keys whose value is a filesystem path: contribute ONLY the final name
# component, never the private absolute prefix (identity must be portable and
# secret-free).
_RETRIEVAL_PATH_TAIL_YAML_KEYS = frozenset(
    {
        ("models", "embedding", "artifact_path"),
        ("models", "reranker", "local_artifact"),
    }
)


def _project_retrieval_yaml(yaml_data: Any) -> Dict[str, Any]:
    """EXPLICIT canonical projection of a freshly-read config mapping.

    Walks the fixed ``_RETRIEVAL_YAML_KEYS`` paths only. Unknown keys —
    including every auth/server/path/logging/metrics key — are dropped
    entirely: they cannot leak a secret and cannot drift retrieval identity.
    A missing section/key contributes ``None`` (present-in-identity), so
    REMOVING a retrieval key is also drift. Path-valued keys contribute only
    their final name component.
    """
    out: Dict[str, Any] = {}
    if not isinstance(yaml_data, dict):
        return out
    for path in _RETRIEVAL_YAML_KEYS:
        node: Any = yaml_data
        missing = False
        for part in path:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                missing = True
                break
        if missing:
            out[".".join(path)] = None
        elif path in _RETRIEVAL_PATH_TAIL_YAML_KEYS and isinstance(node, str) and node:
            out[".".join(path)] = Path(node.replace("\\", "/")).name
        else:
            out[".".join(path)] = _normalize_retrieval_value(node)
    return out


def retrieval_config_digest(cfg: Any) -> str:
    """Secret-free effective retrieval-config digest, FRESHLY READ (P0 #4).

    Two allowlisted inputs, both explicit:

    1. A FRESH re-read of ``config.yaml`` (via config's own loader
       ``_load_yaml_config``) projected through the canonical key allowlist —
       query expansions/groups, keyword routes/category mappings, FTS5
       toggle/parameters, reranker switch + top-k multiplier, supported
       formats/exclusions, chunking/overlap, limits, model switches and
       versions. A loader failure is NEVER swallowed: fresh-unreadable is
       drift/unverifiable (raises :class:`DependencyUnverifiableError`).
    2. The EFFECTIVE derived attributes on the live config singleton
       (``RETRIEVAL_CONFIG_FIELDS``) — the resolved values the running code
       actually uses, independent of YAML layout.

    Path-ish values contribute only their final name component; auth values
    never enter; a HEAD-only vault change never enters (provenance is not
    identity).
    """
    fresh_yaml: Dict[str, Any]
    try:
        from mcp_server import config as _config_module

        raw_yaml = _config_module._load_yaml_config()
    except Exception as exc:
        raise DependencyUnverifiableError(
            f"{DEPENDENCY_UNVERIFIABLE}: config.yaml could not be freshly read for "
            f"retrieval identity: {exc.__class__.__name__}"
        ) from exc
    fresh_yaml = _project_retrieval_yaml(raw_yaml)

    effective: Dict[str, Any] = {}
    for name in RETRIEVAL_CONFIG_FIELDS:
        value = getattr(cfg, name, None)
        if isinstance(value, Path):
            value = value.name
        elif isinstance(value, str) and name.endswith(("_path", "_dir", "_artifact", "_artifact_path")):
            value = Path(value).name if value else None
        effective[name] = _normalize_retrieval_value(value)

    compat: Dict[str, Any] = {}
    try:
        if getattr(cfg, "index_mode", "") == "versioned" and hasattr(cfg, "generation_compatibility"):
            compat = cfg.generation_compatibility() or {}
    except Exception:
        compat = {}
    effective["model_artifact_sha256"] = compat.get("model_artifact_sha256")
    effective["embedding_dimension"] = compat.get("embedding_dimension", getattr(cfg, "embedding_dim", None))

    # Reranker binding (P0 #9): the ENABLED/DISABLED switch is identity, and
    # when enabled the EXACT artifact bytes are identity. An enabled reranker
    # with a missing/unreadable artifact yields the "unavailable" sentinel —
    # a digest that cannot equal the receipt's, so serving fails closed
    # instead of silently degrading to RRF order.
    reranker_enabled = bool(effective.get("reranker_enabled"))
    reranker_artifact = getattr(cfg, "reranker_local_artifact", None)
    if getattr(cfg, "index_mode", "") == "versioned" and reranker_enabled:
        if reranker_artifact:
            try:
                effective["reranker_artifact_sha256"] = digest_tree(Path(str(reranker_artifact)))[0]
            except Exception:
                effective["reranker_artifact_sha256"] = "unavailable"
        else:
            effective["reranker_artifact_sha256"] = "unavailable"
    else:
        effective["reranker_artifact_sha256"] = None if not reranker_enabled else "unconfigured"

    return _stable_json_digest({"yaml": fresh_yaml, "effective": effective})


def installed_record_evidence(distribution_name: str = "knowledge-rag") -> Tuple[str, str, int]:
    """Strict three-column RECORD evidence: (digest, dist_version, rows).

    Parses the installed RECORD as exact three-column CSV (path, hash, size)
    and canonicalizes path+hash+size:

    - hashed rows: sha256 only; the target must be a regular non-symlink
      file; the decoded hash AND the declared decimal size must both match
      the on-disk bytes;
    - hashless rows are admitted ONLY for (a) the distribution's unique own
      ``.dist-info/RECORD`` row and (b) installer-generated ``__pycache__``
      ``.pyc`` files — a hashless ``.pyc`` must exist, and its freshly
      computed sha256+size are folded into the evidence so mutation changes
      the digest and deletion fails;
    - hashless METADATA / entry_points / package source / arbitrary
      dist-info or egg-info rows, and multiple self-RECORD candidates, fail
      closed.

    Raises :class:`DependencyUnverifiableError` when the distribution is
    not installed (a source checkout is NOT admitted evidence) or any check
    fails. Never returns None: absence fails closed.
    """
    import csv as _csv

    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyUnverifiableError(
            f"{DEPENDENCY_UNVERIFIABLE}: distribution {distribution_name!r} is not installed "
            "(no import/RECORD evidence; a source checkout is not admitted)"
        ) from exc
    dist_version = str(getattr(dist, "version", "") or "")
    # PUBLIC API surface only (P0 #5): read_text('RECORD') + locate_file().
    raw = dist.read_text("RECORD")
    if raw is None:
        raise DependencyUnverifiableError(
            f"{DEPENDENCY_UNVERIFIABLE}: installed RECORD missing for {distribution_name!r} "
            "(distribution metadata exposes no RECORD)"
        )
    canonical_rows: List[str] = []
    checked = 0
    self_record_rows = 0
    for row in _csv.reader(raw.splitlines()):
        if not row or not "".join(row).strip():
            continue
        if len(row) != 3:
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD row is not exact three-column CSV: {row!r}"
            )
        path_str, hash_field, size_field = (part.strip() for part in row)
        path_str = path_str.strip('"')
        normalized = path_str.replace("\\", "/")
        if not hash_field:
            # Hashless exemptions: the distribution's OWN RECORD row, and
            # installer-generated .pyc files. Everything else fails closed.
            if normalized.endswith(".dist-info/RECORD"):
                self_record_rows += 1
                if self_record_rows > 1:
                    raise DependencyUnverifiableError(
                        f"{DEPENDENCY_UNVERIFIABLE}: RECORD lists multiple self-RECORD candidates"
                    )
                canonical_rows.append(f"{path_str}\x1fSELF-RECORD")
                continue
            if normalized.endswith(".pyc") and "__pycache__" in normalized:
                target = Path(str(dist.locate_file(path_str)))
                if target.is_symlink() or not target.is_file():
                    raise DependencyUnverifiableError(
                        f"{DEPENDENCY_UNVERIFIABLE}: hashless .pyc listed in RECORD is "
                        f"missing/non-regular: {path_str!r}"
                    )
                canonical_rows.append(f"{path_str}\x1fsha256={_sha256_b64url(target)}\x1f{target.stat().st_size}")
                continue
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD row without a hash is not an admitted exemption: {path_str!r}"
            )
        algo, _, expected = hash_field.partition("=")
        if algo != "sha256" or not _RECORD_HASH_RE.match(expected):
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD row has a malformed hash: {path_str!r}"
            )
        if not size_field.isdigit():
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD row has a missing/non-decimal size: {path_str!r}"
            )
        target = Path(str(dist.locate_file(path_str)))
        if target.is_symlink() or not target.is_file():
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD lists a missing/non-regular file: {path_str!r}"
            )
        if target.stat().st_size != int(size_field):
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD size mismatch for {path_str!r} — "
                f"declared {size_field}, on disk {target.stat().st_size}"
            )
        if _sha256_b64url(target) != expected:
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: RECORD hash mismatch for {path_str!r} — installed bytes drifted"
            )
        checked += 1
        canonical_rows.append(f"{path_str}\x1f{hash_field}\x1f{size_field}")
    if checked < 1:
        raise DependencyUnverifiableError(f"{DEPENDENCY_UNVERIFIABLE}: RECORD contains no verified hash-bearing rows")
    digest = hashlib.sha256(
        (f"installed-record:{distribution_name}:{dist_version}\x00" + "\n".join(sorted(canonical_rows))).encode("utf-8")
    ).hexdigest()
    return digest, dist_version, checked


def _sha256_b64url(path: Path) -> str:
    """RECORD-style sha256=<base64url-no-padding> of a file's bytes."""
    import base64

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_READ_CHUNK), b""):
            h.update(chunk)
    return base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")


# The ONE exact expected lock filename (P0/B): operators generate it with
# ``pip-compile --generate-hashes requirements.in > requirements.lock`` (or
# ``uv pip compile --generate-hashes``). A bare requirements.txt is NEVER
# admitted as lock evidence in versioned mode.
DEPENDENCY_LOCK_FILENAME = "requirements.lock"

# Installed distributions whose RECORD evidence is aggregated into
# ``identity.installed_record_sha256`` — the app itself plus the
# LOAD-BEARING retrieval runtimes whose byte drift would silently change
# embedding/rerank/retrieval behavior: the embedding/rerank runtime, the
# ONNX executor, the vector store, the MCP framework, the PEP 508 parsing
# library the strict lock/parity machinery itself depends on, and NumPy
# (vector ops, argpartition top-k).
#
# BOUNDED SCOPE (deliberate): this aggregate does NOT RECORD-hash every
# transitive of the ~105-package graph on every access. The user contract
# is: strict RECORD evidence for these load-bearing distributions PLUS the
# exact version/hash lock (requirements.lock) covering ALL transitives
# via the marker-aware active-graph parity check, PLUS per-access drift
# blocking for corpus/model/config/index artifacts. Do not claim all
# active dependency bytes are RECORD-hashed. ``tokenizers`` participates
# only when genuinely installed.
RECORD_EVIDENCE_DISTRIBUTIONS: Tuple[str, ...] = (
    "knowledge-rag",
    "fastembed",
    "onnxruntime",
    "chromadb",
    "mcp",
    "packaging",
    "numpy",
)


def installed_runtime_record_evidence(
    distribution_names: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, str], int]:
    """Aggregate strict RECORD evidence across the runtime distributions.

    Returns ``(aggregate_digest, {distribution: version}, total_checked)``.
    Every listed distribution must be installed with a verifiable RECORD
    (re-hashed file bytes, base64url sha256); any missing/unverifiable
    distribution raises :class:`DependencyUnverifiableError` — the aggregate
    never fail-opens. ``tokenizers`` is included automatically when present.
    The lock alone does NOT prove installed bytes; this does.
    """
    names = list(distribution_names if distribution_names is not None else RECORD_EVIDENCE_DISTRIBUTIONS)
    try:
        importlib.metadata.distribution("tokenizers")  # probe only
        if "tokenizers" not in names:
            names.append("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        pass
    per_dist: List[Tuple[str, str, str]] = []  # (name, version, digest)
    versions: Dict[str, str] = {}
    total = 0
    for name in names:
        digest, version, checked = installed_record_evidence(name)
        per_dist.append((name, version, digest))
        versions[name] = version
        total += checked
    aggregate = _stable_json_digest({"distributions": sorted(f"{n}@{v}:{d}" for n, v, d in per_dist)})
    return aggregate, versions, total


def _lock_environment(environment: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Marker-evaluation environment: PEP 508 defaults + ``extra=""``.

    ``extra`` defaults to the EMPTY string everywhere in this module: the
    production runtime lock is generated with ``--strip-extras`` and
    installed without extras, so only platform markers (and explicit
    injections for tests) can activate a lock variant.
    """
    env = dict(packaging.markers.default_environment())
    env["extra"] = ""
    if environment:
        env.update(environment)
    return env


def _fail_lock(detail: str, source_name: str = DEPENDENCY_LOCK_FILENAME) -> DependencyUnverifiableError:
    return DependencyUnverifiableError(f"{DEPENDENCY_UNVERIFIABLE}: {source_name}: {detail}")


def _parse_strict_lock(raw: str, source_name: str = DEPENDENCY_LOCK_FILENAME) -> List[Dict[str, Any]]:
    """STRICT physical + logical lock parser (v4.9.0 review revision).

    Physical layer — continuation state is tracked exactly:
    - a requirement line ending in ``\\`` puts the parser in continuation
      mode; the NEXT physical line may then be ONLY a single
      ``--hash=sha256:<64hex>`` token (itself optionally backslash-continued)
      or a comment/blank line;
    - a ``--hash=`` token after a NON-continued requirement, any unexpected
      token while in continuation mode, and EOF with a pending continuation
      all fail closed;
    - ``-r``/``--requirement`` includes, editable, VCS and direct-URL
      requirements fail closed; bare ``--option`` lines are metadata.

    Logical layer — each assembled entry is parsed with
    ``packaging.requirements.Requirement`` and must be an exact
    ``name==version`` pin: exactly one specifier clause, operator ``==``, no
    ``..*`` wildcard, no extras, no URL, a valid PEP 440 version, and at
    least one valid sha256 hash. A PEP 508 environment marker is VALID and
    preserved (platform variants).

    Returns the list of validated physical variants:
    ``{"name", "version", "marker", "hashes", "text"}``.
    """
    entries: List[Dict[str, Any]] = []
    pending = False  # previous physical line ended with a backslash
    current: Optional[Dict[str, Any]] = None

    for lineno, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            # Comments/blanks are legal inside a continuation block.
            continue
        is_hash_line = line.split()[0].startswith("--hash=")
        if pending and not is_hash_line:
            raise _fail_lock(
                f"line {lineno}: unexpected continuation token after a "
                f"backslash-continued requirement (only --hash=sha256:<64hex> "
                f"continuations are admitted): {line.split()[0]!r}"
            )
        if not pending and is_hash_line:
            raise _fail_lock(
                f"line {lineno}: standalone --hash= continuation after a "
                "non-continued requirement (requirement line must end with \\)"
            )
        # Includes: a lock must be self-contained. Both whitespace forms
        # ("-r file", "--requirement file") AND compact forms ("-rfile",
        # "--requirement=file") are rejected.
        if (
            line.startswith("-r ")
            or line.startswith("--requirement ")
            or line in ("-r", "--requirement")
            or (line.startswith("-r") and len(line) > 2 and not line[2].isspace() and line[2] not in ("=",))
            or line.startswith("--requirement=")
        ):
            raise _fail_lock("lock uses -r/--requirement includes — locks must be self-contained")
        # Editable / VCS / direct-URL requirements are never admitted.
        if line.startswith("-e ") or line.startswith("--editable "):
            raise _fail_lock(f"lock contains an editable requirement: {line!r}")
        if line.split("==")[0].strip().lower().startswith(("git+", "hg+", "svn+", "bzr+")) or " @ " in line:
            raise _fail_lock(f"lock contains a VCS/direct-URL requirement: {line.split(' ')[0]!r}")
        # Bare option lines are lock metadata, not pins — but NEVER while a
        # continuation is pending (guarded above), and NEVER a --hash=
        # continuation line: those carry pin evidence and are processed
        # below (a bare "--" prefix skip here would swallow every hash
        # continuation, leaving ``pending`` permanently set).
        if not is_hash_line and (line.startswith("--") or (line.startswith("-") and " " not in line)):
            continue
        ends_with_backslash = raw_line.rstrip().endswith("\\")
        body = line[:-1].strip() if ends_with_backslash else line
        if is_hash_line:
            tokens = body.split()
            if (
                len(tokens) != 1
                or not _HEX64_RE.match(tokens[0][len("--hash=sha256:") :])
                or not tokens[0].startswith("--hash=sha256:")
            ):
                raise _fail_lock(f"line {lineno}: malformed --hash continuation: {line!r}")
            assert current is not None  # pending implies a requirement under assembly
            current["hashes"].append(tokens[0][len("--hash=sha256:") :])
        else:
            if current is not None:
                entries.append(current)
                current = None
            current = {"text": body, "hashes": []}
        pending = ends_with_backslash
    if pending:
        raise _fail_lock("EOF with a pending backslash continuation (incomplete requirement)")
    if current is not None:
        entries.append(current)
        current = None

    # Logical validation with packaging.
    validated: List[Dict[str, Any]] = []
    for entry in entries:
        if not entry["hashes"]:
            raise _fail_lock(f"pin {entry['text']!r} carries no --hash=sha256:<64hex> token")
        try:
            req = packaging.requirements.Requirement(entry["text"])
        except packaging.requirements.InvalidRequirement as exc:
            raise _fail_lock(f"unparsable requirement {entry['text']!r}: {exc}") from exc
        if req.url is not None:
            raise _fail_lock(f"pin {entry['text']!r} is a direct-URL requirement (not exact pin evidence)")
        if req.extras:
            raise _fail_lock(
                f"pin {entry['text']!r} carries extras {sorted(req.extras)!r} (extras are not pin evidence)"
            )
        specs = list(req.specifier)
        if len(specs) != 1 or specs[0].operator != "==" or specs[0].version.endswith(".*"):
            raise _fail_lock(
                f"pin {entry['text']!r} is not an exact name==version pin "
                "(ranged/wildcard/composite specifiers are not lock evidence)"
            )
        try:
            version = str(packaging.version.Version(specs[0].version))
        except packaging.version.InvalidVersion as exc:
            raise _fail_lock(f"pin {entry['text']!r} has an invalid PEP 440 version") from exc
        validated.append(
            {
                "name": canonicalize_name(req.name),
                "version": version,
                "marker": str(req.marker) if req.marker is not None else None,
                "hashes": list(entry["hashes"]),
                "text": entry["text"],
            }
        )
    return validated


def _dependency_lock_candidates() -> List[Path]:
    """Default lock discovery is FALLBACK, not union (P0 #3).

    If the repo-root ``requirements.lock`` exists, it is the ONE canonical
    source and is returned alone. Only when it does not exist (e.g. a wheel
    install, where no repo root exists) does the packaged
    ``mcp_server/data/requirements.lock`` resource become the canonical
    source. Both copies are NEVER parsed together — the two files are
    byte-identical by release construction, so a union would duplicate
    every pin and trip the multiple-ACTIVE-pins fail-closed guard.
    """
    base = Path(__file__).resolve().parents[1]
    repo_root_lock = base / DEPENDENCY_LOCK_FILENAME
    if repo_root_lock.is_file():
        return [repo_root_lock]
    try:
        # importlib.resources is part of the supported Python >=3.11 baseline.
        import importlib.resources as _resources  # nosemgrep

        resource = _resources.files("mcp_server").joinpath(f"data/{DEPENDENCY_LOCK_FILENAME}")
        if resource.is_file():
            return [Path(str(resource))]
    except (ModuleNotFoundError, FileNotFoundError, OSError):
        pass  # not packaged in this distribution — repo-root lookup stands
    return [repo_root_lock]  # non-existent path -> caller fails closed on read


def dependency_lock_evidence(lock_paths: Optional[List[Path]] = None) -> Tuple[str, str, int]:
    """Strict dependency-lock evidence: (digest, lock_name, variant_count).

    Requires a GENUINE resolved, hash-bearing lock at EXACTLY
    ``requirements.lock`` (pip-compile ``--generate-hashes`` output, whose
    multiline backslash-wrapped requirement blocks each carry ``--hash=``
    tokens). Every entry must be an EXACT ``name==version`` pin with at
    least one valid sha256 hash; a PEP 508 environment marker is VALID and
    preserved (platform variants — at most one variant per name may be
    active for any evaluated environment). ``-r`` includes, editables,
    VCS/direct-URL requirements, ranged/wildcard/composite specifiers,
    extras-bearing pins, and malformed continuations (standalone hashes,
    trailing-backslash EOF, non-hash continuation tokens) fail closed with
    :class:`DependencyUnverifiableError`. The digest hashes the EXACT lock
    bytes; the count is the number of validated physical variants. The
    repo's ranged ``requirements.txt`` is deliberately NOT a candidate in
    versioned mode — nothing is fabricated and no ranged file is blessed.
    """
    candidates = lock_paths if lock_paths is not None else _dependency_lock_candidates()
    last_error: Optional[DependencyUnverifiableError] = None
    for path in candidates:
        p = Path(path)
        if p.is_symlink():
            continue
        if not p.is_file():
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError as exc:
            last_error = DependencyUnverifiableError(f"{DEPENDENCY_UNVERIFIABLE}: lock unreadable: {exc}")
            continue
        try:
            entries = _parse_strict_lock(raw, p.name)
        except DependencyUnverifiableError as exc:
            last_error = exc
            continue
        if entries:
            # Evidence = exact lock BYTES; count = validated physical variants
            # (multiple platform variants of one name each count once).
            return _sha256_file(p), p.name, len(entries)
        last_error = DependencyUnverifiableError(f"{DEPENDENCY_UNVERIFIABLE}: {p.name} pins no requirements")
    raise last_error or DependencyUnverifiableError(
        f"{DEPENDENCY_UNVERIFIABLE}: no dependency lock found at {sorted(str(c) for c in candidates)} "
        f"(expected {DEPENDENCY_LOCK_FILENAME} generated with pip-compile --generate-hashes)"
    )


# ─────────────────────────────────────────────────────────────────────────
# Release/runtime dependency parity (v4.9.0 review revision).
# ``requirements.lock`` is the ONE canonical production install input. The
# ACTIVE graph is computed by a breadth-first traversal of installed
# Requires-Dist metadata (parsed as packaging.Requirement, markers evaluated
# with ``extra=""`` everywhere); every active edge must resolve to exactly
# one ACTIVE exact lock pin matching the installed version. The self
# package is NOT part of the lock and is bound by installed version ==
# ``mcp_server.__version__``. Lock digest and installed RECORD digest stay
# DISTINCT evidence.
# ─────────────────────────────────────────────────────────────────────────

_NAME_NORMALIZE_RE = re.compile(r"[-_.]+")


def canonicalize_name(name: str) -> str:
    """PEP 503 canonical project-name normalization."""
    return _NAME_NORMALIZE_RE.sub("-", name.strip()).lower()


def _active_variants(variants: List[Dict[str, Any]], env: Dict[str, str], source_name: str) -> List[Dict[str, Any]]:
    """Marker-filter one name's physical lock variants to the ACTIVE ones."""
    active: List[Dict[str, Any]] = []
    for variant in variants:
        if variant["marker"] is None:
            active.append(variant)
            continue
        marker = packaging.markers.Marker(variant["marker"])
        if marker.evaluate(env):
            active.append(variant)
    if len(active) > 1:
        raise _fail_lock(
            f"{source_name}: multiple ACTIVE pins for the evaluated environment: {[v['text'] for v in active]}"
        )
    return active


def lock_pin_map(
    lock_paths: Optional[List[Path]] = None,
    environment: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Parse ``requirements.lock`` into canonical name -> ONE active pin.

    Marker-bearing exact pins are valid; variants are kept per canonical
    name and at most ONE may be active for the evaluated environment
    (multiple active variants fail closed). Returns
    ``{canonical_name: {"version": str, "hashes": [sha256hex...], "marker": str|None}}``
    for the active variant of every name.
    """
    resolved = lock_paths if lock_paths is not None else _dependency_lock_candidates()
    env = _lock_environment(environment)
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    found_any = False
    for path in resolved:
        p = Path(path)
        if p.is_symlink() or not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8")
        for variant in _parse_strict_lock(raw, p.name):
            found_any = True
            by_name.setdefault(variant["name"], []).append(variant)
    if not found_any:
        raise _fail_lock(f"no dependency lock parsed at {sorted(str(c) for c in resolved)}")
    pins: Dict[str, Dict[str, Any]] = {}
    for name, variants in by_name.items():
        active = _active_variants(variants, env, name)
        if not active:
            continue  # every variant inactive in this environment — no pin
        pins[name] = {
            "version": active[0]["version"],
            "hashes": active[0]["hashes"],
            "marker": active[0]["marker"],
        }
    if not pins:
        raise _fail_lock("no lock pin is active for the evaluated environment")
    return pins


def _iter_requires(distribution_name: str) -> List[packaging.requirements.Requirement]:
    """Parsed Requires-Dist of an installed distribution (fails closed)."""
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyUnverifiableError(
            f"{DEPENDENCY_UNVERIFIABLE}: distribution {distribution_name!r} is not installed "
            "(active dependency graph cannot be verified)"
        ) from exc
    parsed: List[packaging.requirements.Requirement] = []
    for req_text in dist.requires or []:
        try:
            parsed.append(packaging.requirements.Requirement(req_text))
        except packaging.requirements.InvalidRequirement as exc:
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: {distribution_name} declares an unparsable "
                f"Requires-Dist entry {req_text!r}: {exc}"
            ) from exc
    return parsed


def active_dependency_graph(
    root_distribution: str = "knowledge-rag",
    environment: Optional[Dict[str, str]] = None,
    version_of=None,
    requires_of=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """BFS over installed mandatory Requires-Dist edges from the root.

    The root always starts with ``extra=""`` (production runtime lock is
    generated with ``--strip-extras``); markers are evaluated with the
    current (or injected) environment with ``extra=""`` — so optional
    extra-gated groups are NEVER active. Child ``Requirement.extras`` are
    propagated into the child's visited-extras set and a node is revisited
    only when its extras set GROWS (cycles terminate).
    ``version_of(name) -> str | None`` is injectable for tests (defaults to
    importlib.metadata.version). Returns ``{name: [edges...]}`` with each
    edge ``{"name", "specifier", "extras", "required_by"}``.
    """
    if version_of is None:

        def _default_version_of(name):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None

        version_of = _default_version_of

    resolve_requires = _iter_requires if requires_of is None else requires_of
    env = _lock_environment(environment)
    graph: Dict[str, List[Dict[str, Any]]] = {}
    # Per-distribution set of ACTIVATED extras. The root starts with NONE
    # (production lock is --strip-extras; root optional groups never
    # activate). A mandatory edge ``core[feature]`` ACTIVATES ``feature``
    # on core, so core's ``leaf; extra == "feature"`` edge becomes active
    # and must be traversed. A node is revisited only when its activated
    # extras set GROWS (cycles terminate).
    activated_extras: Dict[str, set] = {canonicalize_name(root_distribution): set()}
    frontier: List[Tuple[str, str]] = [(canonicalize_name(root_distribution), "")]
    while frontier:
        node, via = frontier.pop(0)
        requires = resolve_requires(node)
        installed_version = version_of(node)
        if installed_version is None:
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: active graph node {node!r} (via {via!r}) "
                "is not installed — a mandatory child may never be missing"
            )
        node_extras = activated_extras.get(node, set())
        for req in requires:
            child = canonicalize_name(req.name)
            marker_env = dict(env)
            # Marker is evaluated with extra="" once per every extra
            # ACTIVATED for this distribution (plus the empty extra). An
            # edge is active when its marker holds for ANY activated extra
            # (or for extra="" when the distribution has none activated).
            if req.marker is not None:
                active = any(req.marker.evaluate({**marker_env, "extra": extra}) for extra in ({""} | node_extras))
                if not active:
                    continue  # inactive edge (extra/platform gated) — not in graph
            graph.setdefault(node, []).append(
                {
                    "name": child,
                    "specifier": str(req.specifier),
                    "extras": sorted(req.extras),
                    "required_by": node,
                }
            )
            child_extras = activated_extras.get(child, set()) | set(req.extras)
            if child_extras != activated_extras.get(child):
                activated_extras[child] = child_extras
                frontier.append((child, node))
    return graph


def require_self_version_parity(distribution_name: str = "knowledge-rag") -> str:
    """Fail closed unless the installed ``knowledge-rag`` distribution version
    equals ``mcp_server.__version__``.

    The self package is NOT part of the pip-compile lock (pip-compile only
    resolves its dependencies), so version parity IS the self-identity check —
    used at generation build AND at the runtime drift recheck.
    """
    from . import __version__

    try:
        installed = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyUnverifiableError(
            f"{DEPENDENCY_UNVERIFIABLE}: distribution {distribution_name!r} is not installed "
            "(self-version parity is required evidence)"
        ) from exc
    if installed != __version__:
        raise DependencyUnverifiableError(
            f"{DEPENDENCY_UNVERIFIABLE}: installed {distribution_name}=={installed} does not "
            f"match mcp_server.__version__=={__version__} (reinstall the wheel; the self "
            "package is pinned by version parity, not by requirements.lock)"
        )
    return installed


def verify_runtime_dependency_parity(
    lock_paths: Optional[List[Path]] = None,
    distribution_name: str = "knowledge-rag",
    environment: Optional[Dict[str, str]] = None,
    version_of=None,
    requires_of=None,
) -> Dict[str, int]:
    """Fail-closed lock/installed parity for the ACTIVE runtime graph.

    1. BFS the installed mandatory dependency graph from ``distribution_name``
       (root extra="", markers evaluated with extra="" — optional groups are
       never active; child extras propagate; cycles terminate).
    2. Every ACTIVE edge requires: child installed, installed version
       satisfies the declared specifier, exactly one ACTIVE exact lock pin
       for the child, and installed version == pin version. A mandatory
       child missing or only INACTIVE lock variants fails.
    3. ACTIVE lock pins OUTSIDE the graph: if installed, the version must
       equal the pin exactly; if absent, they are ignored (Windows-only /
       GPU-extra absent passes; a missing mandatory transitive never does).
       INACTIVE variants are ignored entirely.
    4. Self-version parity: installed knowledge-rag == mcp_server.__version__.

    ``version_of``/``requires_of`` are injectable seams (tests); defaults
    use importlib.metadata. Returns ``{"graph_edges": n, "pinned": m,
    "installed_checked": k}``.
    """
    require_self_version_parity(distribution_name)
    pins = lock_pin_map(lock_paths, environment)

    if version_of is None:

        def _default_version_of(name):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None

        version_of = _default_version_of

    graph = active_dependency_graph(distribution_name, environment, version_of, requires_of)

    graph_names = {distribution_name}
    for _parent, edges in graph.items():
        for edge in edges:
            child = edge["name"]
            graph_names.add(child)
            installed_version = version_of(child)
            if installed_version is None:
                raise DependencyUnverifiableError(
                    f"{DEPENDENCY_UNVERIFIABLE}: active mandatory dependency {child!r} "
                    f"(required by {edge['required_by']!r}) is not installed"
                )
            if edge["specifier"]:
                spec = packaging.specifiers.SpecifierSet(edge["specifier"])
                if not spec.contains(installed_version, prereleases=True):
                    raise DependencyUnverifiableError(
                        f"{DEPENDENCY_UNVERIFIABLE}: installed {child}=={installed_version} "
                        f"does not satisfy declared specifier {edge['specifier']!r} "
                        f"(required by {edge['required_by']!r})"
                    )
            pin = pins.get(child)
            if pin is None:
                raise DependencyUnverifiableError(
                    f"{DEPENDENCY_UNVERIFIABLE}: active mandatory dependency {child!r} "
                    f"(required by {edge['required_by']!r}) has no ACTIVE exact lock pin "
                    "(only inactive variants or absent — regenerate requirements.lock)"
                )
            if installed_version != pin["version"]:
                raise DependencyUnverifiableError(
                    f"{DEPENDENCY_UNVERIFIABLE}: lock pins {child}=={pin['version']} but "
                    f"{installed_version} is installed (runtime drift — reinstall from the lock)"
                )

    # Orphan pins: active in the lock but outside the active graph.
    checked = 0
    for name, pin in pins.items():
        if name in graph_names:
            checked += 1
            continue
        installed_version = version_of(name)
        if installed_version is None:
            continue  # platform/extra-inactive pin absent — fine
        checked += 1
        if installed_version != pin["version"]:
            raise DependencyUnverifiableError(
                f"{DEPENDENCY_UNVERIFIABLE}: lock pins {name}=={pin['version']} but "
                f"{installed_version} is installed (runtime drift — reinstall from the lock)"
            )
    edges_total = sum(len(edges) for edges in graph.values())
    return {"graph_edges": edges_total, "pinned": len(pins), "installed_checked": checked}


def chunking_identity(cfg: Any) -> str:
    """Canonical chunking identity: model + prefixes + size/overlap."""
    return _stable_json_digest(
        {
            "embedding_model": getattr(cfg, "embedding_model", None),
            "query_prefix": getattr(cfg, "query_prefix", None),
            "passage_prefix": getattr(cfg, "passage_prefix", None),
            "chunk_size": getattr(cfg, "chunk_size", None),
            "chunk_overlap": getattr(cfg, "chunk_overlap", None),
        }
    )


def model_config_identity(cfg: Any) -> str:
    """Canonical model-config identity: the configuration side of the model
    compatibility binding (the artifact BYTES side is model_artifact_sha256).

    Covers every execution-changing model field: embedding model/dimension/
    prefixes/runtime/pooling AND the reranker binding (enabled flag, logical
    model, exact artifact digest) plus the loader/runtime provider config —
    builder, startup pin, and per-request gate all call this ONE function.
    """
    compat = cfg.generation_compatibility() if hasattr(cfg, "generation_compatibility") else {}
    compat = compat or {}
    return _stable_json_digest(
        {
            "embedding_model": compat.get("embedding_model"),
            "embedding_dimension": compat.get("embedding_dimension"),
            "query_prefix": compat.get("query_prefix"),
            "passage_prefix": compat.get("passage_prefix"),
            "runtime_version": compat.get("runtime_version"),
            "pooling": compat.get("pooling"),
            "reranker_enabled": compat.get("reranker_enabled"),
            "reranker_model": compat.get("reranker_model"),
            "reranker_artifact_sha256": compat.get("reranker_artifact_sha256"),
            "gpu_mode": getattr(cfg, "gpu_mode", None),
        }
    )


def generation_identity(
    cfg: Any,
    *,
    corpus_manifest_sha256: str,
    code_sha256: Optional[str] = None,
) -> Dict[str, str]:
    """The ONE canonical v3 identity block for builder and runtime.

    Assembles every _IDENTITY_KEYS field from freshly recomputed evidence:
    the caller supplies the corpus manifest digest (builder: sealed tree;
    runtime: live source tree), and every other field is derived here from
    the shared evidence functions so builder receipts and runtime
    verification can never disagree field-by-field.
    """
    # Release/runtime reproducibility (v4.9.0): the builder environment must
    # itself satisfy lock parity + self-version parity (the same identity the
    # runtime drift recheck enforces) before it seals a receipt.
    verify_runtime_dependency_parity()
    record_digest, _versions, _checked = installed_runtime_record_evidence()
    artifact_sha = (cfg.generation_compatibility() or {}).get("model_artifact_sha256")
    if not isinstance(artifact_sha, str) or not _HEX64_RE.match(artifact_sha):
        raise VerificationError(
            "generation_identity: compatibility.model_artifact_sha256 is missing/invalid "
            "(versioned mode requires the exact materialized artifact digest)"
        )
    # ONE fresh config read feeds both digests (P0 #6): the legacy
    # ``config_sha256`` field and ``retrieval_config_sha256`` are the SAME
    # canonical retrieval-config digest, never two independent reads that
    # could observe different file states mid-computation.
    config_digest = retrieval_config_digest(cfg)
    return {
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "config_sha256": config_digest,
        "code_sha256": code_sha256 if code_sha256 is not None else code_identity(),
        "retrieval_config_sha256": config_digest,
        "installed_record_sha256": record_digest,
        "dependency_lock_sha256": dependency_lock_evidence()[0],
        "model_artifact_sha256": artifact_sha,
        "model_config_sha256": model_config_identity(cfg),
        "chunking_sha256": chunking_identity(cfg),
    }


def code_identity(module_dir: Optional[Path] = None) -> str:
    """Canonical SOURCE tree digest of the ``mcp_server`` package directory.

    Excludes generated/non-source bytes (P0 #6): ``__pycache__`` directories,
    ``*.pyc``/``*.pyo``/``*.pyd`` bytecode, ``*.log`` logs, and temp/swap
    files — none of them are executable package source, and including them
    would make the identity flap on every import. The digest changes on any
    packaged ``.py``/``.json``/``.yaml``/static-asset byte change.
    """
    base = Path(module_dir) if module_dir is not None else Path(__file__).resolve().parent

    excluded_dirs = {"__pycache__"}
    excluded_suffixes = (".pyc", ".pyo", ".pyd", ".log", ".swp", ".tmp")
    entries: List[Tuple[str, Path]] = []
    for root, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded_dirs)
        root_path = Path(root)
        for n in sorted(names):
            if n.endswith(excluded_suffixes):
                continue
            p = root_path / n
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise UnsafePathError(f"non-source/symlink entry inside package tree: {p}")
            entries.append((p.relative_to(base).as_posix(), p))
    entries.sort(key=lambda item: item[0])

    h = hashlib.sha256(_TREE_DIGEST_VERSION)
    for rel, p in entries:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(p).encode("ascii"))
        h.update(b"\0")
        h.update(str(p.stat().st_size).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def inspect_current_receipt(root: Path) -> Optional[Dict[str, Any]]:
    """LENIENT read of the current receipt for stats/status tooling ONLY.

    Returns a safe summary dict even for v2 receipts (schema_version,
    created_at, status, truncated identity/compatibility digests, backend
    counts, and a ``servable`` flag) — v2 yields ``servable=False`` with
    reason ``schema_v2_rebuild_required``. Never verifies artifacts (stats
    must not touch sealed bytes); never returns the raw receipt; never
    exposes paths or provenance HEAD (HEAD is excluded from stats output).
    Returns ``None`` when the pointer or receipt is missing/unreadable.
    """
    try:
        store = GenerationStore(Path(root), create=False)
        with store._read_lock():
            pointer = store._read_pointer_strict()
            if pointer is None:
                return None
            gdir = store._resolve_generation_dir(pointer["generation_id"])
            receipt = store._read_receipt(gdir, pointer["generation_id"])
    except Exception:
        return None
    version = receipt.get("schema_version")
    ident = receipt.get("identity") or {}
    compat = receipt.get("compatibility") or {}
    backends = receipt.get("backends") or {}
    chroma_ev = backends.get("chroma") or {}
    fts_ev = backends.get("fts5") or {}
    return {
        "schema_version": version if isinstance(version, int) else None,
        "generation_id": receipt.get("generation_id"),
        "receipt_sha256": pointer.get("receipt_sha256"),
        "created_at": receipt.get("created_at"),
        "status": receipt.get("status"),
        "servable": version == RECEIPT_SCHEMA_VERSION,
        "reason": None if version == RECEIPT_SCHEMA_VERSION else "schema_v2_rebuild_required",
        "identity": {
            key: (str(value)[:12] + "...") if isinstance(value, str) and len(value) > 12 else value
            for key, value in ident.items()
        },
        "compatibility": {
            "embedding_model": compat.get("embedding_model"),
            "embedding_dimension": compat.get("embedding_dimension"),
            "model_artifact_sha256": str(compat.get("model_artifact_sha256") or "")[:12] + "...",
        },
        "backends": {
            "chroma_row_count": chroma_ev.get("row_count"),
            "chroma_row_digest": str(chroma_ev.get("row_digest") or "")[:12] + "...",
            "fts5_row_count": fts_ev.get("row_count"),
            "fts5_row_digest": str(fts_ev.get("row_digest") or "")[:12] + "...",
        },
    }


def _entry_violations(directory: Path, allowed: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Return (unexpected, missing) top-level entry names for ``directory``."""
    present = set(os.listdir(directory))
    return present - set(allowed), set(allowed) - present


def _utc_now_iso() -> str:
    """Canonical UTC ISO-8601 timestamp: YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_created_at(value: Any) -> datetime:
    """Parse a canonical, timezone-aware UTC ISO-8601 ``created_at``.

    Accepts exactly ``YYYY-MM-DDTHH:MM:SSZ`` (the form ``_utc_now_iso``
    produces); naive timestamps, non-UTC offsets, fractional seconds, and any
    other text are rejected.
    """
    if not isinstance(value, str) or not _CREATED_AT_RE.match(value):
        raise VerificationError(f"created_at must be canonical UTC ISO-8601 'YYYY-MM-DDTHH:MM:SSZ', got {value!r}")
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise VerificationError(f"created_at is not a valid UTC timestamp: {value!r}") from exc
    return dt.replace(tzinfo=timezone.utc)


def _require_safe_id(generation_id: str) -> str:
    """Validate a generation ID against the strict safe-ID grammar."""
    if not isinstance(generation_id, str) or not _SAFE_ID_RE.match(generation_id):
        raise UnsafePathError(f"unsafe generation id: {generation_id!r}")
    return generation_id


def _require_hex64(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise VerificationError(f"{what} must be 64 lowercase hex chars, got {value!r}")
    return value


def _require_count(value: Any, what: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VerificationError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


def _validate_identity(identity: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Validate the identity block (content-bound + provenance SHAs).

    Every key is a mandatory 64-hex digest in v3 — installed RECORD and
    dependency-lock digests are REQUIRED evidence, never nullable and never
    fail-open (P0 #2). ``config_sha256`` is retained as the legacy provenance
    field; it never controls readiness.
    """
    if not isinstance(identity, dict) or set(identity) != set(_IDENTITY_KEYS):
        got = sorted(identity) if isinstance(identity, dict) else type(identity).__name__
        raise VerificationError(f"identity must have exactly {sorted(_IDENTITY_KEYS)}, got: {got}")
    out: Dict[str, Optional[str]] = {}
    for key in sorted(_IDENTITY_KEYS):
        out[key] = _require_hex64(identity[key], f"identity.{key}")
    return out


def _validate_provenance(provenance: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Validate the provenance block. ``vault_head`` is informational only.

    Nullable ONLY when the source is provably not a Git worktree — a v3
    receipt carries either a 40-hex HEAD or an explicit ``null``; both are
    schema-valid. Provenance NEVER participates in readiness or identity
    comparison (P0 #1): a HEAD-only change is GREEN by construction.
    """
    if not isinstance(provenance, dict) or set(provenance) != set(_PROVENANCE_KEYS):
        got = sorted(provenance) if isinstance(provenance, dict) else type(provenance).__name__
        raise VerificationError(f"provenance must have exactly {sorted(_PROVENANCE_KEYS)}, got: {got}")
    head = provenance["vault_head"]
    if head is None:
        return {"vault_head": None}
    if not isinstance(head, str) or not _VAULT_HEAD_RE.match(head.strip().lower()):
        raise VerificationError(
            f"provenance.vault_head must be a 40-hex (SHA-1) or 64-hex (SHA-256) commit id or null, got: {head!r}"
        )
    return {"vault_head": head.strip().lower()}


def _validate_compatibility(compatibility: Any) -> Dict[str, Any]:
    """Validate the exact compatibility binding object."""
    if not isinstance(compatibility, dict) or set(compatibility) != set(_COMPATIBILITY_KEYS):
        got = sorted(compatibility) if isinstance(compatibility, dict) else type(compatibility).__name__
        raise VerificationError(f"compatibility must have exactly {sorted(_COMPATIBILITY_KEYS)}, got: {got}")
    out: Dict[str, Any] = {}
    for key in ("collection_name", "embedding_model", "runtime_version", "pooling"):
        value = compatibility[key]
        if not isinstance(value, str) or not value:
            raise VerificationError(f"compatibility.{key} must be a non-empty string, got {value!r}")
        out[key] = value
    for key in ("query_prefix", "passage_prefix"):
        value = compatibility[key]
        if not isinstance(value, str):
            raise VerificationError(f"compatibility.{key} must be a string, got {value!r}")
        out[key] = value
    dim = compatibility["embedding_dimension"]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise VerificationError(f"compatibility.embedding_dimension must be an int >= 1, got {dim!r}")
    out["embedding_dimension"] = dim
    chunk_size = compatibility["chunk_size"]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise VerificationError(f"compatibility.chunk_size must be an int >= 1, got {chunk_size!r}")
    chunk_overlap = compatibility["chunk_overlap"]
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise VerificationError(f"compatibility.chunk_overlap must be an int >= 0, got {chunk_overlap!r}")
    if chunk_overlap >= chunk_size:
        raise VerificationError(f"compatibility.chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
    out["chunk_size"] = chunk_size
    out["chunk_overlap"] = chunk_overlap
    out["model_artifact_sha256"] = _require_hex64(
        compatibility["model_artifact_sha256"], "compatibility.model_artifact_sha256"
    )
    # Reranker binding (P0 #9/A): enabled is a hard boolean; when enabled the
    # logical model name must be a non-empty string and the artifact digest a
    # 64-hex digest of the exact materialized directory. When disabled both
    # must be None — the explicit disabled identity state.
    enabled = compatibility["reranker_enabled"]
    if not isinstance(enabled, bool):
        raise VerificationError(f"compatibility.reranker_enabled must be a bool, got {enabled!r}")
    out["reranker_enabled"] = enabled
    rmodel = compatibility["reranker_model"]
    rdigest = compatibility["reranker_artifact_sha256"]
    if enabled:
        if not isinstance(rmodel, str) or not rmodel:
            raise VerificationError(
                f"compatibility.reranker_model must be a non-empty string when reranker is enabled, got {rmodel!r}"
            )
        out["reranker_model"] = rmodel
        out["reranker_artifact_sha256"] = _require_hex64(rdigest, "compatibility.reranker_artifact_sha256")
    else:
        if rmodel is not None or rdigest is not None:
            raise VerificationError(
                "compatibility.reranker_model/reranker_artifact_sha256 must be None "
                f"when reranker is disabled, got model={rmodel!r}, digest={rdigest!r}"
            )
        out["reranker_model"] = None
        out["reranker_artifact_sha256"] = None
    return out


def _validate_chroma_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != set(_CHROMA_EVIDENCE_KEYS):
        got = sorted(evidence) if isinstance(evidence, dict) else type(evidence).__name__
        raise VerificationError(f"chroma evidence must have exactly {sorted(_CHROMA_EVIDENCE_KEYS)}, got: {got}")
    name = evidence["collection_name"]
    if not isinstance(name, str) or not name:
        raise VerificationError(f"chroma.collection_name must be a non-empty string, got {name!r}")
    counts = {
        key: _require_count(evidence[key], f"chroma.{key}", minimum=0)
        for key in ("row_count", "unique_id_count", "hydrated_id_count")
    }
    if not (counts["row_count"] == counts["unique_id_count"] == counts["hydrated_id_count"]):
        raise VerificationError(f"chroma id-count parity violated (all three counts must be equal): {counts}")
    return {
        "collection_name": name,
        **counts,
        "row_digest": _require_hex64(evidence["row_digest"], "chroma.row_digest"),
        "common_row_digest": _require_hex64(evidence["common_row_digest"], "chroma.common_row_digest"),
        "backend_generation_id": _require_hex64(evidence["backend_generation_id"], "chroma.backend_generation_id"),
    }


def _validate_fts_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != set(_FTS_EVIDENCE_KEYS):
        got = sorted(evidence) if isinstance(evidence, dict) else type(evidence).__name__
        raise VerificationError(f"fts5 evidence must have exactly {sorted(_FTS_EVIDENCE_KEYS)}, got: {got}")
    version = evidence["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != FTS_SCHEMA_VERSION:
        raise VerificationError(f"fts5 schema_version must be exactly {FTS_SCHEMA_VERSION}, got {version!r}")
    status = evidence["status"]
    if status != RECEIPT_STATUS_COMPLETE:
        raise VerificationError(f"fts5 status must be {RECEIPT_STATUS_COMPLETE!r}, got {status!r}")
    return {
        "schema_version": version,
        "status": status,
        "row_count": _require_count(evidence["row_count"], "fts5.row_count", minimum=0),
        "row_digest": _require_hex64(evidence["row_digest"], "fts5.row_digest"),
        "source_digest": _require_hex64(evidence["source_digest"], "fts5.source_digest"),
        "verified_digest": _require_hex64(evidence["verified_digest"], "fts5.verified_digest"),
        "backend_generation_id": _require_hex64(evidence["backend_generation_id"], "fts5.backend_generation_id"),
    }


def _require_backend_parity(chroma: Dict[str, Any], fts5: Dict[str, Any]) -> None:
    """Backends must attest the same COMMON logical row universe.

    Parity is over the fields BOTH backends store — (chunk_id, document,
    filename, category) — via ``chroma.common_row_digest``. The complete
    per-backend digests (``row_digest``) are intentionally unlike: Chroma's
    covers embeddings + full retrieval metadata, FTS5's covers its own
    logical rows; each is independently bound and never forced equal.
    """
    if chroma["row_count"] != fts5["row_count"]:
        raise VerificationError(
            f"backend row-count parity violated: chroma={chroma['row_count']} != fts5={fts5['row_count']}"
        )
    canonical = chroma["common_row_digest"]
    if not (fts5["row_digest"] == fts5["source_digest"] == fts5["verified_digest"]):
        raise VerificationError(
            "fts5 digest self-parity violated: row_digest == source_digest == verified_digest required"
        )
    for field in ("row_digest", "source_digest", "verified_digest"):
        if fts5[field] != canonical:
            raise VerificationError(f"backend digest parity violated: fts5.{field} != chroma.common_row_digest")


def _bind_collection_name(compatibility: Dict[str, Any], chroma: Dict[str, Any]) -> None:
    """Chroma evidence must be bound to the receipt's compatibility collection."""
    if chroma["collection_name"] != compatibility["collection_name"]:
        raise VerificationError(
            f"chroma evidence collection_name {chroma['collection_name']!r} != "
            f"compatibility.collection_name {compatibility['collection_name']!r}"
        )


def _parse_pointer(raw: bytes) -> Dict[str, str]:
    """Strictly parse and validate ``current``-pointer bytes. Raises PointerError."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PointerError(f"current pointer is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict) or set(obj) != set(_POINTER_KEYS):
        raise PointerError(
            f"current pointer must be an object with exactly {sorted(_POINTER_KEYS)}, got: "
            f"{sorted(obj) if isinstance(obj, dict) else type(obj).__name__}"
        )
    gid = obj["generation_id"]
    sha = obj["receipt_sha256"]
    if not isinstance(gid, str) or not _SAFE_ID_RE.match(gid):
        raise PointerError(f"current pointer has unsafe generation_id: {gid!r}")
    if not isinstance(sha, str) or not _HEX64_RE.match(sha):
        raise PointerError(f"current pointer has malformed receipt_sha256: {sha!r}")
    return {"generation_id": gid, "receipt_sha256": sha}


# ============================================================================
# STORE
# ============================================================================


class GenerationStore:
    """Crash-safe, versioned generation store rooted at ``root``.

    The store owns the immutable ``generations/<id>`` directories, the
    ``current`` JSON pointer, the ``root/.building-<id>`` staging areas, and a
    stable lock file kept outside ``generations/``. It performs no garbage
    collection and never writes or follows symlinks.
    """

    def __init__(self, root: Path, *, create: bool = False) -> None:
        """Open the store.

        ``create=False`` (default) is the production read-only open: anchors
        are validated and NOTHING is created — missing or invalid anchors are
        rejected. ``create=True`` is the explicit builder initialization.
        """
        self.root = Path(root)
        self.generations_dir = self.root / GENERATIONS_DIRNAME
        self.current_path = self.root / CURRENT_FILENAME
        self.lock_path = self.root / LOCK_FILENAME
        self._tmp_counter = itertools.count()
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._lock_mode: Optional[str] = None  # None | "exclusive" | "shared"
        self._lock_fd: Optional[int] = None
        if create:
            self._ensure_layout()
        else:
            self._validate_layout()

    # ------------------------------------------------------------------ layout

    def _ensure_layout(self) -> None:
        """Builder initialization: create root/ and generations/, validate anchors."""
        if self.root.is_symlink():
            raise UnsafePathError(f"store root is a symlink: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if not stat.S_ISDIR(os.lstat(self.root).st_mode):
            raise UnsafePathError(f"store root is not a directory: {self.root}")
        self._check_generations_anchor(require=False)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self._check_file_anchor(self.current_path, "current pointer")
        self._check_file_anchor(self.lock_path, "lock file")

    def _validate_layout(self) -> None:
        """Read-only open: validate everything, create NOTHING."""
        if self.root.is_symlink():
            raise UnsafePathError(f"store root is a symlink: {self.root}")
        try:
            st = os.lstat(self.root)
        except OSError as exc:
            raise UnsafePathError(f"store root missing or unreadable (read-only open): {self.root}") from exc
        if not stat.S_ISDIR(st.st_mode):
            raise UnsafePathError(f"store root is not a directory: {self.root}")
        self._check_generations_anchor(require=True)
        self._check_file_anchor(self.current_path, "current pointer")
        self._check_file_anchor(self.lock_path, "lock file")

    def _check_generations_anchor(self, require: bool = False) -> None:
        """Fail closed if generations/ is absent (when required), symlinked, or redirected."""
        if self.generations_dir.is_symlink():
            raise UnsafePathError(f"generations directory is a symlink: {self.generations_dir}")
        if not self.generations_dir.exists():
            if require:
                raise UnsafePathError(f"generations directory missing: {self.generations_dir}")
            return
        st = os.lstat(self.generations_dir)
        if not stat.S_ISDIR(st.st_mode):
            raise UnsafePathError(f"generations path is not a directory: {self.generations_dir}")
        expected = Path(os.path.realpath(self.root)) / GENERATIONS_DIRNAME
        if Path(os.path.realpath(self.generations_dir)) != expected:
            raise UnsafePathError(f"generations directory redirected outside the store root: {self.generations_dir}")

    def _check_file_anchor(self, path: Path, what: str) -> None:
        """Fail closed if an optional anchor (current/lock) is symlinked or non-regular."""
        if path.is_symlink():
            raise UnsafePathError(f"{what} is a symlink: {path}")
        if path.exists() and not stat.S_ISREG(os.lstat(path).st_mode):
            raise UnsafePathError(f"{what} is not a regular file: {path}")

    def _check_anchors_locked(self) -> None:
        """Re-verify root/generations/current/lock under the exclusive lock."""
        if self.root.is_symlink() or not stat.S_ISDIR(os.lstat(self.root).st_mode):
            raise UnsafePathError(f"store root anchor invalid: {self.root}")
        self._check_generations_anchor(require=True)
        self._check_file_anchor(self.current_path, "current pointer")
        self._check_file_anchor(self.lock_path, "lock file")

    def _building_dir(self, generation_id: str) -> Path:
        """Staging path: root/.building-<id> — OUTSIDE generations/ by design."""
        return self.root / (BUILDING_PREFIX + _require_safe_id(generation_id))

    def _resolve_generation_dir(self, generation_id: str) -> Path:
        """Safe generation dir: safe ID + symlink + realpath containment checks."""
        _require_safe_id(generation_id)
        lexical = self.generations_dir / generation_id
        if lexical.is_symlink():
            raise UnsafePathError(f"generation directory is a symlink: {lexical}")
        expected = Path(os.path.realpath(self.generations_dir)) / generation_id
        if Path(os.path.realpath(lexical)) != expected:
            raise UnsafePathError(f"generation path escapes the generations root: {lexical}")
        return lexical

    def generation_dir(self, generation_id: str) -> Path:
        """Public safe accessor for a generation's directory path."""
        return self._resolve_generation_dir(generation_id)

    # ------------------------------------------------------------------- lock

    def _open_lock_fd(self) -> int:
        if fcntl is None:
            raise LockUnsupportedError(
                "POSIX file locking (fcntl.flock) is unavailable on this platform; "
                "the generation store refuses to degrade to an in-process lock"
            )
        # The lock file is created lazily HERE (never during __init__) — the
        # documented stable lock path, idempotent across processes.
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(self.lock_path, flags, 0o644)
        except OSError as exc:
            raise UnsafePathError(f"cannot open lock file {self.lock_path}: {exc}") from exc

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        """One stable EXCLUSIVE lock (flock) on a path OUTSIDE generations/.

        Builders only (begin_build/abort_build/publish/activate). Reentrant
        within the acquiring thread: a nested acquisition (either mode)
        reuses the lock already held; anchors are re-checked under the lock.
        There is deliberately no non-POSIX fallback.
        """
        with self._thread_lock:
            outermost = self._lock_depth == 0
            if outermost:
                fd = self._open_lock_fd()
                lock_mod = _fcntl_or_raise()
                lock_mod.flock(fd, lock_mod.LOCK_EX)
                self._lock_fd, self._lock_mode = fd, "exclusive"
            elif self._lock_mode != "exclusive":  # pragma: no cover - misuse guard
                raise GenerationError("cannot acquire the exclusive builder lock while holding a shared read lock")
            self._lock_depth += 1
            try:
                self._check_anchors_locked()
                yield
            finally:
                self._lock_depth -= 1
                if outermost and self._lock_fd is not None:
                    fd, self._lock_fd, self._lock_mode = self._lock_fd, None, None
                    lock_mod = _fcntl_or_raise()
                    try:
                        lock_mod.flock(fd, lock_mod.LOCK_UN)
                    finally:
                        os.close(fd)

    def _open_read_lock_fd(self) -> Optional[int]:
        """Open the EXISTING lock file WITHOUT creating it (production reads).

        Returns ``None`` when the lock file does not exist yet — reads then
        proceed under in-process serialization only (see :meth:`_read_lock`).
        A symlinked or non-regular lock file fails closed.
        """
        if fcntl is None:
            raise LockUnsupportedError(
                "POSIX file locking (fcntl.flock) is unavailable on this platform; "
                "the generation store refuses to degrade to an in-process lock"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.lock_path, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise UnsafePathError(f"cannot open lock file {self.lock_path}: {exc}") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UnsafePathError(f"lock file is not a regular file: {self.lock_path}")
        except Exception:
            os.close(fd)
            raise
        return fd

    @contextmanager
    def _read_lock(self) -> Iterator[None]:
        """SHARED lock for production reads — NEVER creates the lock file.

        Takes LOCK_SH on the existing lock file when present (safe alongside
        concurrent readers, excluded by builders' LOCK_EX). When no lock file
        exists, the read runs under in-process serialization only: this
        process's builders serialize on the same thread lock, and a
        cross-process builder must hold LOCK_EX — which created the lock
        file — before any mutation, so the worst case is a torn read that
        fail-closed full verification resolves to ``None``/an error. Reentrant
        with both read and exclusive acquisitions in the same thread.
        """
        with self._thread_lock:
            outermost = self._lock_depth == 0
            acquired: Optional[int] = None
            if outermost:
                acquired = self._open_read_lock_fd()
                if acquired is not None:
                    self._lock_fd = acquired
                    self._lock_mode = "shared"
                    lock_mod = _fcntl_or_raise()
                    lock_mod.flock(acquired, lock_mod.LOCK_SH)
            self._lock_depth += 1
            try:
                if outermost:
                    self._check_anchors_locked()
                yield
            finally:
                self._lock_depth -= 1
                if outermost and acquired is not None:
                    fd, self._lock_fd, self._lock_mode = acquired, None, None
                    lock_mod = _fcntl_or_raise()
                    try:
                        lock_mod.flock(fd, lock_mod.LOCK_UN)
                    finally:
                        os.close(fd)

    # ------------------------------------------------------------------ build

    def begin_build(self, generation_id: str) -> Path:
        """Create and return ``root/.building-<id>`` for artifact staging.

        Raises :class:`BuildExistsError` if the building directory or a
        published generation with the same ID already exists.
        """
        gid = _require_safe_id(generation_id)
        with self._exclusive_lock():
            final = self.generations_dir / gid
            if final.exists() or final.is_symlink():
                raise BuildExistsError(f"generation {gid!r} is already published: {final}")
            building = self._building_dir(gid)
            if building.is_symlink():
                raise UnsafePathError(f"building directory is a symlink: {building}")
            try:
                building.mkdir(parents=False, exist_ok=False)
            except FileExistsError as exc:
                raise BuildExistsError(f"building directory already exists (stale build?): {building}") from exc
            return building

    def abort_build(self, generation_id: str) -> None:
        """Remove an abandoned ``root/.building-<id>`` staging directory.

        The ONLY destructive operation in the store; never touches published
        generations; refuses to operate through symlinks.
        """
        gid = _require_safe_id(generation_id)
        with self._exclusive_lock():
            building = self._building_dir(gid)
            if building.is_symlink():
                raise UnsafePathError(f"refusing to abort through a symlink: {building}")
            if building.is_dir():
                shutil.rmtree(building)

    # ---------------------------------------------------------------- publish

    def publish(
        self,
        generation_id: str,
        *,
        identity: Dict[str, Any],
        compatibility: Dict[str, Any],
        chroma_evidence: Dict[str, Any],
        fts_evidence: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
        expected_current: Union[None, Dict[str, str], str] = None,
        created_at: Optional[str] = None,
    ) -> ActivationResult:
        """Seal a staged generation and CAS-switch ``current`` to it.

        ``expected_current`` accepts ``None`` (an existing pointer must NOT
        exist), the full pointer identity dict, or the explicit sentinel
        :data:`EXPECTED_CURRENT_ABSENT`. ``None`` and the sentinel both
        require that no pointer exists; the sentinel exists so CAS-conflict
        recovery can distinguish "explicitly checked, none expected" from
        "caller did not check".

        ``provenance`` (v3) records the informational vault Git HEAD — never
        part of identity or readiness.

        All durability-critical work happens under ONE lock acquisition:
        anchor re-checks, staged-entry and artifact verification
        (digests/counts/types), assembly and FULL validation of the receipt
        (schema, compatibility, backend parity, canonical UTC created_at) and
        admission of its encoded size (``<= _MAX_RECEIPT_BYTES``) — strictly
        BEFORE any fsync/rename/CAS — then fsync of every artifact
        file and directory, receipt written LAST, atomic same-filesystem
        rename ``root/.building-<id>`` -> ``generations/<id>``, fsync of
        ``generations/``, and the identity-bound CAS pointer swap.

        ``expected_current`` binds generation_id AND receipt SHA:
        ``None`` requires that no pointer exists yet; a dict
        ``{"generation_id", "receipt_sha256"}`` must equal the current pointer
        exactly (ABA/lost-update safe). On conflict the generation stays
        published and :class:`CurrentConflictError` is raised — recover with
        ``activate()``. If the pointer swap happened but its fsync failed,
        :class:`CommitStateUncertainError` is raised instead of pretending
        the pointer is unchanged.
        """
        gid = _require_safe_id(generation_id)
        expected = self._validate_expected_identity(expected_current)
        prov = _validate_provenance(provenance if provenance is not None else {"vault_head": None})
        ident = _validate_identity(identity)
        compat = _validate_compatibility(compatibility)
        if ident["model_artifact_sha256"] != compat["model_artifact_sha256"]:
            raise VerificationError(
                "publish inputs: identity.model_artifact_sha256 != compatibility.model_artifact_sha256"
            )
        chroma_ev = _validate_chroma_evidence(chroma_evidence)
        fts_ev = _validate_fts_evidence(fts_evidence)
        _require_backend_parity(chroma_ev, fts_ev)
        _bind_collection_name(compat, chroma_ev)

        building = self._building_dir(gid)
        final = self.generations_dir / gid

        with self._exclusive_lock():
            if building.is_symlink() or final.is_symlink():
                raise UnsafePathError(f"symlink in publish path: {building} / {final}")
            if not building.is_dir():
                raise PublishError(f"no building directory for {gid!r} — call begin_build() and stage artifacts first")
            if final.exists():
                raise PublishError(f"generation {gid!r} is already published: {final}")

            # Exact top-level entries: the five artifacts ONLY (receipt comes
            # last, after sealing). WAL/SHM sidecars or any extra entry rejects.
            extras, missing = _entry_violations(building, set(REQUIRED_ARTIFACTS))
            if extras or missing:
                details = []
                if extras:
                    details.append(f"unexpected entries (WAL/SHM/extra artifacts are rejected): {sorted(extras)}")
                if missing:
                    details.append(f"missing entries: {sorted(missing)}")
                raise PublishError(
                    f"staged generation top-level entries must be exactly {sorted(REQUIRED_ARTIFACTS)}: "
                    + "; ".join(details)
                )

            # Verify staged bytes (also rejects symlinks / non-regular files /
            # sqlite sidecars inside tree artifacts). The BEFORE set; the seam
            # below re-hashes AFTER inspection and requires equality.
            artifacts = self._hash_staged_artifacts(building)

            # ------------------------------------------------------------------
            # P0 (pass 10): staged semantic inspection MUST NOT open Chroma in
            # this process — a PersistentClient may mutate SQLite/HNSW bytes
            # or retain handles, so the receipt would bind pre-open bytes
            # while sealing post-open bytes. The seam below inspects Chroma in
            # a short-lived CHILD, waits for its successful exit, accepts only
            # validated JSON evidence, and requires the five-artifact hashes
            # to be UNCHANGED after the child and all readers are closed.
            # ------------------------------------------------------------------
            semantics = self._recompute_staged_semantics(
                building,
                ident,
                chroma_ev,
                fts_ev,
            )
            # Receipt artifact hashes come ONLY from the FINAL, verified-
            # unchanged artifact set (the seam re-hashed after inspection).
            artifacts = semantics["artifacts"]

            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "generation_id": gid,
                "created_at": created_at if created_at is not None else _utc_now_iso(),
                "status": RECEIPT_STATUS_COMPLETE,
                "provenance": prov,
                "identity": ident,
                "compatibility": compat,
                "artifacts": artifacts,
                "backends": {"chroma": chroma_ev, "fts5": fts_ev},
            }

            # Validate the ASSEMBLED receipt (schema, compatibility, parity,
            # canonical UTC created_at) BEFORE any fsync / rename / CAS step.
            self._validate_receipt(receipt, gid)
            payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
            if len(payload) > _MAX_RECEIPT_BYTES:
                raise VerificationError(
                    f"receipt too large: {len(payload)} encoded bytes exceeds the "
                    f"{_MAX_RECEIPT_BYTES}-byte maximum for generation {gid!r}; refusing to seal"
                )

            # Durability: every artifact file + containing directory first...
            _fsync_tree(building)
            # ...then the receipt LAST (all artifacts are already durable).
            _write_file_exclusive(building / RECEIPT_FILENAME, payload)
            _fsync_dir(building)
            _atomic_replace(building, final)
            _fsync_dir(self.generations_dir)

            receipt_sha256 = hashlib.sha256(payload).hexdigest()
            self._cas_current_locked(gid, receipt_sha256, expected)

        return ActivationResult(gid, receipt_sha256, receipt, restart_required=True)

    # ------------------------------------------------- staged inspection seam

    _CHILD_INSPECT_TIMEOUT_S = 120.0  # bound the child; 2 min is ample

    @staticmethod
    def _hash_staged_artifacts(building: Path) -> Dict[str, Dict[str, Any]]:
        """Hash/count the EXACT five staged artifacts (no sidecars, no extras).

        Fail-closed on any extra/missing top-level entry, symlink, or
        non-regular file — the caller invokes this BEFORE and AFTER the
        inspection child so any mutation the child (or a leaked handle)
        caused is detected before the receipt is assembled.
        """
        extras, missing = _entry_violations(building, set(REQUIRED_ARTIFACTS))
        if extras or missing:
            details = []
            if extras:
                details.append(f"unexpected entries (WAL/SHM/extra artifacts are rejected): {sorted(extras)}")
            if missing:
                details.append(f"missing entries: {sorted(missing)}")
            raise PublishError(
                f"staged generation top-level entries must be exactly {sorted(REQUIRED_ARTIFACTS)}: "
                + "; ".join(details)
            )
        artifacts: Dict[str, Dict[str, Any]] = {}
        for name, (rel, kind) in REQUIRED_ARTIFACTS.items():
            p = building / rel
            if p.is_symlink():
                raise UnsafePathError(f"staged artifact {name} is a symlink: {p}")
            if kind == "tree":
                if not p.is_dir():
                    raise PublishError(f"staged artifact {name} missing or not a directory: {p}")
                digest, count = _tree_digest(p)
                if count < 1:
                    raise PublishError(f"staged artifact {name} is empty (no regular files): {p}")
            else:
                if not p.is_file():
                    raise PublishError(f"staged artifact {name} missing or not a regular file: {p}")
                digest, count = _sha256_file(p), 1
            artifacts[name] = {"path": rel, "kind": kind, "count": count, "sha256": digest}
        return artifacts

    def _recompute_staged_semantics(
        self,
        building: Path,
        ident: Dict[str, Any],
        chroma_ev: Dict[str, Any],
        fts_ev: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recompute ALL staged semantic evidence; verify against caller input.

        THE production seam (pass-10 P0). Runs under publish's exclusive
        store lock and:

        1. Hashes the exact five staged artifacts (before-inspection set).
        2. Recomputes the canonical corpus manifest and requires it equals
           ``identity.corpus_manifest_sha256``.
        3. Inspects the staged Chroma collection IN A SHORT-LIVED CHILD
           PROCESS — the parent never opens a PersistentClient, so no parent
           handle can mutate or retain the sealed bytes. The parent waits
           (bounded) for successful exit and accepts ONLY validated JSON
           evidence; timeout/crash/malformed output is a VerificationError.
        4. Independently re-reads the sealed FTS artifact read-only and
           recomputes its row universe; validates the schema-v2 marker and
           its digest binding.
        5. Cross-binds every caller-declared backend field against the
           recomputed values, and common-universe parity across backends.
        6. Re-hashes the exact five staged artifacts and requires
           before == after. Any mutation (by the child or a straggler
           handle) rejects publication; ``current`` stays untouched.

        Receipt artifact hashes / backend generation IDs are derived by the
        caller ONLY from this final unchanged artifact set.
        """
        import subprocess  # local: builder path only
        import sys as _sys

        from mcp_server.fts5_index import (  # local: avoid import cycle
            is_credible_v2_marker,
            read_sealed_fts_row_universe,
        )

        before = self._hash_staged_artifacts(building)

        # -- (2) canonical corpus manifest binds identity -------------------
        sealed_entries = corpus_manifest_entries(building / CORPUS_ARTIFACT)
        if not sealed_entries:
            raise PublishError("staged corpus artifact is empty — refusing to seal")
        sealed_digest = corpus_manifest_digest(sealed_entries)
        if sealed_digest != ident["corpus_manifest_sha256"]:
            raise VerificationError(
                "publish recomputation: staged corpus manifest != identity.corpus_manifest_sha256 "
                f"({sealed_digest[:12]}... != {str(ident['corpus_manifest_sha256'])[:12]}...)"
            )

        # -- (3) Chroma evidence from a short-lived CHILD --------------------
        child_code = (
            "import json,sys\n"
            "from pathlib import Path\n"
            "root=Path(sys.argv[1]); chroma_dir=Path(sys.argv[2]); name=sys.argv[3]\n"
            "sys.path.insert(0,str(root))\n"
            "import chromadb\n"
            "from mcp_server.fts5_index import (capture_chunk_rows, capture_full_chunk_rows, "
            "compute_full_rows_digest, compute_rows_digest)\n"
            "client=chromadb.PersistentClient(path=str(chroma_dir), "
            "settings=chromadb.Settings(anonymized_telemetry=False))\n"
            "try:\n"
            "    col=client.get_collection(name=name)\n"
            "    full=compute_full_rows_digest(capture_full_chunk_rows(col))\n"
            "    common=compute_rows_digest(capture_chunk_rows(col))\n"
            "    print(json.dumps({'full_digest':full[0],'full_count':full[1],"
            "'common_digest':common[0],'unique_id_count':len({r[0] for r in capture_chunk_rows(col)}),"
            "'live_count':int(col.count())}), file=sys.__stdout__)\n"
            "finally:\n"
            "    col=None; client=None\n"
        )
        repo_root = Path(__file__).resolve().parents[1]
        # NEVER open the staged chroma_db directly, not even in the child:
        # copy the tree into a scratch directory beside the staging dir
        # (symlinks=True — real bytes, never hardlinks) and inspect the
        # COPY. Copy fidelity is verified against the before-inspection
        # artifact hash/count, and the context manager removes the scratch
        # on EVERY exit path. Backend generation ids stay bound to the
        # ORIGINAL ``before`` digests, and the final before/after check
        # below still guards the original staged bytes.
        import tempfile  # local: builder path only

        with tempfile.TemporaryDirectory(prefix=".inspect-chroma-", dir=building.parent) as scratch_dir:
            scratch_chroma = Path(scratch_dir) / CHROMA_ARTIFACT
            shutil.copytree(building / CHROMA_ARTIFACT, scratch_chroma, symlinks=True)
            copied_sha, copied_count = _tree_digest(scratch_chroma)
            if copied_sha != before[CHROMA_ARTIFACT]["sha256"] or copied_count != before[CHROMA_ARTIFACT]["count"]:
                raise VerificationError(
                    "publish recomputation: scratch chroma copy digest/count != staged "
                    f"artifact ({copied_sha[:12]}.../{copied_count} != "
                    f"{before[CHROMA_ARTIFACT]['sha256'][:12]}.../{before[CHROMA_ARTIFACT]['count']})"
                )
            try:
                proc = subprocess.run(
                    [
                        _sys.executable,
                        "-c",
                        child_code,
                        str(repo_root),
                        str(scratch_chroma),
                        str(chroma_ev["collection_name"]),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._CHILD_INSPECT_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as exc:
                raise VerificationError(
                    f"publish recomputation: chroma inspection child timed out "
                    f"(>{self._CHILD_INSPECT_TIMEOUT_S:.0f}s) — publication refused"
                ) from exc
            except OSError as exc:
                raise VerificationError(
                    f"publish recomputation: chroma inspection child could not start: {exc}"
                ) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-5:]
            raise VerificationError(
                "publish recomputation: chroma inspection child failed "
                f"(rc={proc.returncode}): {' | '.join(tail) if tail else 'no stderr'}"
            )
        try:
            child_json = json.loads((proc.stdout or "").strip())
        except ValueError as exc:
            raise VerificationError(
                "publish recomputation: chroma inspection child produced malformed JSON — publication refused"
            ) from exc
        if not isinstance(child_json, dict) or not all(
            isinstance(child_json.get(k), (int, str))
            for k in ("full_digest", "full_count", "common_digest", "unique_id_count", "live_count")
        ):
            raise VerificationError(
                "publish recomputation: chroma inspection child evidence has wrong shape — publication refused"
            )

        recomputed_full_digest = str(child_json["full_digest"])
        recomputed_full_count = int(child_json["full_count"])
        recomputed_common_digest = str(child_json["common_digest"])
        recomputed_unique_id_count = int(child_json["unique_id_count"])
        live_count = int(child_json["live_count"])

        # -- (4) FTS: independent read-only recompute ------------------------
        fts_live_digest, fts_live_count = read_sealed_fts_row_universe(building / FTS_ARTIFACT)
        try:
            marker = json.loads((building / FTS_STATE_ARTIFACT).read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            raise VerificationError(f"publish recomputation: FTS state marker unreadable/malformed: {exc}") from exc
        if not is_credible_v2_marker(marker, fts_live_count):
            raise VerificationError(
                "publish recomputation: FTS state marker is not a credible complete schema-v2 marker"
            )
        if marker.get("source_rows_sha256") != fts_live_digest:
            raise VerificationError(
                "publish recomputation: FTS marker digest != independently recomputed FTS row digest"
            )

        # -- (5) cross-bind caller evidence vs recomputed --------------------
        chroma_gid = hashlib.sha256(
            f"chroma:{before['chroma_db']['sha256']}:{recomputed_full_digest}".encode("utf-8")
        ).hexdigest()
        fts_gid = hashlib.sha256(f"fts5:{before[FTS_ARTIFACT]['sha256']}:{fts_live_digest}".encode("utf-8")).hexdigest()
        for field_name, caller_value, recomputed in (
            ("row_count", chroma_ev["row_count"], recomputed_full_count),
            ("unique_id_count", chroma_ev["unique_id_count"], recomputed_unique_id_count),
            ("hydrated_id_count", chroma_ev["hydrated_id_count"], recomputed_full_count),
            ("row_digest", chroma_ev["row_digest"], recomputed_full_digest),
            ("common_row_digest", chroma_ev["common_row_digest"], recomputed_common_digest),
            ("backend_generation_id", chroma_ev["backend_generation_id"], chroma_gid),
            ("row_count", fts_ev["row_count"], fts_live_count),
            ("row_digest", fts_ev["row_digest"], fts_live_digest),
            ("source_digest", fts_ev["source_digest"], fts_live_digest),
            ("verified_digest", fts_ev["verified_digest"], fts_live_digest),
            ("backend_generation_id", fts_ev["backend_generation_id"], fts_gid),
        ):
            if caller_value != recomputed:
                raise VerificationError(
                    f"publish recomputation: backend evidence {field_name} disagrees with staged bytes "
                    f"(caller {str(caller_value)[:12]}... != recomputed {str(recomputed)[:12]}...)"
                )
        if live_count != recomputed_full_count:
            raise VerificationError(
                f"publish recomputation: chroma collection.count() {live_count} != full-row universe {recomputed_full_count}"
            )
        if fts_live_digest != recomputed_common_digest or fts_live_count != recomputed_full_count:
            raise VerificationError("publish recomputation: FTS/common-Chroma parity violated on recomputed evidence")

        # -- (6) post-inspection artifact stability --------------------------
        after = self._hash_staged_artifacts(building)
        if after != before:
            changed = sorted(n for n in before if after.get(n, {}).get("sha256") != before[n]["sha256"])
            raise VerificationError(
                "publish recomputation: staged artifacts mutated during inspection "
                f"(before != after for {changed or 'entry set'}) — publication refused"
            )

        return {
            "artifacts": before,
            "chroma": {
                "row_count": recomputed_full_count,
                "unique_id_count": recomputed_unique_id_count,
                "hydrated_id_count": recomputed_full_count,
                "row_digest": recomputed_full_digest,
                "common_row_digest": recomputed_common_digest,
                "backend_generation_id": chroma_gid,
            },
            "fts5": {
                "row_count": fts_live_count,
                "row_digest": fts_live_digest,
                "source_digest": fts_live_digest,
                "verified_digest": fts_live_digest,
                "backend_generation_id": fts_gid,
            },
        }

    # ----------------------------------------------------------------- verify

    def verify_generation(
        self,
        generation_id: str,
        *,
        expected_identity: Optional[Dict[str, Any]] = None,
        expected_compatibility: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fully verify a published generation; returns the validated receipt.

        Runs under a SHARED store lock (creating nothing) with anchors
        re-checked. Checks: safe ID +
        containment; EXACT top-level entries (five artifacts + receipt, no
        extras — WAL/SHM rejected); receipt existence/size/JSON; exact schema
        with ``status=complete`` and canonical UTC ``created_at``; identity
        SHAs; compatibility binding; per-artifact existence, type,
        symlink/non-regular rejection, recomputed digest and count equality;
        backend evidence schemas, three-count and four-digest parity,
        collection binding; optional identity/compatibility equality gates.
        """
        gid = _require_safe_id(generation_id)
        with self._read_lock():
            return self._verify_generation_locked(
                gid, expected_identity=expected_identity, expected_compatibility=expected_compatibility
            )

    def _verify_generation_locked(
        self,
        gid: str,
        *,
        expected_identity: Optional[Dict[str, Any]] = None,
        expected_compatibility: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Verification body; caller holds the store lock."""
        gdir = self._resolve_generation_dir(gid)
        if not gdir.is_dir():
            raise VerificationError(f"generation directory missing: {gdir}")

        allowed = set(REQUIRED_ARTIFACTS) | {RECEIPT_FILENAME}
        extras, missing = _entry_violations(gdir, allowed)
        if extras or missing:
            details = []
            if extras:
                details.append(f"unexpected entries: {sorted(extras)}")
            if missing:
                details.append(f"missing entries: {sorted(missing)}")
            raise VerificationError(
                f"generation directory entries must be exactly {sorted(allowed)}: " + "; ".join(details)
            )

        receipt = self._read_receipt(gdir, gid)
        self._validate_receipt(receipt, gid)

        for name, meta in receipt["artifacts"].items():
            digest, count = self._artifact_on_disk(gdir, name, meta)
            if digest != meta["sha256"]:
                raise VerificationError(f"artifact {name} digest mismatch: receipt {meta['sha256']} != disk {digest}")
            if count != meta["count"]:
                raise VerificationError(f"artifact {name} count mismatch: receipt {meta['count']} != disk {count}")

        if expected_identity is not None:
            want = _validate_identity(expected_identity)
            if receipt["identity"] != want:
                raise VerificationError(f"identity mismatch: receipt {receipt['identity']} != expected {want}")
        if expected_compatibility is not None:
            want = _validate_compatibility(expected_compatibility)
            if receipt["compatibility"] != want:
                raise VerificationError(
                    f"compatibility mismatch: receipt {receipt['compatibility']} != expected {want}"
                )
        return receipt

    def _read_receipt(self, gdir: Path, gid: str) -> Dict[str, Any]:
        receipt_path = gdir / RECEIPT_FILENAME
        if receipt_path.is_symlink():
            raise UnsafePathError(f"receipt is a symlink: {receipt_path}")
        try:
            raw = _read_regular_file(receipt_path, _MAX_RECEIPT_BYTES)
        except FileNotFoundError as exc:
            raise VerificationError(f"generation receipt missing: {receipt_path}") from exc
        except UnsafePathError:
            raise
        except OSError as exc:
            raise VerificationError(f"generation receipt unreadable: {receipt_path}: {exc}") from exc
        if len(raw) > _MAX_RECEIPT_BYTES:
            raise VerificationError(f"generation receipt too large ({len(raw)} bytes): {receipt_path}")
        try:
            receipt = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise VerificationError(f"generation receipt is not valid JSON: {receipt_path}: {exc}") from exc
        return receipt

    @staticmethod
    def _validate_receipt(receipt: Any, gid: str) -> None:
        """Strict receipt schema check (exact key sets, no extra fields).

        v3 is the publishable/servable schema. v2 receipts fail here (they
        cannot carry the mandatory provenance block or the full identity
        set) — they remain INSPECTABLE through the lenient status reader
        only and must be rebuilt, never served/activated/rolled back.
        """
        if not isinstance(receipt, dict) or set(receipt) != set(_RECEIPT_KEYS):
            got = sorted(receipt) if isinstance(receipt, dict) else type(receipt).__name__
            raise VerificationError(f"receipt must be an object with exactly {sorted(_RECEIPT_KEYS)}, got: {got}")
        version = receipt["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != RECEIPT_SCHEMA_VERSION:
            raise VerificationError(f"unsupported receipt schema_version: {version!r} (want {RECEIPT_SCHEMA_VERSION})")
        if receipt["generation_id"] != gid:
            raise VerificationError(f"receipt generation_id {receipt['generation_id']!r} != directory {gid!r}")
        _parse_created_at(receipt["created_at"])  # canonical timezone-aware UTC ISO-8601
        if receipt["status"] != RECEIPT_STATUS_COMPLETE:
            raise VerificationError(f"receipt status must be {RECEIPT_STATUS_COMPLETE!r}, got {receipt['status']!r}")
        _validate_provenance(receipt["provenance"])
        ident = _validate_identity(receipt["identity"])
        compat = _validate_compatibility(receipt["compatibility"])
        # Cross-binding: the identity block and the compatibility object must
        # agree about the exact model artifact and chunking that built this
        # generation. A disagreement is rejected even though each is
        # individually schema-valid.
        identity_artifact = ident["model_artifact_sha256"] or ""
        if identity_artifact != compat["model_artifact_sha256"]:
            raise VerificationError(
                "identity/compatibility model_artifact_sha256 mismatch: "
                f"identity {identity_artifact[:12]}... != "
                f"compatibility {compat['model_artifact_sha256'][:12]}..."
            )
        compat_chunking = _stable_json_digest(
            {
                "embedding_model": compat["embedding_model"],
                "query_prefix": compat["query_prefix"],
                "passage_prefix": compat["passage_prefix"],
                "chunk_size": compat["chunk_size"],
                "chunk_overlap": compat["chunk_overlap"],
            }
        )
        identity_chunking = ident["chunking_sha256"] or ""
        if identity_chunking != compat_chunking:
            raise VerificationError(
                "identity/compatibility chunking mismatch: identity "
                f"{identity_chunking[:12]}... != compatibility-derived "
                f"{compat_chunking[:12]}..."
            )

        artifacts = receipt["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_ARTIFACTS):
            raise VerificationError(
                f"receipt artifacts must have exactly {sorted(REQUIRED_ARTIFACTS)}, got: "
                f"{sorted(artifacts) if isinstance(artifacts, dict) else type(artifacts).__name__}"
            )
        for name, (want_path, want_kind) in REQUIRED_ARTIFACTS.items():
            meta = artifacts[name]
            if not isinstance(meta, dict) or set(meta) != set(_ARTIFACT_KEYS):
                raise VerificationError(f"receipt artifact {name} must have exactly {sorted(_ARTIFACT_KEYS)}")
            if meta["path"] != want_path:
                raise VerificationError(f"receipt artifact {name} path must be {want_path!r}, got {meta['path']!r}")
            if meta["kind"] != want_kind:
                raise VerificationError(f"receipt artifact {name} kind must be {want_kind!r}, got {meta['kind']!r}")
            minimum = 1 if want_kind in _TREE_KINDS else 1
            _require_count(meta["count"], f"receipt artifact {name} count", minimum=minimum)
            _require_hex64(meta["sha256"], f"receipt artifact {name} sha256")

        backends = receipt["backends"]
        if not isinstance(backends, dict) or set(backends) != {"chroma", "fts5"}:
            raise VerificationError("receipt backends must have exactly ['chroma', 'fts5']")
        chroma_ev = _validate_chroma_evidence(backends["chroma"])
        fts_ev = _validate_fts_evidence(backends["fts5"])
        _require_backend_parity(chroma_ev, fts_ev)
        _bind_collection_name(compat, chroma_ev)

    def _artifact_on_disk(self, gdir: Path, name: str, meta: Dict[str, Any]) -> Tuple[str, int]:
        """Recompute an artifact's (digest, count) from disk with strict typing."""
        p = gdir / meta["path"]
        if p.is_symlink():
            raise UnsafePathError(f"artifact {name} is a symlink: {p}")
        if meta["kind"] == "tree":
            if not p.is_dir():
                raise VerificationError(f"artifact {name} missing or not a directory: {p}")
            return _tree_digest(p)
        if not p.is_file():
            raise VerificationError(f"artifact {name} missing or not a regular file: {p}")
        return _sha256_file(p), 1

    # --------------------------------------------------------------- pointer

    def require_current(
        self,
        *,
        expected_identity: Optional[Dict[str, Any]] = None,
        expected_compatibility: Optional[Dict[str, Any]] = None,
    ) -> CurrentGeneration:
        """Strict production read of ``current`` — ONE lock, FULL verification.

        Runs under a single SHARED store lock (creating nothing) with anchors
        re-checked, and raises precise errors instead of returning ``None``:

        - :class:`PointerError` — pointer missing/unreadable/oversized/malformed
          or naming a generation directory that does not exist.
        - :class:`UnsafePathError` — symlinked pointer, receipt, or generation.
        - :class:`VerificationError` — receipt SHA mismatch, receipt schema,
          artifact digests/counts, backend parity, or an expected-identity /
          expected-compatibility gate that does not hold.

        ``expected_identity`` accepts EITHER the pointer CAS identity
        ``{"generation_id", "receipt_sha256"}`` (checked against the pointer
        bytes read here) OR the Phase-A receipt identity block
        ``{corpus_manifest_sha256, config_sha256, code_sha256}`` (checked by
        full verification). ``expected_compatibility`` is the exact 10-key
        compatibility binding the current generation must carry.
        """
        with self._read_lock():
            if self.current_path.is_symlink():
                raise UnsafePathError(f"current pointer is a symlink: {self.current_path}")
            try:
                raw = _read_regular_file(self.current_path, _MAX_POINTER_BYTES)
            except FileNotFoundError as exc:
                raise PointerError(f"current pointer is missing: {self.current_path}") from exc
            except OSError as exc:
                raise PointerError(f"current pointer unreadable: {self.current_path}: {exc}") from exc
            if len(raw) > _MAX_POINTER_BYTES:
                raise PointerError(f"current pointer exceeds {_MAX_POINTER_BYTES} bytes: {self.current_path}")
            pointer = _parse_pointer(raw)
            gid = pointer["generation_id"]
            gdir = self._resolve_generation_dir(gid)
            if not gdir.is_dir():
                raise PointerError(f"current pointer names missing generation {gid!r}: {gdir}")
            receipt_path = gdir / RECEIPT_FILENAME
            if receipt_path.is_symlink():
                raise UnsafePathError(f"receipt is a symlink: {receipt_path}")
            try:
                payload = _read_regular_file(receipt_path, _MAX_RECEIPT_BYTES)
            except OSError as exc:
                raise VerificationError(f"generation receipt unreadable: {receipt_path}: {exc}") from exc
            if len(payload) > _MAX_RECEIPT_BYTES:
                raise VerificationError(f"generation receipt too large ({len(payload)} bytes): {receipt_path}")
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != pointer["receipt_sha256"]:
                raise VerificationError(
                    f"current pointer receipt SHA mismatch: pointer {pointer['receipt_sha256'][:12]}... "
                    f"!= disk {actual_sha[:12]}... for generation {gid!r}"
                )
            receipt = json.loads(payload.decode("utf-8"))
            self._validate_receipt(receipt, gid)
            expected_block: Optional[Dict[str, Any]] = None
            if expected_identity is not None:
                if isinstance(expected_identity, dict) and set(expected_identity) == set(_POINTER_KEYS):
                    want_gid = expected_identity.get("generation_id")
                    want_sha = expected_identity.get("receipt_sha256")
                    if not isinstance(want_gid, str) or not _SAFE_ID_RE.match(want_gid):
                        raise VerificationError(f"expected_identity has unsafe generation_id: {want_gid!r}")
                    if not isinstance(want_sha, str) or not _HEX64_RE.match(want_sha):
                        raise VerificationError(f"expected_identity has malformed receipt_sha256: {want_sha!r}")
                    if expected_identity != pointer:
                        raise VerificationError(f"current pointer identity mismatch: {expected_identity} != {pointer}")
                else:
                    expected_block = expected_identity
            # Full verification: exact entries, artifact digests/counts,
            # backend parity — via the same code path as verify_generation.
            self._verify_generation_locked(
                gid, expected_identity=expected_block, expected_compatibility=expected_compatibility
            )
            return CurrentGeneration(gid, pointer["receipt_sha256"], gdir, receipt)

    def resolve_current(self) -> Optional[CurrentGeneration]:
        """Optional-flavored :meth:`require_current` (compatibility API).

        Runs the identical single-lock fully-verified read and returns
        ``None`` — never partially trusted state — for every failure mode
        listed there.
        """
        try:
            return self.require_current()
        except (OSError, ValueError, GenerationError):
            return None

    def current_identity(self) -> Optional[Dict[str, str]]:
        """Strictly read the pointer's CAS identity; None only when absent.

        Runs under a SHARED store lock (creating nothing); a symlinked
        ``current`` is REJECTED
        (:class:`UnsafePathError`), not silently treated as absent.
        Convenience for Phase B: pass the result as ``expected_current``.
        """
        with self._read_lock():
            if self.current_path.is_symlink():
                raise UnsafePathError(f"current pointer is a symlink: {self.current_path}")
            try:
                raw = _read_regular_file(self.current_path, _MAX_POINTER_BYTES)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise PointerError(f"current pointer unreadable: {self.current_path}: {exc}") from exc
            if len(raw) > _MAX_POINTER_BYTES:
                raise PointerError(f"current pointer exceeds {_MAX_POINTER_BYTES} bytes")
            return _parse_pointer(raw)

    def _read_pointer_strict(self) -> Optional[Dict[str, str]]:
        """Read the pointer (locked callers only); None when it does not exist."""
        if self.current_path.is_symlink():
            raise UnsafePathError(f"current pointer is a symlink: {self.current_path}")
        try:
            raw = _read_regular_file(self.current_path, _MAX_POINTER_BYTES)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise PointerError(f"current pointer unreadable: {self.current_path}: {exc}") from exc
        if len(raw) > _MAX_POINTER_BYTES:
            raise PointerError(f"current pointer exceeds {_MAX_POINTER_BYTES} bytes")
        return _parse_pointer(raw)

    @staticmethod
    def _validate_expected_identity(
        expected_current: Union[None, Dict[str, str], str],
    ) -> Dict[str, str]:
        """Normalize an EXPLICIT pointer expectation (fail-closed on None).

        Three distinguishable meanings (P1-6):

        * dict — the exact pointer identity that must currently hold;
        * :data:`EXPECTED_CURRENT_ABSENT` — the pointer was checked and MUST
          be absent (CAS-conflict recovery with a lost pointer);
        * ``None`` — NO expectation was captured. REJECTED: an unchecked
          CAS can clobber a concurrent activation, so callers must always
          pass either the captured identity or the explicit absent-sentinel.
        """
        if expected_current is None:
            raise CurrentConflictError(
                "expected_current is required: pass the captured pointer identity, or the "
                f"explicit sentinel {EXPECTED_CURRENT_ABSENT!r} when current was checked absent"
            )
        if expected_current == EXPECTED_CURRENT_ABSENT:
            return EXPECTED_CURRENT_ABSENT  # type: ignore[return-value]  # sentinel, not an identity
        if not isinstance(expected_current, dict) or set(expected_current) != set(_POINTER_KEYS):
            raise CurrentConflictError(
                f"expected_current must be {EXPECTED_CURRENT_ABSENT!r} or a dict with exactly {sorted(_POINTER_KEYS)}"
            )
        gid = expected_current["generation_id"]
        sha = expected_current["receipt_sha256"]
        if not isinstance(gid, str) or not _SAFE_ID_RE.match(gid):
            raise CurrentConflictError(f"expected_current has unsafe generation_id: {gid!r}")
        if not isinstance(sha, str) or not _HEX64_RE.match(sha):
            raise CurrentConflictError(f"expected_current has malformed receipt_sha256: {sha!r}")
        return {"generation_id": gid, "receipt_sha256": sha}

    def _cas_current_locked(self, gid: str, receipt_sha256: str, expected: Dict[str, str]) -> None:
        """Identity-bound compare-and-swap of ``current``. Caller holds the lock.

        ``expected`` is ALWAYS an explicit expectation (never None): either a
        dict binding generation_id AND receipt SHA (ABA/lost-update safe), or
        the :data:`EXPECTED_CURRENT_ABSENT` sentinel requiring that no
        pointer exists. A malformed existing pointer raises
        :class:`PointerError` rather than being clobbered. If the atomic
        replace succeeds but the follow-up fsync of the root fails,
        :class:`CommitStateUncertainError` carries the intended full pointer
        identity with ``may_have_committed=True``.
        """
        current = self._read_pointer_strict()
        if expected == EXPECTED_CURRENT_ABSENT:
            if current is not None:
                raise CurrentConflictError(
                    f"expected no current pointer but it names {current['generation_id']!r} "
                    f"(receipt {current['receipt_sha256'][:12]}...)"
                )
        else:
            if current is None:
                raise CurrentConflictError(f"expected current pointer {expected['generation_id']!r}, but none exists")
            if current != expected:
                raise CurrentConflictError(
                    f"CAS mismatch: expected {expected['generation_id']!r}/"
                    f"{expected['receipt_sha256'][:12]}..., "
                    f"current is {current['generation_id']!r}/{current['receipt_sha256'][:12]}..."
                )

        payload = (
            json.dumps({"generation_id": gid, "receipt_sha256": receipt_sha256}, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")

        tmp_path: Optional[Path] = None
        try:
            for _ in range(8):
                candidate = (
                    self.root / f"{POINTER_TMP_PREFIX}{os.getpid()}.{next(self._tmp_counter)}{POINTER_TMP_SUFFIX}"
                )
                try:
                    _write_file_exclusive(candidate, payload)
                except FileExistsError:
                    continue
                tmp_path = candidate
                break
            if tmp_path is None:  # pragma: no cover - pathological tmp collisions
                raise PublishError("could not allocate a pointer temp file")
            _atomic_replace(tmp_path, self.current_path)
            tmp_path = None
            try:
                _fsync_dir(self.root)
            except DurabilityError as exc:
                raise CommitStateUncertainError(
                    f"current pointer was replaced but fsync of the store root failed; "
                    f"commit state uncertain (may_have_committed=True) for generation {gid!r}",
                    generation_id=gid,
                    receipt_sha256=receipt_sha256,
                ) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------- activate/rollback

    def activate(
        self,
        generation_id: str,
        *,
        expected_current: Union[None, Dict[str, str], str] = None,
        expected_identity: Optional[Dict[str, Any]] = None,
        expected_compatibility: Optional[Dict[str, Any]] = None,
    ) -> ActivationResult:
        """Verify an existing generation, then CAS-switch ``current`` to it.

        ONE lock is held across verification AND the CAS: anchors re-checked,
        full verification (schema, compatibility, artifacts, parity), receipt
        SHA re-read from the verified bytes, then the identity-bound swap.
        Failed verification leaves the pointer byte-identical.

        ``expected_current=EXPECTED_CURRENT_ABSENT`` activates a sealed
        generation when ``current`` is absent (CAS-conflict recovery after a
        lost pointer); a dict binds the exact identity; ``None`` (no
        expectation captured) is REJECTED fail-closed.
        """
        gid = _require_safe_id(generation_id)
        expected = self._validate_expected_identity(expected_current)
        if expected_identity is not None:
            _validate_identity(expected_identity)
        if expected_compatibility is not None:
            _validate_compatibility(expected_compatibility)
        with self._exclusive_lock():
            receipt = self._verify_generation_locked(
                gid, expected_identity=expected_identity, expected_compatibility=expected_compatibility
            )
            receipt_sha256 = _sha256_file(self._resolve_generation_dir(gid) / RECEIPT_FILENAME)
            self._cas_current_locked(gid, receipt_sha256, expected)
        return ActivationResult(gid, receipt_sha256, receipt, restart_required=True)

    def rollback(
        self,
        generation_id: str,
        *,
        expected_current: Dict[str, str],
        expected_identity: Optional[Dict[str, Any]] = None,
        expected_compatibility: Optional[Dict[str, Any]] = None,
    ) -> ActivationResult:
        """Roll ``current`` back to a previously published generation.

        ``expected_current`` (the full CAS identity being rolled back FROM) is
        mandatory and MUST be the exact dict — rollback never creates a
        pointer, never accepts the absent-sentinel, and aborts if the pointer
        moved concurrently. The target is fully verified first — a partial,
        incompatible, or v2-schema generation is rejected and the pointer is
        left byte-identical.
        """
        if not isinstance(expected_current, dict) or expected_current == EXPECTED_CURRENT_ABSENT:
            raise CurrentConflictError(
                "rollback requires the exact expected_current dict {'generation_id','receipt_sha256'}"
            )
        return self.activate(
            generation_id,
            expected_current=expected_current,
            expected_identity=expected_identity,
            expected_compatibility=expected_compatibility,
        )
