"""Offline generation lifecycle CLI (schema v3, spec E).

Commands
--------
status    Show the current pointer identity + receipt summary (fail-closed).
build     Build + activate a NEW immutable generation from the live corpus.
activate  CAS-switch ``current`` to an already-published generation.
rollback  CAS-switch ``current`` back to a previously published generation.

Every mutating command runs OFFLINE: it never starts the MCP server, never
touches a sealed generation's bytes, and reports ``restart_required=true``
because the running server pins one generation for its process lifetime.

Build pipeline (fail-closed at each boundary):

 1. require ``indexing.mode=versioned`` + exact compatibility inputs
    (materialized, symlink-free model artifacts; RECORD + lock evidence);
 2. capture the current CAS identity — a captured ``None`` becomes the
    EXPLICIT ``EXPECTED_CURRENT_ABSENT`` sentinel (never unchecked None);
 3. capture the source corpus manifest BEFORE (live tree, shared selector);
 4. ``GenerationStore.begin_build`` -> ``root/.building-<id>``;
 5. copy exactly the captured entries into ``<staging>/corpus`` (never
    re-walk: the copied set IS the captured set);
 6. verify ``sealed == live-before`` (sorted path+sha256 parity), then run
    population in an ISOLATED CHILD PROCESS so every Chroma/orchestrator/FTS
    writer handle dies before the parent seals anything (no second FTS
    writer can ever exist);
 7. parent re-derives backend evidence from the STAGED ARTIFACTS read-only
    (never trusts child-reported counts/digests): Chroma row universe via a
    read-only client, FTS rows via the sealed reader, explicit backend
    generation IDs bound into the evidence;
 8. synchronous FTS5 build in the child, sealed via
    ``Fts5LexicalIndex.seal_for_publication`` (checkpoint + DELETE journal +
    close); WAL/SHM sidecars make the publish boundary reject loudly;
 9. metadata artifact (POSIX-relative sources) + credible v2 FTS marker;
10. re-walk the LIVE source tree (live-after) and require
    ``live-before == sealed == live-after`` — any drift aborts staging only
    and leaves ``current`` byte-identical (TOCTOU bounded);
11. assemble the canonical v3 identity via ``generations.generation_identity``
    (the ONE shared builder), add provenance, and publish through the store
    (full receipt validation + durability + CAS). Build means
    BUILD+ACTIVATE, so success reports ``restart_required=true``;
12. any pre-publication failure cleans ONLY ``.building-<id>``; a CAS
    conflict preserves the sealed generation and prints the exact
    ``activate`` recovery command; sealed generations are never deleted.

Dependency evidence stays fail-closed against the repo's genuine
``requirements.lock`` (pip-compile ``--generate-hashes`` output, shipped in
the repository, the sdist, and the wheel's ``mcp_server/data/``). Builds
fail with ``dependency_unverifiable`` when the lock is missing, malformed,
or disagrees with the installed environment (unpinned default dependency,
installed-version drift, or installed knowledge-rag version != mcp_server
``__version__``). No hashes are invented.

Chroma immutability is an application-level boundary plus receipt
verification; no OS-level read-only Chroma client is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp_server.config import Config, config  # noqa: E402
from mcp_server.fts5_index import (  # noqa: E402
    capture_chunk_rows,
    capture_full_chunk_rows,
    compute_full_rows_digest,
    compute_rows_digest,
    is_credible_v2_marker,
    read_sealed_fts_row_universe,
)
from mcp_server.generations import (  # noqa: E402
    BUILDING_PREFIX,
    CHROMA_ARTIFACT,
    CORPUS_ARTIFACT,
    EXPECTED_CURRENT_ABSENT,
    FTS_ARTIFACT,
    FTS_STATE_ARTIFACT,
    METADATA_ARTIFACT,
    CommitStateUncertainError,
    CurrentConflictError,
    DependencyUnverifiableError,
    GenerationError,
    GenerationStore,
    corpus_manifest_digest,
    corpus_manifest_entries,
    generation_identity,
    inspect_current_receipt,
    vault_head_or_none,
)


def _print(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_generation_config() -> Config:
    """Import-time ``config`` may be legacy; require the versioned Config."""
    if config.index_mode != "versioned":
        raise SystemExit(
            f"[GENERATION] indexing.mode must be 'versioned' for generation operations (current: {config.index_mode!r})"
        )
    return config


# ---------------------------------------------------------------------------
# Shared corpus selection (ONE selector — generations.corpus_manifest_entries)
# ---------------------------------------------------------------------------


def _corpus_selector_kwargs() -> Dict[str, Any]:
    """Effective supported formats/exclusions for the shared selector."""
    return {
        "supported_suffixes": set(config.supported_formats or []),
        "exclude_patterns": list(getattr(config, "exclude_patterns", None) or []),
    }


def _live_manifest() -> List[Tuple[str, str]]:
    """Sorted (rel_posix, sha256) entries of the LIVE source tree."""
    src = Path(str(config.source_documents_dir))
    return corpus_manifest_entries(src, **_corpus_selector_kwargs())


def _sealed_manifest(staging: Path) -> List[Tuple[str, str]]:
    return corpus_manifest_entries(staging / CORPUS_ARTIFACT, **_corpus_selector_kwargs())


def _copy_captured_entries(entries: List[Tuple[str, str]], source: Path, dest: Path) -> int:
    """Copy EXACTLY the captured manifest entries (no re-walk).

    Each source file is re-verified against its captured sha256 immediately
    before the copy; any mismatch aborts (bounded TOCTOU). Symlinks and
    special files are rejected.
    """
    import stat as stat_module

    copied = 0
    for rel, sha in entries:
        src = source / rel
        st = src.lstat()
        if stat_module.S_ISLNK(st.st_mode) or not stat_module.S_ISREG(st.st_mode):
            raise SystemExit(f"[GENERATION] non-regular corpus entry: {src}")
        if _sha256_file_str(src) != sha:
            raise SystemExit(f"[GENERATION] source file changed during build (bounded TOCTOU): {rel}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied += 1
    return copied


def _sha256_file_str(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Isolated child population (all writer handles die with the child)
# ---------------------------------------------------------------------------

_CHROMA_SQLITE_SIDECAR_NAMES = ("chroma_db/chroma.sqlite3-wal", "chroma_db/chroma.sqlite3-shm")


def _teardown_chroma_sqlite_sidecars(
    staging: Path,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
) -> Dict[str, Any]:
    """Deterministic chroma teardown squeeze (runs in the population child).

    Called AFTER ``orch.close(strict=True)`` and BEFORE the sidecar sweep.
    Sequence (criteria kr-teardown-fix):

    1. ``gc.collect()`` — drop any lingering SharedSystemClient references so
       the shared System's sqlite connections can actually die;
    2. if the chroma WAL/SHM sidecars still exist, a short-lived
       ``sqlite3.connect`` on the staging chroma db issues
       ``PRAGMA wal_checkpoint(TRUNCATE)`` and closes. Population is
       complete, so there are no writers; if some foreign connection is
       still alive the connect attempt hits its 0.2 s busy timeout, the
       checkpoint is not taken, and the squeeze yields (the sweep then
       honestly fails the build with exit 14);
    3. bounded wait (<= ``timeout`` seconds, ~``poll_interval`` step) for the
       sidecars to disappear; on disappearance — proceed immediately.

    ``clear_system_cache`` is deliberately NOT used: in pinned chromadb
    1.5.9 it swaps the cache for an empty dict WITHOUT ``system.stop()``,
    orphaning a live System and making teardown worse
    (see ``shared_system_client.py:126-129``).

    Returns ``{"cleared": bool, "remaining": [...], "waited_s": float}``;
    ``cleared=False`` means the sidecars survived and the sweep is expected
    to fail the build — this function never weakens the gate itself.
    """
    import gc
    import sqlite3

    sidecar_paths = [staging / name for name in _CHROMA_SQLITE_SIDECAR_NAMES]

    def _remaining() -> List[str]:
        return [p.name for p in sidecar_paths if p.exists()]

    gc.collect()

    if _remaining():
        db_path = staging / "chroma_db" / "chroma.sqlite3"
        if db_path.exists():
            try:
                # timeout=0.2: never wait out a foreign holder's transaction
                # (default 5s busy timeout would stall the squeeze); busy
                # yields after 0.2 s, only the bounded-wait loop waits (P3).
                conn = sqlite3.connect(str(db_path), timeout=0.2)
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    conn.close()
            except sqlite3.Error:
                # Busy/locked foreign connection: yield silently — the sweep
                # remains the honest gate and fails the build (exit 14).
                pass

    started = time.monotonic()
    deadline = started + timeout
    while True:
        remaining = _remaining()
        if not remaining:
            return {"cleared": True, "remaining": [], "waited_s": round(time.monotonic() - started, 3)}
        if time.monotonic() >= deadline:
            return {"cleared": False, "remaining": remaining, "waited_s": round(time.monotonic() - started, 3)}
        time.sleep(poll_interval)


_CHILD_SCRIPT = r"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
STAGING = Path(sys.argv[2])
GID = sys.argv[3]

sys.path.insert(0, str(ROOT))

from mcp_server import config as config_module

cfg = config_module.config
if cfg.index_mode != "versioned":
    print("[CHILD] config is not versioned", file=sys.stderr)
    sys.exit(10)


class _StagingGeneration:
    def __init__(self, staging, gid):
        self.generation_id = gid
        self.receipt_sha256 = ""
        self.generation_dir = staging
        self.receipt = {}


cfg.bind_generation(_StagingGeneration(STAGING, GID), building=True)

from mcp_server.server import KnowledgeOrchestrator

orch = KnowledgeOrchestrator()
stats = orch.index_all(force=True)
errors = int(stats.get("errors", 0))
docs = int(stats.get("indexed", 0))
chunks = int(stats.get("chunks_added", 0))

# Synchronous FTS build inside the SAME child (one FTS writer only).
from mcp_server.fts5_index import Fts5LexicalIndex
from mcp_server.generations import FTS_ARTIFACT, FTS_STATE_ARTIFACT
from mcp_server.fts5_index import capture_chunk_rows, compute_rows_digest

rows = capture_chunk_rows(orch.collection)
row_digest, row_count = compute_rows_digest(rows)
# strict=True: an uncheckpointable WAL/close failure FAILS the build — the
# staging tree must never proceed to seal/publish with an open WAL.
orch.close(strict=True)  # exactly ONE FTS writer: the explicit one below
fts = Fts5LexicalIndex(
    db_path=STAGING / FTS_ARTIFACT, state_path=STAGING / FTS_STATE_ARTIFACT
)
try:
    result = fts.rebuild_content_bound(rows)
    if result.get("status") != "complete":
        print(f"[CHILD] FTS rebuild status={result.get('status')}", file=sys.stderr)
        sys.exit(11)
    if result.get("verified_fts_rows_sha256") != row_digest or int(
        result.get("docs_indexed", -1)
    ) != int(row_count):
        print("[CHILD] FTS/chroma parity failed", file=sys.stderr)
        sys.exit(12)
    fts.seal_for_publication()
finally:
    fts.close()

# Deterministic handle teardown (P0 #2): the orchestrator and the explicit
# FTS writer are closed; fail the build if any SQLite sidecar remains in the
# staging tree — publish() would reject it anyway, but failing here names
# the offending file for the operator. The sweep rejects EVERY canonical
# suffix generations.py enforces (-wal/-shm/-journal and .wal/.shm/.journal),
# including names like chroma.sqlite3-wal.
from mcp_server.generation_cli import _teardown_chroma_sqlite_sidecars

_squeeze = _teardown_chroma_sqlite_sidecars(STAGING)
if not _squeeze["cleared"]:
    print(
        f"[CHILD] chroma sidecar squeeze yielded: {_squeeze['remaining']} "
        f"(waited {_squeeze['waited_s']}s)",
        file=sys.stderr,
    )
_SQLITE_SIDECAR_SUFFIXES = ("-", ".")
sidecars = [
    str(p.relative_to(STAGING))
    for p in STAGING.rglob("*")
    if any(
        p.name.endswith(sep + ext)
        for sep in _SQLITE_SIDECAR_SUFFIXES
        for ext in ("wal", "shm", "journal")
    )
]
if sidecars:
    print(f"[CHILD] backend sidecars remain after close: {sidecars}", file=sys.stderr)
    sys.exit(14)

# Protocol JSON explicitly targets sys.__stdout__: importing mcp_server
# redirects sys.stdout to stderr (MCP stdio safety), which would send this
# payload to stderr and leave the parent without population stats.
print(
    json.dumps(
        {
            "errors": errors,
            "docs": docs,
            "chunks": chunks,
            "row_count": row_count,
            "row_digest": row_digest,
        }
    ),
    file=sys.__stdout__,
)
sys.exit(0 if (errors == 0 and docs > 0 and chunks > 0) else 13)
"""


def _run_population_child(staging: Path, gid: str) -> Dict[str, int]:
    """Run corpus population in an isolated child process.

    ALL Chroma/orchestrator/FTS writer handles are created and destroyed
    inside the child; the parent never opens a writer handle on staging.
    """
    cmd = [sys.executable, "-c", _CHILD_SCRIPT, str(_ROOT), str(staging), gid]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _print(f"[GENERATION] population child failed (rc={proc.returncode}):")
        for line in (proc.stderr or "").strip().splitlines()[-20:]:
            _print(f"  {line}")
        raise SystemExit(f"[GENERATION] population child exited {proc.returncode}")
    stats: Dict[str, int] = {}
    for line in (proc.stdout or "").strip().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            stats = {k: int(v) for k, v in payload.items() if isinstance(v, (int, float))}
    if not stats:
        raise SystemExit("[GENERATION] population child produced no stats")
    return stats


# ---------------------------------------------------------------------------
# Parent-side, read-only evidence derivation from the STAGED artifacts
# ---------------------------------------------------------------------------


def _chroma_evidence_from_staging(staging: Path) -> Dict[str, Any]:
    """Derive Chroma evidence from a byte-verified SCRATCH COPY of the staged
    chroma_db — never by opening the original staged tree itself.

    PersistentClient opens its sqlite tree read-WRITE at the filesystem
    level; opening the ORIGINAL staged chroma_db "read-only" is impossible
    with it and risks materializing WAL/SHM sidecars or touching bytes in
    the tree that is about to be sealed. The parent therefore:

    1. snapshots the staged chroma_db tree digest (the ORIGINAL bytes);
    2. copies the tree to a temporary scratch directory;
    3. verifies the copy is byte-identical to the snapshot;
    4. opens the COPY with a persistent client, recomputes the complete
       and common row universes, and EXPLICITLY CLOSES the client;
    5. re-verifies the ORIGINAL staged tree digest is UNCHANGED (the
       original must remain byte-identical before and after evidence
       derivation) and binds ``backend_generation_id`` to the ORIGINAL
       artifact bytes.

    Never trusts child-reported counts/digests — the parent recomputes
    everything. The canonical Chroma digest covers the COMPLETE row
    universe (ids, documents, packed embeddings, normalized full retrieval
    metadata); the COMMON digest (id, document, filename, category) is
    computed separately for FTS parity.
    """
    import shutil
    import tempfile

    import chromadb

    from mcp_server.generations import digest_tree

    db_dir = staging / CHROMA_ARTIFACT
    original_digest, _orig_count = digest_tree(db_dir)
    with tempfile.TemporaryDirectory(prefix="kr-chroma-evidence-") as scratch_root:
        scratch_db = Path(scratch_root) / CHROMA_ARTIFACT
        shutil.copytree(db_dir, scratch_db)
        copy_digest, _copy_count = digest_tree(scratch_db)
        if copy_digest != original_digest:
            raise SystemExit(
                "[GENERATION] chroma scratch copy is not byte-identical to the staged tree — refusing to seal"
            )
        client = chromadb.PersistentClient(path=str(scratch_db), settings=chromadb.Settings(anonymized_telemetry=False))
        try:
            collection = client.get_collection(config.collection_name)
            full_rows = capture_full_chunk_rows(collection)
            common_rows = capture_chunk_rows(collection)
        finally:
            # Explicit close: release the shared System reference and the
            # database connections before the scratch tree vanishes.
            client.close()
    # The ORIGINAL staged tree must be byte-identical before and after.
    post_digest, _post_count = digest_tree(db_dir)
    if post_digest != original_digest:
        raise SystemExit("[GENERATION] staged chroma_db changed during evidence derivation — refusing to seal")
    full_digest, full_count = compute_full_rows_digest(full_rows)
    common_digest, _common_count = compute_rows_digest(common_rows)
    unique_ids = {row[0] for row in common_rows}
    if len(unique_ids) != full_count:
        raise SystemExit(f"[GENERATION] chroma id universe not unique ({len(unique_ids)} != {full_count})")
    return {
        "collection_name": config.collection_name,
        "row_count": full_count,
        "unique_id_count": len(unique_ids),
        "hydrated_id_count": full_count,
        "row_digest": full_digest,
        "common_row_digest": common_digest,
        "backend_generation_id": _backend_generation_id("chroma", staging, full_digest),
    }


def _fts_evidence_from_staging(staging: Path, chroma_common_digest: str, row_count: int) -> Dict[str, Any]:
    """Derive FTS evidence from the sealed database via the read-only reader.

    Parity compares ONLY the genuinely common logical row fields — the
    (id, document, filename, category) universe Chroma and FTS5 both store —
    never the complete per-backend digests, which are intentionally unlike
    (Chroma's includes embeddings + full metadata).
    """
    db_path = staging / FTS_ARTIFACT
    fts_digest, fts_count = read_sealed_fts_row_universe(db_path)
    if fts_digest != chroma_common_digest or fts_count != row_count:
        raise SystemExit("[GENERATION] sealed FTS rows disagree with the common Chroma row universe — refusing to seal")
    state = json.loads((staging / FTS_STATE_ARTIFACT).read_text(encoding="utf-8"))
    if not is_credible_v2_marker(state, fts_count):
        raise SystemExit("[GENERATION] FTS state marker is not a credible complete schema-v2 marker")
    return {
        "schema_version": 2,
        "status": "complete",
        "row_count": fts_count,
        "row_digest": fts_digest,
        "source_digest": fts_digest,
        "verified_digest": fts_digest,
        "backend_generation_id": _backend_generation_id("fts5", staging, fts_digest),
    }


def _backend_generation_id(kind: str, staging: Path, digest: str) -> str:
    """Explicit per-backend generation id bound into the evidence.

    Deterministic from the STAGED artifact digest so the sealed bytes and the
    receipt agree; both backends additionally carry the row digest. The
    digest function matches the receipt artifact hash EXACTLY: the canonical
    tree digest for the chroma_db DIRECTORY, the canonical FILE digest for
    the regular-file FTS artifact — never ``digest_tree`` on a file, whose
    semantics differ (GenerationStore.publish recomputes and compares these).
    """
    from mcp_server.generations import _sha256_file, digest_tree

    artifact = staging / (CHROMA_ARTIFACT if kind == "chroma" else FTS_ARTIFACT)
    if kind == "chroma":
        artifact_sha, _count = digest_tree(artifact)
    else:
        artifact_sha = _sha256_file(artifact)
    return hashlib.sha256(f"{kind}:{artifact_sha}:{digest}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------


def _build_command(args: argparse.Namespace) -> int:
    _load_generation_config()
    store = GenerationStore(config.data_dir, create=True)

    from mcp_server.generations import new_generation_id

    gid = args.generation_id or new_generation_id()

    captured = store.current_identity()
    expected_current = captured if captured is not None else EXPECTED_CURRENT_ABSENT

    started = time.time()
    _print(f"[GENERATION] building {gid} (mode=versioned, restart required after activation)")

    # Source-before (live) — the controlling parity input.
    live_before = _live_manifest()
    if not live_before:
        raise SystemExit(
            f"[GENERATION] no supported corpus files under {config.source_documents_dir} — "
            "refusing to seal an empty corpus"
        )

    staging = store.begin_build(gid)
    try:
        # -- corpus: copy exactly the captured entries ----------------------
        corpus_dest = staging / CORPUS_ARTIFACT
        corpus_dest.mkdir(parents=True, exist_ok=True)
        copied = _copy_captured_entries(live_before, Path(str(config.source_documents_dir)), corpus_dest)

        sealed = _sealed_manifest(staging)
        if sealed != live_before:
            raise SystemExit("[GENERATION] sealed corpus != live-before manifest — aborting (parity)")
        corpus_sha = corpus_manifest_digest(sealed)
        _print(f"[GENERATION] corpus sealed: {copied} files (sha={corpus_sha[:12]}...)")

        # -- BuildInputSnapshot (P0 #1): EVERY controlling identity captured
        # BEFORE population. The child indexes exactly these sealed corpus
        # bytes with exactly this configuration; the receipt may never bind
        # later bytes than those used to build. Recomputed and compared
        # post-population below.
        identity_before = generation_identity(config, corpus_manifest_sha256=corpus_sha)
        compat_before = config.generation_compatibility()
        provenance_before = {"vault_head": vault_head_or_none(Path(str(config.source_documents_dir)))}
        _print(
            f"[GENERATION] build-input snapshot bound (identity sha={identity_before['corpus_manifest_sha256'][:12]}..., "
            f"code={identity_before['code_sha256'][:12]}..., lock={identity_before['dependency_lock_sha256'][:12]}...)"
        )

        # -- population (isolated child; all writer handles die) ------------
        pop = _run_population_child(staging, gid)
        _print(
            f"[GENERATION] populated (child): docs={pop.get('docs')} "
            f"chunks={pop.get('chunks')} rows={pop.get('row_count')}"
        )

        # -- parent-derived backend evidence (read-only, from staged bytes) --
        chroma_evidence = _chroma_evidence_from_staging(staging)
        fts_evidence = _fts_evidence_from_staging(
            staging, chroma_evidence["common_row_digest"], chroma_evidence["row_count"]
        )

        # -- metadata artifact sanity ---------------------------------------
        metadata_path = staging / METADATA_ARTIFACT
        if not metadata_path.is_file():
            raise SystemExit(f"[GENERATION] builder did not produce {METADATA_ARTIFACT}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bad_sources = [d for d, info in metadata.items() if Path(str(info.get("source", ""))).is_absolute()]
        if bad_sources:
            raise SystemExit(
                f"[GENERATION] metadata records absolute source paths (must be POSIX-relative): {bad_sources[:3]}"
            )

        # -- live-after parity + snapshot equality (TOCTOU bound) ------------
        live_after = _live_manifest()
        if live_after != live_before:
            raise SystemExit("[GENERATION] live corpus drifted during build (before != after) — aborting")
        # P0 #1: recompute EVERY controlling input after population and
        # require EXACT equality with the pre-population snapshot. A build
        # may never seal a receipt binding bytes/config that changed while
        # the child was indexing the earlier snapshot.
        identity_after = generation_identity(
            config, corpus_manifest_sha256=corpus_manifest_digest(_sealed_manifest(staging))
        )
        if identity_after != identity_before:
            drifted = sorted(k for k in identity_before if identity_after.get(k) != identity_before[k])
            raise SystemExit(f"[GENERATION] controlling build inputs drifted during population: {drifted} — aborting")
        compat_after = config.generation_compatibility()
        if compat_after != compat_before:
            raise SystemExit("[GENERATION] compatibility changed during population — aborting")

        # -- publish with the PRE-POPULATION snapshot (never later bytes) ----
        activation = store.publish(
            gid,
            identity=identity_before,
            compatibility=compat_before,
            chroma_evidence=chroma_evidence,
            fts_evidence=fts_evidence,
            provenance=provenance_before,
            expected_current=expected_current,
        )
        _print(
            f"[GENERATION] published + activated {gid} "
            f"(receipt {activation.receipt_sha256[:12]}...) in {time.time() - started:.1f}s"
        )
        _print("[GENERATION] restart_required=true")
        print(json.dumps(activation.to_dict(), indent=2))
        return 0
    except CurrentConflictError as exc:
        # The generation IS sealed; only the pointer CAS lost. Print the exact
        # recovery command — never delete the sealed generation.
        _print(f"[GENERATION] CAS conflict: {exc}")
        _print(f"[GENERATION] recovery: knowledge-rag-generation activate {gid}")
        return 2
    except CommitStateUncertainError as exc:
        _print(
            f"[GENERATION] commit state uncertain for {exc.generation_id} "
            f"(may_have_committed={exc.may_have_committed}) — re-read current: "
            "knowledge-rag-generation status"
        )
        return 3
    except DependencyUnverifiableError as exc:
        _print(f"[GENERATION] dependency evidence unverifiable: {exc}")
        store.abort_build(gid)
        return 4
    except BaseException as exc:
        # Pre-publication failure: clean ONLY the staging tree. The sealed
        # generations and the pointer are untouched.
        if not isinstance(exc, SystemExit):
            _print(f"[GENERATION] build failed: {exc.__class__.__name__}: {exc}")
        store.abort_build(gid)
        _print(f"[GENERATION] removed staging {BUILDING_PREFIX}{gid}")
        raise


def _status_command(args: argparse.Namespace) -> int:
    _load_generation_config()
    store = GenerationStore(config.data_dir, create=False)
    try:
        current = store.require_current()
    except GenerationError:
        # STRICT v3 resolution failed (missing/invalid/stale/v2). Fall back
        # to the LENIENT inspector — status must report safely, never dump
        # raw exception text or topology (P0 #10).
        summary = inspect_current_receipt(config.data_dir)
        if summary is None:
            _print("[GENERATION] current is not resolvable and no receipt is inspectable")
            return 1
        payload = dict(summary)
        payload["drifted_from_pinned"] = (
            config.active_generation_id is not None and summary.get("generation_id") != config.active_generation_id
        )
        print(json.dumps(payload, indent=2))
        return 1
    receipt = current.receipt
    print(
        json.dumps(
            {
                "generation_id": current.generation_id,
                "receipt_sha256": current.receipt_sha256,
                "schema_version": receipt.get("schema_version"),
                "servable": receipt.get("schema_version") == 3,
                "reason": None,
                "created_at": receipt.get("created_at"),
                "compatibility": receipt.get("compatibility"),
                "chroma": (receipt.get("backends") or {}).get("chroma"),
                "fts5": (receipt.get("backends") or {}).get("fts5"),
                "artifacts": {
                    name: {"count": meta.get("count"), "sha256": meta.get("sha256", "")[:12] + "..."}
                    for name, meta in (receipt.get("artifacts") or {}).items()
                },
                "drifted_from_pinned": (
                    config.active_generation_id is not None and current.generation_id != config.active_generation_id
                ),
            },
            indent=2,
        )
    )
    return 0


def _activate_command(args: argparse.Namespace) -> int:
    _load_generation_config()
    store = GenerationStore(config.data_dir, create=False)
    expected = store.current_identity()
    if expected is None:
        raise SystemExit("[GENERATION] no current pointer — activate requires an existing pointer")
    compat = config.generation_compatibility()
    try:
        result = store.activate(
            args.generation_id,
            expected_current=expected,
            expected_compatibility=compat,
        )
    except CommitStateUncertainError as exc:
        _print(
            f"[GENERATION] commit state uncertain ({exc.generation_id}); "
            "re-read current: knowledge-rag-generation status"
        )
        return 3
    _print(f"[GENERATION] {result.summary()}")
    _print("[GENERATION] restart_required=true")
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _rollback_command(args: argparse.Namespace) -> int:
    _load_generation_config()
    store = GenerationStore(config.data_dir, create=False)
    expected = store.current_identity()
    if expected is None:
        raise SystemExit("[GENERATION] no current pointer — rollback requires an existing pointer")
    try:
        result = store.rollback(
            args.generation_id,
            expected_current=expected,
            expected_compatibility=config.generation_compatibility(),
        )
    except CommitStateUncertainError as exc:
        _print(
            f"[GENERATION] commit state uncertain ({exc.generation_id}); "
            "re-read current: knowledge-rag-generation status"
        )
        return 3
    _print(f"[GENERATION] {result.summary()}")
    _print("[GENERATION] restart_required=true")
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # mcp_server/__init__.py redirects stdout to stderr for MCP stdio safety;
    # this is a terminal application, so restore the package's original
    # terminal stdout first (same pattern as vault_rag_cli.main). Diagnostics
    # stay on stderr via _print().
    package = sys.modules.get("mcp_server")
    sys.stdout = getattr(package, "_original_stdout", None) or sys.__stdout__

    parser = argparse.ArgumentParser(
        prog="knowledge-rag-generation",
        description="Offline immutable-generation lifecycle (build / status / activate / rollback).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show the verified current generation")
    p_status.set_defaults(func=_status_command)

    p_build = sub.add_parser("build", help="Build + activate a new generation from the live corpus")
    _build_args(p_build)

    p_activate = sub.add_parser("activate", help="CAS-switch current to a published generation")
    p_activate.add_argument("generation_id")
    p_activate.set_defaults(func=_activate_command)

    p_rollback = sub.add_parser("rollback", help="CAS-switch current back to a published generation")
    p_rollback.add_argument("generation_id")
    p_rollback.set_defaults(func=_rollback_command)

    args = parser.parse_args(argv)
    return args.func(args)


def _build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--generation-id", default=None, help="Explicit generation id (default: gen-<uuid4>)")
    p.set_defaults(func=_build_command)


if __name__ == "__main__":  # pragma: no cover — CLI entrypoint
    raise SystemExit(main())
