"""Deterministic tests for the versioned generation store (Phase A, rev 4).

Proves the reviewed contract:
- GenerationStore(root) is a read-only open that creates NOTHING and rejects
  missing/invalid anchors; create=True is the explicit builder init;
- staging lives at root/.building-<id> (OUTSIDE generations/);
- exact top-level entries: five artifacts in staging, five + generation.json
  after sealing — WAL/SHM/any extra rejected, including inside tree artifacts;
- receipt (schema_version 2): status=complete, canonical UTC created_at,
  identity SHAs, exact 10-key compatibility object, per-artifact hashes,
  Chroma evidence (collection_name + three equal counts + row_digest),
  FTS evidence (schema_version exactly 2, complete, three digests) with
  four-digest parity against Chroma and equal row counts, collection binding;
- the ASSEMBLED receipt is validated BEFORE any fsync/rename/CAS;
- resolve_current performs FULL verification under the lock, verified-only;
- fsync of every artifact file + directory, receipt written LAST;
- CommitStateUncertainError (may_have_committed=True + intended identity)
  when the root fsync fails after the pointer replace;
- one stable exclusive lock across verify/rename/CAS; anchors re-checked;
- CAS binds generation_id AND receipt SHA (ABA / lost-update safe);
- fail-closed symlink policy; no silent in-process lock fallback;
- current_identity rejects a symlinked current; restart_required surfaced;
- rev 4: oversize receipts (> _MAX_RECEIPT_BYTES) rejected BEFORE any
  durability or pointer operation;
- rev 4: read paths take a SHARED lock on the EXISTING lock file and create
  nothing (exact directory entries proven unchanged);
- rev 4: created_at=None is the supported auto-now input; rollback
  invariants asserted on exact pointer bytes; receipt-SHA pinning and
  altered-compatibility detection asserted at their real boundaries (pointer
  resolution / the explicit expected-compatibility gate).

Everything runs on tmp_path with fixed byte payloads: no network, no models,
no Chroma/FTS imports, no sleeps.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_server import generations as gens
from mcp_server.generations import (
    BuildExistsError,
    CommitStateUncertainError,
    CurrentConflictError,
    DurabilityError,
    GenerationStore,
    LockUnsupportedError,
    PointerError,
    PublishError,
    UnsafePathError,
    VerificationError,
)

HEX = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
STABLE_DIGEST = lambda payload: hashlib.sha256(  # noqa: E731
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
).hexdigest()


def chunking_digest_for(compat: dict) -> str:
    """Compatibility-derived chunking digest (mirrors the receipt cross-bind)."""
    return STABLE_DIGEST(
        {
            "embedding_model": compat["embedding_model"],
            "query_prefix": compat["query_prefix"],
            "passage_prefix": compat["passage_prefix"],
            "chunk_size": compat["chunk_size"],
            "chunk_overlap": compat["chunk_overlap"],
        }
    )


IDENTITY = {
    "corpus_manifest_sha256": HEX("corpus-manifest-v1"),
    "config_sha256": HEX("effective-config-v1"),
    "code_sha256": HEX("code-identity-v1"),
    "retrieval_config_sha256": HEX("retrieval-config-v1"),
    "installed_record_sha256": HEX("installed-record-v1"),
    "dependency_lock_sha256": HEX("dependency-lock-v1"),
    "model_artifact_sha256": HEX("model-artifact-v1"),
    "model_config_sha256": HEX("model-config-v1"),
    "chunking_sha256": STABLE_DIGEST(
        {
            "embedding_model": "bge-small-en-v1.5",
            "query_prefix": "Represent this sentence for searching: ",
            "passage_prefix": "Represent this sentence: ",
            "chunk_size": 512,
            "chunk_overlap": 64,
        }
    ),
}
IDENTITY_V2 = dict(IDENTITY, config_sha256=HEX("effective-config-v2"))

COMPAT = {
    "collection_name": "knowledge_rag_v1",
    "embedding_model": "bge-small-en-v1.5",
    "embedding_dimension": 384,
    "query_prefix": "Represent this sentence for searching: ",
    "passage_prefix": "Represent this sentence: ",
    "model_artifact_sha256": HEX("model-artifact-v1"),
    "runtime_version": "sentence-transformers 3.0.1",
    "pooling": "cls",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "reranker_enabled": False,
    "reranker_model": None,
    "reranker_artifact_sha256": None,
}
COMPAT_V2 = dict(COMPAT, embedding_model="bge-base-en-v1.5", embedding_dimension=768)

ROW_DIGEST = HEX("canonical-row-universe-v1")
FULL_DIGEST = HEX("chroma-full-row-universe-v1")
BACKEND_GID = HEX("backend-generation-v1")
ROWS = 3

CHROMA_ROWS = {
    "chroma.sqlite3": b"chroma-sqlite-payload-v1\n",
    "collections.bin": b"collections-payload-v1\n",
}
CORPUS_ROWS = {"doc-a.md": b"# doc a\n", "doc-b.md": b"# doc b\n"}
FTS_BYTES = b"fts5-index-payload-v1\x00\x01\x02\n"
FTS_STATE_BYTES = b"fts5-migration-state-v1\n"
METADATA_BYTES = b'{"indexed_documents": 2}\n'

EXPECTED_ARTIFACT_FILES = frozenset(
    {"corpus", "chroma_db", "fts5_index.db", "fts5_migration.state", "index_metadata.json"}
)


# ============================================================================
# Helpers / fixtures
# ============================================================================


@pytest.fixture()
def store(tmp_path: Path) -> GenerationStore:
    return GenerationStore(tmp_path, create=True)


def compat(over: dict | None = None) -> dict:
    out = dict(COMPAT)
    out.update(over or {})
    return out


def chroma_ev(
    count: int = ROWS,
    digest: str = FULL_DIGEST,
    common: str = ROW_DIGEST,
    collection: str = COMPAT["collection_name"],
    **over,
) -> dict:
    ev = {
        "collection_name": collection,
        "row_count": count,
        "unique_id_count": count,
        "hydrated_id_count": count,
        "row_digest": digest,
        "common_row_digest": common,
        "backend_generation_id": BACKEND_GID,
    }
    ev.update(over)
    return ev


def fts_ev(count: int = ROWS, digest: str = ROW_DIGEST, **over) -> dict:
    ev = {
        "schema_version": 2,
        "status": "complete",
        "row_count": count,
        "row_digest": digest,
        "source_digest": digest,
        "verified_digest": digest,
        "backend_generation_id": BACKEND_GID,
    }
    ev.update(over)
    return ev


def stage_generation(
    building: Path,
    *,
    chroma: dict | None = None,
    corpus: dict | None = None,
    fts: bytes = FTS_BYTES,
    fts_state: bytes = FTS_STATE_BYTES,
    metadata: bytes = METADATA_BYTES,
) -> Path:
    """Stage deterministic bytes for ALL required artifacts."""
    for rel, payload in ((gens.CHROMA_ARTIFACT, chroma or CHROMA_ROWS), (gens.CORPUS_ARTIFACT, corpus or CORPUS_ROWS)):
        target_dir = building / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in payload.items():
            (target_dir / name).write_bytes(blob)
    (building / gens.FTS_ARTIFACT).write_bytes(fts)
    (building / gens.FTS_STATE_ARTIFACT).write_bytes(fts_state)
    (building / gens.METADATA_ARTIFACT).write_bytes(metadata)
    return building


def publish_generation(
    store: GenerationStore,
    gid: str,
    *,
    expected_current: dict | None | str = None,
    identity: dict | None = None,
    compatibility: dict | None = None,
    chroma: dict | None = None,
    corpus: dict | None = None,
    fts: bytes = FTS_BYTES,
    chroma_evidence: dict | None = None,
    fts_evidence: dict | None = None,
    provenance: dict | None = None,
    created_at: str | None = None,
) -> gens.ActivationResult:
    """Publish a well-formed generation.

    First publication defaults to the EXPLICIT absent sentinel — production
    rejects ``None`` (unchecked CAS); tests must never weaken that contract.
    A str resolves to that generation's full pointer identity — but never
    the sentinel (``store_receipt_sha`` must not be called for it).

    ``is None`` checks (never ``or``) keep EMPTY negative fixtures (e.g.
    ``chroma={}``) meaningful as explicit overrides.

    The fast schema/CAS/durability path monkeypatches ONLY
    ``GenerationStore._recompute_staged_semantics`` around this helper: real
    file/tree hashing, receipt validation, fsync, rename and CAS stay live;
    the seam returns deterministic declared evidence built from the REAL
    staged artifact hashes so cross-checks remain honest.
    """
    if expected_current is None:
        expected_current = gens.EXPECTED_CURRENT_ABSENT
    if isinstance(expected_current, str) and expected_current != gens.EXPECTED_CURRENT_ABSENT:
        expected_current = {
            "generation_id": expected_current,
            "receipt_sha256": store_receipt_sha(store, expected_current),
        }
    building = store.begin_build(gid)
    stage_generation(building, chroma=chroma, corpus=corpus, fts=fts)
    # Bind the DEFAULT identity's corpus digest to the ACTUAL staged corpus
    # bytes (controller item 2) — a normal publication's controlling corpus
    # identity must be the real manifest of the sealed tree. Tests that
    # deliberately exercise corpus/identity mismatch pass an explicit
    # ``identity`` and are left untouched.
    staged_corpus_digest = gens.corpus_manifest_digest(gens.corpus_manifest_entries(building / gens.CORPUS_ARTIFACT))
    effective_compat = COMPAT if compatibility is None else compatibility
    if identity is None:
        # Default identity is fully consistent with the EFFECTIVE
        # compatibility: the corpus digest binds the ACTUAL staged bytes,
        # and the artifact/chunking digests are derived from the
        # compatibility object the receipt will carry (receipt validation
        # cross-binds both, so a synthetic constant cannot sneak through).
        identity = dict(
            IDENTITY,
            corpus_manifest_sha256=staged_corpus_digest,
            model_artifact_sha256=effective_compat["model_artifact_sha256"],
            chunking_sha256=chunking_digest_for(effective_compat),
        )
    # Fast schema/CAS/durability path (controller-approved seam): the REAL
    # staged-artifact hashing stays live inside the fake (the receipt still
    # binds the actual staged bytes), while the heavy child-process Chroma/
    # FTS semantic inspection is replaced by the caller's declared evidence.
    # Real file/tree hashing, receipt validation, fsync, rename and CAS in
    # publish() remain fully active.
    with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
        return store.publish(
            gid,
            identity=identity,
            compatibility=effective_compat,
            chroma_evidence=chroma_ev() if chroma_evidence is None else chroma_evidence,
            fts_evidence=fts_ev() if fts_evidence is None else fts_evidence,
            provenance=provenance,
            expected_current=expected_current,
            created_at=created_at,
        )


def store_receipt_sha(store: GenerationStore, gid: str) -> str:
    return gens._sha256_file(store.generations_dir / gid / gens.RECEIPT_FILENAME)


def _fast_semantics(self, building, ident, chroma_ev, fts_ev):
    """Test seam for ``_recompute_staged_semantics`` (controller-approved).

    Skips ONLY the heavy child-process Chroma inspection and the FTS
    read-only recompute; the REAL staged five-artifact hashing runs (both
    before and after — staging is untouched here, so equality holds), and
    backend evidence is the caller's declared values. publish()'s REAL
    receipt validation, artifact-hash binding, fsync, rename and CAS stay
    fully active — this fake can still fail those honestly.
    """
    before = self._hash_staged_artifacts(building)
    after = self._hash_staged_artifacts(building)
    if after != before:
        raise VerificationError("fast semantics: staged artifacts mutated")
    chroma_gid = hashlib.sha256(
        f"chroma:{before[gens.CHROMA_ARTIFACT]['sha256']}:{chroma_ev['row_digest']}".encode()
    ).hexdigest()
    fts_gid = hashlib.sha256(f"fts5:{before[gens.FTS_ARTIFACT]['sha256']}:{fts_ev['row_digest']}".encode()).hexdigest()
    return {
        "artifacts": before,
        "chroma": dict(chroma_ev, backend_generation_id=chroma_gid),
        "fts5": dict(fts_ev, backend_generation_id=fts_gid),
    }


def pointer_identity(store: GenerationStore, gid: str) -> dict:
    return {"generation_id": gid, "receipt_sha256": store_receipt_sha(store, gid)}


def seed_current(store: GenerationStore) -> bytes:
    """Publish g1 as current; return the exact pointer bytes for comparisons."""
    publish_generation(store, "g1")
    return store.current_path.read_bytes()


def symlink_or_skip(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - restricted hosts
        pytest.skip("platform cannot create symlinks")


def fifo_or_skip(path: Path) -> None:
    try:
        os.mkfifo(path)
    except (OSError, AttributeError, NotImplementedError):  # pragma: no cover
        pytest.skip("platform cannot create FIFOs")


def no_tmp_pointer_files(store: GenerationStore) -> bool:
    return not list(store.root.glob(gens.POINTER_TMP_PREFIX + "*" + gens.POINTER_TMP_SUFFIX))


def directory_snapshot(root: Path) -> list[str]:
    """Exact recursive listing: sorted relative POSIX paths of every entry."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _tamper_receipt(store: GenerationStore, gid: str, mutate, baseline: bytes | None = None) -> None:
    """Rewrite a receipt, by default starting from a pristine snapshot."""
    path = store.generations_dir / gid / gens.RECEIPT_FILENAME
    raw = baseline if baseline is not None else path.read_bytes()
    receipt = json.loads(raw)
    mutate(receipt)
    path.write_bytes(json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n")


def crash_replace_when(fail_if):
    """Wrap gens._atomic_replace to raise when fail_if(dst) is true (crash sim)."""
    real = gens._atomic_replace

    def crashing(src: Path, dst: Path) -> None:
        if fail_if(dst):
            raise RuntimeError(f"simulated crash before rename onto {dst}")
        real(src, dst)

    return crashing


# ============================================================================
# (P0-2) read-only open vs explicit builder init
# ============================================================================


def test_default_open_on_missing_root_creates_nothing(tmp_path: Path):
    missing = tmp_path / "absent-root"
    with pytest.raises(UnsafePathError):
        GenerationStore(missing)  # default read-only open
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_default_open_rejects_missing_generations_dir(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(UnsafePathError):
        GenerationStore(root)
    assert not (root / gens.GENERATIONS_DIRNAME).exists()
    assert not (root / gens.LOCK_FILENAME).exists()
    assert not (root / gens.CURRENT_FILENAME).exists()


def test_readonly_open_leaves_exact_directory_entries_untouched(tmp_path: Path):
    builder = GenerationStore(tmp_path, create=True)
    # Pointer-less store: no lock file exists yet, and reads must not
    # materialize one (shared lock on the EXISTING file, or the no-file path).
    before = directory_snapshot(tmp_path)
    assert gens.LOCK_FILENAME not in before
    readonly = GenerationStore(tmp_path)
    assert readonly.resolve_current() is None
    assert readonly.current_identity() is None
    with pytest.raises(VerificationError):
        readonly.verify_generation("g1")
    assert directory_snapshot(tmp_path) == before

    # Populated store (lock + current + one generation): reads still change
    # NOTHING anywhere under the root — exact entry set, byte for byte.
    publish_generation(builder, "g1")
    before = directory_snapshot(tmp_path)
    assert gens.LOCK_FILENAME in before  # created by the BUILDER, not the reader
    reader = GenerationStore(tmp_path)
    assert reader.resolve_current() is not None
    assert reader.current_identity() == pointer_identity(builder, "g1")
    assert reader.verify_generation("g1")["generation_id"] == "g1"
    assert directory_snapshot(tmp_path) == before


def test_create_true_builds_layout(tmp_path: Path):
    store = GenerationStore(tmp_path, create=True)
    assert store.root.is_dir()
    assert store.generations_dir.is_dir()
    assert not store.current_path.exists()


def test_symlinked_store_root_rejected(tmp_path: Path):
    real = tmp_path / "real-root"
    real.mkdir()
    symlink_or_skip(real, tmp_path / "link-root")
    with pytest.raises(UnsafePathError):
        GenerationStore(tmp_path / "link-root", create=True)
    with pytest.raises(UnsafePathError):
        GenerationStore(tmp_path / "link-root")


def test_symlinked_generations_dir_rejected_at_init(tmp_path: Path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    symlink_or_skip(elsewhere, root / gens.GENERATIONS_DIRNAME)
    with pytest.raises(UnsafePathError):
        GenerationStore(root, create=True)
    with pytest.raises(UnsafePathError):
        GenerationStore(root)


# ============================================================================
# (6) collision-resistant IDs + strict safe-ID validation
# ============================================================================


def test_new_generation_id_is_collision_resistant_and_safe():
    ids = {gens.new_generation_id() for _ in range(64)}
    assert len(ids) == 64
    for gid in ids:
        assert re.fullmatch(r"gen-[0-9a-f]{32}", gid)
        gens._require_safe_id(gid)


@pytest.mark.parametrize(
    "bad",
    ["../escape", "a/b", "..", ".", ".hidden", "-dash", "_under", "", "x" * 129, "gen one", "gen\nid", "a\\b", None, 7],
)
def test_unsafe_generation_ids_rejected(store, bad):
    for call in (store.generation_dir, store.begin_build, store.verify_generation, store.activate):
        with pytest.raises(UnsafePathError):
            call(bad)


@pytest.mark.parametrize("good", ["g1", "gen-2026.08.14", "A" * 128, "0start", "x.y-z_w", gens.new_generation_id()])
def test_safe_generation_ids_accepted(store, good):
    assert store.generation_dir(good) == store.generations_dir / good


def test_building_prefixed_ids_can_never_collide(store):
    with pytest.raises(UnsafePathError):
        store.generation_dir(".building-g1")


# ============================================================================
# staging at root/.building-<id> + exact-entry policy
# ============================================================================


def test_begin_build_stages_at_root_outside_generations(store):
    building = store.begin_build("g1")
    assert building == store.root / ".building-g1"
    assert building.is_dir()
    assert list(store.generations_dir.iterdir()) == []
    store.abort_build("g1")
    assert not building.exists()


def test_begin_build_conflicts(store):
    building = store.begin_build("g1")
    with pytest.raises(BuildExistsError):
        store.begin_build("g1")
    stage_generation(building)
    with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    with pytest.raises(BuildExistsError):
        store.begin_build("g1")


def test_publish_rejects_extra_top_level_entries(store):
    building = store.begin_build("g1")
    stage_generation(building)
    (building / "chroma.sqlite3-wal").write_bytes(b"stale wal")
    with pytest.raises(PublishError, match="unexpected entries"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")
    assert not (store.generations_dir / "g1").exists()


def test_publish_rejects_missing_top_level_entries(store):
    building = store.begin_build("g1")
    stage_generation(building)
    (building / gens.FTS_STATE_ARTIFACT).unlink()
    with pytest.raises(PublishError, match="missing entries"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_publish_rejects_sqlite_sidecars_inside_tree_artifacts(store):
    building = store.begin_build("g1")
    stage_generation(building)
    (building / gens.CHROMA_ARTIFACT / "chroma.sqlite3-shm").write_bytes(b"shm")
    with pytest.raises(UnsafePathError, match="sidecar"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_verify_rejects_extra_top_level_entries_in_published_generation(store):
    publish_generation(store, "g1")
    (store.generations_dir / "g1" / "fts5_index.db-wal").write_bytes(b"wal")
    with pytest.raises(VerificationError, match="unexpected entries"):
        store.verify_generation("g1")
    assert store.resolve_current() is None


def test_verify_rejects_missing_artifact_entry(store):
    publish_generation(store, "g1")
    (store.generations_dir / "g1" / gens.METADATA_ARTIFACT).unlink()
    with pytest.raises(VerificationError, match="missing entries"):
        store.verify_generation("g1")
    assert store.resolve_current() is None


# ============================================================================
# (5) fail-closed symlink policy + non-regular files
# ============================================================================


def test_symlinked_generation_dir_rejected(store, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(outside, store.generations_dir / "evil")
    with pytest.raises(UnsafePathError):
        store.generation_dir("evil")
    with pytest.raises(UnsafePathError):
        store.verify_generation("evil")


def test_symlinked_artifact_rejected_even_with_identical_bytes(store, tmp_path: Path):
    publish_generation(store, "g1")
    outside = tmp_path / "outside-fts.db"
    outside.write_bytes(FTS_BYTES)
    fts_path = store.generations_dir / "g1" / gens.FTS_ARTIFACT
    fts_path.unlink()
    symlink_or_skip(outside, fts_path)
    with pytest.raises(UnsafePathError):
        store.verify_generation("g1")
    assert store.resolve_current() is None


def test_symlink_inside_tree_artifact_rejected(store, tmp_path: Path):
    publish_generation(store, "g1")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"whatever")
    symlink_or_skip(outside, store.generations_dir / "g1" / gens.CHROMA_ARTIFACT / "link.bin")
    with pytest.raises(UnsafePathError):
        store.verify_generation("g1")
    assert store.resolve_current() is None


def test_non_regular_file_in_tree_artifact_rejected(store):
    publish_generation(store, "g1")
    fifo_or_skip(store.generations_dir / "g1" / gens.CORPUS_ARTIFACT / "pipe")
    with pytest.raises(UnsafePathError):
        store.verify_generation("g1")
    assert store.resolve_current() is None


def test_symlinked_current_fails_closed(store, tmp_path: Path):
    old_pointer = seed_current(store)
    outside = tmp_path / "outside-pointer"
    outside.write_bytes(old_pointer)
    store.current_path.unlink()
    symlink_or_skip(outside, store.current_path)
    assert store.resolve_current() is None
    with pytest.raises(UnsafePathError):
        store.current_identity()
    with pytest.raises(UnsafePathError):  # anchors re-checked under the lock
        store.begin_build("g2")


def test_symlinked_lock_rejected(store, tmp_path: Path):
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"")
    symlink_or_skip(outside, store.lock_path)
    with pytest.raises(UnsafePathError):
        store.begin_build("g1")


def test_no_symlinks_created_by_store(store):
    publish_generation(store, "g1")
    for root, dirs, names in os.walk(store.root, followlinks=False):
        for n in dirs + names:
            assert not (Path(root) / n).is_symlink(), f"store created a symlink: {root}/{n}"


def test_missing_fcntl_raises_lock_unsupported(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gens, "fcntl", None)
    bare = GenerationStore(tmp_path, create=True)
    with pytest.raises(LockUnsupportedError):
        with bare._exclusive_lock():
            pass  # pragma: no cover
    with pytest.raises(LockUnsupportedError):
        with bare._read_lock():
            pass  # pragma: no cover


def test_lock_file_lives_outside_generations(store):
    publish_generation(store, "g1")
    assert store.lock_path.parent == store.root
    assert store.lock_path.name == gens.LOCK_FILENAME
    assert store.lock_path.exists()
    assert gens.LOCK_FILENAME not in [p.name for p in store.generations_dir.iterdir()]


# ============================================================================
# (1) exact layout + receipt schema (rev 3)
# ============================================================================


def test_exact_generation_layout(store):
    publish_generation(store, "g1")
    gdir = store.generations_dir / "g1"
    assert {p.name for p in gdir.iterdir()} == EXPECTED_ARTIFACT_FILES | {gens.RECEIPT_FILENAME}
    assert (gdir / gens.CORPUS_ARTIFACT).is_dir()
    assert (gdir / gens.CHROMA_ARTIFACT).is_dir()
    for name in (gens.FTS_ARTIFACT, gens.FTS_STATE_ARTIFACT, gens.METADATA_ARTIFACT):
        assert stat.S_ISREG(os.lstat(gdir / name).st_mode), name


def test_valid_publish_receipt_contract(store):
    result = publish_generation(store, "g1")
    receipt = result.receipt

    assert receipt["schema_version"] == gens.RECEIPT_SCHEMA_VERSION == 3
    assert receipt["status"] == "complete"
    assert receipt["generation_id"] == "g1"
    # Expected identity = the fixture identity bound to the ACTUAL staged
    # corpus manifest (the helper derives corpus_manifest_sha256 from the
    # staged bytes; the old synthetic constant is not the sealed digest).
    staged_identity = dict(
        IDENTITY,
        corpus_manifest_sha256=gens.corpus_manifest_digest(
            gens.corpus_manifest_entries(store.generations_dir / "g1" / gens.CORPUS_ARTIFACT)
        ),
    )
    assert receipt["identity"] == staged_identity
    assert receipt["compatibility"] == COMPAT
    assert set(receipt["artifacts"]) == set(gens.REQUIRED_ARTIFACTS)

    chroma_meta = receipt["artifacts"][gens.CHROMA_ARTIFACT]
    assert chroma_meta["kind"] == "tree" and chroma_meta["count"] == len(CHROMA_ROWS)
    corpus_meta = receipt["artifacts"][gens.CORPUS_ARTIFACT]
    assert corpus_meta["kind"] == "tree" and corpus_meta["count"] == len(CORPUS_ROWS)
    fts_meta = receipt["artifacts"][gens.FTS_ARTIFACT]
    assert fts_meta["kind"] == "file" and fts_meta["count"] == 1
    assert fts_meta["sha256"] == hashlib.sha256(FTS_BYTES).hexdigest()
    assert receipt["artifacts"][gens.FTS_STATE_ARTIFACT]["sha256"] == hashlib.sha256(FTS_STATE_BYTES).hexdigest()
    assert receipt["artifacts"][gens.METADATA_ARTIFACT]["sha256"] == hashlib.sha256(METADATA_BYTES).hexdigest()

    assert receipt["backends"]["chroma"] == chroma_ev()
    assert receipt["backends"]["fts5"] == fts_ev()

    assert gens._parse_created_at(receipt["created_at"]).tzinfo is not None
    assert store.verify_generation("g1") == receipt
    current = store.resolve_current()
    assert current is not None and current.receipt == receipt
    assert current.compatibility() == COMPAT


def test_publish_accepts_explicit_canonical_created_at(store):
    publish_generation(store, "g1", created_at="2026-08-14T21:43:58Z")
    receipt = store.verify_generation("g1")
    assert receipt["created_at"] == "2026-08-14T21:43:58Z"


def test_publish_auto_now_created_at_is_canonical_utc(store):
    publish_generation(store, "g1")  # created_at=None is the supported auto-now input
    receipt = store.verify_generation("g1")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", receipt["created_at"])
    parsed = gens._parse_created_at(receipt["created_at"])
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    "bad_created_at",
    ["2026-08-14T21:43:58+00:00", "2026-08-14 21:43:58", "2026-08-14T21:43:58.123Z", "", "not-a-date", 7],
)
def test_non_canonical_created_at_rejected_everywhere(store, bad_created_at):
    # At publish input (assembled-receipt validation, BEFORE durability):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="created_at"):
        with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
            store.publish(
                "g1",
                identity=IDENTITY,
                compatibility=COMPAT,
                chroma_evidence=chroma_ev(),
                fts_evidence=fts_ev(),
                created_at=bad_created_at,
                expected_current=gens.EXPECTED_CURRENT_ABSENT,
            )
    store.abort_build("g1")
    # And in an existing receipt:
    publish_generation(store, "g1")
    baseline = (store.generations_dir / "g1" / gens.RECEIPT_FILENAME).read_bytes()
    _tamper_receipt(store, "g1", lambda r: r.update(created_at=bad_created_at), baseline)
    with pytest.raises(VerificationError, match="created_at"):
        store.verify_generation("g1")
    assert store.resolve_current() is None


# ============================================================================
# (P0-1) compatibility object
# ============================================================================


@pytest.mark.parametrize(
    "over",
    [
        {"embedding_model": ""},
        {"embedding_dimension": 0},
        {"embedding_dimension": True},
        {"chunk_overlap": 512},  # >= chunk_size
        {"chunk_size": 32, "chunk_overlap": 64},
        {"model_artifact_sha256": "not-hex"},
        {"pooling": ""},
    ],
)
def test_invalid_compatibility_rejected_at_publish(store, over):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="compatibility"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=compat(over),
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_compatibility_key_set_must_be_exact(store):
    building = store.begin_build("g1")
    stage_generation(building)
    dropped = {k: v for k, v in COMPAT.items() if k != "pooling"}
    with pytest.raises(VerificationError, match="compatibility"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=dropped,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    with pytest.raises(VerificationError, match="compatibility"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=compat({"extra": 1}),
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_receipt_compatibility_tamper_contract(store):
    publish_generation(store, "g1")
    baseline = (store.generations_dir / "g1" / gens.RECEIPT_FILENAME).read_bytes()
    old_pointer = store.current_path.read_bytes()
    # chunk_size=999 still passes every per-field compatibility rule
    # (chunk_overlap 64 < 999), but it diverges from the receipt's
    # identity.chunking_sha256 cross-binding — caught below.
    _tamper_receipt(store, "g1", lambda r: r["compatibility"].update(chunk_size=999), baseline)
    # Receipt validation cross-binds identity.chunking_sha256 to the
    # compatibility object: chunk_size=999 changes the compatibility-derived
    # chunking digest, so bare verify_generation (schema + cross-bind) fails
    # closed even before any expected-compatibility input.
    with pytest.raises(VerificationError, match="chunking mismatch"):
        store.verify_generation("g1")
    # And the pointer's receipt-SHA pin fails resolution closed.
    assert store.resolve_current() is None
    assert store.current_path.read_bytes() == old_pointer  # pointer untouched


# ============================================================================
# (P0-1) backend evidence: counts, digests, binding
# ============================================================================


def test_chroma_three_count_parity_enforced(store):
    def attempt(ev) -> None:
        building = store.begin_build("g1")
        stage_generation(building)
        try:
            store.publish(
                "g1",
                identity=IDENTITY,
                compatibility=COMPAT,
                chroma_evidence=ev,
                fts_evidence=fts_ev(),
                expected_current=gens.EXPECTED_CURRENT_ABSENT,
            )
        finally:
            if building.exists():
                store.abort_build("g1")

    with pytest.raises(VerificationError, match="id-count parity"):
        attempt(chroma_ev(unique_id_count=4))
    with pytest.raises(VerificationError, match="id-count parity"):
        attempt(chroma_ev(hydrated_id_count=2))
    assert not (store.generations_dir / "g1").exists()
    assert store.resolve_current() is None


def test_chroma_collection_binding_enforced(store):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="collection_name"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(collection="other_collection"),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


@pytest.mark.parametrize(
    "over",
    [{"schema_version": 1}, {"schema_version": 3}, {"status": "building"}, {"schema_version": 2, "status": "failed"}],
)
def test_fts_schema_and_status_enforced(store, over):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(**over),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


@pytest.mark.parametrize("field", ["row_digest", "source_digest", "verified_digest"])
def test_fts_four_digest_parity_enforced(store, field):
    building = store.begin_build("g1")
    stage_generation(building)
    bad = fts_ev()
    bad[field] = HEX("divergent")
    with pytest.raises(VerificationError, match="digest .*parity"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=bad,
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_backend_row_count_parity_enforced(store):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="row-count parity"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(count=3),
            fts_evidence=fts_ev(count=4),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_receipt_backend_evidence_key_sets_exact(store):
    publish_generation(store, "g1")
    baseline = (store.generations_dir / "g1" / gens.RECEIPT_FILENAME).read_bytes()
    _tamper_receipt(store, "g1", lambda r: r["backends"]["fts5"].pop("verified_digest"), baseline)
    with pytest.raises(VerificationError, match="fts5 evidence"):
        store.verify_generation("g1")
    _tamper_receipt(store, "g1", lambda r: r["backends"]["chroma"].update(extra=1), baseline)
    with pytest.raises(VerificationError, match="chroma evidence"):
        store.verify_generation("g1")
    _tamper_receipt(store, "g1", lambda r: r["backends"]["chroma"].pop("hydrated_id_count"), baseline)
    with pytest.raises(VerificationError, match="chroma evidence"):
        store.verify_generation("g1")


def test_receipt_tamper_detects_count_and_digest_divergence(store):
    publish_generation(store, "g1")
    baseline = (store.generations_dir / "g1" / gens.RECEIPT_FILENAME).read_bytes()
    _tamper_receipt(store, "g1", lambda r: r["backends"]["fts5"].update(row_count=4), baseline)
    with pytest.raises(VerificationError, match="row-count parity"):
        store.verify_generation("g1")
    _tamper_receipt(store, "g1", lambda r: r["backends"]["fts5"].update(source_digest=HEX("x")), baseline)
    with pytest.raises(VerificationError, match="digest .*parity"):
        store.verify_generation("g1")


def test_publish_input_identity_enforced(store):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="identity"):
        store.publish(
            "g1",
            identity={"corpus_manifest_sha256": HEX("c")},
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    with pytest.raises(VerificationError, match="identity"):
        store.publish(
            "g1",
            identity=dict(IDENTITY, code_sha256="nope"),
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


# ============================================================================
# receipt validation BEFORE durability; missing/hash mismatch
# ============================================================================


def test_receipt_validated_before_any_durability_step(store, monkeypatch):
    order: list[str] = []
    real_tree, real_write, real_replace = gens._fsync_tree, gens._write_file_exclusive, gens._atomic_replace
    monkeypatch.setattr(gens, "_fsync_tree", lambda base: (order.append("fsync_tree"), real_tree(base))[1])
    monkeypatch.setattr(
        gens,
        "_write_file_exclusive",
        lambda path, payload: (order.append(f"write:{path.name}"), real_write(path, payload))[1],
    )
    monkeypatch.setattr(
        gens,
        "_atomic_replace",
        lambda src, dst: (order.append(f"replace:{dst.name}"), real_replace(src, dst))[1],
    )

    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="created_at"):
        with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
            store.publish(
                "g1",
                identity=IDENTITY,
                compatibility=COMPAT,
                chroma_evidence=chroma_ev(),
                fts_evidence=fts_ev(),
                created_at="2026-08-14T21:43:58+00:00",
                expected_current=gens.EXPECTED_CURRENT_ABSENT,
            )
    assert order == []  # nothing durable happened before validation failed
    assert building.is_dir()
    assert not (store.generations_dir / "g1").exists()
    store.abort_build("g1")

    # Valid publish: SEMANTIC ordering only (no brittle index assertions).
    publish_generation(store, "g2")
    assert order[0] == "fsync_tree"  # artifacts durable before anything else
    receipt_write = order.index(f"write:{gens.RECEIPT_FILENAME}")
    final_rename = order.index("replace:g2")  # building -> final
    tmp_writes = [i for i, e in enumerate(order) if e.startswith(f"write:{gens.POINTER_TMP_PREFIX}")]
    pointer_replace = order.index(f"replace:{gens.CURRENT_FILENAME}")
    assert receipt_write < final_rename  # receipt sealed last, before the rename
    assert final_rename < pointer_replace  # generation final BEFORE the pointer swaps
    # The pointer temp file is created+fsynced BETWEEN the final-generation
    # rename and the pointer replace.
    assert len(tmp_writes) == 1
    assert final_rename < tmp_writes[0] < pointer_replace


def test_oversize_receipt_rejected_before_durability_and_pointer(store, monkeypatch):
    old_pointer = seed_current(store)
    order: list[str] = []
    real_tree = gens._fsync_tree
    real_write = gens._write_file_exclusive
    real_replace = gens._atomic_replace
    monkeypatch.setattr(gens, "_fsync_tree", lambda base: (order.append("fsync_tree"), real_tree(base))[1])
    monkeypatch.setattr(
        gens,
        "_write_file_exclusive",
        lambda path, payload: (order.append(f"write:{path.name}"), real_write(path, payload))[1],
    )
    monkeypatch.setattr(
        gens, "_atomic_replace", lambda src, dst: (order.append(f"replace:{dst.name}"), real_replace(src, dst))[1]
    )
    building = store.begin_build("g2")
    stage_generation(building)
    # Corpus-bound identity (the default derive) so the test reaches the
    # SIZE gate rather than failing early on a synthetic corpus digest —
    # all real receipt-size/durability/CAS logic stays live via the seam.
    g2_identity = dict(
        IDENTITY,
        corpus_manifest_sha256=gens.corpus_manifest_digest(
            gens.corpus_manifest_entries(building / gens.CORPUS_ARTIFACT)
        ),
    )
    with monkeypatch.context() as m:
        m.setattr(gens, "_MAX_RECEIPT_BYTES", 8)  # any valid receipt is now oversize
        with pytest.raises(VerificationError, match="receipt too large"):
            with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
                store.publish(
                    "g2",
                    identity=g2_identity,
                    compatibility=COMPAT,
                    chroma_evidence=chroma_ev(),
                    fts_evidence=fts_ev(),
                    expected_current=pointer_identity(store, "g1"),
                )
    # Rejection happened BEFORE any durability or pointer operation.
    assert order == []
    assert store.current_path.read_bytes() == old_pointer  # byte-identical
    assert building.is_dir()  # staging intact for retry/abort
    assert not (store.generations_dir / "g2").exists()
    assert no_tmp_pointer_files(store)
    store.abort_build("g2")


def test_missing_generation_and_receipt_fail(store):
    with pytest.raises(VerificationError):
        store.verify_generation("never-published")
    publish_generation(store, "g1")
    (store.generations_dir / "g1" / gens.RECEIPT_FILENAME).unlink()
    # Fail-closed exact-entry validation: the sealed generation without its
    # receipt is missing a required entry.
    with pytest.raises(VerificationError, match="missing entries"):
        store.verify_generation("g1")
    assert store.resolve_current() is None


def test_receipt_sha_pinning_enforced_at_pointer_resolution(store):
    old_pointer = seed_current(store)
    _tamper_receipt(store, "g1", lambda r: r["compatibility"].update(pooling="mean"))
    # The tampered receipt changes the receipt SHA: the pointer's pin no
    # longer matches, so resolution fails closed.
    assert store.resolve_current() is None
    # The pointer bytes themselves are untouched — only the hash pin broke.
    assert store.current_path.read_bytes() == old_pointer
    # Receipt-SHA pinning is the POINTER's contract: generic verify_generation
    # does not infer historical bytes, and pooling="mean" is schema-valid, so
    # bare verification passes (and returns the tampered compatibility).
    assert store.verify_generation("g1")["compatibility"]["pooling"] == "mean"


def test_artifact_digest_mismatch_detected(store):
    publish_generation(store, "g1")
    (store.generations_dir / "g1" / gens.FTS_ARTIFACT).write_bytes(b"tampered")
    with pytest.raises(VerificationError, match="digest mismatch"):
        store.verify_generation("g1")
    assert store.resolve_current() is None


def test_malformed_pointer_fails_closed(store):
    seed_current(store)
    store.current_path.write_bytes(b"{not json")
    assert store.resolve_current() is None
    with pytest.raises(PointerError):
        store.current_identity()


def test_pointer_extra_keys_rejected(store):
    seed_current(store)
    store.current_path.write_bytes(json.dumps({"generation_id": "g1", "receipt_sha256": HEX("x"), "extra": 1}).encode())
    assert store.resolve_current() is None
    with pytest.raises(PointerError):
        store.current_identity()


# ============================================================================
# durability errors + uncertain commits
# ============================================================================


def test_durability_error_before_rename_preserves_pointer_and_staging(store, monkeypatch):
    old_pointer = seed_current(store)
    real = gens._fsync_tree

    def failing(base: Path):
        if base.name.startswith(gens.BUILDING_PREFIX):
            raise DurabilityError("simulated artifact fsync failure")
        return real(base)

    monkeypatch.setattr(gens, "_fsync_tree", failing)
    with pytest.raises(DurabilityError):
        publish_generation(store, "g2", expected_current="g1")
    monkeypatch.undo()
    assert store.current_path.read_bytes() == old_pointer
    assert (store.root / ".building-g2").is_dir()  # staging preserved for retry
    assert not (store.generations_dir / "g2").exists()


def test_root_fsync_failure_after_replace_raises_uncertain(store, monkeypatch):
    real = gens._fsync_dir

    def failing(path: Path):
        if path == store.root:
            raise DurabilityError("simulated root fsync failure")
        return real(path)

    monkeypatch.setattr(gens, "_fsync_dir", failing)
    with pytest.raises(CommitStateUncertainError) as excinfo:
        publish_generation(store, "g1")
    monkeypatch.undo()
    err = excinfo.value
    assert err.may_have_committed is True
    intended = err.pointer_identity()
    assert intended["generation_id"] == "g1"
    assert intended["receipt_sha256"] == store_receipt_sha(store, "g1")
    # The replace DID happen — never claim the pointer is unchanged.
    pointer = store.current_identity()
    assert pointer == intended
    # And the store is fully consistent: recovery is just re-reading.
    assert store.resolve_current() is not None
    assert store.resolve_current().generation_id == "g1"


def test_activate_root_fsync_failure_also_uncertain(store, monkeypatch):
    seed_current(store)  # g1 current
    publish_generation(store, "g2", expected_current="g1")
    real = gens._fsync_dir

    def failing(path: Path):
        if path == store.root:
            raise DurabilityError("simulated root fsync failure")
        return real(path)

    monkeypatch.setattr(gens, "_fsync_dir", failing)
    with pytest.raises(CommitStateUncertainError) as excinfo:
        store.activate("g1", expected_current=pointer_identity(store, "g2"))
    monkeypatch.undo()
    assert excinfo.value.pointer_identity()["generation_id"] == "g1"
    assert store.resolve_current().generation_id == "g1"


# ============================================================================
# crash safety (single _atomic_replace choke point)
# ============================================================================


def test_crash_before_final_rename_preserves_old_bytes(store, monkeypatch):
    old_pointer = seed_current(store)
    monkeypatch.setattr(
        gens,
        "_atomic_replace",
        crash_replace_when(lambda dst: dst.name == "g2"),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish_generation(store, "g2", expected_current="g1")
    monkeypatch.undo()
    assert store.current_path.read_bytes() == old_pointer
    assert (store.root / ".building-g2").is_dir()
    assert not (store.generations_dir / "g2").exists()
    # Recovery: clear staging, retry the publish.
    store.abort_build("g2")
    publish_generation(store, "g2", expected_current="g1")
    assert store.resolve_current().generation_id == "g2"


def test_crash_before_pointer_replace_preserves_old_bytes(store, monkeypatch):
    old_pointer = seed_current(store)
    monkeypatch.setattr(
        gens,
        "_atomic_replace",
        crash_replace_when(lambda dst: dst.name == gens.CURRENT_FILENAME),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish_generation(store, "g2", expected_current="g1")
    monkeypatch.undo()
    # Old pointer bytes intact; g2 IS published (rename happened) — recover
    # with an explicit activate.
    assert store.current_path.read_bytes() == old_pointer
    assert (store.generations_dir / "g2").is_dir()
    result = store.activate("g2", expected_current=pointer_identity(store, "g1"))
    assert result.generation_id == "g2"
    assert store.resolve_current().generation_id == "g2"
    assert no_tmp_pointer_files(store)


# ============================================================================
# CAS identity binding
# ============================================================================


def test_publish_conflict_when_pointer_exists(store):
    seed_current(store)  # g1
    # Explicit absent-sentinel against an EXISTING pointer: the genuine CAS
    # conflict (production must never accept an unchecked None here).
    with pytest.raises(CurrentConflictError):
        publish_generation(store, "g2", expected_current=gens.EXPECTED_CURRENT_ABSENT)
    # g2 stays published; the pointer still names g1.
    assert store.resolve_current().generation_id == "g1"
    assert (store.generations_dir / "g2").is_dir()


def test_publish_rejects_unchecked_expected_current(store, tmp_path):
    """``None`` (no expectation captured) is rejected — never a silent clobber.

    Calls ``store.publish`` DIRECTLY: the ``publish_generation`` helper
    intentionally converts ``None`` to the explicit absent sentinel, so it
    cannot exercise the production None-rejection path.
    """
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(CurrentConflictError):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=None,  # type: ignore[arg-type]
        )


def test_cas_aba_rejected(store):
    publish_generation(store, "g1")
    publish_generation(store, "g2", expected_current="g1")
    stale = {"generation_id": "g1", "receipt_sha256": HEX("stale-but-well-formed")}
    with pytest.raises(CurrentConflictError):
        store.activate("g1", expected_current=stale)
    assert store.resolve_current().generation_id == "g2"


def test_cas_malformed_expected_identity_rejected(store):
    old_pointer = seed_current(store)
    # Each case stages its OWN generation ID (cleaned up after) so no case is
    # shadowed by BuildExistsError — every raw value must reach
    # CurrentConflictError from expected-identity validation.
    for i, bad in enumerate(({}, {"generation_id": "g1"}, {"generation_id": "g1", "receipt_sha256": "nothex"}, "g1")):
        gid = f"gcas-{i}"
        building = store.begin_build(gid)
        stage_generation(building)
        with pytest.raises(CurrentConflictError):
            store.publish(
                gid,
                identity=IDENTITY,
                compatibility=COMPAT,
                chroma_evidence=chroma_ev(),
                fts_evidence=fts_ev(),
                expected_current=bad,
            )
        store.abort_build(gid)
    assert store.current_path.read_bytes() == old_pointer


def test_current_identity_matches_resolve(store):
    assert store.current_identity() is None
    publish_generation(store, "g1")
    identity = store.current_identity()
    assert identity == pointer_identity(store, "g1")
    assert store.resolve_current().identity() == identity


# ============================================================================
# activate / rollback gates
# ============================================================================


def test_activate_rejects_unexpected_pointer_state(store):
    publish_generation(store, "g1")
    # Absent-sentinel against an EXISTING pointer: genuine CAS conflict.
    with pytest.raises(CurrentConflictError):
        store.activate("g1", expected_current=gens.EXPECTED_CURRENT_ABSENT)


def test_activate_missing_generation_leaves_pointer(store):
    old_pointer = seed_current(store)
    with pytest.raises(VerificationError):
        store.activate("missing-gen", expected_current=pointer_identity(store, "g1"))
    assert store.current_path.read_bytes() == old_pointer


def test_activate_expected_compatibility_gate(store):
    seed_current(store)  # g1 with COMPAT
    with pytest.raises(VerificationError, match="compatibility mismatch"):
        store.activate("g1", expected_current=pointer_identity(store, "g1"), expected_compatibility=COMPAT_V2)
    assert store.resolve_current().generation_id == "g1"


def test_activate_expected_identity_gate(store):
    seed_current(store)
    with pytest.raises(VerificationError, match="identity mismatch"):
        store.activate("g1", expected_current=pointer_identity(store, "g1"), expected_identity=IDENTITY_V2)
    assert store.resolve_current().generation_id == "g1"


def test_rollback_rejects_partial_generation(store):
    seed_current(store)  # g1 current
    publish_generation(store, "g2", expected_current="g1")
    g2_pointer = store.current_path.read_bytes()
    # Partially corrupt g1 (the rollback target).
    (store.generations_dir / "g1" / gens.CORPUS_ARTIFACT / "doc-a.md").write_bytes(b"corrupted")
    with pytest.raises(VerificationError):
        store.rollback("g1", expected_current=pointer_identity(store, "g2"))
    # Pointer left byte-identical to the successful g2 swap.
    assert store.current_path.read_bytes() == g2_pointer
    assert store.resolve_current().generation_id == "g2"


def test_rollback_rejects_incompatible_generation(store):
    seed_current(store)  # g1 with COMPAT
    publish_generation(store, "v2", expected_current="g1", compatibility=COMPAT_V2)
    pre_attempt = store.current_path.read_bytes()  # exact pre-attempt pointer bytes
    expected = pointer_identity(store, "v2")
    with pytest.raises(VerificationError, match="compatibility mismatch"):
        store.rollback("g1", expected_current=expected, expected_compatibility=COMPAT_V2)
    assert store.current_path.read_bytes() == pre_attempt  # byte-identical
    assert store.resolve_current().generation_id == "v2"


def test_rollback_requires_expected_current(store):
    seed_current(store)
    # Absent-sentinel against an EXISTING pointer: genuine CAS conflict.
    with pytest.raises(CurrentConflictError):
        store.rollback("g1", expected_current=gens.EXPECTED_CURRENT_ABSENT)


def test_successful_rollback_and_result_surface(store):
    seed_current(store)  # g1
    g1_pointer = store.current_path.read_bytes()  # bytes rollback must restore
    publish_generation(store, "g2", expected_current="g1")
    assert store.current_path.read_bytes() != g1_pointer
    result = store.rollback("g1", expected_current=pointer_identity(store, "g2"))
    assert result.generation_id == "g1"
    assert result.restart_required is True
    assert result.receipt["compatibility"] == COMPAT
    assert result.to_dict()["restart_required"] is True
    assert "restart_required=true" in result.summary()
    assert store.resolve_current().generation_id == "g1"
    # Success invariant: the pointer bytes are EXACTLY the originally
    # captured g1 pointer bytes (deterministic CAS serialization), not merely
    # an equivalent generation ID.
    assert store.current_path.read_bytes() == g1_pointer


# ============================================================================
# no GC; abort only touches staging
# ============================================================================


def test_no_garbage_collection(store):
    seed_current(store)
    publish_generation(store, "g2", expected_current="g1")
    publish_generation(store, "g3", expected_current="g2")
    for gid in ("g1", "g2", "g3"):
        assert (store.generations_dir / gid).is_dir()
    store.abort_build("nonexistent")  # no-op, no error


# ============================================================================
# (v3) reranker binding / provenance / sentinel CAS / v2 inspect-only
# ============================================================================


def test_reranker_disabled_is_explicit_none_state(store):
    publish_generation(store, "g1")
    receipt = store.verify_generation("g1")
    assert receipt["compatibility"]["reranker_enabled"] is False
    assert receipt["compatibility"]["reranker_model"] is None
    assert receipt["compatibility"]["reranker_artifact_sha256"] is None


@pytest.mark.parametrize(
    "over",
    [
        {"reranker_enabled": "false"},  # must be a hard bool
        {"reranker_enabled": True, "reranker_model": None},  # enabled needs a model
        {"reranker_enabled": True, "reranker_model": "xenova/mini"},  # needs artifact digest
        {"reranker_enabled": True, "reranker_model": "xenova/mini", "reranker_artifact_sha256": "not-hex"},
        {"reranker_enabled": False, "reranker_model": "xenova/mini"},  # disabled must be None-None
    ],
)
def test_reranker_binding_validation(store, over):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=compat(over),
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_reranker_enabled_with_exact_artifact_binds(store):
    building = store.begin_build("g1")
    stage_generation(building)
    enabled = compat(
        {
            "reranker_enabled": True,
            "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
            "reranker_artifact_sha256": HEX("reranker-artifact-v1"),
        }
    )
    with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
        result = store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=enabled,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    assert result.receipt["compatibility"]["reranker_artifact_sha256"] == HEX("reranker-artifact-v1")


def test_provenance_vault_head_recorded_and_40hex_accepted(store):
    head40 = "a" * 40
    publish_generation(store, "g1", provenance={"vault_head": head40})
    receipt = store.verify_generation("g1")
    assert receipt["provenance"] == {"vault_head": head40}


def test_provenance_64hex_sha256_object_accepted(store):
    head64 = HEX("head-object")
    publish_generation(store, "g1", provenance={"vault_head": head64})
    assert store.verify_generation("g1")["provenance"]["vault_head"] == head64


def test_provenance_invalid_head_rejected(store):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError, match="provenance"):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            provenance={"vault_head": "z" * 40},
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")


def test_first_publish_requires_explicit_absent_sentinel_or_none(store):
    # EXPECTED_CURRENT_ABSENT is accepted for the FIRST publish.
    publish_generation(store, "g1", expected_current=gens.EXPECTED_CURRENT_ABSENT)
    assert store.resolve_current().generation_id == "g1"


def test_v2_receipt_is_inspectable_but_never_servable(store, tmp_path):
    """A schema-v2 receipt stays readable for stats but servable=False."""
    publish_generation(store, "g1")
    receipt_path = store.generations_dir / "g1" / gens.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt["schema_version"] = 2
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n")
    # v2 fails strict resolution (pointer pins the v3 receipt sha).
    assert store.resolve_current() is None
    # ...but the lenient inspector still summarizes it.
    summary = gens.inspect_current_receipt(store.root)
    assert summary is not None
    assert summary["schema_version"] == 2
    assert summary["servable"] is False
    assert summary["reason"] == "schema_v2_rebuild_required"


def test_backend_generation_ids_required_and_bound(store):
    building = store.begin_build("g1")
    stage_generation(building)
    with pytest.raises(VerificationError):
        store.publish(
            "g1",
            identity=IDENTITY,
            compatibility=COMPAT,
            chroma_evidence=chroma_ev(backend_generation_id=None),
            fts_evidence=fts_ev(),
            expected_current=gens.EXPECTED_CURRENT_ABSENT,
        )
    store.abort_build("g1")
    assert not (store.root / ".building-g1").exists()
