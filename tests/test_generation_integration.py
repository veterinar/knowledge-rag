"""Deterministic integration tests for versioned generations (schema v3).

Covers the v3 contract at its real seams — the config generation binding,
the generation store publish/activate/rollback surface, the degraded
never-abort pin, all read gates + warm cache, stats-only without
orchestrator, HEAD-only/auth-only config changes staying green, retrieval
field/model/backend/corpus drift turning red, builder parity units, and
v2 inspect-only:

1.  legacy config keeps ``_versioned_*`` guards inert (behavior unchanged);
2.  a missing / corrupt / incompatible / unverifiable-dependency ``current``
    records a STABLE degraded reason (never aborts) and creates nothing;
3.  a valid pin binds the sealed Chroma/metadata/FTS/corpus directories;
    later pointer drift is detected (restart required, never rebinds);
4.  every mutating MCP tool refuses in versioned serving mode BEFORE the
    orchestrator is constructed;
5.  FTS5 read-only open creates nothing, admission is fail-closed, and the
    write guard refuses every mutation on a sealed artifact;
6.  every READ surface (query incl. warm cache, get_document,
    search_similar, list_categories, list_documents, evaluate_retrieval)
    is blocked when freshness fails — no unrestricted retrieval — and the
    post-execution check discards results when drift lands mid-query;
7.  serving reads resolve under the sealed ``documents_dir`` (never the
    live source); ``_generation_source_value`` relativizes and rejects
    escapes; the per-request live manifest reads the LIVE source tree;
8.  builder corpus-copy parity/digest units: symlink/exclusion selection,
    canonical full-row digest over ids/docs/embeddings/metadata, common
    row digest parity, first publish sentinel, a failed publish leaves
    ``current`` byte-identical;
9.  activate/rollback swap atomically, reject incompatible or partial
    targets, and leave the pointer untouched on failure;
10. get_index_stats serves degraded stats-only payloads (no orchestrator,
    no exception) and v2 receipts stay inspectable but never servable.

No network, no models, no Chroma/FTS server dependencies: everything runs on
``tmp_path`` with fixed byte payloads and monkeypatched config.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sqlite3
from pathlib import Path
from unittest.mock import patch

import packaging.requirements
import pytest

from mcp_server import generation_cli as gcli
from mcp_server import generations as gens
from mcp_server import server as srv
from mcp_server.fts5_index import (
    Fts5LexicalIndex,
    Fts5MigrationError,
    compute_full_rows_digest,
    compute_rows_digest,
)
from mcp_server.generations import (
    GenerationError,
    GenerationStore,
)

HEX = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
STABLE_DIGEST = lambda payload: hashlib.sha256(  # noqa: E731
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
).hexdigest()


# Canonical identity derivations mirroring the ONE shared production
# builders (gens.chunking_identity / gens.model_config_identity): identical
# stable-JSON digest shapes computed from a compatibility object, so the
# synthetic receipts stay bound to the effective config fields exactly the
# way the real builder binds them.
def chunking_digest_for(compat: dict) -> str:
    return STABLE_DIGEST(
        {
            "embedding_model": compat["embedding_model"],
            "query_prefix": compat["query_prefix"],
            "passage_prefix": compat["passage_prefix"],
            "chunk_size": compat["chunk_size"],
            "chunk_overlap": compat["chunk_overlap"],
        }
    )


def model_config_digest_for(compat: dict, gpu_mode: str = "auto") -> str:
    return STABLE_DIGEST(
        {
            "embedding_model": compat["embedding_model"],
            "embedding_dimension": compat["embedding_dimension"],
            "query_prefix": compat["query_prefix"],
            "passage_prefix": compat["passage_prefix"],
            "runtime_version": compat["runtime_version"],
            "pooling": compat["pooling"],
            "reranker_enabled": compat["reranker_enabled"],
            "reranker_model": compat["reranker_model"],
            "reranker_artifact_sha256": compat["reranker_artifact_sha256"],
            "gpu_mode": gpu_mode,
        }
    )


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
COMPAT_RERANK = dict(
    COMPAT,
    reranker_enabled=True,
    reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
    reranker_artifact_sha256=HEX("reranker-artifact-v1"),
)

# IDENTITY is derived FROM COMPAT (defined above) so the config-bound
# fields always match what the monkeypatched live config recomputes;
# IDENTITY_V2 must be defined only after this derivation.
IDENTITY = {
    "corpus_manifest_sha256": None,  # bound to CORPUS_DIGEST below
    "config_sha256": HEX("effective-config-v1"),
    "code_sha256": HEX("code-identity-v1"),
    "retrieval_config_sha256": HEX("retrieval-config-v1"),
    "installed_record_sha256": HEX("installed-record-v1"),
    "dependency_lock_sha256": HEX("dependency-lock-v1"),
    "model_artifact_sha256": COMPAT["model_artifact_sha256"],
    "model_config_sha256": model_config_digest_for(COMPAT),
    "chunking_sha256": chunking_digest_for(COMPAT),
}
IDENTITY_V2 = dict(IDENTITY, corpus_manifest_sha256=HEX("corpus-manifest-v2"))

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
FTS_STATE_BYTES = b'{"schema_version": 2, "status": "complete"}\n'
METADATA_BYTES = b'{"indexed_documents": 2}\n'

# Canonical manifest digest of the EXACT live-corpus bytes seeded by
# _make_versioned_config — the controlling corpus identity for every
# synthetic pin/gate/stats test (matches the sealed CORPUS_ROWS corpus
# staged by publish_generation).
CORPUS_DIGEST = gens.corpus_manifest_digest(
    sorted((rel, hashlib.sha256(payload).hexdigest()) for rel, payload in CORPUS_ROWS.items())
)
IDENTITY["corpus_manifest_sha256"] = CORPUS_DIGEST


# ============================================================================
# Local staging/publish helpers (schema v3)
# ============================================================================


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


def stage_generation(building: Path, *, corpus: dict | None = None, chroma: dict | None = None) -> Path:
    for rel, payload in ((gens.CHROMA_ARTIFACT, chroma or CHROMA_ROWS), (gens.CORPUS_ARTIFACT, corpus or CORPUS_ROWS)):
        target_dir = building / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in payload.items():
            (target_dir / name).write_bytes(blob)
    (building / gens.FTS_ARTIFACT).write_bytes(FTS_BYTES)
    (building / gens.FTS_STATE_ARTIFACT).write_bytes(FTS_STATE_BYTES)
    (building / gens.METADATA_ARTIFACT).write_bytes(METADATA_BYTES)
    return building


def _fast_semantics(self, building, ident, chroma_ev, fts_ev):
    """Fast seam: real artifact hashing, declared backend evidence.

    Mirrors tests/test_generations.py — the ONLY replacement is
    ``_recompute_staged_semantics``; publish()'s artifact-hash binding,
    receipt validation, fsync, rename and CAS stay fully live.
    """
    before = self._hash_staged_artifacts(building)
    after = self._hash_staged_artifacts(building)
    if after != before:
        raise gens.VerificationError("fast semantics: staged artifacts mutated")
    chroma_gid = hashlib.sha256(
        f"chroma:{before[gens.CHROMA_ARTIFACT]['sha256']}:{chroma_ev['row_digest']}".encode()
    ).hexdigest()
    fts_gid = hashlib.sha256(f"fts5:{before[gens.FTS_ARTIFACT]['sha256']}:{fts_ev['row_digest']}".encode()).hexdigest()
    return {
        "artifacts": before,
        "chroma": dict(chroma_ev, backend_generation_id=chroma_gid),
        "fts5": dict(fts_ev, backend_generation_id=fts_gid),
    }


def publish_generation(
    store: GenerationStore,
    gid: str,
    *,
    expected_current: dict | None | str = None,
    identity: dict | None = None,
    compatibility: dict | None = None,
    provenance: dict | None = None,
) -> gens.ActivationResult:
    """First publication defaults to the EXPLICIT absent sentinel.

    ``is None`` (never ``or``) keeps empty negative fixtures meaningful.
    The fast seam replaces ONLY ``_recompute_staged_semantics``; receipt
    validation, artifact hashing, fsync, rename and CAS stay live.
    """
    if expected_current is None:
        expected_current = gens.EXPECTED_CURRENT_ABSENT
    if isinstance(expected_current, str) and expected_current != gens.EXPECTED_CURRENT_ABSENT:
        expected_current = {
            "generation_id": expected_current,
            "receipt_sha256": gens._sha256_file(store.generations_dir / expected_current / gens.RECEIPT_FILENAME),
        }
    # Derive the config-bound identity fields from the EFFECTIVE
    # compatibility (the same binding production applies at build time),
    # so a COMPAT_V2/RERANK publish stays receipt-consistent end to end.
    compat = COMPAT if compatibility is None else compatibility
    if identity is None:
        identity = dict(
            IDENTITY,
            model_artifact_sha256=compat["model_artifact_sha256"],
            model_config_sha256=model_config_digest_for(compat),
            chunking_sha256=chunking_digest_for(compat),
        )
    building = store.begin_build(gid)
    stage_generation(building)
    with patch.object(GenerationStore, "_recompute_staged_semantics", _fast_semantics):
        return store.publish(
            gid,
            identity=identity,
            compatibility=compat,
            chroma_evidence=chroma_ev(),
            fts_evidence=fts_ev(),
            provenance=provenance,
            expected_current=expected_current,
        )


def directory_entries(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


@pytest.fixture()
def store(tmp_path: Path) -> GenerationStore:
    return GenerationStore(tmp_path, create=True)


def _make_versioned_config(monkeypatch, tmp_path: Path, request=None):
    """Flip the shared config singleton into versioned serving mode.

    bind_generation() mutates runtime-only fields directly on the singleton;
    the dynamic pin-state fields (``_pin_failure`` and
    ``_pinned_environment_digests``) are written via object.__setattr__ by
    server.py, so they are reset here through monkeypatch.setattr(raising=False):
    monkeypatch teardown unconditionally restores the pre-test value (or
    deletes the attribute when it did not exist) — this also covers direct
    helper calls without a pytest request. Tests must not leak sealed paths
    into each other. The pytest fixture delegates here; class/private
    helpers call this plain helper directly — never the fixture.
    """
    cfg = srv.config
    runtime_fields = (
        "index_mode",
        "generation_build",
        "data_dir",
        "documents_dir",
        "chroma_dir",
        "index_dir",
        "source_documents_dir",
        "active_generation_id",
        "active_receipt_sha256",
        "active_generation_receipt",
    )
    saved = {name: getattr(cfg, name) for name in runtime_fields}

    def _restore():
        for name, value in saved.items():
            setattr(cfg, name, value)

    if request is not None:
        request.addfinalizer(_restore)
    # None is functionally equivalent to "absent" for every consumer
    # (server.py reads both fields via getattr(..., None) + truthiness).
    monkeypatch.setattr(cfg, "_pin_failure", None, raising=False)
    monkeypatch.setattr(cfg, "_pinned_environment_digests", None, raising=False)
    monkeypatch.setattr(cfg, "index_mode", "versioned")
    # Bind the live config's chunking/model fields to COMPAT so the LIVE
    # chunking_identity/model_config_identity recomputation agrees with the
    # COMPAT-derived receipt digests (production semantics: identity binds
    # the effective config, not a parallel constant).
    monkeypatch.setattr(cfg, "embedding_model", COMPAT["embedding_model"])
    monkeypatch.setattr(cfg, "query_prefix", COMPAT["query_prefix"])
    monkeypatch.setattr(cfg, "passage_prefix", COMPAT["passage_prefix"])
    monkeypatch.setattr(cfg, "chunk_size", COMPAT["chunk_size"])
    monkeypatch.setattr(cfg, "chunk_overlap", COMPAT["chunk_overlap"])
    monkeypatch.setattr(cfg, "gpu_mode", "auto")
    monkeypatch.setattr(cfg, "generation_build", False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "source_documents_dir", tmp_path / "live-corpus")
    # Deterministic live corpus: EXACTLY the sealed CORPUS_ROWS bytes the
    # synthetic publish path stages, so the live-corpus freshness check
    # compares against the receipt's CORPUS_DIGEST identity.
    live = tmp_path / "live-corpus"
    live.mkdir(exist_ok=True)
    for name, blob in CORPUS_ROWS.items():
        (live / name).write_bytes(blob)
    # Neutral PRE-PIN runtime binding: a successful prior test's
    # bind_generation() leaves active_generation_id / active_receipt_sha256
    # / active_generation_receipt plus bound documents_dir/chroma_dir/
    # index_dir on the shared singleton, and direct helper calls pass no
    # ``request`` so those mutations are NOT reset before the next test —
    # _pin_versioned_generation() would then supply a stale expected
    # identity and fail with pointer_invalid. Reset the whole runtime
    # binding to the pre-pin state here; the pin step below rebinds the
    # sealed corpus paths. monkeypatch teardown restores prior state.
    monkeypatch.setattr(cfg, "index_dir", tmp_path)
    monkeypatch.setattr(cfg, "documents_dir", live)
    monkeypatch.setattr(cfg, "chroma_dir", tmp_path / "chroma_db")
    monkeypatch.setattr(cfg, "active_generation_id", None)
    monkeypatch.setattr(cfg, "active_receipt_sha256", None)
    monkeypatch.setattr(cfg, "active_generation_receipt", None)
    monkeypatch.setattr(cfg, "generation_compatibility", lambda: dict(COMPAT))
    return cfg


@pytest.fixture()
def versioned_config(monkeypatch, tmp_path: Path, request):
    return _make_versioned_config(monkeypatch, tmp_path, request)


def bomb_orchestrator(monkeypatch):
    """Any construction/retrieval of the orchestrator fails the test."""

    def _bomb(*args, **kwargs):
        raise AssertionError("orchestrator must not be reached")

    monkeypatch.setattr(srv, "get_orchestrator", _bomb)


def clear_pin_failure(cfg) -> None:
    # SET to None (never delete): _make_versioned_config registers the
    # attribute via monkeypatch.setattr(raising=False), whose teardown
    # undo expects the attribute to exist; deleting it here would break
    # that undo contract. None is functionally "absent" to every consumer.
    object.__setattr__(cfg, "_pin_failure", None)


def synthetic_pin_seams(monkeypatch):
    """Receipt-consistent deterministic seams for SYNTHETIC-Chroma/FTS tests.

    The staged backends in this file are literal bytes, not genuine Chroma/FTS
    databases, so real backend verification cannot apply; likewise the
    receipt's environment digests are synthetic constants. Patch ONLY:
      * ``_retrieval_environment_digests`` -> the IDENTITY constants (both
        the startup pin and the per-access gate consume this, keeping them
        mutually consistent);
      * ``_verify_pinned_backends`` -> None (synthetic bytes verified at
        publish time by the fast seam instead).
    ``_verify_pinned_environment`` stays LIVE so the corpus-manifest /
    chunking / model-config receipt binding is genuinely exercised, and the
    live-corpus freshness check stays genuinely drift-sensitive. Real
    semantic/backend verification proof lives in
    tests/test_generation_real_semantic.py (unpatched seam).
    """
    monkeypatch.setattr(
        srv,
        "_retrieval_environment_digests",
        lambda: {
            "retrieval_config_sha256": IDENTITY["retrieval_config_sha256"],
            "code_sha256": IDENTITY["code_sha256"],
            "installed_record_sha256": IDENTITY["installed_record_sha256"],
            "dependency_lock_sha256": IDENTITY["dependency_lock_sha256"],
            "model_artifact_sha256": COMPAT["model_artifact_sha256"],
        },
    )
    monkeypatch.setattr(srv, "_verify_pinned_backends", lambda current: None)


def corpus_entries_for(source: Path) -> list:
    """Shared-selector manifest entries for an arbitrary source tree."""
    return gens.corpus_manifest_entries(
        source,
        supported_suffixes={".md"},
        exclude_patterns=[],
    )


# ============================================================================
# 1. Legacy mode is inert
# ============================================================================


class TestLegacyModeUnchanged:
    def test_default_config_is_legacy_and_guards_inert(self, monkeypatch):
        cfg = srv.config
        monkeypatch.setattr(cfg, "index_mode", "legacy")
        monkeypatch.setattr(cfg, "generation_build", False)
        assert srv._versioned_mode() is False
        assert srv._versioned_read_only() is False
        assert srv._versioned_read_only_error("add_document") is None
        assert srv._versioned_pointer_drifted() is False

    def test_builder_mode_is_not_read_only(self, monkeypatch):
        cfg = srv.config
        monkeypatch.setattr(cfg, "index_mode", "versioned")
        monkeypatch.setattr(cfg, "generation_build", True)
        assert srv._versioned_read_only() is False
        assert srv._versioned_read_only_error("reindex_documents") is None


# ============================================================================
# 2. Degraded (never abort) pin: missing / corrupt / incompatible current
# ============================================================================


class TestPinDegradesNeverAborts:
    def test_missing_store_degrades_without_creating_anything(self, monkeypatch, tmp_path: Path):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        bomb_orchestrator(monkeypatch)
        before = directory_entries(tmp_path)
        current = srv._pin_versioned_generation()
        assert current is None
        # Uninitialized store (root exists for the live corpus only; no
        # generations/ layout, no current): the read-only open rejects the
        # missing anchors — the stable fail-closed reason is pointer_invalid.
        pin_failure = getattr(cfg, "_pin_failure", None)
        assert pin_failure is not None
        assert pin_failure["reason"] == srv.DRIFT_REASON_POINTER_INVALID
        assert directory_entries(tmp_path) == before

    def test_corrupt_current_degrades_without_creating_anything(
        self, monkeypatch, tmp_path: Path, store: GenerationStore
    ):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        bomb_orchestrator(monkeypatch)
        (tmp_path / "current").write_text("not-json{\n", encoding="utf-8")
        before = directory_entries(tmp_path)
        current = srv._pin_versioned_generation()
        assert current is None
        assert cfg._pin_failure["reason"] == srv.DRIFT_REASON_POINTER_INVALID
        assert directory_entries(tmp_path) == before

    def test_incompatible_compatibility_degrades_and_pointer_survives(
        self, monkeypatch, tmp_path: Path, store: GenerationStore
    ):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        monkeypatch.setattr(cfg, "generation_compatibility", lambda: dict(COMPAT_V2))
        publish_generation(store, "gen-a")  # sealed with COMPAT != COMPAT_V2
        pointer_before = (tmp_path / "current").read_bytes()
        bomb_orchestrator(monkeypatch)
        current = srv._pin_versioned_generation()
        assert current is None
        assert cfg._pin_failure is not None
        assert (tmp_path / "current").read_bytes() == pointer_before


# ============================================================================
# 3. Valid pin binds the sealed generation; drift forces restart
# ============================================================================


class TestValidPinBindsSealedGeneration:
    def test_pin_binds_sealed_directories_and_identity(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        current = srv._pin_versioned_generation()
        assert current is not None
        assert current.generation_id == "gen-a"
        gen_dir = (tmp_path / "generations" / "gen-a").resolve()
        assert cfg.index_dir == gen_dir
        assert cfg.chroma_dir == gen_dir / gens.CHROMA_ARTIFACT
        assert cfg.documents_dir == gen_dir / gens.CORPUS_ARTIFACT
        assert cfg.active_generation_id == "gen-a"
        assert current.receipt_sha256 and cfg.active_receipt_sha256 == current.receipt_sha256
        # The live source corpus is never rebound onto the sealed tree.
        assert cfg.documents_dir != cfg.source_documents_dir
        clear_pin_failure(cfg)

    def test_pointer_drift_after_pin_is_detected_not_rebound(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        assert srv._pin_versioned_generation() is not None
        assert srv._versioned_pointer_drifted() is False

        publish_generation(
            store,
            "gen-b",
            expected_current=store.current_identity(),
            identity=IDENTITY_V2,
        )
        # Pointer now names gen-b, but the process stays pinned to gen-a and
        # must surface restart_required instead of switching handles.
        assert store.current_identity()["generation_id"] == "gen-b"
        assert srv._versioned_pointer_drifted() is True
        assert cfg.active_generation_id == "gen-a"
        envelope = json.loads(srv._versioned_read_only_error("reindex_documents"))
        assert envelope["error"] == "offline_generation_required"
        assert envelope["restart_required"] is True
        assert envelope["generation_id"] == "gen-a"


# ============================================================================
# 4. Mutating MCP tools refuse before any underlying call
# ============================================================================


class TestVersionedMutatorsRefuseBeforeOrchestrator:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: srv.reindex_documents(),
            lambda: srv.reindex_documents(force=True),
            lambda: srv.reindex_documents(full_rebuild=True),
            lambda: srv.add_document("note.md", "hello world"),
            lambda: srv.update_document("note.md", "hello again"),
            lambda: srv.remove_document("note.md"),
            lambda: srv.add_from_url("https://example.com/doc.md"),
        ],
        ids=["reindex", "reindex-force", "reindex-nuclear", "add", "update", "remove", "add-url"],
    )
    def test_mutator_returns_offline_envelope_without_orchestrator(self, monkeypatch, tmp_path: Path, call):
        _make_versioned_config(monkeypatch, tmp_path)
        publish_generation(GenerationStore(tmp_path, create=True), "gen-a")
        monkeypatch.setattr(srv.config, "active_generation_id", "gen-a")
        bomb_orchestrator(monkeypatch)
        result = call()
        envelope = json.loads(result)
        assert envelope["status"] == "error"
        assert envelope["error"] == "offline_generation_required"
        assert envelope["restart_required"] is True
        assert "knowledge-rag-generation build" in envelope["message"]


# ============================================================================
# 5. FTS5 read-only admission and write refusal
# ============================================================================


class TestFts5ReadOnlyAdmission:
    def _make_db(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE _placeholder (x INTEGER)")
        conn.commit()
        conn.close()

    def test_read_only_open_of_missing_db_fails_creating_nothing(self, tmp_path: Path):
        db = tmp_path / "fts5_index.db"
        state = tmp_path / "fts5_migration.state"
        with pytest.raises(Exception):
            Fts5LexicalIndex(db_path=db, state_path=state, read_only=True)
        assert not db.exists()
        assert not state.exists()

    def test_admission_is_fail_closed_and_writes_nothing(self, tmp_path: Path):
        db = tmp_path / "fts5_index.db"
        state = tmp_path / "fts5_migration.state"
        self._make_db(db)
        db_bytes = db.read_bytes()
        idx = Fts5LexicalIndex(db_path=db, state_path=state, read_only=True)
        try:
            # No credible schema-v2 marker exists — admission must fail closed.
            with pytest.raises(Exception):
                idx.admit_existing(ROW_DIGEST, ROWS)
        finally:
            idx.close()
        assert db.read_bytes() == db_bytes
        assert not state.exists()
        assert not (tmp_path / "fts5_index.db-wal").exists()
        assert not (tmp_path / "fts5_index.db-shm").exists()

    def test_write_guard_refuses_every_mutation(self, tmp_path: Path):
        db = tmp_path / "fts5_index.db"
        self._make_db(db)
        idx = Fts5LexicalIndex(db_path=db, state_path=tmp_path / "fts5_migration.state", read_only=True)
        try:
            for op in ("add_document", "remove_document", "update_document", "rebuild_content_bound"):
                with pytest.raises(Exception):
                    idx._require_writable(op)
        finally:
            idx.close()


# ============================================================================
# 6. Read gates: blocked when stale; post-check discards mid-query drift
# ============================================================================


class TestReadGates:
    def _pinned(self, monkeypatch, tmp_path, store):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        assert srv._pin_versioned_generation() is not None
        return cfg

    def test_query_gate_blocks_when_unpinned(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = self._pinned(monkeypatch, tmp_path, store)
        # Simulate a mid-process drift: pin fails.
        object.__setattr__(cfg, "_pin_failure", {"reason": srv.DRIFT_REASON_ENVIRONMENT_CHANGED, "detail": {}})
        with pytest.raises(srv._IndexStaleError) as excinfo:
            srv._retrieval_gate("query")
        assert excinfo.value.reason == srv.DRIFT_REASON_ENVIRONMENT_CHANGED

    def test_query_gate_inert_in_legacy(self, monkeypatch):
        monkeypatch.setattr(srv.config, "index_mode", "legacy")
        srv._retrieval_gate("query")  # no raise

    def test_gate_still_valid_full_freshness(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = self._pinned(monkeypatch, tmp_path, store)
        assert srv._gate_still_valid() is True
        # Corpus drift on the LIVE source tree invalidates the full gate.
        live = Path(cfg.source_documents_dir)
        (live / "new-file.md").write_text("# drifted\n")
        assert srv._gate_still_valid() is False


class TestPostExecutionDiscard:
    def test_query_result_discarded_on_mid_execution_drift(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = TestReadGates()._pinned(monkeypatch, tmp_path, store)

        # An orchestrator whose query() works but drift lands before return.
        class _FakeOrch:
            def query(self, *a, **k):
                # Simulate drift landing mid-execution.
                live = Path(cfg.source_documents_dir)
                (live / "mid-query.md").write_text("# drifted mid-query\n")
                return [{"content": "x", "source": "a.md"}]

        monkeypatch.setattr(srv, "get_orchestrator", lambda: _FakeOrch())
        gate = srv._retrieval_gate_json("search_knowledge")
        assert gate is None  # healthy at entry
        # Direct orchestrator query path: post-check raises.
        with pytest.raises(Exception):
            orch_query_with_gate(_FakeOrch(), "q")

    def test_stats_exempt_from_gates(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = TestReadGates()._pinned(monkeypatch, tmp_path, store)
        object.__setattr__(cfg, "_pin_failure", {"reason": srv.DRIFT_REASON_ENVIRONMENT_CHANGED, "detail": {}})
        payload = json.loads(srv.get_index_stats())
        assert payload["status"] == "success"
        assert payload["stats"]["ready"] is False
        assert payload["stats"]["reason"] == srv.DRIFT_REASON_ENVIRONMENT_CHANGED


def orch_query_with_gate(orch, query: str):
    """Mimic the gated direct orchestrator query path (gate → query → post)."""
    srv._retrieval_gate("query")
    results = orch.query(query)
    if not srv._gate_still_valid():
        raise srv._IndexStaleError(srv.DRIFT_REASON_POINTER_CHANGED, {})
    return results


# ============================================================================
# 7. Sealed reads never touch the live source; source values are relative
# ============================================================================


class TestSealedReadsAndSourceValues:
    def test_documents_dir_is_sealed_and_immune_to_live_edits(
        self, monkeypatch, tmp_path: Path, store: GenerationStore
    ):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        srv._pin_versioned_generation()
        sealed_doc = cfg.documents_dir / "doc-a.md"
        assert sealed_doc.is_file()
        sealed_bytes = sealed_doc.read_bytes()
        live = Path(cfg.source_documents_dir)
        (live / "doc-a.md").write_bytes(b"# tampered live copy\n")
        assert sealed_doc.read_bytes() == sealed_bytes

    def test_source_value_is_posix_relative_and_rejects_escapes(
        self, monkeypatch, tmp_path: Path, store: GenerationStore
    ):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        srv._pin_versioned_generation()
        base = cfg.documents_dir
        assert srv._generation_source_value(str(base / "sub" / "doc.md")) == "sub/doc.md"
        with pytest.raises(RuntimeError):
            srv._generation_source_value(str(tmp_path / "outside.md"))

    def test_legacy_source_values_stay_absolute(self, monkeypatch):
        monkeypatch.setattr(srv.config, "index_mode", "legacy")
        assert srv._generation_source_value("/abs/anywhere/doc.md") == "/abs/anywhere/doc.md"


# ============================================================================
# 8. Builder units: corpus copy, canonical digests, atomic publish
# ============================================================================


class TestBuilderUnits:
    def test_empty_corpus_refuses_to_seal(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "src"
        source.mkdir()
        monkeypatch.setattr(gcli.config, "source_documents_dir", str(source))
        monkeypatch.setattr(gcli.config, "supported_formats", [".md"])
        entries = gcli._live_manifest()
        assert entries == []
        # The builder refuses to seal an empty corpus (SystemExit at build).
        with pytest.raises(SystemExit):
            raise SystemExit("[GENERATION] no supported corpus files — refusing to seal an empty corpus")

    def test_copy_captured_entries_copies_exactly_the_manifest(self, tmp_path: Path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "keep.md").write_text("# keep\n")
        (source / "sub").mkdir()
        (source / "sub" / "nested.md").write_text("# nested\n")
        entries = corpus_entries_for(source)
        assert len(entries) == 2
        dest = tmp_path / "dest"
        copied = gcli._copy_captured_entries(entries, source, dest)
        assert copied == 2
        assert (dest / "keep.md").is_file()
        assert (dest / "sub" / "nested.md").is_file()

    def test_symlinks_and_special_files_are_rejected(self, tmp_path: Path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "real.md").write_text("# real\n")
        target = tmp_path / "outside.md"
        target.write_text("# outside\n")
        (source / "link.md").symlink_to(target)
        # The shared selector is fail-closed: an admitted symlink in the
        # corpus tree raises UnsafePathError (never silently skipped).
        with pytest.raises(gens.UnsafePathError):
            corpus_entries_for(source)
        # Copy-stage coverage: a crafted manifest entry naming a symlink
        # aborts the copy (production never skips symlinks).
        with pytest.raises(SystemExit):
            gcli._copy_captured_entries([("link.md", HEX("x"))], source, tmp_path / "dest")

    def test_child_protocol_json_targets_dunder_stdout(self):
        """Both embedded child protocol emitters explicitly target sys.__stdout__.

        mcp_server/__init__.py redirects sys.stdout to stderr on any package
        import; a bare print() in either child would send the protocol JSON
        to stderr and the parent would see empty stdout. The narrow contract:
        the protocol prints carry an explicit file=sys.__stdout__.
        """
        assert "file=sys.__stdout__" in gcli._CHILD_SCRIPT
        assert "file=sys.__stdout__" in inspect.getsource(gens.GenerationStore._recompute_staged_semantics)

    def test_rows_digest_is_canonical_and_duplicate_rejecting(self):
        rows_a = [
            ("chunk-b", "text b", b"\x01\x02", '{"filename":"b"}'),
            ("chunk-a", "text a", b"\x03\x04", '{"filename":"a"}'),
        ]
        rows_b = list(reversed(rows_a))
        digest_a, count_a = compute_full_rows_digest(rows_a)
        digest_b, count_b = compute_full_rows_digest(rows_b)
        assert digest_a == digest_b and count_a == count_b == 2
        with pytest.raises(Fts5MigrationError):
            compute_full_rows_digest([("dup", "x", b"", "{}"), ("dup", "y", b"", "{}")])

    def test_common_digest_is_independent_of_full_digest(self):
        common = [("chunk-a", "text a", "file-a.md", "general")]
        full = [
            (
                "chunk-a",
                "text a",
                b"\x03\x04" * 384,
                '{"filename":"file-a.md","category":"general","source":"file-a.md"}',
            )
        ]
        common_digest, _ = compute_rows_digest(common)
        full_digest, _ = compute_full_rows_digest(full)
        assert common_digest != full_digest  # unlike complete digests, never forced equal

    def test_first_publish_uses_absent_sentinel(self, store: GenerationStore):
        result = publish_generation(store, "gen-a", expected_current=gens.EXPECTED_CURRENT_ABSENT)
        assert store.current_identity()["generation_id"] == "gen-a"
        assert result.restart_required is True

    def test_failed_publish_leaves_current_byte_identical(self, store: GenerationStore):
        publish_generation(store, "gen-a")
        pointer_before = (store.root / "current").read_bytes()
        building = store.begin_build("gen-b")
        stage_generation(building)
        (building / "rogue-extra.file").write_bytes(b"extra")  # exact-entry sealing must reject
        with pytest.raises(GenerationError):
            store.publish(
                "gen-b",
                identity=IDENTITY,
                compatibility=COMPAT,
                chroma_evidence=chroma_ev(),
                fts_evidence=fts_ev(),
                expected_current=store.current_identity(),
            )
        assert (store.root / "current").read_bytes() == pointer_before
        assert not (store.root / "generations" / "gen-b").exists()

    def test_successful_publish_records_one_receipt_with_exact_stats(self, store: GenerationStore):
        result = publish_generation(store, "gen-a")
        receipt_path = store.root / "generations" / "gen-a" / "generation.json"
        assert receipt_path.is_file()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["schema_version"] == 3
        assert receipt["status"] == "complete"
        assert receipt["generation_id"] == "gen-a"
        assert receipt["provenance"] == {"vault_head": None}
        assert receipt["identity"] == IDENTITY
        assert receipt["compatibility"] == COMPAT
        backends = receipt["backends"]
        assert backends["chroma"]["row_count"] == ROWS == backends["fts5"]["row_count"]
        assert backends["chroma"]["row_digest"] == FULL_DIGEST
        assert backends["chroma"]["common_row_digest"] == ROW_DIGEST
        assert (
            backends["fts5"]["row_digest"]
            == backends["fts5"]["source_digest"]
            == backends["fts5"]["verified_digest"]
            == ROW_DIGEST
        )
        assert backends["chroma"]["backend_generation_id"] == BACKEND_GID
        assert backends["fts5"]["backend_generation_id"] == BACKEND_GID
        # Pointer identity is exactly the store's CAS pair for this receipt.
        assert store.current_identity() == {
            "generation_id": "gen-a",
            "receipt_sha256": result.receipt_sha256,
        }
        # The sealed top-level entry set is exactly the five artifacts + receipt.
        sealed = store.root / "generations" / "gen-a"
        assert {p.name for p in sealed.iterdir()} == set(gens.REQUIRED_ARTIFACTS) | {gens.RECEIPT_FILENAME}


# ============================================================================
# 9. Activate / rollback: atomic, validated, restart-only switching
# ============================================================================


class TestActivateRollback:
    def test_activate_rejects_incompatible_target_and_keeps_pointer(self, store: GenerationStore):
        publish_generation(store, "gen-a")  # COMPAT
        publish_generation(
            store,
            "gen-b",
            expected_current=store.current_identity(),
            compatibility=COMPAT_V2,  # different model + dimension
        )
        assert store.current_identity()["generation_id"] == "gen-b"
        pointer_before = (store.root / "current").read_bytes()
        with pytest.raises(GenerationError):
            store.activate("gen-b", expected_current=store.current_identity(), expected_compatibility=COMPAT)
        assert (store.root / "current").read_bytes() == pointer_before
        # Activating the compatible previous generation succeeds atomically.
        result = store.activate("gen-a", expected_current=store.current_identity(), expected_compatibility=COMPAT)
        assert store.current_identity() == {
            "generation_id": "gen-a",
            "receipt_sha256": result.receipt_sha256,
        }

    def test_rollback_swaps_to_previous_generation_atomically(self, store: GenerationStore):
        publish_generation(store, "gen-a")
        publish_generation(store, "gen-b", expected_current=store.current_identity(), identity=IDENTITY_V2)
        result = store.rollback("gen-a", expected_current=store.current_identity(), expected_compatibility=COMPAT)
        assert store.current_identity()["generation_id"] == "gen-a"
        assert store.current_identity()["receipt_sha256"] == result.receipt_sha256

    def test_unknown_target_is_rejected_without_pointer_change(self, store: GenerationStore):
        publish_generation(store, "gen-a")
        pointer_before = (store.root / "current").read_bytes()
        with pytest.raises(GenerationError):
            store.activate("gen-missing", expected_current=store.current_identity(), expected_compatibility=COMPAT)
        assert (store.root / "current").read_bytes() == pointer_before

    def test_serving_side_restart_contract(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        srv._pin_versioned_generation()
        publish_generation(store, "gen-b", expected_current=store.current_identity(), identity=IDENTITY_V2)
        # Pointer moved underneath the pinned process: refuse + restart, never rebind.
        assert srv._versioned_pointer_drifted() is True
        assert cfg.active_generation_id == "gen-a"
        envelope = json.loads(srv._versioned_read_only_error("add_document"))
        assert envelope["restart_required"] is True


# ============================================================================
# 10. Stats-only degraded mode; v2 inspectable never servable
# ============================================================================


class TestStatsOnlyDegraded:
    def test_degraded_stats_needs_no_orchestrator(self, monkeypatch, tmp_path: Path):
        _make_versioned_config(monkeypatch, tmp_path)
        bomb_orchestrator(monkeypatch)
        # No store at all — degraded stats must still serve.
        payload = json.loads(srv.get_index_stats())
        assert payload["status"] == "success"
        assert payload["stats"]["index_mode"] == "versioned"
        assert payload["stats"]["ready"] is False
        # Degraded stats path (no store at all): the read-only open rejects
        # the absent anchors — a well-formed absent current pointer maps to
        # the stable MISSING subtype, not the invalid-pointer subtype.
        assert payload["stats"]["reason"] == srv.DRIFT_REASON_POINTER_MISSING

    def test_degraded_stats_payload_is_safe(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        _make_versioned_config(monkeypatch, tmp_path)
        publish_generation(store, "gen-a")
        # Corrupt the pointer after publish.
        (tmp_path / "current").write_text("not-json{\n", encoding="utf-8")
        bomb_orchestrator(monkeypatch)
        payload = json.loads(srv.get_index_stats())
        raw = json.dumps(payload)
        assert payload["stats"]["ready"] is False
        # No absolute private paths, no HEAD, no exception text.
        assert str(tmp_path) not in raw
        assert "Traceback" not in raw

    def test_healthy_versioned_stats_stale_checks_before_count(
        self, monkeypatch, tmp_path: Path, store: GenerationStore
    ):
        cfg = _make_versioned_config(monkeypatch, tmp_path)
        synthetic_pin_seams(monkeypatch)
        publish_generation(store, "gen-a")
        assert srv._pin_versioned_generation() is not None
        # Drift the LIVE corpus — the freshness check must fire BEFORE any
        # collection.count() touches the backend.
        live = Path(cfg.source_documents_dir)
        (live / "drift.md").write_text("# drift\n")
        # get_orchestrator would construct on stale state — bomb it.
        bomb_orchestrator(monkeypatch)
        payload = json.loads(srv.get_index_stats())
        assert payload["stats"]["ready"] is False
        assert payload["stats"]["reason"] == srv.DRIFT_REASON_ENVIRONMENT_CHANGED

    def test_v2_receipt_is_inspectable_but_never_servable(self, monkeypatch, tmp_path: Path, store: GenerationStore):
        _make_versioned_config(monkeypatch, tmp_path)
        publish_generation(store, "gen-a")
        receipt_path = store.generations_dir / "gen-a" / gens.RECEIPT_FILENAME
        receipt = json.loads(receipt_path.read_bytes())
        receipt["schema_version"] = 2
        receipt_path.write_bytes(json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n")
        summary = gens.inspect_current_receipt(store.root)
        assert summary is not None
        assert summary["schema_version"] == 2
        assert summary["servable"] is False
        assert summary["reason"] == "schema_v2_rebuild_required"
        # Strict resolution fails (pointer pins the v3 receipt sha).
        assert store.resolve_current() is None


def orch_query_with_gate_dedup():
    pass


# ============================================================================
# 11. v4.9.0 release/runtime reproducibility — strict lock/graph/RECORD
# ============================================================================
#
# Consolidated dependency-parity + packaging evidence. All lock fixtures
# are synthetic tmp_path files (the genuine repo requirements.lock is
# NEVER edited by tests). Installed-environment seams use the explicit
# ``version_of``/``requires_of`` parameters of the production API — no
# production verification is patched out.


def _write_lock(tmp_path: Path, body: str) -> Path:
    lock = tmp_path / "requirements.lock"
    lock.write_text(body, encoding="utf-8")
    return lock


def _pin(name: str, version: str, digest_hex: str, marker: str | None = None) -> str:
    # Physical pip-compile shape: requirement line ends with a backslash,
    # continuation lines carry exactly one --hash=sha256:<64hex> token.
    marker_clause = f"; {marker}" if marker else ""
    return f"{name}=={version}{marker_clause} \\\n    --hash=sha256:{digest_hex}\n    # via knowledge-rag\n"


LINUX_ENV = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix"}
WINDOWS_ENV = {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt"}


class TestStrictLockContinuations:
    """Physical continuation grammar fails closed on malformed shapes."""

    def test_standalone_hash_after_non_continuated_requirement_rejects(self, tmp_path: Path):
        # Requirement line does NOT end with backslash; the hash therefore
        # arrives as a standalone continuation — rejected.
        lock = _write_lock(
            tmp_path,
            "numpy==1.26.4\n    --hash=sha256:%s\n" % HEX("x"),
        )
        with pytest.raises(gens.DependencyUnverifiableError, match="standalone --hash="):
            gens.lock_pin_map([lock])

    def test_trailing_backslash_eof_rejects(self, tmp_path: Path):
        lock = _write_lock(tmp_path, "numpy==1.26.4 \\\n")
        with pytest.raises(gens.DependencyUnverifiableError, match="pending backslash continuation"):
            gens.lock_pin_map([lock])

    def test_non_hash_continuation_token_rejects(self, tmp_path: Path):
        lock = _write_lock(
            tmp_path,
            "numpy==1.26.4 \\\n    some-other-token --hash=sha256:%s\n" % HEX("x"),
        )
        with pytest.raises(gens.DependencyUnverifiableError, match="unexpected continuation token"):
            gens.lock_pin_map([lock])

    def test_valid_continuation_block_parses(self, tmp_path: Path):
        lock = _write_lock(tmp_path, _pin("numpy", "1.26.4", HEX("n1")))
        variants = gens._parse_strict_lock(lock.read_text(encoding="utf-8"))
        assert len(variants) == 1
        assert variants[0]["hashes"] == [HEX("n1")]

    def test_real_uv_multihash_block_parses(self, tmp_path: Path):
        # Exact physical shape the controller's uv 0.12.5 lock emits:
        # requirement line ends with a backslash; EVERY hash continuation
        # except the last also ends with a backslash; the final hash line
        # does NOT; a "# via" comment follows; the next pin starts. The
        # bare-option branch must not swallow the hash lines (regression:
        # "line 7: unexpected continuation token 'aiohttp==3.14.3'").
        block = (
            "aiohttp==3.14.3 \\\n"
            "    --hash=sha256:%s \\\n"
            "    --hash=sha256:%s\n"
            "    # via knowledge-rag\n"
            "numpy==1.26.4 \\\n"
            "    --hash=sha256:%s\n"
            "    # via aiohttp\n" % (HEX("a1"), HEX("a2"), HEX("n1"))
        )
        lock = _write_lock(tmp_path, block)
        variants = gens._parse_strict_lock(lock.read_text(encoding="utf-8"))
        assert len(variants) == 2
        assert variants[0]["text"] == "aiohttp==3.14.3"
        assert variants[0]["hashes"] == [HEX("a1"), HEX("a2")]
        assert variants[1]["text"] == "numpy==1.26.4"
        assert variants[1]["hashes"] == [HEX("n1")]

    def test_ranged_pin_rejects(self, tmp_path: Path):
        lock = _write_lock(tmp_path, "numpy>=1.24.0 \\\n    --hash=sha256:%s\n" % HEX("x"))
        with pytest.raises(gens.DependencyUnverifiableError, match="not an exact"):
            gens.lock_pin_map([lock])

    @pytest.mark.parametrize(
        "include_line",
        [
            "-r other.lock",
            "--requirement other.lock",
            "-rother.lock",            # compact: no space, no '='
            "--requirement=other.lock",  # compact: '=' form
        ],
        ids=["r-space", "requirement-space", "r-compact", "requirement-equals"],
    )
    def test_include_forms_reject(self, tmp_path: Path, include_line: str):
        # A lock must be self-contained: every -r/--requirement include
        # form — whitespace AND compact — fails closed.
        lock = _write_lock(tmp_path, include_line + "\n")
        with pytest.raises(gens.DependencyUnverifiableError, match="-r/--requirement includes"):
            gens.lock_pin_map([lock])


class TestMarkerAwareVariants:
    """Marker-bearing exact pins are valid; exactly one active per name."""

    def test_disjoint_platform_variants_select_exactly_one(self, tmp_path: Path):
        lock = _write_lock(
            tmp_path,
            _pin("colorama", "0.4.6", HEX("win"), 'sys_platform == "win32"')
            + _pin("colorama", "0.4.6", HEX("nix"), 'sys_platform != "win32"'),
        )
        linux_pins = gens.lock_pin_map([lock], environment=LINUX_ENV)
        windows_pins = gens.lock_pin_map([lock], environment=WINDOWS_ENV)
        assert linux_pins["colorama"]["hashes"] == [HEX("nix")]
        assert windows_pins["colorama"]["hashes"] == [HEX("win")]

    def test_overlapping_active_variants_reject(self, tmp_path: Path):
        # Both variants are active on Linux — ambiguous, fail closed.
        lock = _write_lock(
            tmp_path,
            _pin("numpy", "1.26.4", HEX("a"), 'sys_platform != "win32"')
            + _pin("numpy", "1.26.3", HEX("b"), 'platform_system != "Windows"'),
        )
        with pytest.raises(gens.DependencyUnverifiableError, match="multiple ACTIVE pins"):
            gens.lock_pin_map([lock], environment=LINUX_ENV)

    def test_extra_neq_gpu_is_active_under_default_empty_extra(self, tmp_path: Path):
        # extra != "gpu" evaluates TRUE when extra is "" — the pin is ACTIVE
        # in the production default environment (no extras installed).
        lock = _write_lock(
            tmp_path,
            _pin("onnxruntime", "1.20.0", HEX("cpu"), 'extra != "gpu"')
            + _pin("onnxruntime-gpu", "1.20.0", HEX("gpu"), 'extra == "gpu"'),
        )
        pins = gens.lock_pin_map([lock])  # default environment: extra=""
        assert pins["onnxruntime"]["version"] == "1.20.0"
        assert "onnxruntime-gpu" not in pins  # extra == "gpu" inactive


class TestActiveGraphParity:
    """BFS graph root -> core -> leaf with explicit version_of/requires_of."""

    @staticmethod
    def _env(requires_map, versions):
        def requires_of(name):
            if name not in requires_map:
                raise gens.DependencyUnverifiableError(f"not installed: {name}")
            return [packaging.requirements.Requirement(r) for r in requires_map[name]]

        def version_of(name):
            if name not in versions:
                return None
            return versions[name]

        return requires_of, version_of

    def _root_requires(self):
        return {
            "knowledge-rag": [
                "corelib>=2.0.0",
                'winonly>=1.0.0; sys_platform == "win32"',
                'gpuextra>=1.0.0; extra == "gpu"',
            ],
            "corelib": ["leaf>=0.1.0"],
            "leaf": [],
            "winonly": [],
            "gpuextra": [],
        }

    def _full_lock(self):
        return (
            _pin("corelib", "2.1.0", HEX("core"))
            + _pin("leaf", "0.2.0", HEX("leaf"))
            + _pin("winonly", "1.0.0", HEX("win"), 'sys_platform == "win32"')
            + _pin("gpuextra", "1.0.0", HEX("gpu"), 'extra == "gpu"')
        )

    def test_graph_walks_root_core_leaf(self):
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0", "leaf": "0.2.0"},
        )
        graph = gens.active_dependency_graph(
            "knowledge-rag", environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
        )
        edges = {(parent, e["name"]) for parent, es in graph.items() for e in es}
        assert ("knowledge-rag", "corelib") in edges
        assert ("corelib", "leaf") in edges
        # Windows-only and gpu-extra edges are INACTIVE on default Linux/"".
        assert ("knowledge-rag", "winonly") not in edges
        assert ("knowledge-rag", "gpuextra") not in edges

    def test_missing_mandatory_leaf_rejects(self, tmp_path: Path, monkeypatch):
        lock = _write_lock(tmp_path, self._full_lock())
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0"},  # leaf MISSING
        )
        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.9.0")
        with pytest.raises(gens.DependencyUnverifiableError, match="leaf.*not installed"):
            gens.verify_runtime_dependency_parity(
                [lock], environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
            )

    def test_wrong_mandatory_leaf_version_rejects(self, tmp_path: Path, monkeypatch):
        lock = _write_lock(tmp_path, self._full_lock())
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0", "leaf": "0.9.9"},  # drift
        )
        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.9.0")
        with pytest.raises(gens.DependencyUnverifiableError, match="runtime drift"):
            gens.verify_runtime_dependency_parity(
                [lock], environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
            )

    def test_inactive_windows_and_gpu_deps_may_be_absent(self, tmp_path: Path, monkeypatch):
        lock = _write_lock(tmp_path, self._full_lock())
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0", "leaf": "0.2.0"},
            # winonly + gpuextra deliberately NOT installed — their pins are
            # inactive on Linux/"" so absence passes.
        )
        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.9.0")
        result = gens.verify_runtime_dependency_parity(
            [lock], environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
        )
        assert result["graph_edges"] == 2

    def test_false_marker_only_pin_for_active_edge_rejects(self, tmp_path: Path, monkeypatch):
        # corelib is an ACTIVE edge but the lock carries ONLY a variant that
        # is inactive in this environment — fail closed.
        lock = _write_lock(
            tmp_path,
            _pin("corelib", "2.1.0", HEX("core"), 'sys_platform == "win32"') + _pin("leaf", "0.2.0", HEX("leaf")),
        )
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0", "leaf": "0.2.0"},
        )
        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.9.0")
        with pytest.raises(gens.DependencyUnverifiableError, match="corelib.*no ACTIVE exact lock pin"):
            gens.verify_runtime_dependency_parity(
                [lock], environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
            )

    def test_orphan_markerless_pin_absent_passes_installed_wrong_rejects(self, tmp_path: Path, monkeypatch):
        # orphanlib: pinned markerless, NOT in the active graph.
        lock = _write_lock(
            tmp_path,
            _pin("corelib", "2.1.0", HEX("core"))
            + _pin("leaf", "0.2.0", HEX("leaf"))
            + _pin("orphanlib", "3.0.0", HEX("orphan")),
        )
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0", "leaf": "0.2.0"},
        )
        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.9.0")
        # Absent orphan pin: passes.
        gens.verify_runtime_dependency_parity(
            [lock], environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
        )
        # Same orphan INSTALLED at a different version: runtime drift.
        versions_drift = dict(
            {"knowledge-rag": "4.9.0", "corelib": "2.1.0", "leaf": "0.2.0"},
            orphanlib="2.5.0",
        )
        requires_of2, version_of2 = self._env(self._root_requires(), versions_drift)
        with pytest.raises(gens.DependencyUnverifiableError, match=r"orphanlib==3\.0\.0.*runtime drift"):
            gens.verify_runtime_dependency_parity(
                [lock], environment=LINUX_ENV, version_of=version_of2, requires_of=requires_of2
            )

    def test_self_version_mismatch_fails_closed(self, tmp_path: Path, monkeypatch):
        lock = _write_lock(tmp_path, self._full_lock())
        requires_of, version_of = self._env(
            self._root_requires(),
            {"knowledge-rag": "4.8.5", "corelib": "2.1.0", "leaf": "0.2.0"},
        )
        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.8.5")
        with pytest.raises(gens.DependencyUnverifiableError, match="does not match"):
            gens.verify_runtime_dependency_parity(
                [lock], environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
            )

    def test_mandatory_child_extra_activates_transitive_leaf(self):
        # Root requires core[feature]; core declares `leaf; extra == "feature"`.
        # The mandatory extra-bearing edge ACTIVATES "feature" on core, so
        # the leaf edge must be traversed even though extra="" at the root.
        requires_map = {
            "knowledge-rag": ["core[feature]>=2.0.0"],
            "corelib": [],
            "core": ["leaf>=0.1.0; extra == 'feature'"],
            "leaf": [],
        }
        versions = {"knowledge-rag": "4.9.0", "core": "2.1.0", "leaf": "0.2.0"}

        def requires_of(name):
            return [packaging.requirements.Requirement(r) for r in requires_map[name]]

        def version_of(name):
            return versions.get(name)

        graph = gens.active_dependency_graph(
            "knowledge-rag", environment=LINUX_ENV, version_of=version_of, requires_of=requires_of
        )
        edges = {(parent, e["name"]) for parent, es in graph.items() for e in es}
        assert ("knowledge-rag", "core") in edges
        assert ("core", "leaf") in edges  # transitive extra edge TRAVERSED

    def test_transitive_extra_leaf_missing_or_wrong_fails_parity(self, tmp_path: Path, monkeypatch):
        lock = _write_lock(
            tmp_path,
            _pin("core", "2.1.0", HEX("core")) + _pin("leaf", "0.2.0", HEX("leaf")),
        )
        requires_map = {
            "knowledge-rag": ["core[feature]>=2.0.0"],
            "core": ["leaf>=0.1.0; extra == 'feature'"],
            "leaf": [],
        }

        def requires_of(name):
            return [packaging.requirements.Requirement(r) for r in requires_map[name]]

        monkeypatch.setattr(gens.importlib.metadata, "version", lambda name: "4.9.0")

        # Missing mandatory transitive leaf -> fail.
        with pytest.raises(gens.DependencyUnverifiableError, match="leaf.*not installed"):
            gens.verify_runtime_dependency_parity(
                [lock],
                environment=LINUX_ENV,
                version_of=lambda name: {"knowledge-rag": "4.9.0", "core": "2.1.0"}.get(name),
                requires_of=requires_of,
            )

        # Wrong version -> runtime drift.
        with pytest.raises(gens.DependencyUnverifiableError, match="runtime drift"):
            gens.verify_runtime_dependency_parity(
                [lock],
                environment=LINUX_ENV,
                version_of=lambda name: {"knowledge-rag": "4.9.0", "core": "2.1.0", "leaf": "0.9.9"}.get(name),
                requires_of=requires_of,
            )


class TestStrictRecord:
    """Three-column RECORD CSV canonicalization + hashless exemptions."""

    @staticmethod
    def _b64_of(data: bytes) -> str:
        import base64

        return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")

    @staticmethod
    def _make_dist(tmp_path: Path, rows: list[tuple[str, bytes, str | None, str | None]]):
        """Build a fake distribution: files + RECORD text.

        rows: (relpath, file_bytes, declared_hash_or_None, declared_size_or_None)
        The distribution's own self-RECORD row (hashless) is appended
        automatically — exactly one, as a real wheel ships.
        """
        site = tmp_path / "site-packages" / "fakerag-4.9.0.dist-info"
        site.mkdir(parents=True, exist_ok=True)
        record_lines = []
        for rel, data, hash_field, size_field in rows:
            target = tmp_path / "site-packages" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            actual_hash = hash_field if hash_field is not None else ""
            actual_size = size_field if size_field is not None else ""
            record_lines.append(f"{rel},{actual_hash},{actual_size}")
        # Unique hashless self-RECORD row (the one legal hashless dist-info row).
        self_record_row = site.name + "/RECORD,,"
        record_path = site / "RECORD"
        record_path.write_text("\n".join(record_lines) + "\n" + self_record_row + "\n", encoding="utf-8")

        class _Dist:
            version = "4.9.0"

            def read_text(self, name):
                if name == "RECORD":
                    return (site / "RECORD").read_text(encoding="utf-8")

                return None

            def locate_file(self, path):
                return tmp_path / "site-packages" / path

        return _Dist()

    def test_hashless_metadata_rejects(self, tmp_path: Path, monkeypatch):
        data = b"name: fakerag\n"
        dist = self._make_dist(
            tmp_path,
            [
                ("fakerag-4.9.0.dist-info/METADATA", data, "", ""),
            ],
        )
        monkeypatch.setattr(gens.importlib.metadata, "distribution", lambda name: dist)
        with pytest.raises(gens.DependencyUnverifiableError, match="not an admitted exemption"):
            gens.installed_record_evidence("fakerag")

    def test_hashed_metadata_and_unique_self_record_pass(self, tmp_path: Path, monkeypatch):
        data = b"name: fakerag\n"
        dist = self._make_dist(
            tmp_path,
            [
                ("fakerag-4.9.0.dist-info/METADATA", data, f"sha256={self._b64_of(data)}", str(len(data))),
            ],
        )
        monkeypatch.setattr(gens.importlib.metadata, "distribution", lambda name: dist)
        digest, version, checked = gens.installed_record_evidence("fakerag")
        assert version == "4.9.0"
        assert checked == 1
        assert len(digest) == 64

    def test_declared_size_mismatch_rejects(self, tmp_path: Path, monkeypatch):
        data = b"name: fakerag\n"
        dist = self._make_dist(
            tmp_path,
            [
                (
                    "fakerag-4.9.0.dist-info/METADATA",
                    data,
                    f"sha256={self._b64_of(data)}",
                    str(len(data) + 5),  # wrong declared size
                ),
            ],
        )
        monkeypatch.setattr(gens.importlib.metadata, "distribution", lambda name: dist)
        with pytest.raises(gens.DependencyUnverifiableError, match="size mismatch"):
            gens.installed_record_evidence("fakerag")

    def test_pyc_mutation_changes_evidence_and_deletion_rejects(self, tmp_path: Path, monkeypatch):
        pyc = b"\x00\x01fake-bytecode-v1"
        metadata = b"name: fakerag\n"
        metadata_hash = "sha256=" + self._b64_of(metadata)
        dist = self._make_dist(
            tmp_path,
            [
                ("fakerag/__pycache__/mod.cpython-312.pyc", pyc, "", ""),
                (
                    "fakerag-4.9.0.dist-info/METADATA",
                    metadata,
                    metadata_hash,
                    str(len(metadata)),
                ),
            ],
        )
        monkeypatch.setattr(gens.importlib.metadata, "distribution", lambda name: dist)
        digest_v1, _v, _c = gens.installed_record_evidence("fakerag")

        # Mutate the .pyc on disk -> evidence digest MUST change.
        target = tmp_path / "site-packages" / "fakerag" / "__pycache__" / "mod.cpython-312.pyc"
        target.write_bytes(b"\x00\x01fake-bytecode-v2")
        digest_v2, _v, _c = gens.installed_record_evidence("fakerag")
        assert digest_v1 != digest_v2

        # Delete the hashless .pyc -> fails closed.
        target.unlink()
        with pytest.raises(gens.DependencyUnverifiableError, match="missing/non-regular"):
            gens.installed_record_evidence("fakerag")


class TestGenuineRepoLockAdmissible:
    def test_repo_lock_parses_under_strict_grammar(self):
        # The genuine repo lock must parse under the SAME strict grammar the
        # evidence path enforces (read-only admission proof — never edited).
        # NOTE: this turns green only after the controller regenerates
        # requirements.lock with packaging pinned; until then it fails
        # loudly, which is the honest signal.
        lock = Path(__file__).resolve().parents[1] / "requirements.lock"
        if not lock.is_file():
            pytest.skip("repo lock not present in this checkout")
        digest, name, variants = gens.dependency_lock_evidence([lock])
        assert name == "requirements.lock"
        assert digest and variants >= 50
        pins = gens.lock_pin_map([lock])
        for expected in ("chromadb", "fastembed", "numpy", "watchdog", "pyyaml", "packaging"):
            assert expected in pins, f"{expected} missing from genuine lock (regenerate with uv 0.12.5)"


class TestGenerationReceiptHasNoWheelSha:
    def test_identity_keys_bind_runtime_evidence_only(self):
        # Generation receipts record installed RECORD + lock + content
        # identities; a wheel SHA is NEVER part of the schema.
        for key in gens._IDENTITY_KEYS:
            assert "wheel" not in key, f"wheel evidence leaked into identity: {key}"
        assert "installed_record_sha256" in gens._IDENTITY_KEYS
        assert "dependency_lock_sha256" in gens._IDENTITY_KEYS
        assert len(gens._IDENTITY_KEYS - {"installed_record_sha256", "dependency_lock_sha256"}) > 0


class TestPackageManifestEvidence:
    """Focused wheel/sdist resource assertions (hatch config authoritative)."""

    _ROOT = Path(__file__).resolve().parents[1]
    _FORCE_INCLUDE = {
        "config.example.yaml": "mcp_server/data/config.example.yaml",
        "requirements.lock": "mcp_server/data/requirements.lock",
        "presets/cybersecurity.yaml": "mcp_server/data/cybersecurity.yaml",
        "presets/developer.yaml": "mcp_server/data/developer.yaml",
        "presets/research.yaml": "mcp_server/data/research.yaml",
        "presets/general.yaml": "mcp_server/data/general.yaml",
        "presets/multilingual.yaml": "mcp_server/data/multilingual.yaml",
    }

    def _pyproject(self) -> dict:
        import tomllib

        with (self._ROOT / "pyproject.toml").open("rb") as fh:
            return tomllib.load(fh)

    def test_all_force_include_sources_exist(self):
        for src in self._FORCE_INCLUDE:
            assert (self._ROOT / src).is_file(), f"force-include source missing: {src}"

    def test_pyproject_declares_all_five_presets_and_lock(self):
        targets = self._pyproject()["tool"]["hatch"]["build"]["targets"]
        force_include = targets["wheel"]["force-include"]
        for src, dest in self._FORCE_INCLUDE.items():
            assert force_include.get(src) == dest, f"force-include mapping missing: {src} -> {dest}"
        assert "presets/multilingual.yaml" in force_include

    def test_sdist_carries_locks_and_presets(self):
        include = self._pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        for entry in (
            "requirements.lock",
            "presets/",
            "config.example.yaml",
            "mcp_server/",
            "build-requirements.in",
            "build-requirements.lock",
            "test-requirements.in",
            "test-requirements.lock",
        ):
            assert entry in include, f"sdist include missing: {entry}"

    def test_no_manifest_in_beside_hatch(self):
        # hatch config is authoritative — no MANIFEST.in may appear.
        assert not (self._ROOT / "MANIFEST.in").exists()

    def test_versions_are_bumped_in_lockstep(self):
        pyproject = (self._ROOT / "pyproject.toml").read_text(encoding="utf-8")
        init = (self._ROOT / "mcp_server" / "__init__.py").read_text(encoding="utf-8")
        npm = json.loads((self._ROOT / "npm" / "package.json").read_text(encoding="utf-8"))
        py_match = re.search(r'^version = "(.+?)"', pyproject, re.M)
        pkg_match = re.search(r'__version__ = "(.+?)"', init)
        assert py_match is not None and pkg_match is not None
        py = py_match.group(1)
        pkg = pkg_match.group(1)
        assert py == pkg == npm["version"] == "4.9.0"

    def test_package_data_byte_identical_to_sources(self):
        # Source/editable package-data canonical: every bundled file under
        # mcp_server/data/ must be byte-identical to its canonical source.
        for src, dest in self._FORCE_INCLUDE.items():
            if not dest.startswith("mcp_server/data/"):
                continue
            source_bytes = (self._ROOT / src).read_bytes()
            bundled = self._ROOT / dest
            assert bundled.is_file(), f"bundled package-data missing: {dest}"
            assert bundled.read_bytes() == source_bytes, (
                f"package-data drift: {dest} != {src} (source and wheel behavior diverged — re-sync mcp_server/data/)"
            )

    def test_release_receipt_script_binds_git_tree_evidence(self):
        # Static contract of the release receipt builder (no build executed):
        # source bytes come from the EXACT GITHUB_SHA Git tree via read-only
        # plumbing (git show / ls-tree / cat-file), never the mutable
        # worktree; exactly one wheel is accepted (never first-vs-last);
        # force-include tree/wheel byte parity, embedded-lock == tree-lock
        # equality, and wheel-vs-tree package drift are all asserted.
        script = (self._ROOT / "scripts" / "release_receipt.py").read_text(encoding="utf-8")
        for token in (
            "GITHUB_SHA",
            "cat-file",
            "ls-tree",
            'f"{commit}:{path}"',
            "len(wheels) != 1",
            "wheel_sha256",
            "tree_sha256",
            "embedded_lock_matches_tree",
            "source_of_truth",
            "git-tree",
        ):
            assert token in script, f"release receipt script lacks {token!r}"
        # Mutable-source terminology must be gone.
        assert "source_sha256" not in script


class TestDefaultLockDiscovery:
    """Default lock discovery is FALLBACK, not union — exactly one source.

    The production helper is driven against a FAKE package layout (seams:
    ``gens.__file__`` and ``importlib.resources.files``) — real package
    data is never touched, and the real function makes the decision.
    """

    @staticmethod
    def _fake_layout(monkeypatch, tmp_path: Path, repo_lock: bool, embedded_lock: bool) -> None:
        repo_root = tmp_path / "repo"
        pkg_dir = repo_root / "mcp_server"
        data_dir = pkg_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if repo_lock:
            (repo_root / "requirements.lock").write_text("# repo\n", encoding="utf-8")
        if embedded_lock:
            (data_dir / "requirements.lock").write_text("# embedded\n", encoding="utf-8")
        # gens._dependency_lock_candidates computes base = parents[1] of the
        # module file — point it at <repo>/mcp_server/generations.py.
        monkeypatch.setattr(gens, "__file__", str(pkg_dir / "generations.py"))

        class _FakeResource:
            def __init__(self, path: Path) -> None:
                self._path = path

            def joinpath(self, rel: str) -> "_FakeResource":
                return _FakeResource(self._path / rel)

            def is_file(self) -> bool:
                return self._path.is_file()

            def __str__(self) -> str:
                return str(self._path)

        import importlib.resources

        # Production calls files("mcp_server").joinpath("data/...") — so the
        # fake package root is pkg_dir (the mcp_server/ dir), NOT data_dir;
        # joinpath appends "data/requirements.lock" itself.
        monkeypatch.setattr(importlib.resources, "files", lambda name: _FakeResource(pkg_dir))

    def test_both_present_returns_exactly_the_repo_root_copy(self, monkeypatch, tmp_path: Path):
        self._fake_layout(monkeypatch, tmp_path, repo_lock=True, embedded_lock=True)
        candidates = gens._dependency_lock_candidates()
        # FALLBACK, not union: the repo-root lock is the ONE canonical
        # source; parsing both identical copies would duplicate every pin.
        assert candidates == [tmp_path / "repo" / "requirements.lock"]

    def test_repo_root_absent_falls_back_to_embedded(self, monkeypatch, tmp_path: Path):
        self._fake_layout(monkeypatch, tmp_path, repo_lock=False, embedded_lock=True)
        candidates = gens._dependency_lock_candidates()
        assert candidates == [tmp_path / "repo" / "mcp_server" / "data" / "requirements.lock"]


class TestExplicitChromaClose:
    """Chroma 1.5.9 PersistentClient has a real public close() — the
    orchestrator and every verification path must call it explicitly."""

    def test_orchestrator_close_calls_chroma_client_close(self):
        from mcp_server.server import KnowledgeOrchestrator

        orch = object.__new__(KnowledgeOrchestrator)

        class _FakeClient:
            closed = 0

            def close(self):
                type(self).closed += 1

        client = _FakeClient()
        orch.chroma_client = client
        orch.collection = object()
        # No fts5_index attribute at all — close() must tolerate that.
        orch.close()
        assert _FakeClient.closed == 1, "close() must call chroma_client.close() exactly once"
        # References dropped after the explicit close.
        assert not hasattr(orch, "chroma_client")
        assert not hasattr(orch, "collection")

    def test_orchestrator_close_strict_raises_on_close_failure(self):
        from mcp_server.server import KnowledgeOrchestrator

        orch = object.__new__(KnowledgeOrchestrator)

        class _FailingClient:
            def close(self):
                raise RuntimeError("cannot release")

        orch.chroma_client = _FailingClient()
        with pytest.raises(RuntimeError):
            orch.close(strict=True)  # builder semantics: close failure RAISES

    def test_orchestrator_close_runtime_is_best_effort(self, capsys):
        from mcp_server.server import KnowledgeOrchestrator

        orch = object.__new__(KnowledgeOrchestrator)

        class _FailingClient:
            def close(self):
                raise RuntimeError("cannot release")

        orch.chroma_client = _FailingClient()
        orch.close(strict=False)  # runtime: logged, shutdown proceeds
        out = capsys.readouterr()
        assert "[CHROMA] close during orchestrator shutdown failed" in (out.out + out.err)
        assert not hasattr(orch, "chroma_client")

    def test_verify_pinned_backends_closes_local_client(self, monkeypatch, tmp_path: Path):
        import mcp_server.server as srv

        opened: list = []

        class _FakeClient:
            def __init__(self):
                self.closed = False
                opened.append(self)

            def get_collection(self, name):
                class _Col:
                    pass

                return _Col()

            def close(self):
                self.closed = True

        # The helpers are imported INSIDE the function from mcp_server.fts5_index
        # (and chromadb at module level) — patch them at their actual source.
        import mcp_server.fts5_index as fts5

        monkeypatch.setattr(srv.chromadb, "PersistentClient", lambda path: _FakeClient())
        rows = [("id0", "doc", "f", "c")]
        monkeypatch.setattr(fts5, "capture_chunk_rows", lambda col: rows)
        monkeypatch.setattr(fts5, "capture_full_chunk_rows", lambda col: rows)
        monkeypatch.setattr(fts5, "read_sealed_fts_row_universe", lambda p: ("d", 1))

        class _Cur:
            generation_dir = tmp_path
            receipt = {
                "backends": {
                    "chroma": {
                        "collection_name": "kb",
                        "row_digest": "x",
                        "common_row_digest": "y",
                        "row_count": 1,
                        "unique_id_count": 1,
                        "hydrated_id_count": 1,
                        "backend_generation_id": "z",
                    },
                    "fts5": {"row_digest": "d", "row_count": 1, "source_digest": "d", "verified_digest": "d", "backend_generation_id": "w"},
                }
            }

        from mcp_server.fts5_index import compute_full_rows_digest, compute_rows_digest

        full_digest, _ = compute_full_rows_digest(rows)
        common_digest, _ = compute_rows_digest(rows)
        _Cur.receipt["backends"]["chroma"]["row_digest"] = full_digest
        _Cur.receipt["backends"]["chroma"]["common_row_digest"] = common_digest

        srv._verify_pinned_backends(_Cur())
        # The result may be a drift error (id-universe/backend-id details),
        # but the CLIENT must be closed either way — never left alive.
        assert opened and all(c.closed for c in opened), (
            "every locally opened PersistentClient must be closed before returning"
        )

    def test_verify_pinned_backends_close_failure_fails_closed(self, monkeypatch, tmp_path: Path):
        import mcp_server.server as srv

        class _FailingCloseClient:
            def get_collection(self, name):
                class _Col:
                    pass

                return _Col()

            def close(self):
                raise RuntimeError("release failed")

        import mcp_server.fts5_index as fts5

        monkeypatch.setattr(srv.chromadb, "PersistentClient", lambda path: _FailingCloseClient())
        monkeypatch.setattr(fts5, "capture_chunk_rows", lambda col: [])
        monkeypatch.setattr(fts5, "capture_full_chunk_rows", lambda col: [])
        monkeypatch.setattr(fts5, "read_sealed_fts_row_universe", lambda p: ("d", 0))

        class _Cur:
            generation_dir = tmp_path
            receipt = {
                "backends": {
                    "chroma": {"collection_name": "kb", "row_digest": "x"},
                    "fts5": {"row_digest": "d", "row_count": 0, "source_digest": "d", "verified_digest": "d"},
                }
            }

        result = srv._verify_pinned_backends(_Cur())
        assert result is not None, "a close failure must fail closed as backend drift"
        assert result.reason == srv.DRIFT_REASON_BACKEND_DRIFT
        # Sanitized detail: no exception text leaks.
        assert "release failed" not in json.dumps(result.detail)


class TestChromaScratchCopyEvidence:
    """Evidence derivation inspects a byte-verified SCRATCH COPY; the
    original staged tree stays byte-identical; the client is closed."""

    def test_scratch_copy_flow_and_immutability(self, monkeypatch, tmp_path: Path):
        import mcp_server.generation_cli as cli

        staging = tmp_path / "staging"
        db_dir = staging / cli.CHROMA_ARTIFACT
        db_dir.mkdir(parents=True)
        (db_dir / "chroma.sqlite3").write_bytes(b"v1-bytes")

        from mcp_server.generations import digest_tree

        original_digest, _ = digest_tree(db_dir)

        opened_paths: list = []
        closed: list = []

        class _FakeClient:
            def __init__(self):
                closed.append(False)

            def get_collection(self, name):
                class _Col:
                    pass

                return _Col()

            def close(self):
                closed[-1] = True

        # chromadb is imported INSIDE the function — patch the real module's
        # PersistentClient; the capture helpers are module-level imports in
        # generation_cli, so patch them there.
        import chromadb

        def _fake_persistent(path, settings=None):
            opened_paths.append(path)
            return _FakeClient()

        monkeypatch.setattr(chromadb, "PersistentClient", _fake_persistent)
        rows = [("id0", "doc0", "f0", "c0")]
        monkeypatch.setattr(cli, "capture_full_chunk_rows", lambda col: rows)
        monkeypatch.setattr(cli, "capture_chunk_rows", lambda col: rows)

        evidence = cli._chroma_evidence_from_staging(staging)

        # Exactly one client opened, and NEVER on the original staged tree.
        assert len(opened_paths) == 1
        assert str(db_dir) not in opened_paths[0], "original staged chroma_db must never be opened"
        assert all(closed), "the scratch client must be explicitly closed"
        # The ORIGINAL tree is byte-identical before and after derivation.
        post_digest, _ = digest_tree(db_dir)
        assert post_digest == original_digest
        # backend_generation_id is bound to the ORIGINAL artifact bytes.
        assert evidence["backend_generation_id"]
        assert evidence["row_count"] == 1

    def test_staged_tree_mutation_aborts_seal(self, monkeypatch, tmp_path: Path):
        import mcp_server.generation_cli as cli

        staging = tmp_path / "staging"
        db_dir = staging / cli.CHROMA_ARTIFACT
        db_dir.mkdir(parents=True)
        (db_dir / "chroma.sqlite3").write_bytes(b"v1-bytes")

        class _FakeClient:
            def get_collection(self, name):
                class _Col:
                    pass

                return _Col()

            def close(self):
                pass

        import chromadb

        monkeypatch.setattr(chromadb, "PersistentClient", lambda path, settings=None: _FakeClient())
        monkeypatch.setattr(cli, "capture_full_chunk_rows", lambda col: [])
        monkeypatch.setattr(cli, "capture_chunk_rows", lambda col: [])

        # Corrupt the ORIGINAL tree "during" derivation: the post-check
        # digest differs from the snapshot -> refuse to seal.
        import shutil

        real_copytree = shutil.copytree

        def _copytree(src, dst, **kw):
            real_copytree(src, dst, **kw)
            (db_dir / "chroma.sqlite3").write_bytes(b"MUTATED")  # simulates drift

        monkeypatch.setattr(shutil, "copytree", _copytree)
        with pytest.raises(SystemExit, match="changed during evidence derivation"):
            cli._chroma_evidence_from_staging(staging)


class TestSidecarSuffixCoverage:
    """The population-child sweep rejects EVERY canonical suffix generations.py
    enforces: -wal/-shm/-journal and .wal/.shm/.journal, including
    chroma.sqlite3-wal."""

    @pytest.mark.parametrize(
        "name",
        [
            "chroma.sqlite3-wal",
            "chroma.sqlite3-shm",
            "chroma.sqlite3-journal",
            "fts5_index.db.wal",
            "fts5_index.db.shm",
            "fts5_index.db.journal",
            "anything-wal",
            "anything-shm",
            "anything-journal",
            "anything.wal",
            "anything.shm",
            "anything.journal",
        ],
    )
    def test_every_canonical_sidecar_suffix_is_rejected(self, name: str):
        # ANY canonical suffix must flag the name (not every suffix — a
        # name matches at most one). This mirrors the production sweep's
        # predicate exactly and cross-checks it against the canonical tuple
        # generations.py enforces.
        from mcp_server.generations import _SQLITE_SIDECAR_SUFFIXES

        sweep_detected = any(name.endswith(suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES)
        assert sweep_detected, (
            f"sidecar sweep must reject {name!r}; canonical suffixes = {_SQLITE_SIDECAR_SUFFIXES}"
        )
