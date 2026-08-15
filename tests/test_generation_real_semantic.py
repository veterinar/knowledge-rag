"""ONE unpatched, real semantic publication test (controller pass-10 item 3).

Builds a tiny GENUINE Chroma collection (3 literal rows) and a GENUINE FTS5
database + schema-v2 state marker from the SAME common rows, then calls the
REAL ``GenerationStore.publish`` with the REAL
``_recompute_staged_semantics`` seam — no monkeypatching of semantic
inspection anywhere in this file.

Proves:
- the child-process Chroma inspection + read-only FTS recompute succeed on
  genuine artifacts and agree with the caller's evidence;
- row counts/digests in the receipt match independent recomputation;
- ``verify_generation`` succeeds on the sealed generation;
- NO sqlite/FTS sidecars remain, and the Chroma artifact tree digest is
  UNCHANGED across the inspection child (before == after);
- one deliberately WRONG caller digest fails BEFORE any activation/current
  change.

Offline, local, deterministic, small. No network, no models, no sleeps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
import pytest

from mcp_server import generations as gens
from mcp_server.fts5_index import (
    Fts5LexicalIndex,
    capture_chunk_rows,
    capture_full_chunk_rows,
    compute_full_rows_digest,
    compute_rows_digest,
)
from mcp_server.generations import GenerationStore, VerificationError

# --- three literal rows shared by both backends -----------------------------

ROWS = [
    ("chunk-001", "feline diabetes mellitus insulin therapy", "vet-feline.md", "feline"),
    ("chunk-002", "canine hip dysplasia orthopedic screening", "vet-canine.md", "canine"),
    ("chunk-003", "feline hyperthyroidism methimazole dosing", "vet-feline.md", "feline"),
]

HEX = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
STABLE_DIGEST = lambda payload: hashlib.sha256(  # noqa: E731
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
).hexdigest()

COMPAT = {
    "collection_name": "knowledge_rag_v1",
    "embedding_model": "bge-small-en-v1.5",
    "embedding_dimension": 4,
    "query_prefix": "q: ",
    "passage_prefix": "p: ",
    "model_artifact_sha256": HEX("model-artifact-real-v1"),
    "runtime_version": "fastembed 0.8.0",
    "pooling": "cls",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "reranker_enabled": False,
    "reranker_model": None,
    "reranker_artifact_sha256": None,
}

IDENTITY = {
    "corpus_manifest_sha256": None,  # bound to the actual staged corpus below
    "config_sha256": HEX("effective-config-real-v1"),
    "code_sha256": HEX("code-identity-real-v1"),
    "retrieval_config_sha256": HEX("retrieval-config-real-v1"),
    "installed_record_sha256": HEX("installed-record-real-v1"),
    "dependency_lock_sha256": HEX("dependency-lock-real-v1"),
    "model_artifact_sha256": HEX("model-artifact-real-v1"),
    "model_config_sha256": HEX("model-config-real-v1"),
    "chunking_sha256": STABLE_DIGEST(
        {
            "embedding_model": "bge-small-en-v1.5",
            "query_prefix": "q: ",
            "passage_prefix": "p: ",
            "chunk_size": 512,
            "chunk_overlap": 64,
        }
    ),
}


def _build_chroma(building: Path) -> None:
    client = chromadb.PersistentClient(path=str(building / gens.CHROMA_ARTIFACT))
    col = client.create_collection(name=COMPAT["collection_name"])
    for chunk_id, content, filename, category in ROWS:
        col.add(
            ids=[chunk_id],
            documents=[content],
            metadatas=[{"filename": filename, "category": category}],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
        )


def _build_fts(building: Path, rows: list) -> str:
    fts = Fts5LexicalIndex(
        db_path=building / gens.FTS_ARTIFACT,
        state_path=building / gens.FTS_STATE_ARTIFACT,
    )
    try:
        result = fts.rebuild_content_bound(rows)
        assert result.get("status") == "complete", result
        fts.seal_for_publication()
    finally:
        fts.close()
    return str(result["verified_fts_rows_sha256"])


def _stage_real(store: GenerationStore, gid: str) -> tuple[Path, dict, dict]:
    """Stage genuine artifacts; return (building, chroma_ev, fts_ev)."""
    building = store.begin_build(gid)
    _build_chroma(building)
    # Common rows straight from the sealed Chroma bytes — both backends are
    # built from the SAME row universe.
    client = chromadb.PersistentClient(path=str(building / gens.CHROMA_ARTIFACT))
    col = client.get_collection(name=COMPAT["collection_name"])
    common_rows = capture_chunk_rows(col)
    fts_digest = _build_fts(building, common_rows)

    (building / gens.METADATA_ARTIFACT).write_text(
        json.dumps({"indexed_documents": 2, "rows": len(common_rows)}), encoding="utf-8"
    )
    # corpus: two small literal files, POSIX-relative sources
    corpus_dir = building / gens.CORPUS_ARTIFACT
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "vet-feline.md").write_text("# feline\n", encoding="utf-8")
    (corpus_dir / "vet-canine.md").write_text("# canine\n", encoding="utf-8")

    # close local handles before returning
    col = None
    client = None

    # Recompute evidence the same way the seam's child does.
    client = chromadb.PersistentClient(path=str(building / gens.CHROMA_ARTIFACT))
    col = client.get_collection(name=COMPAT["collection_name"])
    full_rows = capture_full_chunk_rows(col)
    full = compute_full_rows_digest(full_rows)
    common = compute_rows_digest(capture_chunk_rows(col))
    col = None
    client = None

    chroma_sha, _ = gens.digest_tree(building / gens.CHROMA_ARTIFACT)
    fts_sha = gens._sha256_file(building / gens.FTS_ARTIFACT)
    chroma_ev = {
        "collection_name": COMPAT["collection_name"],
        "row_count": full[1],
        "unique_id_count": len({r[0] for r in common_rows}),
        "hydrated_id_count": full[1],
        "row_digest": full[0],
        "common_row_digest": common[0],
        "backend_generation_id": hashlib.sha256(f"chroma:{chroma_sha}:{full[0]}".encode()).hexdigest(),
    }
    fts_ev = {
        "schema_version": 2,
        "status": "complete",
        "row_count": len(common_rows),
        "row_digest": fts_digest,
        "source_digest": fts_digest,
        "verified_digest": fts_digest,
        "backend_generation_id": hashlib.sha256(f"fts5:{fts_sha}:{fts_digest}".encode()).hexdigest(),
    }
    return building, chroma_ev, fts_ev


def test_real_semantic_publication_end_to_end(tmp_path: Path):
    store = GenerationStore(tmp_path, create=True)

    building, chroma_ev, fts_ev = _stage_real(store, "g-real-1")
    chroma_before = gens.digest_tree(building / gens.CHROMA_ARTIFACT)

    identity = dict(IDENTITY)
    identity["corpus_manifest_sha256"] = gens.corpus_manifest_digest(
        gens.corpus_manifest_entries(building / gens.CORPUS_ARTIFACT)
    )

    # REAL publish — no seam patch anywhere in this file.
    result = store.publish(
        "g-real-1",
        identity=identity,
        compatibility=COMPAT,
        chroma_evidence=chroma_ev,
        fts_evidence=fts_ev,
        expected_current=gens.EXPECTED_CURRENT_ABSENT,
    )

    receipt = result.receipt
    assert receipt["backends"]["chroma"]["row_count"] == 3
    assert receipt["backends"]["fts5"]["row_count"] == 3
    assert receipt["backends"]["chroma"]["row_digest"] == chroma_ev["row_digest"]
    assert receipt["backends"]["fts5"]["row_digest"] == fts_ev["row_digest"]

    # Full verification of the sealed generation succeeds.
    verified = store.verify_generation("g-real-1")
    assert verified["backends"]["chroma"]["common_row_digest"] == chroma_ev["common_row_digest"]

    # No sidecars anywhere in the sealed tree.
    sealed = store.generations_dir / "g-real-1"
    for p in sealed.rglob("*"):
        assert p.suffix not in {"-wal", "-shm", "-journal"}, p
        assert not p.name.endswith((".db-wal", ".db-shm")), p

    # Chroma artifact digest UNCHANGED across the inspection child.
    chroma_after = gens.digest_tree(sealed / gens.CHROMA_ARTIFACT)
    assert chroma_after == chroma_before

    # The pointer moved to the new generation.
    current = store.resolve_current()
    assert current is not None and current.generation_id == "g-real-1"

    # --- deliberately WRONG caller digest fails BEFORE activation ----------
    building2, chroma_ev2, fts_ev2 = _stage_real(store, "g-real-2")
    identity2 = dict(identity)  # same corpus bytes → same manifest digest
    bad_chroma = dict(chroma_ev2, row_digest=HEX("deliberately-wrong-digest"))
    pointer_before = store.current_path.read_bytes()
    with pytest.raises(VerificationError):
        store.publish(
            "g-real-2",
            identity=identity2,
            compatibility=COMPAT,
            chroma_evidence=bad_chroma,
            fts_evidence=fts_ev2,
            expected_current={
                "generation_id": "g-real-1",
                "receipt_sha256": gens._sha256_file(store.generations_dir / "g-real-1" / gens.RECEIPT_FILENAME),
            },
        )
    # current is byte-identical: the wrong digest never activated anything.
    assert store.current_path.read_bytes() == pointer_before
    current = store.resolve_current()
    assert current is not None and current.generation_id == "g-real-1"
    # failed publish leaves NO sealed generation and no staging leftovers
    assert not (store.generations_dir / "g-real-2").exists()
