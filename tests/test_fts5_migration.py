"""Task 05 tests: FTS5 lazy migration lifecycle + CRUD sync (ADR-008).

Covers the assigned catalog:
- Unit — TestCRUDSync: UT-049, UT-050, UT-051, UT-052
- Integration — TestCRUDSyncIntegration: IT-CRUD-001 .. IT-CRUD-004
- Package B (v4.8.3 Gate 0): B04–B10 + corrective P1 detectors (the legacy
  positional-resume lifecycle is retired: rebuild-from-zero only)

The tests exercise ``Fts5LexicalIndex`` directly (real SQLite, real FTS5
tables, real marker files) so the SQLite locking behaviour Windows CI is
sensitive to actually runs. Orchestrator-level hooks use a lightweight
stub (``_build_sync_orch``) that mounts the real ``_fts5_sync_add`` +
``_fts5_sync_remove_by_doc_id`` bound methods so the CRUD-sync
integration path is unmediated.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import List, Tuple

import pytest

import mcp_server.fts5_index as fts5_module
from mcp_server.fts5_index import (
    Fts5LexicalIndex,
    Fts5MigrationError,
    Fts5MigrationState,
    compute_rows_digest,
)
from mcp_server.server import KnowledgeOrchestrator

Row = Tuple[str, str, str, str]
_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(tmp_path, *, marker_status: str | None = None, docs_indexed: int = 0) -> Fts5LexicalIndex:
    db_path = tmp_path / "fts5_index.db"
    state_path = tmp_path / "fts5_migration.state"
    if marker_status is not None:
        Fts5MigrationState(state_path).write(
            {
                "status": marker_status,
                "docs_total": 100,
                "docs_indexed": docs_indexed,
                "started_at": "2026-08-07T12:00:00Z",
                "completed_at": None,
                "error": None,
            }
        )
    return Fts5LexicalIndex(db_path=db_path, state_path=state_path)


def _gen_rows(n: int, prefix: str = "chunk") -> List[Row]:
    return [(f"{prefix}_{i:04d}", f"content mentioning CVE-2021-{i:04d}", f"f{i}.md", "security") for i in range(n)]


# ===========================================================================
# TestCRUDSync — UT-049, UT-050, UT-051, UT-052
# ===========================================================================


class TestCRUDSync:
    def test_ut049_add_document_visible_in_search(self, tmp_path):
        """UT-049: add_document inserts a row + subsequent search hits it."""
        index = _make_index(tmp_path, marker_status="complete", docs_indexed=0)
        try:
            index.add_document("chunk_1", "Content mentioning CVE-2021-4034", "file.md", "security")
            hits = index.search("CVE-2021-4034", top_k=5)
            ids = [chunk_id for chunk_id, _ in hits]
            assert "chunk_1" in ids
        finally:
            index.close()

    def test_ut050_remove_document_clears_from_search(self, tmp_path):
        """UT-050: remove_document deletes the row + search no longer returns it."""
        index = _make_index(tmp_path, marker_status="complete", docs_indexed=0)
        try:
            index.add_document("chunk_1", "Content mentioning CVE-2021-4034", "file.md", "security")
            assert index.search("CVE-2021-4034", top_k=5)
            index.remove_document("chunk_1")
            assert index.search("CVE-2021-4034", top_k=5) == []
        finally:
            index.close()

    def test_ut051_add_document_error_caught_and_metric_incremented(self, tmp_path, monkeypatch):
        """UT-051: orchestrator CRUD hook swallows FTS5 error + bumps error counter."""
        from mcp_server.metrics import FAST_PATH_ERRORS_TOTAL

        orch, capture = _build_sync_orch(tmp_path, monkeypatch)
        # Force the FTS5 write to raise OperationalError.
        capture.raise_on_add = sqlite3.OperationalError("disk full")

        before = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')
        # Must NOT raise even though the FTS5 write blows up.
        orch._fts5_sync_add(  # noqa: SLF001 — bound helper under test
            ["chunk_a"], ["content"], [{"filename": "f.md", "category": "sec"}]
        )
        after = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')
        assert after > before

    def test_ut052_concurrent_add_document_distinct_chunk_ids(self, tmp_path):
        """UT-052: two threads adding distinct chunk_ids both persist (RLock serializes)."""
        index = _make_index(tmp_path, marker_status="complete", docs_indexed=0)
        try:
            errors: list[BaseException] = []

            def worker(chunk_id: str, content: str) -> None:
                try:
                    index.add_document(chunk_id, content, "f.md", "security")
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t1 = threading.Thread(target=worker, args=("chunk_A", "CVE-2021-1111 alpha"))
            t2 = threading.Thread(target=worker, args=("chunk_B", "CVE-2021-2222 beta"))
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            assert errors == []
            ids_alpha = {cid for cid, _ in index.search("CVE-2021-1111", top_k=5)}
            ids_beta = {cid for cid, _ in index.search("CVE-2021-2222", top_k=5)}
            assert "chunk_A" in ids_alpha
            assert "chunk_B" in ids_beta
        finally:
            index.close()


# ===========================================================================
# TestCRUDSyncIntegration — IT-CRUD-001 .. IT-CRUD-004
# ===========================================================================


class TestCRUDSyncIntegration:
    def test_it_crud_001_add_update_remove_reflected_in_search(self, tmp_path, monkeypatch):
        """IT-CRUD-001: add → search → update → search → remove → search sequence."""
        orch, capture = _build_sync_orch(tmp_path, monkeypatch)
        capture.raise_on_add = None

        orch._fts5_sync_add(  # noqa: SLF001
            ["chunk_1"], ["CVE-2021-4034 exploit primer"], [{"filename": "a.md", "category": "sec"}]
        )
        assert capture.index.search("CVE-2021-4034", top_k=5)

        # Update: remove then add same chunk_id with new content.
        capture.index.update_document("chunk_1", "New content mentioning CVE-2999-9999", "b.md", "sec")
        assert capture.index.search("CVE-2999-9999", top_k=5)
        assert capture.index.search("CVE-2021-4034", top_k=5) == []

        capture.index.remove_document("chunk_1")
        assert capture.index.search("CVE-2999-9999", top_k=5) == []

    def test_it_crud_002_concurrent_add_document_via_helper(self, tmp_path, monkeypatch):
        """IT-CRUD-002: 5 threads driving _fts5_sync_add — all rows visible after."""
        orch, capture = _build_sync_orch(tmp_path, monkeypatch)
        capture.raise_on_add = None
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                orch._fts5_sync_add(  # noqa: SLF001
                    [f"chunk_{i}"],
                    [f"content mentioning CVE-2021-{i:04d}"],
                    [{"filename": f"{i}.md", "category": "sec"}],
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert errors == []
        with capture.index._fts5_lock:  # noqa: SLF001
            total = capture.index._conn.execute("SELECT COUNT(*) FROM fts5_documents").fetchone()[0]  # noqa: SLF001
        assert total == 5

    def test_it_crud_003_disk_full_error_swallowed_no_raise(self, tmp_path, monkeypatch):
        """IT-CRUD-003: SQLite OperationalError never propagates to the CRUD tool."""
        from mcp_server.metrics import FAST_PATH_ERRORS_TOTAL

        orch, capture = _build_sync_orch(tmp_path, monkeypatch)
        capture.raise_on_add = sqlite3.OperationalError("disk full")
        before = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')

        # No exception should escape.
        orch._fts5_sync_add(  # noqa: SLF001
            ["chunk_z"], ["content"], [{"filename": "z.md", "category": "sec"}]
        )
        after = _metric_value(FAST_PATH_ERRORS_TOTAL, 'error_class="Fts5CrudSyncError"')
        assert after > before

    def test_it_crud_004_feature_off_skips_fts5_calls(self, tmp_path, monkeypatch):
        """IT-CRUD-004: config.fts5_enabled=False → zero add_document calls made."""
        orch, capture = _build_sync_orch(tmp_path, monkeypatch, fts5_enabled=False)
        capture.raise_on_add = RuntimeError("should never be invoked")

        # Even with a booby-trapped index, no exception because the hook exits early.
        orch._fts5_sync_add(  # noqa: SLF001
            ["chunk_x"], ["content"], [{"filename": "x.md", "category": "sec"}]
        )
        assert capture.add_calls == 0


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _CaptureIndex:
    """Wrapper around a real ``Fts5LexicalIndex`` that can inject failures."""

    def __init__(self, index: Fts5LexicalIndex) -> None:
        self.index = index
        self.raise_on_add: BaseException | None = None
        self.add_calls = 0

    def add_document(self, chunk_id: str, content: str, filename: str, category: str) -> None:
        self.add_calls += 1
        if self.raise_on_add is not None:
            raise self.raise_on_add
        self.index.add_document(chunk_id, content, filename, category)

    def remove_document(self, chunk_id: str) -> None:
        self.index.remove_document(chunk_id)


_SYNC_ORCH_INDEXES: list = []  # real indexes closed by _cleanup_sync_orch below


def _build_sync_orch(tmp_path, monkeypatch, *, fts5_enabled: bool = True):
    """CRUD-sync stub orchestrator; leak fix: config patch via monkeypatch, index
    closed by the autouse yield fixture (the old node-scan finalizer never ran)."""
    import mcp_server.server as srv
    from mcp_server.server import KnowledgeOrchestrator

    Fts5MigrationState(tmp_path / "fts5_migration.state").write(
        {
            "status": "complete",
            "docs_total": 0,
            "docs_indexed": 0,
            "started_at": "2026-08-07T12:00:00Z",
            "completed_at": "2026-08-07T12:00:00Z",
            "error": None,
        }
    )
    real_index = Fts5LexicalIndex(
        db_path=tmp_path / "fts5_index.db",
        state_path=tmp_path / "fts5_migration.state",
    )
    _SYNC_ORCH_INDEXES.append(real_index)
    capture = _CaptureIndex(real_index)

    orch = object.__new__(KnowledgeOrchestrator)
    orch.fts5_index = capture  # helpers accept the duck-typed wrapper
    orch.collection = types.SimpleNamespace(get=lambda where, include: {"ids": []})

    # Bind the real hooks so behaviour under test is production-shaped.
    orch._fts5_sync_add = types.MethodType(  # noqa: SLF001
        KnowledgeOrchestrator._fts5_sync_add, orch
    )
    orch._fts5_sync_remove_by_doc_id = types.MethodType(  # noqa: SLF001
        KnowledgeOrchestrator._fts5_sync_remove_by_doc_id, orch
    )

    monkeypatch.setattr(srv.config, "fts5_enabled", fts5_enabled)
    return orch, capture


@pytest.fixture(autouse=True)
def _cleanup_sync_orch():
    yield  # close every index created by _build_sync_orch (yield teardown)
    while _SYNC_ORCH_INDEXES:
        _SYNC_ORCH_INDEXES.pop().close()


def _metric_value(name: str, label: str) -> float:
    """Read a labelled counter from the shared metrics collector."""
    from mcp_server.metrics import get_metrics

    prefix = f"{name}{{{label}}} " if label else f"{name} "
    for line in get_metrics().exposition().split("\n"):
        if line.startswith(prefix):
            return float(line.split()[-1])
    return 0.0


# --- Package B (v4.8.3 Gate 0): B04–B10; B01–B03 + marker-v2 in test_v483_hotfix.py ---


def _fake_collection(rows: List[Row], *, shuffle: bool = False, fault: str | None = None):
    rows = [tuple(str(v) for v in row) for row in rows]
    all_ids = ([row[0] for row in rows] + ([rows[0][0]] if fault == "duplicate_id" else [])
               + (["chunk_alien"] if fault == "overlong" else []))

    def count():
        if fault == "count":
            raise RuntimeError("chroma count unavailable")
        return len(rows) + (1 if fault == "duplicate_id" else 0)

    def get(ids=None, include=(), limit=None, offset=0):
        if ids is None:
            return {"ids": all_ids[offset : offset + (limit or len(all_ids))]}
        if fault == "read":
            raise RuntimeError("chroma get failed")
        sel = [row for row in rows if row[0] in set(ids)]
        sel = sel[1:] + sel[:1] if shuffle else (sel[1:] if fault == "short_source" else sel)  # true derangement
        if fault in ("dup_hydrated", "extra_hydrated"):
            sel = sel + (sel[:1] if fault == "dup_hydrated" else [("chunk_alien", "x", "y", "z")])
        docs = [row[1] for row in sel]
        return {"ids": [row[0] for row in sel], "documents": docs[:-1] if fault == "truncated" else docs,
                "metadatas": [{"filename": row[2], "category": row[3]} for row in sel]}

    return types.SimpleNamespace(count=count, get=get)


def _block_finalize(monkeypatch: pytest.MonkeyPatch) -> Tuple[threading.Event, threading.Event]:
    entered, release = threading.Event(), threading.Event()
    real_finalize = Fts5LexicalIndex._finalize_rebuild

    def _paused(self, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=10.0), "finalize barrier never released"
        return real_finalize(self, *args, **kwargs)

    monkeypatch.setattr(Fts5LexicalIndex, "_finalize_rebuild", _paused)
    return entered, release


_STALE_STAGING_SQL = ('CREATE VIRTUAL TABLE IF NOT EXISTS "fts5_documents_staging_g1" '
                      "USING fts5(chunk_id UNINDEXED, content, filename, category)")


def _worker_shell(index, collection):
    orch = object.__new__(KnowledgeOrchestrator)
    orch._index_lock = threading.RLock()
    orch._fts5_dispatch_lock = threading.Lock()
    orch._fts5_dispatch_active = False
    orch._fts5_dispatch_pending = False
    orch.fts5_index = index
    orch.collection = collection
    return orch


@pytest.mark.parametrize("fault", ["count", "read", "short_source", "duplicate_id", "overlong", "dup_hydrated", "extra_hydrated", "truncated", "sql"])
def test_b04_b10_fts5_snapshot_faults_publish_failed_non_ready(tmp_path, monkeypatch, fault):
    index = _make_index(tmp_path)
    if fault == "sql":
        monkeypatch.setattr(Fts5LexicalIndex, "_populate_staging", _sql_boom)
    orch = _worker_shell(index, _fake_collection(_gen_rows(12), fault=None if fault == "sql" else fault))
    KnowledgeOrchestrator._fts5_rebuild_worker(orch)  # synchronous in-test; admission inside the worker
    try:
        state = index.state.read()
        assert (state["status"] == "failed" and state["generation"] == 1 and index.is_ready() is False
                and index.search_if_ready("CVE-2021-0001") is None)
    finally:
        index.close()


def test_b05_to_b10_fts5_generation_lifecycle_and_locks(tmp_path, monkeypatch):
    """B05 zero-rebuild; B06 lock; B07 single-flight; B08 barrier; B09 restore; B10 stale (+D2/D4)."""
    rows = _gen_rows(120)
    index = _make_index(tmp_path, marker_status="in_progress", docs_indexed=40)
    entered, release = _block_finalize(monkeypatch)
    try:
        for row in reversed(rows[5:40]):  # TQ-2: corrupted/REORDERED partial prefix, not the canonical first 40
            index.add_document(*row)
        index.add_document("chunk_0007", "CORRUPTED stale content", "f.md", "security")
        with index._fts5_lock:  # noqa: SLF001 — stale staging residue from an interrupted run (TQ-2)
            index._conn.execute(_STALE_STAGING_SQL)  # noqa: SLF001
            index._conn.execute('INSERT INTO "fts5_documents_staging_g1" VALUES (?, ?, ?, ?)', ("stale_1", "STALE relic", "s.md", "x"))  # noqa: SLF001
            index._conn.commit()  # noqa: SLF001
        assert index.is_ready() is False, "unversioned partial marker must not be credible (B05)"
        release.set()  # B05 rebuild runs through unblocked; reversed input rows (D4)
        result = index.rebuild_content_bound(list(reversed(rows)))
        assert result["source_rows_sha256"] == compute_rows_digest(rows)[0] == result["verified_fts_rows_sha256"]
        assert index.count() == 120 and index.is_ready() is True, "rebuild-from-zero, no duplicates (B05)"
        with index._fts5_lock:  # noqa: SLF001
            rowid_order = [r[0] for r in index._conn.execute("SELECT chunk_id FROM fts5_documents ORDER BY rowid")]  # noqa: SLF001
        assert rowid_order == sorted(rowid_order, key=lambda c: c.encode("utf-8")), "canonical rowid order (D4)"
        assert index.search_if_ready("CORRUPTED") == [] and index.search_if_ready("STALE") == [], "no residue (B05/TQ-2)"
        assert index.search_if_ready("CVE-2021-0001"), "credible generation must serve (B08)"
        release.clear()
        entered.clear()
        orch = _worker_shell(index, _fake_collection([("new_1", "candidate NEWTOKEN", "n.md", "sec")]))
        orch._index_all_impl = lambda *args, **kwargs: {"ran": True}
        worker = threading.Thread(target=KnowledgeOrchestrator._fts5_rebuild_worker, args=(orch,))
        worker.start()
        assert entered.wait(timeout=5.0)  # worker paused immediately before complete/ready (Locks §2)
        assert index.begin_rebuild() is None, "ordinary starts are single-flight (B07)"
        assert KnowledgeOrchestrator.index_all(orch).get("skipped_reason") == "reindex_already_running", "B06"
        assert index.search_if_ready("NEWTOKEN") is None, "candidate must not serve pre-publication (B08)"
        gen2 = index.invalidate_generation()
        assert index.search_if_ready("CVE-2021-0001") is None, "invalidated generation must not serve (B08)"
        assert index.begin_rebuild() == gen2, "successor admitted after invalidation (B07)"
        epoch_before = index._mutation_epoch  # noqa: SLF001 — TQ-3 witness
        release.set()
        worker.join(timeout=10.0)
        assert index._mutation_epoch == epoch_before, "a stale generation must never execute the swap ALTER (TQ-3)"  # noqa: SLF001
        state = index.state.read()
        assert state["status"] == "invalidated" and state["generation"] == gen2 and index.is_ready() is False, \
            "durable invalidation; late complete dropped (D2/B07)"
        assert index.rebuild_content_bound(_gen_rows(8, "succ"), generation=gen2)["status"] == "complete"
        index.publish_rebuild_failure(1, RuntimeError("stale late failure"))  # B10 stale guard (gen1)
        state = index.state.read()
        assert state["status"] == "complete" and state["generation"] == gen2 and index.is_ready() is True
        assert index.search_if_ready("NEWTOKEN") == [], "stale candidate never becomes visible (B08/B10)"
        prior_marker, real_write = index.state.read(), index.state.write
        prior_live = index._live_digest()  # noqa: SLF001 — TQ-3: exact prior rows must survive
        armed = {"on": True}

        def failing_complete_write(payload):  # P1-1: fail after the swap commit, at complete publication
            if armed["on"] and payload.get("status") == "complete":
                armed["on"] = False
                raise OSError("disk full during publication")
            real_write(payload)
        index.state.write = failing_complete_write
        with pytest.raises(Fts5MigrationError, match="complete-marker publication failed"):
            index.rebuild_content_bound(_gen_rows(8, "succ"), generation=gen2)  # same-source candidate
        del index.state.write
        assert index.state.read() == prior_marker and index.is_ready() is True, "same-source prior restored (P1-1/P1-2)"
        assert index.search_if_ready("CVE-2021-0002") and index.count() == 8, "prior rows survive publication failure"
        assert index._live_digest()[:2] == prior_live[:2], "exact prior rows/digest preserved (B09/TQ-3)"  # noqa: SLF001
    finally:
        release.set()
        index.close()


def test_p1_8_fts5_dispatch_daemon_and_thread_start_rollback(tmp_path, monkeypatch):
    """P1-8: daemon rebuild worker; a failed Thread.start consumes no admission."""
    index = _make_index(tmp_path)
    orch = _worker_shell(index, _fake_collection(_gen_rows(3)))

    def failing_start(self):
        raise RuntimeError("thread start failed")
    monkeypatch.setattr(threading.Thread, "start", failing_start)
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)  # must neither raise nor consume admission
    monkeypatch.undo()
    assert index.begin_rebuild() == 1, "admission must not be wedged by a failed dispatch"
    index.end_rebuild(1)
    real_thread_cls, instances = threading.Thread, []

    class RecordingThread(real_thread_cls):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", RecordingThread)
    hold, release_capture = threading.Event(), threading.Event()

    def gated_capture(collection, _real=fts5_module.capture_chunk_rows):
        hold.set()
        assert release_capture.wait(timeout=10.0), "capture gate never released"
        return _real(collection)

    monkeypatch.setattr("mcp_server.server.capture_chunk_rows", gated_capture)
    barrier = threading.Barrier(2)

    def dispatch():
        barrier.wait(timeout=5.0)
        KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)

    callers = [real_thread_cls(target=dispatch) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=5.0)
    try:
        assert hold.wait(timeout=5.0)
        KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)  # live reservation blocks a third dispatch too
        assert len(instances) == 1 and instances[0].daemon is True, "one reserved daemon worker (D1/P1-8)"
        release_capture.set()
        instances[0].join(timeout=10.0)
        for successor in list(instances[1:]):  # BC-05 successor from the pending third request, if any
            successor.join(timeout=10.0)
        assert index.is_ready() is True
        assert _metric_value("knowledge_rag_fast_path_migration_docs_indexed", "") == 3.0, "indexed after complete (D6)"
        monkeypatch.setattr(Fts5LexicalIndex, "_populate_staging", _sql_boom)
        KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(_gen_rows(3, "other"))))
        assert _metric_value("knowledge_rag_fast_path_migration_docs_total", "") == 3.0, "total = captured snapshot (D6)"
        assert _metric_value("knowledge_rag_fast_path_migration_docs_indexed", "") == 0.0, "failure resets indexed (D6)"
        KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(_gen_rows(3), fault="count")))
        assert (_metric_value("knowledge_rag_fast_path_migration_docs_total", "") == 0.0
                and _metric_value("knowledge_rag_fast_path_migration_docs_indexed", "") == 0.0), \
            "capture failure resets BOTH gauges (D6)"
    finally:
        index.close()


def test_p1_7_fts5_builder_binds_source_and_target_to_cli_root(tmp_path, monkeypatch):
    """P1-7/D5: --data-dir binds Chroma source + FTS target; lab paths never consulted; fail-closed."""
    import importlib.util

    import chromadb

    import mcp_server.server as srv

    root, lab = tmp_path / "root", tmp_path / "lab-sentinel"
    monkeypatch.setattr(srv.config, "data_dir", lab, raising=False)
    monkeypatch.setattr(srv.config, "chroma_dir", lab / "chroma_db", raising=False)
    collection = chromadb.PersistentClient(path=str(root / "chroma_db")).get_or_create_collection(name="knowledge_base")
    collection.add(ids=["c1", "c2", "c3"], documents=["alpha TOKROOT", "beta", "gamma"],
                   metadatas=[{"filename": "a.md", "category": "x"}] * 3, embeddings=[[0.0, 0.1]] * 3)
    spec = importlib.util.spec_from_file_location("build_fts5_index_under_test",
                                                  _REPO_ROOT / "scripts" / "build_fts5_index.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.delenv("KNOWLEDGE_RAG_DIR", raising=False)
    with pytest.raises(SystemExit, match="data root does not exist"):
        builder.main(["--data-dir", str(tmp_path / "missing-root")])
    bare = tmp_path / "bare-root"
    bare.mkdir()
    (bare / "fts5_index.db").write_bytes(b"PRIOR")
    with pytest.raises(SystemExit, match="chroma source does not exist"):
        builder.main(["--data-dir", str(bare)])
    assert (bare / "fts5_index.db").read_bytes() == b"PRIOR", "prior FTS preserved on fail-closed exit (D5)"
    assert not (bare / "chroma_db").exists(), "fail-closed: no empty corpus (D5)"
    assert builder.main(["--data-dir", str(root)]) == 0
    assert (root / "fts5_index.db").exists() and not lab.exists(), "root-bound target; lab never consulted (P1-7)"
    (tmp_path / "config.yaml").write_text("search:\n  collection_name: par_name\n", encoding="utf-8")
    (root / "config.yaml").write_text("search:\n  collection_name: loc_name\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="ambiguous collection_name"):
        builder.main(["--data-dir", str(root)])  # conflicting adjacent/root-local configs fail closed (D5-r5)
    project = tmp_path / "project"  # D5-r5: canonical layout — project/config.yaml beside project/data
    (project / "data" / "chroma_db").mkdir(parents=True)
    (project / "config.yaml").write_text("search:\n  collection_name: vet_custom\n", encoding="utf-8")
    sub = chromadb.PersistentClient(path=str(project / "data" / "chroma_db")).get_or_create_collection(name="vet_custom")
    sub.add(ids=["c1"], documents=["alpha SUBTOKEN"], metadatas=[{"filename": "a", "category": "x"}], embeddings=[[0.1, 0.2]])
    env = {**os.environ, "KNOWLEDGE_RAG_DIR": str(tmp_path / "sub-lab-sentinel")}
    proc = subprocess.run([sys.executable, str(_REPO_ROOT / "scripts" / "build_fts5_index.py"),
                           "--data-dir", str(project / "data")], env=env, cwd=str(_REPO_ROOT),
                          capture_output=True, text=True, timeout=55)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-500:]
    assert (project / "data" / "fts5_index.db").exists() and not (tmp_path / "sub-lab-sentinel").exists(), \
        "canonical project/config.yaml collection_name honored; lab binding overridden (D5-r5)"
    assert (not (project / "data" / "data").exists() and not (project / "data" / "documents").exists()
            and not (project / "data" / "models").exists()), "no nested config-created roots (D5)"
    reopened = Fts5LexicalIndex(db_path=project / "data" / "fts5_index.db",
                                state_path=project / "data" / "fts5_migration.state")
    try:  # TQ-4: exact row count, token retrieval and digest — not only file existence
        assert reopened.count() == 1 and reopened.search("SUBTOKEN"), "exact rows must be retrievable (TQ-4)"
        marker = reopened.state.read()
        assert marker["docs_total"] == 1 and marker["source_rows_sha256"] == compute_rows_digest(
            [("c1", "alpha SUBTOKEN", "a", "x")])[0], "exact digest (TQ-4)"
    finally:
        reopened.close()


def _sql_boom(self, staging, rows, *args):
    raise sqlite3.OperationalError("disk I/O error")


def test_d2_d3_durable_invalidation_generation_authority_and_rollback(tmp_path, monkeypatch):
    """D2/P1-2: durable reset, reopen restore, early pre-marker barrier; D3: restore needs exact live match."""
    index = _make_index(tmp_path)
    index.rebuild_content_bound([(f"c{i}", f"OLDTOKEN {i}", "f.md", "x") for i in range(3)])
    gen2 = index.invalidate_generation()
    state = index.state.read()
    assert state["generation"] == gen2 and state["status"] != "complete", "invalidation must persist durably (D2)"
    index.close()
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5_index.db", state_path=tmp_path / "fts5_migration.state")
    try:
        assert index._generation == gen2 and index.is_ready() is False, "reopen restores counter, non-ready (D2)"  # noqa: SLF001
        assert index.rebuild_content_bound([(f"c{i}", f"OLDTOKEN {i}", "f.md", "x") for i in range(3)], generation=index.begin_rebuild())["status"] == "complete"
        assert index.is_ready() is True, "retry after reopen must succeed (D2)"
        started, resume = threading.Event(), threading.Event()

        def delaying_digest(rows_arg, _real=compute_rows_digest):
            if not started.is_set():
                started.set()
                assert resume.wait(timeout=10.0), "early barrier never released"
            return _real(rows_arg)
        monkeypatch.setattr(fts5_module, "compute_rows_digest", delaying_digest)
        delayed = threading.Thread(target=index.rebuild_content_bound, args=([("d1", "DELAYED", "f.md", "x")],), kwargs={"generation": gen2})
        delayed.start()
        assert started.wait(timeout=5.0)
        gen3 = index.invalidate_generation()
        assert index.rebuild_content_bound([(f"n{i}", f"GEN3TOKEN {i}", "f.md", "x") for i in range(3)], generation=gen3)["status"] == "complete"
        resume.set()
        delayed.join(timeout=10.0)
        state = index.state.read()
        assert state["status"] == "complete" and state["generation"] == gen3 and index.count() == 3 \
            and index.is_ready() is True, "delayed pre-marker write dropped; live DB stays gen3 (P1-2/D2)"
        reverse_name = "_reverse_swap" if hasattr(Fts5LexicalIndex, "_reverse_swap") else "_reverse_swap_quiet"
        monkeypatch.setattr(Fts5LexicalIndex, reverse_name, lambda self, staging, backup: True)  # false success: bytes stay candidate
        real_write, armed = index.state.write, {"armed": True}

        def failing_write(payload):
            if armed["armed"] and payload.get("status") == "complete":
                armed["armed"] = False
                raise OSError("publication failed")
            real_write(payload)
        index.state.write = failing_write
        with pytest.raises(Fts5MigrationError, match="complete-marker publication failed"):
            index.rebuild_content_bound([(f"n{i}", f"NEWTOKEN {i}", "f.md", "x") for i in range(3)], generation=gen3)
        del index.state.write
        assert index.is_ready() is False and index.state.read()["status"] != "complete", "restore requires exact live match (D3)"
        assert index.search_if_ready("NEWTOKEN") is None and index.search_if_ready("GEN3TOKEN") is None
    finally:
        resume.set()
        index.close()


def test_p1_1_startup_dispatch_is_nonblocking_bounded(tmp_path, monkeypatch):
    """P1-1: startup dispatch returns after bounded admission; the exact
    source-vs-live verification runs behind the worker; hybrid serves meanwhile."""
    index = _make_index(tmp_path)
    index.rebuild_content_bound(_gen_rows(10))
    index.close()
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5_index.db", state_path=tmp_path / "fts5_migration.state")
    orch = _worker_shell(index, _fake_collection(_gen_rows(10)))
    orch._fts5_startup_dispatch_done = False
    monkeypatch.setattr("mcp_server.server.config.fts5_enabled", True)
    entered, release = threading.Event(), threading.Event()

    def gated_digest(rows_arg, _real=compute_rows_digest):
        entered.set()
        assert release.wait(timeout=10.0), "verification gate never released"
        return _real(rows_arg)

    monkeypatch.setattr(fts5_module, "compute_rows_digest", gated_digest)
    dispatcher = threading.Thread(target=KnowledgeOrchestrator._dispatch_fts5_startup_rebuild, args=(orch,))
    dispatcher.start()
    dispatcher.join(timeout=3.0)
    try:
        assert not dispatcher.is_alive(), "startup dispatch must not block on full verification (P1-1)"
        assert index.is_ready() is False, "hybrid fallback until the worker verifies (P1-1)"
        assert entered.wait(timeout=5.0)
    finally:
        release.set()
        dispatcher.join(timeout=10.0)
    for worker in [t for t in threading.enumerate() if t.name == "fts5-rebuild"]:
        worker.join(timeout=10.0)
    assert index.is_ready() is True, "promotion only after source-verified equality (P1-1/P1-2)"
    index.close()


def test_p1_2_reset_with_new_source_never_restores_or_promotes_old(tmp_path):
    """P1-2 production shape: FTS A ready; canonical Chroma becomes B; a stale
    credible A marker must not promote for B, and a reset + candidate
    publication failure must never make old FTS-A ready for B."""
    rows_a = [(f"a{i}", f"ATOKEN {i}", "f.md", "x") for i in range(3)]
    rows_b = [(f"b{i}", f"BTOKEN {i}", "f.md", "x") for i in range(3)]
    rows_c = [(f"c{i}", f"CTOKEN {i}", "f.md", "x") for i in range(3)]
    index = _make_index(tmp_path)
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_a)))
    assert index.is_ready() is True and index.search_if_ready("ATOKEN")
    index.close()  # restart with corpus already B (e.g., lost invalidation) and credible marker A
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5_index.db", state_path=tmp_path / "fts5_migration.state")
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_b)))
    assert index.is_ready() is True and index.search_if_ready("BTOKEN"), "verifier must rebuild, not promote"
    assert index.search_if_ready("ATOKEN") == [], "credible-but-stale marker must not promote old bytes (P1-2)"
    try:
        gen_c = index.invalidate_generation()  # production reset order: Chroma already changed to C
        real_write, armed = index.state.write, {"on": True}

        def failing_write(payload):
            if armed["on"] and payload.get("status") == "complete":
                armed["on"] = False
                raise OSError("publication failed")
            real_write(payload)

        index.state.write = failing_write
        KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_c)))
        del index.state.write
        assert index.is_ready() is False, "old FTS-B must never become ready for corpus C (P1-2)"
        assert index.search_if_ready("BTOKEN") is None and index.search_if_ready("CTOKEN") is None
        assert index.state.read()["status"] != "complete", "durable state stays nonready/failed (P1-2)"
        KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_c)))  # retryable
        assert index.is_ready() is True and index.search_if_ready("CTOKEN")
        assert gen_c == index.state.read()["generation"]
    finally:
        index.close()


def test_bc04_verify_and_publish_requires_exact_source_and_generation(tmp_path):
    """BC-04: the reusable generation-guarded publisher promotes only on exact
    live count/distinct/digest equality with the given source identity."""
    index = _make_index(tmp_path)
    try:
        rows = _gen_rows(6)
        index.rebuild_content_bound(rows)
        source_digest, total = compute_rows_digest(rows)
        assert index.verify_and_publish(source_digest, total, index._generation) is True  # noqa: SLF001 — idempotent
        wrong_digest = compute_rows_digest(_gen_rows(6, "zz"))[0]
        assert index.verify_and_publish(wrong_digest, total, index._generation) is False, "mismatch must not publish"  # noqa: SLF001
        assert index.verify_and_publish(source_digest, total, index._generation + 1) is False, "non-current generation"  # noqa: SLF001
        assert index.is_ready() is True and index.state.read()["status"] == "complete"
        index.invalidate_generation()
        assert index.is_ready() is False
        assert index.verify_and_publish(source_digest, total, index._generation) is True, "live==source recovers reset"  # noqa: SLF001
        assert index.is_ready() is True
    finally:
        index.close()


def test_bc05_recovery_request_while_worker_finishes_starts_one_successor(tmp_path, monkeypatch):
    """BC-05: a dispatch in the writer-release→flag-clear window must start
    exactly one deterministic successor — never zero, never parallel."""
    index = _make_index(tmp_path)
    orch = _worker_shell(index, _fake_collection(_gen_rows(3)))
    in_finish, resume_finish = threading.Event(), threading.Event()
    real_end = Fts5LexicalIndex.end_rebuild

    def gated_end(self, generation):
        real_end(self, generation)
        in_finish.set()
        assert resume_finish.wait(timeout=10.0), "finish gate never released"

    monkeypatch.setattr(Fts5LexicalIndex, "end_rebuild", gated_end)
    real_thread_cls, workers = threading.Thread, []

    class Rec(real_thread_cls):
        def __init__(self, *args, **kwargs):
            workers.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", Rec)
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)
    assert in_finish.wait(timeout=10.0)  # _index_lock already released; dispatch flag still set
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch, material=True)  # material recovery inside the window (BC-05/P1-C)
    assert len(workers) == 1, "no parallel worker inside the finish window (BC-05)"
    resume_finish.set()
    workers[0].join(timeout=10.0)
    for successor in list(workers[1:]):
        successor.join(timeout=10.0)
    try:
        assert len(workers) == 2, "exactly one deterministic successor after the active worker clears (BC-05)"
        assert index.is_ready() is True
    finally:
        index.close()


def test_p1c_ordinary_dispatch_coalesces_material_reset_spawns_one_successor(tmp_path, monkeypatch):
    """P1-C: ordinary duplicates coalesce into the live worker (no successor);
    a production reset during a blocked worker invalidates immediately and
    yields exactly one material successor with an advanced durable generation."""
    index = _make_index(tmp_path)
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(_gen_rows(3))))
    assert index.is_ready() is True
    orch = _worker_shell(index, _fake_collection(_gen_rows(3)))
    monkeypatch.setattr("mcp_server.server.config.fts5_enabled", True)
    hold, release_capture = threading.Event(), threading.Event()

    def gated_capture(collection, _real=fts5_module.capture_chunk_rows):
        hold.set()
        assert release_capture.wait(timeout=10.0)
        return _real(collection)

    monkeypatch.setattr("mcp_server.server.capture_chunk_rows", gated_capture)
    real_thread_cls, workers = threading.Thread, []

    class Rec(real_thread_cls):
        def __init__(self, *args, **kwargs):
            workers.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", Rec)
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)
    assert hold.wait(timeout=5.0)
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)  # ordinary duplicates while a worker is live
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)
    release_capture.set()
    workers[0].join(timeout=10.0)
    for extra in list(workers[1:]):
        extra.join(timeout=10.0)
    assert len(workers) == 1, "ordinary duplicates must coalesce into the live worker (P1-C)"
    assert index.is_ready() is True
    hold.clear()
    release_capture.clear()
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch)
    assert hold.wait(timeout=5.0)
    orch._fts5_startup_dispatch_done = True
    KnowledgeOrchestrator._fts5_reset_and_rebuild(orch)  # production reset while the old generation is blocked
    assert index.is_ready() is False, "reset invalidates immediately (P1-C; REDs if the invalidate call is removed)"
    release_capture.set()
    for worker in list(workers[1:]):
        worker.join(timeout=10.0)
    for worker in list(workers):
        worker.join(timeout=10.0)
    assert len(workers) == 3, "exactly one material successor — never W2+W3 twins (P1-C/BC-05)"
    state = index.state.read()
    assert state["status"] == "complete" and state["generation"] == 2 and index.is_ready() is True
    index.close()
