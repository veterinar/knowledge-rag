"""v4.8.3 hotfix regression suite.

Each test reproduces exactly one production bug found in v4.8.0/4.8.1/4.8.2
before the fix, then asserts the fix. Test names encode the GH issue when
one exists and the internal fix number otherwise.

Cross-reference:
- fix1_index_lock_instance_level   — internal (nuclear_rebuild + concurrent reindex)
- b03_fts5_snapshot_capture        — Package B (map by returned ID, no positional zip)
- marker_v2_rejection              — Package B (schema-v2 credibility, no unversioned ready)
- gh161_write_dispatch_isolates_staging  — GH #161 (grishkovei)
- gh162_checkpoint_only_committed_docs   — GH #162 (grishkovei)
- gh163_force_flag_propagates_through    — GH #163 (grishkovei)
- b01/b02 — Package B (digest integrity; handles-only init + one startup dispatch)
- fix6_format_skips_orphan_hits          — internal (Chroma-FTS5 drift residue)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# fix1 — _index_lock must be per-instance so a GC'd orch can't wedge others
# ---------------------------------------------------------------------------


def test_fix1_index_lock_is_per_instance() -> None:
    """Two orch instances must own distinct locks; class-level singleton wedged v4.8.2."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        a = KnowledgeOrchestrator()
        a._index_lock = threading.Lock()
        b = KnowledgeOrchestrator()
        b._index_lock = threading.Lock()
        assert a._index_lock is not b._index_lock


# --- B03 (Package B): snapshot capture maps explicit-ID responses by id ---


def test_b03_fts5_snapshot_capture_maps_by_returned_id() -> None:
    """B03: shuffled explicit-ID responses map by returned id, never positional zip."""
    from test_fts5_migration import _fake_collection

    from mcp_server.fts5_index import capture_chunk_rows, compute_rows_digest

    rows = [(f"chunk_{i:03d}", f"content {i}", f"f{i}.md", "sec") for i in range(7)]
    captured = capture_chunk_rows(_fake_collection(list(reversed(rows)), shuffle=True), batch_size=3)
    assert captured == sorted(rows, key=lambda row: row[0].encode("utf-8")), "canonical global order (P1-6)"
    assert compute_rows_digest(captured) == compute_rows_digest(rows)


# --- Marker v2 rejection (Package B): replaces the unversioned-ready assert ---


@pytest.mark.parametrize("case", ["legacy_v1", "malformed", "missing_digest", "count_mismatch", "digest_mismatch",
                                  "fake_hex_equal", "live_content_corrupt", "live_id_corrupt",
                                  "bool_generation", "zero_generation"])
def test_marker_v2_rejection_blocks_serving_readiness(tmp_path: Any, case: str) -> None:
    """Non-credible markers and live-byte corruption stay non-ready on reopen (P1-5)."""
    from mcp_server.fts5_index import Fts5LexicalIndex

    marker_path = tmp_path / "fts5_migration.state"
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5.db", state_path=marker_path)
    index.rebuild_content_bound([(f"chunk_{i}", f"content token{i}", "f.md", "sec") for i in range(9)])
    credible = index.state.read()
    index.close()
    if case.startswith("live_"):
        column, value = ("content", "corrupted") if case == "live_content_corrupt" else ("chunk_id", "chunk_zz")
        connection = sqlite3.connect(str(tmp_path / "fts5.db"))
        connection.execute(f"UPDATE fts5_documents SET {column} = ? WHERE rowid = 1", (value,))
        connection.commit()
        connection.close()
    elif case == "malformed":
        marker_path.write_text("{not json", encoding="utf-8")
    else:
        payloads = {"legacy_v1": {"status": "complete", "docs_total": 9, "docs_indexed": 9},
                    "missing_digest": {**credible, "verified_fts_rows_sha256": None},
                    "count_mismatch": {**credible, "docs_total": 14, "docs_indexed": 14},
                    "digest_mismatch": {**credible, "verified_fts_rows_sha256": "0" * 64},
                    "fake_hex_equal": {**credible, "source_rows_sha256": "z" * 64, "verified_fts_rows_sha256": "z" * 64},
                    "bool_generation": {**credible, "generation": True},
                    "zero_generation": {**credible, "generation": 0}}
        marker_path.write_text(json.dumps(payloads[case]), encoding="utf-8")
    reopened = Fts5LexicalIndex(db_path=tmp_path / "fts5.db", state_path=marker_path)
    try:
        assert reopened.is_ready() is False and reopened.search_if_ready("token3") is None
        with reopened._fts5_lock:  # noqa: SLF001 — credibility + live bytes must both reject (P1-5, strict ints)
            assert not reopened._live_matches(reopened.state.read())  # noqa: SLF001
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# GH #161 — write dispatch must isolate staging from live query path
# ---------------------------------------------------------------------------


def test_gh161_write_collection_routes_to_staging_when_active() -> None:
    """During nuclear_rebuild populate, writes hit staging; reads hit production."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        prod = MagicMock(name="prod_collection")
        staging = MagicMock(name="staging_collection")
        orch.collection = prod
        orch._staging_target = None

        # No staging active → writes route to production.
        assert orch._write_collection is prod

        # Staging active → writes route to staging, but self.collection stays prod.
        orch._staging_target = staging
        assert orch._write_collection is staging
        assert orch.collection is prod, "query path must keep seeing production during populate"

        # Post-swap cleanup restores default routing.
        orch._staging_target = None
        assert orch._write_collection is prod


# ---------------------------------------------------------------------------
# GH #162 — checkpoint must serialize only docs committed by current run
# ---------------------------------------------------------------------------


def test_gh162_tracking_only_records_committed_this_run() -> None:
    """Checkpoint IDs must not include unprocessed docs from prior metadata."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        orch._reindex_progress = {"operation": "smart_reindex"}
        orch._indexed_docs = {}  # required by _seed_chunks_total_estimate
        stats = {"total_files": 3}
        tracking = orch._init_reindex_tracking(
            resume_state={"doc_ids": ["prior_run_committed"], "chunks_processed": 42},
            stats=stats,
        )

        assert "committed_this_run" in tracking, "tracking must expose committed_this_run set"
        assert tracking["committed_this_run"] == {"prior_run_committed"}, (
            "resume must fold in prior-run committed IDs so they survive interruption"
        )
        assert tracking["chunks_processed"] == 42


# ---------------------------------------------------------------------------
# GH #163 — force=True must propagate through reindex_all to index_all
# ---------------------------------------------------------------------------


def test_gh163_reindex_all_accepts_and_propagates_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """reindex_all(force=True) must call index_all(force=True), not force=False."""
    from mcp_server.server import KnowledgeOrchestrator

    calls: list[dict[str, Any]] = []

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        orch._index_lock = threading.RLock()
        orch.bm25_index = MagicMock()
        orch._bm25_initialized = False
        orch.query_cache = MagicMock()
        orch._ensure_bm25_index = lambda: None
        orch._save_metadata = lambda: None
        orch._indexed_docs = {}

        def fake_index_all(**kwargs: Any) -> dict[str, int]:
            calls.append(kwargs)
            return {
                "indexed": 0,
                "updated": 0,
                "deleted": 0,
                "chunks_added": 0,
                "errors": 0,
                "total_files": 0,
                "skipped": 0,
            }

        orch.index_all = fake_index_all
        # Skip filesystem work — nuclear_rebuild path irrelevant for this test.
        with patch("shutil.rmtree"), patch("pathlib.Path.iterdir", return_value=[]):
            orch.reindex_all(force=True)

    assert calls, "reindex_all must invoke index_all"
    assert calls[0].get("force") is True, f"index_all must receive force=True, got {calls[0]!r}"


# --- B01/B02 (Package B): digest integrity; constructor/main dispatch ---


@pytest.mark.parametrize("tamper", ["wrong_id", "wrong_content"])
def test_b01_fts5_digest_integrity_rejects_equal_count_corruption(tmp_path, monkeypatch, tamper):
    """B01: same-count wrong-ID/content read-backs fail digest verification."""
    from mcp_server.fts5_index import Fts5LexicalIndex, Fts5MigrationError

    rows = [(f"chunk_{i:03d}", f"content {i}", "f.md", "sec") for i in range(20)]
    real_populate = Fts5LexicalIndex._populate_staging

    def _tampering(self: Any, staging: str, ordered: Any, *args: Any) -> Any:
        bad = list(ordered)
        cid, content, fn, cat = bad[7]
        bad[7] = ("chunk_999", content, fn, cat) if tamper == "wrong_id" else (cid, "corrupted", fn, cat)
        return real_populate(self, staging, bad, *args)

    monkeypatch.setattr(Fts5LexicalIndex, "_populate_staging", _tampering)
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5.db", state_path=tmp_path / "m.state")
    try:
        with pytest.raises(Fts5MigrationError, match="verification failed"):
            index.rebuild_content_bound(rows)
        assert (index.state.read() or {}).get("status") == "failed" and index.is_ready() is False
    finally:
        index.close()


@pytest.mark.parametrize("corpus", ["populated", "empty"])
def test_b02_fts5_constructor_handles_only_and_main_dispatches_once(tmp_path, monkeypatch, corpus):
    """B02: handles-only __init__; main(): primary decision (incl. empty-corpus
    initial index) -> exactly one FTS dispatch -> watcher/transport."""
    import sys
    from contextlib import nullcontext

    from test_vetclub_runtime_regressions import _FakeObserver, _FakeWatcher, _main_invocation, _production_orchestrator

    import mcp_server as package
    import mcp_server.instance_lock as instance_lock
    import mcp_server.preflight as preflight
    from mcp_server import server as srv

    monkeypatch.setattr(srv.config, "fts5_enabled", True)
    orch = _production_orchestrator(tmp_path, monkeypatch)
    try:
        assert orch.fts5_index is not None and orch.fts5_index.is_ready() is False
        assert not (tmp_path / "data" / "fts5_migration.state").exists(), "no complete 0/0 marker from __init__"
        assert not any(t.name == "fts5-rebuild" for t in threading.enumerate()), "no dispatch from __init__"
    finally:
        orch.fts5_index.close()

    order: list = []
    fake = SimpleNamespace(collection=SimpleNamespace(count=lambda: 1 if corpus == "populated" else 0),
                           _check_dimension_mismatch=lambda: (order.append("primary_decision"), False)[1],
                           _dispatch_fts5_startup_rebuild=lambda: order.append("fts5_dispatch"))
    fake.index_all = lambda: (order.append("initial_index"), {"indexed": 0, "chunks_added": 0})[1]
    monkeypatch.setattr(instance_lock, "single_instance_lock", nullcontext)
    monkeypatch.setattr(preflight, "run_preflight", lambda: None)
    monkeypatch.setattr(srv, "get_orchestrator", lambda: fake)
    monkeypatch.setattr(srv, "DocumentWatcher", lambda *args, **kwargs: _FakeWatcher(order))
    monkeypatch.setattr(srv, "Observer", lambda: _FakeObserver(order))
    monkeypatch.setattr(srv, "_serve_with_lifecycle", lambda transport, observer, watcher: order.append("serve"))
    monkeypatch.setattr(package, "_original_stdout", sys.stdout)
    monkeypatch.delenv("KNOWLEDGE_RAG_WATCHER_DISABLED", raising=False)
    with _main_invocation():
        srv.main()
    expected_head = ["primary_decision"] + (["initial_index"] if corpus == "empty" else [])
    assert order == expected_head + ["fts5_dispatch", "observer.schedule", "observer.start", "serve"]


# ---------------------------------------------------------------------------
# fix6 — _format_fts5_results must skip orphan hits (chroma missing chunk_id)
# ---------------------------------------------------------------------------


def test_fix6_format_skips_orphan_chunk_ids() -> None:
    """FTS5 hits pointing to Chroma-missing chunks must be filtered, not padded."""
    from mcp_server.server import KnowledgeOrchestrator

    with patch.object(KnowledgeOrchestrator, "__init__", lambda self: None):
        orch = KnowledgeOrchestrator()
        # Only 'chunk_alive' exists in Chroma; 'chunk_orphan' is FTS5 residue.
        orch.collection = SimpleNamespace(
            get=lambda **kw: {
                "ids": ["chunk_alive"],
                "documents": ["real content"],
                "metadatas": [{"source": "real.md", "filename": "real.md", "category": "redteam"}],
            }
        )
        hits = [("chunk_alive", 10.0), ("chunk_orphan", 5.0)]
        results = orch._format_fts5_results(hits, max_results=5, category_filter=None)

    assert len(results) == 1
    assert results[0]["source"] == "real.md"
    assert all(r["content"] for r in results), "no result may have empty content"


def test_p1_3_fts5_explicit_dispatch_raises_on_reset_race_auto_falls_back(tmp_path, monkeypatch):
    """P1-3: explicit dispatch raises Fts5NotReadyError on the reset race; auto falls back."""
    from mcp_server import server as srv
    from mcp_server.fts5_index import Fts5LexicalIndex, Fts5NotReadyError

    index = Fts5LexicalIndex(db_path=tmp_path / "fts5.db", state_path=tmp_path / "m.state")
    index.rebuild_content_bound([("c1", "alpha token", "f.md", "x")])
    orch = object.__new__(srv.KnowledgeOrchestrator)
    orch.fts5_index, orch.query_router = index, SimpleNamespace(classify=lambda q: "lexical")
    monkeypatch.setattr(srv.config, "fts5_enabled", True)
    monkeypatch.setattr(index, "is_ready", lambda: True)  # stale gate answer -> TOCTOU window
    index.invalidate_generation()  # the reset lands between the gate and the search
    try:
        with pytest.raises(Fts5NotReadyError):
            srv.KnowledgeOrchestrator._maybe_dispatch_fts5(orch, "alpha", 5, None, "fts5")
        result, path = srv.KnowledgeOrchestrator._maybe_dispatch_fts5(orch, "alpha", 5, None, "auto")
        assert result is None and path == "fallback"
    finally:
        index.close()


def test_p1a_fts5_crash_between_swap_and_publication_recovers_on_reopen(tmp_path, monkeypatch):
    """P1-A: candidate live + stale backup + in_progress marker after a simulated
    crash must not wedge same-generation retries: a matching live is blessed and
    cleaned; a changed source rebuilds past the stale backup name."""
    from test_fts5_migration import _fake_collection, _make_index, _worker_shell

    from mcp_server.fts5_index import Fts5LexicalIndex
    from mcp_server.server import KnowledgeOrchestrator

    index = _make_index(tmp_path)
    rows_a = [(f"a{i}", f"ATOK {i}", "f.md", "x") for i in range(3)]
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_a)))
    real_vp = Fts5LexicalIndex.verify_and_publish

    def crash_at_publication(self, source_digest, total, generation, started_at=None):
        raise KeyboardInterrupt  # simulated process loss after the swap commit, before publication

    rows_b = [(f"b{i}", f"BTOK {i}", "f.md", "x") for i in range(3)]
    monkeypatch.setattr(Fts5LexicalIndex, "verify_and_publish", crash_at_publication)
    with pytest.raises(KeyboardInterrupt):
        index.rebuild_content_bound(rows_b, generation=index._generation)  # noqa: SLF001
    monkeypatch.setattr(Fts5LexicalIndex, "verify_and_publish", real_vp)
    index.close()
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5_index.db", state_path=tmp_path / "fts5_migration.state")
    assert index.is_ready() is False
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_b)))
    assert index.is_ready() is True and index.search_if_ready("BTOK 1"), "matching live must be blessed (P1-A)"
    rows_c = [(f"c{i}", f"CTOK {i}", "f.md", "x") for i in range(3)]
    monkeypatch.setattr(Fts5LexicalIndex, "verify_and_publish", crash_at_publication)
    with pytest.raises(KeyboardInterrupt):
        index.rebuild_content_bound(rows_c, generation=index._generation)  # noqa: SLF001
    monkeypatch.setattr(Fts5LexicalIndex, "verify_and_publish", real_vp)
    index.close()
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5_index.db", state_path=tmp_path / "fts5_migration.state")
    rows_d = [(f"d{i}", f"DTOK {i}", "f.md", "x") for i in range(3)]
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows_d)))
    assert index.is_ready() is True and index.search_if_ready("DTOK 1"), "stale backup must not wedge retries (P1-A)"
    with index._fts5_lock:  # noqa: SLF001 — no stale generation artifacts remain after recovery
        leftovers = [r[0] for r in index._conn.execute(  # noqa: SLF001
            "SELECT name FROM sqlite_master WHERE name LIKE 'fts5_documents_backup%' OR name LIKE 'fts5_documents_staging%'").fetchall()]
    assert leftovers == [], leftovers
    index.close()


def test_p1b_fts5_publisher_rejects_live_mutation_during_paged_scan(tmp_path, monkeypatch):
    """P1-B: a same-count live mutation during the paged digest scan or before
    publication must reject publication (mutation epoch)."""
    from test_fts5_migration import _gen_rows, _make_index

    import mcp_server.fts5_index as fts5_module
    from mcp_server.fts5_index import Fts5LexicalIndex, compute_rows_digest

    index = _make_index(tmp_path)
    rows = _gen_rows(5)
    index.rebuild_content_bound(rows)
    source_digest, total = compute_rows_digest(rows)
    index.close()
    index = Fts5LexicalIndex(db_path=tmp_path / "fts5_index.db", state_path=tmp_path / "fts5_migration.state")
    in_scan, resume = threading.Event(), threading.Event()

    def gated_digest(rows_arg, _real=compute_rows_digest):
        in_scan.set()
        assert resume.wait(timeout=10.0)
        return _real(rows_arg)

    monkeypatch.setattr(fts5_module, "compute_rows_digest", gated_digest)
    outcome: list = []
    publisher = threading.Thread(target=lambda: outcome.append(index.verify_and_publish(source_digest, total, 1)))
    publisher.start()
    assert in_scan.wait(timeout=5.0)  # pages read; mutate live now with a same-count update
    index.update_document("chunk_0002", "MUTATED same-count content", "f.md", "security")
    resume.set()
    publisher.join(timeout=10.0)
    assert outcome == [False], "publication must be rejected after a mid-scan live mutation (P1-B)"
    assert index.is_ready() is False, "digest A must never be published over mutated bytes B (P1-B)"
    index.close()


def test_tq1_fts5_verification_read_failure_demotes_ready_generation(tmp_path, monkeypatch):
    """TQ-1: a live read error during an ordinary worker verification must
    publish failed and demote readiness for the current generation."""
    import sqlite3 as _sqlite3

    from test_fts5_migration import _fake_collection, _gen_rows, _make_index, _worker_shell

    from mcp_server.fts5_index import Fts5LexicalIndex
    from mcp_server.server import KnowledgeOrchestrator

    index = _make_index(tmp_path)
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(_gen_rows(3))))
    assert index.is_ready() is True

    def broken_live(self):
        raise _sqlite3.OperationalError("disk I/O error during live read")

    monkeypatch.setattr(Fts5LexicalIndex, "_live_digest", broken_live)
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(_gen_rows(3))))
    assert index.is_ready() is False, "verification read failure must demote readiness (TQ-1)"
    assert index.state.read()["status"] == "failed"
    index.close()


def test_p1c_fts5_material_reset_survives_thread_start_failure(tmp_path, monkeypatch):
    """P1-C: a material reset around a failed Thread.start is retried exactly
    once — never stranded, never a tight retry loop."""
    from test_fts5_migration import _fake_collection, _gen_rows, _make_index, _worker_shell

    from mcp_server import server as srv
    from mcp_server.server import KnowledgeOrchestrator

    index = _make_index(tmp_path)
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(_gen_rows(3))))
    orch = _worker_shell(index, _fake_collection(_gen_rows(3)))
    orch._fts5_startup_dispatch_done = True
    monkeypatch.setattr(srv.config, "fts5_enabled", True)
    attempts = {"n": 0}
    real_start = threading.Thread.start

    def flaky_start(thread_self):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("thread start failed")
        real_start(thread_self)

    monkeypatch.setattr(threading.Thread, "start", flaky_start)
    KnowledgeOrchestrator._fts5_reset_and_rebuild(orch)
    monkeypatch.setattr(threading.Thread, "start", real_start)
    assert attempts["n"] == 2, "exactly one bounded retry — not stranded, no tight loop (P1-C)"
    for worker in [t for t in threading.enumerate() if t.name == "fts5-rebuild"]:
        worker.join(timeout=10.0)
    assert index.is_ready() is True and index.state.read()["generation"] == 2, "material reset must not be lost (P1-C)"
    index.close()


def test_fts5_legacy_start_migration_background_shim_rebuilds_from_zero(tmp_path):
    """API-surface compatibility: the legacy entry keeps its exact signature but
    delegates to content-bound rebuild-from-zero; resume_from is never a cursor."""
    from test_fts5_migration import _gen_rows, _make_index

    index = _make_index(tmp_path, marker_status="in_progress", docs_indexed=40)
    rows = _gen_rows(60)
    thread = index.start_migration_background(lambda: iter(rows), 60, resume_from=40, on_progress=None)
    thread.join(timeout=15.0)
    try:
        assert not thread.is_alive() and index.count() == 60, "rebuild-from-zero; resume_from ignored (shim)"
        assert index.is_ready() is True and index.state.read()["schema_version"] == 2
    finally:
        index.close()


def test_final_v483_package_b_rollout_lifecycle(tmp_path, monkeypatch):
    """Ali final rollout gate — one integration pass over the six correctives:
    (F6) one-shot startup dispatch; (F1) post-swap digest holds no outer lock;
    (F2) failing initial marker write preserves prior credible marker/readiness;
    (F5) publisher live-read failure fails the candidate generation closed;
    (F3) queued material successor start-failure gets one bounded retry;
    (F4) legacy shim delegates to the authoritative path, resume_from is
    compatibility-only and docs_total mismatch fails closed."""
    import mcp_server.fts5_index as fts5_module
    from mcp_server import server as srv
    from mcp_server.fts5_index import Fts5LexicalIndex
    from mcp_server.server import KnowledgeOrchestrator
    from test_fts5_migration import _fake_collection, _gen_rows, _make_index, _worker_shell

    monkeypatch.setattr(srv.config, "fts5_enabled", True)
    index = _make_index(tmp_path)
    rows = _gen_rows(6)
    orch = _worker_shell(index, _fake_collection(rows))

    # (F6) startup dispatch is truly one-shot
    orch._fts5_startup_dispatch_done = False
    dispatched: list = []
    real_start_worker = KnowledgeOrchestrator._start_fts5_rebuild_worker
    monkeypatch.setattr(KnowledgeOrchestrator, "_start_fts5_rebuild_worker",
                        lambda self, material=False: dispatched.append(material))
    KnowledgeOrchestrator._dispatch_fts5_startup_rebuild(orch)
    KnowledgeOrchestrator._dispatch_fts5_startup_rebuild(orch)
    assert dispatched == [False], "startup dispatch must be truly one-shot (F6)"
    monkeypatch.setattr(KnowledgeOrchestrator, "_start_fts5_rebuild_worker", real_start_worker)

    # baseline build through the real worker (authoritative admission + _index_lock)
    KnowledgeOrchestrator._fts5_rebuild_worker(orch)
    assert index.is_ready() is True and index.search_if_ready("CVE-2021-0001")

    # (F1) post-swap digest/publication holds no outer lock: a concurrent search
    # from another thread returns promptly instead of blocking on the hold.
    probe: dict = {}
    real_vp = Fts5LexicalIndex.verify_and_publish

    def probing_vp(self, source_digest, total, generation, started_at=None):
        prober = threading.Thread(target=lambda: probe.setdefault("hit", self.search_if_ready("CVE-2021-0001")),
                                  daemon=True)
        prober.start()
        prober.join(timeout=2.0)
        probe["blocked"] = prober.is_alive()
        return real_vp(self, source_digest, total, generation, started_at)

    monkeypatch.setattr(Fts5LexicalIndex, "verify_and_publish", probing_vp)
    assert index.rebuild_content_bound(rows, generation=index._generation)["status"] == "complete"  # noqa: SLF001
    monkeypatch.setattr(Fts5LexicalIndex, "verify_and_publish", real_vp)
    assert probe["blocked"] is False, "search must return promptly during the post-swap digest (F1)"
    assert index.is_ready() is True

    # (F2) failing initial in_progress write preserves prior credible marker + readiness
    prior_marker = index.state.read()

    def failing_write(payload):
        raise OSError("disk full")

    index.state.write = failing_write
    outcome = index.rebuild_content_bound(rows, generation=index._generation)  # noqa: SLF001
    del index.state.write
    assert outcome["status"] == "failed" and index.state.read() == prior_marker and index.is_ready() is True, \
        "initial-marker failure must preserve the prior credible marker and readiness (F2)"

    # (F5) publisher live-read failure fails the candidate generation closed
    real_live = Fts5LexicalIndex._live_digest

    def broken_live(self):
        raise sqlite3.OperationalError("disk I/O error during live read")

    monkeypatch.setattr(Fts5LexicalIndex, "_live_digest", broken_live)
    KnowledgeOrchestrator._fts5_rebuild_worker(_worker_shell(index, _fake_collection(rows)))
    monkeypatch.setattr(Fts5LexicalIndex, "_live_digest", real_live)
    assert index.is_ready() is False and index.state.read()["status"] == "failed", \
        "publisher live-read failure must leave the generation failed/non-ready (F5)"

    # (F3) queued material successor whose first Thread.start fails retries once
    hold, release_capture = threading.Event(), threading.Event()

    def gated_capture(collection, _real=fts5_module.capture_chunk_rows):
        hold.set()
        assert release_capture.wait(timeout=10.0)
        return _real(collection)

    monkeypatch.setattr("mcp_server.server.capture_chunk_rows", gated_capture)
    orch2 = _worker_shell(index, _fake_collection(rows))
    orch2._fts5_startup_dispatch_done = True
    KnowledgeOrchestrator._start_fts5_rebuild_worker(orch2)
    assert hold.wait(timeout=5.0)
    KnowledgeOrchestrator._fts5_reset_and_rebuild(orch2)  # queue the material successor
    fails = {"n": 0}
    real_start = threading.Thread.start

    def flaky_start(thread_self):
        fails["n"] += 1
        if fails["n"] == 1:
            raise RuntimeError("thread start failed")
        real_start(thread_self)

    monkeypatch.setattr(threading.Thread, "start", flaky_start)
    release_capture.set()
    for worker in [t for t in threading.enumerate() if t.name == "fts5-rebuild"]:
        worker.join(timeout=10.0)
    monkeypatch.setattr(threading.Thread, "start", real_start)
    for worker in [t for t in threading.enumerate() if t.name == "fts5-rebuild"]:
        worker.join(timeout=10.0)
    assert fails["n"] == 2, "queued material successor gets exactly one bounded retry (F3)"
    assert index.is_ready() is True and index.state.read()["generation"] == 2, "successor completed generation 2 (F3)"

    # (F4) legacy shim: authoritative path, resume_from compat-only, docs_total fail-closed
    thread = index.start_migration_background(lambda: iter(rows), 6, resume_from=4)
    thread.join(timeout=15.0)
    assert not thread.is_alive() and index.count() == 6 and index.is_ready() is True, \
        "shim delegates to the content-bound path; resume_from is never a cursor (F4)"
    thread = index.start_migration_background(lambda: iter(rows), 99)
    thread.join(timeout=15.0)
    assert index.is_ready() is False and index.state.read()["status"] == "failed", \
        "docs_total mismatch must fail closed (F4)"
    index.close()
