"""Pass 1 regression tests: reindex/watcher concurrency (VetClub runtime corrective).

Twelve focused behavioral cases: admission race / already_running (through
real production __init__ state) / rollback, reindex_all lock envelope, single
scheduler identity, retry modes under bounded backoff, timing semantics, stop
semantics, recoverable admission, nuclear_rebuild one-writer envelope,
transport lifecycle cleanup, and main() watcher startup/cleanup modes."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional
from unittest.mock import patch

import pytest

from mcp_server import server as srv
from mcp_server.server import DocumentWatcher, KnowledgeOrchestrator


def _index_stats(errors: int = 0) -> Dict[str, Any]:
    return {
        "total_files": 1,
        "indexed": 1,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "errors": errors,
        "chunks_added": 1,
        "chunks_removed": 0,
        "dedup_skipped": 0,
        "categories": {"general": 1},
    }


def _bare_orchestrator() -> KnowledgeOrchestrator:
    """Orchestrator shell with only the reindex-admission state under test."""
    orch = object.__new__(KnowledgeOrchestrator)
    orch._reindex_progress = {"active": False}
    orch._reindex_admission_lock = threading.Lock()
    return orch


def _indexing_shell() -> KnowledgeOrchestrator:
    """Orchestrator shell with only the index-lock state under test."""
    orch = object.__new__(KnowledgeOrchestrator)
    orch._index_lock = threading.RLock()
    orch._index_all_impl = lambda *_args, **_kwargs: _index_stats()
    return orch


def _watcher(index_all: Any, debounce: float) -> DocumentWatcher:
    return DocumentWatcher(lambda: SimpleNamespace(index_all=index_all), debounce_seconds=debounce)


def _wait_until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "condition not reached within timeout"


class _FakeEmbeddings:
    """ChromaDB-interface embedding stub (no model download)."""

    _dim, is_legacy = 384, False

    def __call__(self, input: Any) -> Any:
        return [[0.1] * 384 for _ in input]

    name = staticmethod(lambda: "fake-test-embeddings")
    build_from_config = staticmethod(lambda config: _FakeEmbeddings())
    get_config = staticmethod(lambda: {})
    validate_config_update = staticmethod(lambda old_config, new_config: None)


def _production_orchestrator(tmp_path: Path, monkeypatch: Any) -> KnowledgeOrchestrator:
    """Real KnowledgeOrchestrator() through production __init__: tmp dirs, fake embeddings."""
    for sub in ("documents", "data/chroma_db", "models"):
        (tmp_path / sub).mkdir(parents=True)
    monkeypatch.setattr(srv.config, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(srv.config, "data_dir", tmp_path / "data")
    monkeypatch.setattr(srv.config, "chroma_dir", tmp_path / "data" / "chroma_db")
    monkeypatch.setattr(srv.config, "models_cache_dir", tmp_path / "models")
    monkeypatch.setattr(srv.config, "transport", "stdio")
    with patch("mcp_server.server.FastEmbedEmbeddings", _FakeEmbeddings):
        return KnowledgeOrchestrator()


def test_simultaneous_background_reindex_admits_exactly_one(monkeypatch: Any) -> None:
    """Two racing callers must admit one run; the loser must not clobber progress."""
    orch = _bare_orchestrator()
    index_calls: list = []
    release_run = threading.Event()

    def fake_index_all(force: bool = False) -> Dict[str, Any]:
        # Stays active until both callers returned: the loser sees a live owner.
        index_calls.append(threading.current_thread().name)
        assert release_run.wait(timeout=5.0)
        return _index_stats()

    orch.index_all = fake_index_all
    # Holds both callers inside the check-then-set window; under atomic
    # admission the loser never gets here and the winner's barrier times out.
    barrier = threading.Barrier(2)
    original_fresh = KnowledgeOrchestrator._fresh_reindex_progress

    def racing_fresh(mode: str, resume_state: Any) -> Dict[str, Any]:
        try:
            barrier.wait(timeout=1.0)
        except threading.BrokenBarrierError:
            pass
        return original_fresh(mode, resume_state)

    monkeypatch.setattr(KnowledgeOrchestrator, "_fresh_reindex_progress", staticmethod(racing_fresh))
    statuses: list = []
    threads = [
        threading.Thread(target=lambda: statuses.append(orch.start_reindex_background("incremental"))) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    release_run.set()
    _wait_until(lambda: not orch._reindex_progress.get("active"))
    assert [s["status"] for s in statuses].count("started") == 1
    assert [s["status"] for s in statuses].count("already_running") == 1
    assert len(index_calls) == 1
    assert orch._reindex_progress.get("result") == _index_stats()
    assert orch._reindex_progress.get("error") is None


def test_second_caller_reports_already_running_without_second_run(monkeypatch: Any, tmp_path: Path) -> None:
    """Behavior 2 through REAL production __init__ state: instance _index_lock
    is RLock-capable, _reindex_admission_lock exists and serializes admission."""
    orch = _production_orchestrator(tmp_path, monkeypatch)

    # Same-thread nonblocking reacquire: a plain Lock fails here.
    assert orch._index_lock.acquire(blocking=False)
    reacquired = orch._index_lock.acquire(blocking=False)
    assert reacquired, "production _index_lock is not same-thread re-entrant"
    orch._index_lock.release()
    orch._index_lock.release()

    entered, release = threading.Event(), threading.Event()
    calls: list = []

    def blocking_index_all(force: bool = False) -> Dict[str, Any]:
        calls.append(1)
        entered.set()
        assert release.wait(timeout=5.0)
        return _index_stats()

    orch.index_all = blocking_index_all

    admitted: list = []
    caller = threading.Thread(target=lambda: admitted.append(orch.start_reindex_background("incremental")))
    with orch._reindex_admission_lock:
        caller.start()
        caller.join(timeout=0.3)
        assert caller.is_alive(), "start_reindex_background bypassed the admission lock"
    caller.join(timeout=5.0)
    assert admitted and admitted[0]["status"] == "started"
    assert entered.wait(timeout=5.0)
    owner_progress = orch._reindex_progress
    second = orch.start_reindex_background("incremental")
    assert second["status"] == "already_running"
    assert second["progress"]["operation"] == "incremental"
    assert orch._reindex_progress is owner_progress
    release.set()
    _wait_until(lambda: not orch._reindex_progress.get("active"))
    assert len(calls) == 1


def test_thread_start_failure_rolls_back_admission(monkeypatch: Any) -> None:
    """A failed Thread.start must not leave phantom ownership behind."""
    orch = _bare_orchestrator()
    orch.index_all = lambda force=False: _index_stats()
    real_thread = threading.Thread

    class FailingThread:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread creation failed")

    monkeypatch.setattr(srv.threading, "Thread", FailingThread)
    with pytest.raises(RuntimeError, match="thread creation failed"):
        orch.start_reindex_background("incremental")
    # Rolled back: inactive, with the error recorded like any failed run.
    assert orch._reindex_progress.get("active") is False
    assert orch._reindex_progress.get("error") == "thread creation failed"
    # A subsequent start is not blocked by phantom ownership.
    monkeypatch.setattr(srv.threading, "Thread", real_thread)
    assert orch.start_reindex_background("incremental")["status"] == "started"
    _wait_until(lambda: not orch._reindex_progress.get("active"))
    assert orch._reindex_progress.get("result") == _index_stats()


def test_reindex_all_holds_index_lock_and_leaves_chroma_dirs_alone(monkeypatch: Any, tmp_path: Path) -> None:
    """Busy during BM25/cache postprocessing; Chroma segment dirs untouched."""
    segment = tmp_path / "12345678-1234-1234-1234-123456789abc"
    segment.mkdir()
    monkeypatch.setattr(srv.config, "chroma_dir", tmp_path)
    orch = _indexing_shell()
    orch._bm25_initialized = True
    orch._ensure_bm25_index = lambda: None
    orch.query_cache = SimpleNamespace(invalidate=lambda: None)
    entered, release = threading.Event(), threading.Event()

    class BlockingBM25:
        def clear(self) -> None:
            entered.set()
            assert release.wait(timeout=5.0)

    orch.bm25_index = BlockingBM25()
    results: list = []
    worker = threading.Thread(target=lambda: results.append(orch.reindex_all()))
    worker.start()
    assert entered.wait(timeout=5.0)
    try:
        concurrent = orch.index_all()
    finally:
        release.set()
        worker.join(timeout=5.0)
    assert concurrent.get("skipped_reason") == "reindex_already_running"
    assert not worker.is_alive()
    assert segment.is_dir()
    assert results and results[0]["orphan_folders_cleaned"] == 0


def test_watcher_single_scheduler_identity_and_no_timers(monkeypatch: Any) -> None:
    """One scheduler admitted under racing events; one identity; no Timers."""
    timer_instances: list = []
    real_timer = threading.Timer

    class RecordingTimer(real_timer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            timer_instances.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(srv.threading, "Timer", RecordingTimer)
    callback_threads: list = []
    concurrent, peak = [0], [0]
    in_cycle1, release_cycle1 = threading.Event(), threading.Event()

    def index_all(**_kwargs: Any) -> Dict[str, Any]:
        callback_threads.append(threading.current_thread())
        concurrent[0] += 1
        peak[0] = max(peak[0], concurrent[0])
        try:
            if len(callback_threads) == 1:
                in_cycle1.set()
                assert release_cycle1.wait(timeout=5.0)
            return _index_stats()
        finally:
            concurrent[0] -= 1

    watcher = _watcher(index_all, 0.01)
    try:
        barrier = threading.Barrier(2)

        def send(path: str) -> None:
            barrier.wait(timeout=5.0)
            watcher._schedule_reindex(path)

        senders = [threading.Thread(target=send, args=(f"{i}.md",)) for i in range(2)]
        for t in senders:
            t.start()
        for t in senders:
            t.join(timeout=5.0)
        assert (
            len([t for t in threading.enumerate() if t.name == "knowledge-rag-watcher"]) == 1
        )  # racing events admitted one scheduler
        assert in_cycle1.wait(timeout=5.0)
        # Mid-cycle event: must be served by the SAME scheduler in a successor cycle.
        watcher._schedule_reindex("late.md")
        release_cycle1.set()
        _wait_until(lambda: len(callback_threads) >= 2 and not watcher._pending_paths)
        assert len(callback_threads) == 2
        assert watcher._retry_attempt == 0
        assert peak[0] == 1
        assert timer_instances == []
        assert len({t.ident for t in callback_threads}) == 1
        assert callback_threads[0] is watcher._scheduler
    finally:
        watcher.stop(timeout=5.0)
    assert watcher._scheduler is not None
    assert not watcher._scheduler.is_alive()


@pytest.mark.parametrize("failure", ["already_running", "exception", "errors"])
def test_watcher_retry_modes_requeue_exact_batch_and_reset_after_success(failure: str) -> None:
    """Each failure mode requeues the exact batch; success consumes and resets."""
    entered, release = threading.Event(), threading.Event()
    calls: list = []

    def index_all(**_kwargs: Any) -> Dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=5.0)
            if failure == "already_running":
                return {"skipped_reason": "reindex_already_running"}
            if failure == "exception":
                raise OSError("transient disk I/O error")
            return _index_stats(errors=2)
        return _index_stats()

    watcher = _watcher(index_all, 0.01)
    try:
        # Pre-set backoff so the retry delay (debounce * 2**3) is clearly
        # longer than the debounce a mid-cycle event would establish.
        watcher._retry_attempt = 3
        watcher._schedule_reindex("first.md")
        assert entered.wait(timeout=5.0)
        watcher._schedule_reindex("second.md")
        release.set()
        _wait_until(lambda: watcher._retry_attempt == 4)
        with watcher._cond:
            assert watcher._pending_paths == {"first.md", "second.md"}
            assert watcher._due_at is not None
            remaining = watcher._due_at - time.monotonic()
        assert remaining > watcher._debounce
        _wait_until(lambda: not watcher._pending_paths and watcher._retry_attempt == 0)
        assert len(calls) == 2
    finally:
        watcher.stop(timeout=5.0)


def test_scheduler_timing_semantics() -> None:
    """Due-time stability, empty wake, orphaned pending, capped/floored backoff."""
    assert srv.RETRY_BACKOFF_CAP_SECONDS == 300.0
    # First event establishes the due time; later events only extend the batch.
    idle = _watcher(lambda **_kwargs: _index_stats(), 60.0)
    try:
        idle._schedule_reindex("a.md")
        with idle._cond:
            first_due = idle._due_at
        idle._schedule_reindex("b.md")
        with idle._cond:
            assert first_due is not None
            assert idle._due_at == first_due
            assert idle._pending_paths == {"a.md", "b.md"}
            # 60 * 2**10 is capped at the product bound; attempt 0 gives the debounce.
            idle._retry_attempt = 10
            assert idle._retry_delay_locked() == srv.RETRY_BACKOFF_CAP_SECONDS
            idle._retry_attempt = 0
            assert idle._retry_delay_locked() == 60.0
    finally:
        idle.stop(timeout=5.0)
    # The effective ceiling never undercuts the configured debounce.
    slow = _watcher(lambda **_kwargs: _index_stats(), 600.0)
    with slow._cond:
        slow._retry_attempt = 5
        assert slow._retry_delay_locked() == 600.0
    slow.stop(timeout=5.0)
    # A due wake with nothing pending must not reach index_all.
    empty_calls: list = []
    empty = _watcher(lambda **_kwargs: empty_calls.append(1) or _index_stats(), 60.0)
    try:
        with empty._cond:
            empty._ensure_scheduler_locked()
            empty._due_at = time.monotonic()
            empty._cond.notify_all()
        _wait_until(lambda: empty._due_at is None)
        assert empty_calls == []
    finally:
        empty.stop(timeout=5.0)
    # Paths injected without an event still get a successor cycle on success.
    orphan_calls: list = []
    orphan_box: list = []

    def orphan_index_all(**_kwargs: Any) -> Dict[str, Any]:
        orphan_calls.append(1)
        if len(orphan_calls) == 1:
            orphan_box[0]._pending_paths.add("late.md")
        return _index_stats()

    orphan = _watcher(orphan_index_all, 0.01)
    orphan_box.append(orphan)
    try:
        orphan._schedule_reindex("veterinary.md")
        _wait_until(lambda: len(orphan_calls) == 2 and not orphan._pending_paths)
        assert orphan._retry_attempt == 0
    finally:
        orphan.stop(timeout=5.0)


def test_stop_is_bounded_idempotent_terminal_and_safe_from_scheduler_thread() -> None:
    """stop(): bounded join, idempotent, terminal, self-join-safe."""
    calls: list = []
    watcher = _watcher(lambda **_kwargs: calls.append(1) or _index_stats(), 60.0)
    watcher._schedule_reindex("before.md")
    with watcher._cond:
        scheduler = watcher._scheduler
    assert scheduler is not None and scheduler.is_alive()
    watcher.stop(timeout=5.0)
    assert not scheduler.is_alive()
    watcher.stop(timeout=5.0)  # idempotent
    watcher._schedule_reindex("after-stop.md")  # terminal: ignored
    assert watcher._pending_paths == {"before.md"}
    assert watcher._scheduler is scheduler
    assert calls == []
    # Orchestrator callbacks run on the scheduler thread; stop() from there
    # must not self-join.
    stopped_inside = threading.Event()
    inner_box: list = []

    def self_stopping_index_all(**_kwargs: Any) -> Dict[str, Any]:
        inner_box[0].stop(timeout=5.0)  # a self-join would raise RuntimeError
        stopped_inside.set()
        return _index_stats()

    inner = _watcher(self_stopping_index_all, 0.01)
    inner_box.append(inner)
    inner._schedule_reindex("a.md")
    assert stopped_inside.wait(timeout=5.0)
    assert inner._scheduler is not None
    _wait_until(lambda: not inner._scheduler.is_alive())
    inner._schedule_reindex("late.md")
    assert "late.md" not in inner._pending_paths


def test_watchdog_dispatch_survives_scheduler_start_failure(monkeypatch: Any, tmp_path: Path) -> None:
    """Start failure must not escape on_created; the next event recovers the batch."""
    from watchdog.events import FileCreatedEvent

    calls: list = []
    watcher = _watcher(lambda **_kwargs: calls.append(1) or _index_stats(), 0.01)
    real_start = threading.Thread.start

    def failing_start(self: threading.Thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    first = str(tmp_path / "first.md")
    watcher.on_created(FileCreatedEvent(first))  # public boundary: must not raise
    with watcher._cond:
        assert watcher._scheduler is None
        assert watcher._pending_paths == {first}
        assert watcher._due_at is not None
    monkeypatch.setattr(threading.Thread, "start", real_start)
    try:
        watcher.on_created(FileCreatedEvent(str(tmp_path / "second.md")))
        _wait_until(lambda: bool(calls) and not watcher._pending_paths)
        assert len(calls) == 1  # one cycle consumed both preserved paths
        assert len([t for t in threading.enumerate() if t.name == "knowledge-rag-watcher"]) == 1
    finally:
        watcher.stop(timeout=5.0)
    assert not watcher._scheduler.is_alive()


@pytest.mark.parametrize("swap", [True, False])
def test_nuclear_rebuild_holds_writer_lock_for_complete_lifecycle(swap: bool) -> None:
    """Fails if the lock is released before the mode's rebuild callback returns."""
    orch = _indexing_shell()
    entered, release = threading.Event(), threading.Event()
    results: list = []
    wrong_mode: list = []

    def blocking_rebuild() -> Dict[str, Any]:
        entered.set()
        assert release.wait(timeout=5.0)
        return _index_stats()

    if swap:
        orch._rebuild_via_swap = blocking_rebuild
        orch._rebuild_destructive = lambda: wrong_mode.append("destructive")
    else:
        orch._rebuild_destructive = blocking_rebuild
        orch._rebuild_via_swap = lambda: wrong_mode.append("swap")
    worker = threading.Thread(target=lambda: results.append(orch.nuclear_rebuild(swap=swap)))
    worker.start()
    assert entered.wait(timeout=5.0)
    try:
        concurrent = orch.index_all()  # mid-callback: the writer lock must be held
    finally:
        release.set()
        worker.join(timeout=5.0)
    assert concurrent.get("skipped_reason") == "reindex_already_running"
    assert results and results[0] == _index_stats()
    assert wrong_mode == []
    assert not worker.is_alive()
    # Foreign lock owner: the same mode returns busy and mutates nothing.
    mutations: list = []
    orch._rebuild_via_swap = lambda: mutations.append("swap") or _index_stats()
    orch._rebuild_destructive = lambda: mutations.append("destructive") or _index_stats()
    owned, release_owner = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with orch._index_lock:
            owned.set()
            assert release_owner.wait(timeout=5.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert owned.wait(timeout=5.0)
    try:
        busy = orch.nuclear_rebuild(swap=swap)
    finally:
        release_owner.set()
        holder.join(timeout=5.0)
    assert busy.get("skipped_reason") == "reindex_already_running"
    assert mutations == []


class _FakeObserver:
    def __init__(
        self,
        log: list,
        start_error: Optional[BaseException] = None,
        stop_error: Optional[BaseException] = None,
        alive_after_join: bool = False,
    ) -> None:
        self._log = log
        self._start_error = start_error
        self._stop_error = stop_error
        self._alive = alive_after_join
        self.daemon = False

    def schedule(self, *_args: Any, **_kwargs: Any) -> None:
        self._log.append("observer.schedule")

    def start(self) -> None:
        self._log.append("observer.start")
        if self._start_error is not None:
            raise self._start_error

    def stop(self) -> None:
        self._log.append("observer.stop")
        if self._stop_error is not None:
            raise self._stop_error

    def join(self, timeout: Any = None) -> None:
        self._log.append(("observer.join", timeout))

    def is_alive(self) -> bool:
        return self._alive


class _FakeWatcher:
    def __init__(self, log: list, scheduler: Any = None) -> None:
        self._log = log
        self._scheduler = scheduler

    def stop(self, timeout: Any = None) -> None:
        self._log.append(("watcher.stop", timeout))


_BOUND = srv.WATCHER_SHUTDOWN_TIMEOUT_SECONDS
_CLEANUP = ["observer.stop", ("observer.join", _BOUND), ("watcher.stop", _BOUND)]


@pytest.mark.parametrize("outcome", ["return", "raise"])
def test_lifecycle_cleanup_order_and_original_exception_authority(monkeypatch: Any, capsys: Any, outcome: str) -> None:
    """Fixed cleanup order; injected cleanup failure masks nothing, blocks nothing."""
    log: list = []
    if outcome == "return":
        monkeypatch.setattr(srv, "_run_transport", lambda transport: log.append(("transport", transport)))
        srv._serve_with_lifecycle("stdio", _FakeObserver(log, alive_after_join=True), _FakeWatcher(log))
        assert log == [("transport", "stdio"), *_CLEANUP]
        assert "observer still alive past shutdown bound" in capsys.readouterr().err
        return

    def boom(transport: str) -> None:
        raise RuntimeError("transport crashed")

    monkeypatch.setattr(srv, "_run_transport", boom)
    observer = _FakeObserver(log, stop_error=RuntimeError("observer stop failed"))
    stuck_scheduler = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="transport crashed"):
        srv._serve_with_lifecycle("sse", observer, _FakeWatcher(log, scheduler=stuck_scheduler))
    # Injected observer.stop failure: bounded join and watcher stop still run
    # as independent attempts; the transport's exception stays authoritative.
    assert log == _CLEANUP
    err = capsys.readouterr().err
    assert "observer stop failed" in err
    assert "scheduler still alive past shutdown bound" in err


@contextmanager
def _main_invocation() -> Iterator[None]:
    old_argv, old_stdout = sys.argv, sys.stdout
    try:
        sys.argv = ["knowledge-rag"]
        yield
    finally:
        sys.argv, sys.stdout = old_argv, old_stdout


@pytest.mark.parametrize(
    "mode", ["disabled", "enabled", "startup_exception", "startup_baseexception", "metrics_failure"]
)
def test_main_watcher_startup_modes(monkeypatch: Any, mode: str) -> None:
    """Exact objects reach the wrapper; every startup failure cleans immediately."""
    import mcp_server as package
    import mcp_server.instance_lock as instance_lock
    import mcp_server.metrics as metrics
    import mcp_server.preflight as preflight

    log: list = []
    created: Dict[str, list] = {"watcher": [], "observer": []}
    start_error = {
        "startup_exception": RuntimeError("observer start failed"),
        "startup_baseexception": KeyboardInterrupt(),
    }.get(mode)
    stop_error = RuntimeError("observer stop failed") if mode == "startup_baseexception" else None

    def make_watcher(*_args: Any, **_kwargs: Any) -> _FakeWatcher:
        watcher = _FakeWatcher(log)
        created["watcher"].append(watcher)
        return watcher

    def make_observer() -> _FakeObserver:
        observer = _FakeObserver(log, start_error=start_error, stop_error=stop_error)
        created["observer"].append(observer)
        return observer

    called: Dict[str, Any] = {}

    def record_serve(transport: Any, observer: Any, watcher: Any) -> None:
        called.update(transport=transport, observer=observer, watcher=watcher)

    orch = SimpleNamespace(collection=SimpleNamespace(count=lambda: 1), _check_dimension_mismatch=lambda: False)
    monkeypatch.setattr(instance_lock, "single_instance_lock", nullcontext)
    monkeypatch.setattr(preflight, "run_preflight", lambda: None)
    monkeypatch.setattr(srv, "get_orchestrator", lambda: orch)
    monkeypatch.setattr(srv, "DocumentWatcher", make_watcher)
    monkeypatch.setattr(srv, "Observer", make_observer)
    monkeypatch.setattr(srv, "_serve_with_lifecycle", record_serve)
    monkeypatch.setattr(package, "_original_stdout", sys.stdout)
    monkeypatch.setattr(srv.config, "transport", "sse" if mode == "metrics_failure" else "stdio")
    monkeypatch.setattr(srv.config, "metrics_enabled", mode == "metrics_failure")
    if mode == "disabled":
        monkeypatch.setenv("KNOWLEDGE_RAG_WATCHER_DISABLED", "1")
    else:
        monkeypatch.delenv("KNOWLEDGE_RAG_WATCHER_DISABLED", raising=False)
    if mode == "metrics_failure":

        def failing_metrics(_port: Any) -> None:
            raise RuntimeError("metrics startup failed")

        monkeypatch.setattr(metrics, "start_metrics_server", failing_metrics)
    with _main_invocation():
        if mode == "startup_baseexception":
            with pytest.raises(KeyboardInterrupt):
                srv.main()
        elif mode == "metrics_failure":
            with pytest.raises(RuntimeError, match="metrics startup failed"):
                srv.main()
        else:
            srv.main()
    if mode == "disabled":
        assert called == {"transport": "stdio", "observer": None, "watcher": None}
        assert log == []
        assert created == {"watcher": [], "observer": []}
    elif mode == "enabled":
        # The exact created objects reach the lifecycle wrapper, unstopped.
        assert created["watcher"] and created["observer"]
        assert called["transport"] == "stdio"
        assert called["observer"] is created["observer"][0]
        assert called["watcher"] is created["watcher"][0]
        assert log == ["observer.schedule", "observer.start"]
    elif mode == "startup_exception":
        # Immediate cleanup of the partial stack; the server continues disabled.
        assert log == ["observer.schedule", "observer.start", *_CLEANUP]
        assert called == {"transport": "stdio", "observer": None, "watcher": None}
    elif mode == "startup_baseexception":
        # Stop failed, join+watcher-stop still ran, original BaseException wins.
        assert log == ["observer.schedule", "observer.start", *_CLEANUP]
        assert called == {}
    else:  # metrics_failure: after a clean start, pre-transport failure
        assert log == ["observer.schedule", "observer.start", *_CLEANUP]
        assert called == {}


# ── v4.9.0: typed watcher config wiring ────────────────────────────────────


def test_watcher_uses_config_fields_not_hardcoded_debounce(monkeypatch: Any) -> None:
    """main() constructs DocumentWatcher from config fields (v4.9.0)."""
    import mcp_server as package
    import mcp_server.instance_lock as instance_lock
    import mcp_server.preflight as preflight

    captured: Dict[str, Any] = {}

    def capture_watcher(_get_orch, **kwargs):
        captured.update(kwargs)
        return _FakeWatcher([])

    orch = SimpleNamespace(collection=SimpleNamespace(count=lambda: 1), _check_dimension_mismatch=lambda: False)
    monkeypatch.setattr(instance_lock, "single_instance_lock", nullcontext)
    monkeypatch.setattr(preflight, "run_preflight", lambda: None)
    monkeypatch.setattr(srv, "get_orchestrator", lambda: orch)
    monkeypatch.setattr(srv, "DocumentWatcher", capture_watcher)
    monkeypatch.setattr(srv, "Observer", lambda: _FakeObserver([]))
    monkeypatch.setattr(srv, "_serve_with_lifecycle", lambda transport, observer, watcher: None)
    monkeypatch.setattr(package, "_original_stdout", sys.stdout)
    monkeypatch.setattr(srv.config, "transport", "stdio")
    monkeypatch.setattr(srv.config, "metrics_enabled", False)
    monkeypatch.delenv("KNOWLEDGE_RAG_WATCHER_DISABLED", raising=False)
    monkeypatch.setattr(srv.config, "watch_for_changes", True)
    monkeypatch.setattr(srv.config, "watch_debounce_seconds", 2.5)
    with _main_invocation():
        srv.main()
    assert captured == {"debounce_seconds": 2.5}


def test_watcher_disabled_by_watch_for_changes_false(monkeypatch: Any, capsys: Any) -> None:
    """advanced.watch_for_changes=false skips construction entirely."""
    import mcp_server as package
    import mcp_server.instance_lock as instance_lock
    import mcp_server.preflight as preflight

    created: list = []

    monkeypatch.setattr(instance_lock, "single_instance_lock", nullcontext)
    monkeypatch.setattr(preflight, "run_preflight", lambda: None)
    monkeypatch.setattr(
        srv,
        "get_orchestrator",
        lambda: SimpleNamespace(collection=SimpleNamespace(count=lambda: 1), _check_dimension_mismatch=lambda: False),
    )
    monkeypatch.setattr(
        srv,
        "DocumentWatcher",
        lambda *a, **k: created.append(k) or _FakeWatcher([]),
    )
    monkeypatch.setattr(srv, "Observer", lambda: _FakeObserver([]))
    monkeypatch.setattr(srv, "_serve_with_lifecycle", lambda transport, observer, watcher: None)
    monkeypatch.setattr(package, "_original_stdout", sys.stdout)
    monkeypatch.setattr(srv.config, "transport", "stdio")
    monkeypatch.setattr(srv.config, "metrics_enabled", False)
    monkeypatch.delenv("KNOWLEDGE_RAG_WATCHER_DISABLED", raising=False)
    monkeypatch.setattr(srv.config, "watch_for_changes", False)
    with _main_invocation():
        srv.main()
    assert created == []
    # Package init redirects stdout->stderr only in the normal MCP context;
    # under pytest either stream may carry the message — check both.
    captured = capsys.readouterr()
    assert "advanced.watch_for_changes=false" in (captured.out + captured.err)


# ── v4.9.0 advanced watcher fields (isolated from the live config singleton) ─


def _isolated_config(monkeypatch: Any, tmp_path: Path, yaml_payload: Any) -> Any:
    """Construct a Config against a CONTROLLED YAML payload + tmp BASE_DIR.

    The module-level ``_yaml``/``BASE_DIR`` are monkeypatched so nothing is
    read from (or created under) the operator's live VetClub config. A live
    configured ``debounce=5.0`` is perfectly valid — these tests never
    assume the singleton's values.
    """
    import mcp_server.config as config_module

    monkeypatch.setattr(config_module, "_yaml", yaml_payload, raising=False)
    monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
    cfg = config_module.Config()
    monkeypatch.setattr(config_module, "_yaml", {}, raising=False)
    return cfg


def test_watch_fields_defaults_isolated(monkeypatch: Any, tmp_path: Path) -> None:
    """No YAML at all -> documented defaults: watcher on, debounce 10.0."""
    cfg = _isolated_config(monkeypatch, tmp_path, {})
    assert cfg.watch_for_changes is True
    assert cfg.watch_debounce_seconds == 10.0
    assert isinstance(cfg.watch_debounce_seconds, float)


def test_watch_fields_read_from_yaml(monkeypatch: Any, tmp_path: Path) -> None:
    """advanced.* is honored; a configured debounce of 5 (int) stays 5.0."""
    cfg = _isolated_config(
        monkeypatch,
        tmp_path,
        {"advanced": {"watch_for_changes": False, "watch_debounce_seconds": 5}},
    )
    assert cfg.watch_for_changes is False
    assert cfg.watch_debounce_seconds == 5.0


def test_watch_fields_validation_rejects_invalid_debounce(capsys: Any) -> None:
    """Non-finite / non-positive / wrong-typed debounce falls back to 10.0."""
    from mcp_server.config import Config

    for bad in (0, -1.5, float("inf"), float("nan"), "10", True, None):
        cfg = Config.__new__(Config)
        setattr(cfg, "watch_for_changes", True)
        setattr(cfg, "watch_debounce_seconds", bad)
        Config._validate_advanced(cfg)
        assert cfg.watch_debounce_seconds == 10.0, f"debounce {bad!r} must fall back"
    captured = capsys.readouterr()
    assert "invalid" in (captured.out + captured.err)


def test_watch_fields_validation_accepts_finite_positive() -> None:
    """Finite positive debounce is coerced to float and kept; valid bool untouched."""
    from mcp_server.config import Config

    for good in (0.5, 5, 30.0):
        cfg = Config.__new__(Config)
        setattr(cfg, "watch_for_changes", False)
        setattr(cfg, "watch_debounce_seconds", good)
        Config._validate_advanced(cfg)
        assert cfg.watch_debounce_seconds == float(good)
        assert cfg.watch_for_changes is False


def test_watch_for_changes_wrong_type_falls_back_to_true(capsys: Any) -> None:
    """Non-bool watch_for_changes resets to True (watcher on by default)."""
    from mcp_server.config import Config

    cfg = Config.__new__(Config)
    setattr(cfg, "watch_for_changes", "no")
    setattr(cfg, "watch_debounce_seconds", 10.0)
    Config._validate_advanced(cfg)
    assert cfg.watch_for_changes is True
    captured = capsys.readouterr()
    assert "watch_for_changes" in (captured.out + captured.err)


# ── v4.9.0 release/runtime reproducibility — static contract assertions ─────
# These read ONLY committed files (no builds, no network, no execution of
# workflows): they pin the shape of the reproducibility contract so an
# accidental edit anywhere in the chain fails a focused test.

_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


class TestPythonCapabilityContract:
    def test_requires_python_capped_below_3_14_with_3_13_classifier(self):
        raw = _read("pyproject.toml")
        assert 'requires-python = ">=3.11,<3.14"' in raw
        assert '"Programming Language :: Python :: 3.13"' in raw
        assert '"Programming Language :: Python :: 3.14"' not in raw

    def test_packaging_is_a_direct_runtime_dependency(self):
        raw = _read("pyproject.toml")
        assert '"packaging>=24.0",' in raw
        # and in the contribution list
        assert "packaging>=24.0" in _read("requirements.txt")


class TestInstallerContract:
    def test_install_py_uses_lock_require_hashes_then_project_no_deps(self):
        raw = _read("install.py")
        assert 'RUNTIME_LOCK_FILE = "requirements.lock"' in raw
        # Reproducible source-install contract: both hash locks, temporary
        # builder venv, --no-isolation wheel build, exact wheel --no-deps.
        # (Finer detail lives in the comprehensive reproducibility test.)
        assert '"--require-hashes", "-r", str(runtime_lock)' in raw
        assert '"--no-isolation"' in raw
        assert '"--no-deps", str(wheel_path)' in raw
        # The ranged requirements.txt must never be a production input.
        assert '"-r", str(req)' not in raw
        assert '"--no-deps", str(install_path)' not in raw
        # Supported Pythons capped at 3.13.
        assert 'SUPPORTED_PY = {"3.11", "3.12", "3.13"}' in raw


class TestDockerContract:
    def test_dockerfile_consumes_lock_and_wheel_only(self):
        raw = _read("Dockerfile")
        assert "COPY requirements.lock ./requirements.lock" in raw
        assert "COPY dist/*.whl ./dist/" in raw
        assert "--require-hashes" in raw
        assert "--no-deps dist/*.whl" in raw
        assert "pip check" in raw
        # The source tree must NOT be copied over the installed package.
        assert "COPY mcp_server/ ./mcp_server/" not in raw
        assert "COPY presets/ ./presets/" not in raw
        # No generic model pre-download: exact model artifacts are
        # deployment inputs verified by the runtime receipt.
        assert "TextEmbedding(" not in raw
        assert "TextCrossEncoder(" not in raw
        assert "deployment inputs" in raw.lower()


class TestLockContract:
    def test_marker_aware_universal_lock_contract(self):
        # The strict parser + graph parity live in generations.py and are
        # behaviorally tested in test_generation_integration.py; here we
        # pin that the load-bearing library is imported and the
        # environment seam defaults extra to "".
        raw = _read("mcp_server/generations.py")
        assert "import packaging.requirements" in raw
        assert "import packaging.markers" in raw
        assert "default_environment" in raw
        assert 'env["extra"] = ""' in raw

    def test_build_lock_input_declared_and_sdist_carries_it(self):
        build_in = _read("build-requirements.in")
        assert "build" in build_in
        assert "hatchling" in build_in
        # Only build + hatchling (comments excluded).
        substantive = [
            line.strip() for line in build_in.splitlines() if line.strip() and not line.strip().startswith("#")
        ]
        # pip is hash-pinned so builder venvs end with an exact pip (no
        # floating `pip install --upgrade pip` in the build path).
        assert sorted(substantive) == ["build", "hatchling", "pip"]
        raw = _read("pyproject.toml")
        assert '"build-requirements.in"' in raw
        assert '"build-requirements.lock"' in raw

    def test_test_tooling_is_hash_locked_without_automatic_reruns(self):
        test_in = _read("test-requirements.in")
        substantive = [
            line.strip()
            for line in test_in.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for tool in ("pytest", "pytest-cov", "hypothesis", "psutil"):
            assert any(line == tool or line.startswith(tool) for line in substantive), (
                f"{tool} missing from test-requirements.in"
            )
        # NO automatic reruns: rerunfailures is absent from the SUBSTANTIVE
        # package lines (comments may mention it in prose) and CI never
        # passes --reruns (a rerun is controller-authorized only after a
        # confirmed infra-flake or byte change).
        assert not any("rerunfailures" in line for line in substantive)
        assert "--reruns" not in _read(".github/workflows/ci.yml")


class TestWorkflowContract:
    """Exactly one PR-head full suite; reproducible build; docker artifact."""

    def test_ci_has_one_full_suite_lane_on_pr_head_only(self):
        raw = _read(".github/workflows/ci.yml")
        # The one full-suite lane: pull_request EVENT only (github.sha on a
        # PR is the synthetic merge SHA and must not be compared), with the
        # EXACT head bytes checked out explicitly.
        assert "full-test:" in raw
        assert "if: github.event_name == 'pull_request'" in raw
        full_test = raw.split("full-test:")[1]
        assert "github.event.pull_request.head.sha == github.sha" not in full_test
        assert "ref: ${{ github.event.pull_request.head.sha }}" in full_test
        assert "persist-credentials: false" in full_test
        # Matrix lane is a lock/install/import smoke — NO pytest in it.
        assert "lock-smoke:" in raw
        lock_smoke = raw.split("lock-smoke:")[1].split("full-test:")[0]
        assert "pytest tests/" not in lock_smoke
        assert "SMOKE_OK" in lock_smoke
        # Full-suite lane installs BOTH locks with --require-hashes.
        assert "--require-hashes -r requirements.lock" in full_test
        assert "--require-hashes -r test-requirements.lock" in full_test
        # 3 OS x 3 Python matrix retained.
        assert "ubuntu-latest" in raw
        assert "windows-latest" in raw
        assert "macos-latest" in raw
        assert '"3.11"' in raw and '"3.12"' in raw and '"3.13"' in raw

    def test_ci_smoke_is_windows_portable(self):
        raw = _read(".github/workflows/ci.yml")
        smoke = raw.split("lock-smoke:")[1].split("full-test:")[0]
        # No Unix-only venv paths in the matrix lane.
        assert "/tmp/" not in smoke
        assert "shell: python" in smoke
        # Cross-platform bin/Scripts computation, not hard-coded paths.
        assert '"Scripts" if os.name == "nt" else "bin"' in smoke
        assert '"python.exe" if os.name == "nt" else "python"' in smoke
        # Build uses the build lock + --no-isolation (no floating tools).
        assert '"-r", "build-requirements.lock"' in smoke
        assert '"--no-isolation"' in smoke
        # No floating pip upgrade anywhere in CI.
        assert "--upgrade" not in raw.replace("--upgrade_deps", "")

    def test_release_does_not_rerun_full_suite_and_builds_reproducibly(self):
        raw = _read(".github/workflows/release.yml")
        assert "pytest" not in raw  # release verifies the wheel, never reruns CI
        assert '"-r", "build-requirements.lock"' not in raw  # shell form used:
        assert "--require-hashes -r build-requirements.lock" in raw
        assert "python -m build --no-isolation" in raw
        # No floating pip upgrade in the release path either.
        assert "pip install --upgrade" not in raw
        # Version identity gate retained.
        assert "verify-versions" in raw
        # Exact-wheel verification in a clean venv: EXACTLY ONE wheel,
        # never first-vs-last selection.
        assert "verify-wheel" in raw
        assert "release-verify-venv" in raw
        verify = raw.split("verify-wheel:")[1].split("publish-npm:")[0]
        assert "assert len(wheels) == 1" in verify

    def test_publish_jobs_wait_for_wheel_verification(self):
        raw = _read(".github/workflows/release.yml")
        pypi = raw.split("publish-pypi:")[1].split("verify-wheel:")[0]
        docker = raw.split("publish-docker:")[1].split("yank-superseded:")[0]
        assert "needs: [build, verify-wheel]" in pypi
        assert "needs: [build, verify-wheel]" in docker

    def test_release_docker_uses_exact_artifact_and_owner_namespace(self):
        raw = _read(".github/workflows/release.yml")
        docker = raw.split("publish-docker:")[1].split("yank-superseded:")[0]
        # Downloads the EXACT python-dist artifact before docker build.
        assert "name: python-dist" in docker
        assert "path: dist/" in docker
        # Actual repository owner namespace, not a hard-coded owner.
        assert "ghcr.io/${{ github.repository_owner }}/knowledge-rag" in docker
        # build-push action has an id so the deployment receipt can bind the
        # pushed IMAGE DIGEST; the receipt binds digest + exact wheel SHA
        # (read from the downloaded release-verification receipt) and fails
        # when any value is missing.
        assert "id: docker_push" in docker
        assert "name: release-evidence" in docker
        assert "steps.docker_push.outputs.digest" in docker
        assert "docker-deployment" in docker
        assert "image_digest" in docker
        assert "wheel_sha256" in docker

    def test_release_receipt_binds_git_tree_evidence(self):
        script = _read("scripts/release_receipt.py")
        # Source bytes come from the EXACT GITHUB_SHA Git tree via read-only
        # plumbing — never the mutable worktree.
        for token in (
            "GITHUB_SHA",
            "ls-tree",
            'f"{commit}:{path}"',
            "cat-file",
            "source_of_truth",
            "git-tree",
        ):
            assert token in script, f"release receipt lacks git-tree plumbing {token!r}"
        # Exactly-one-wheel gate (never first-vs-last).
        assert "len(wheels) != 1" in script
        # Tree-bound evidence terminology, not mutable-source terms.
        for token in (
            "wheel_sha256",
            "tree_sha256",
            "embedded_lock_matches_tree",
            "wheel package-file set mismatch vs git tree",
            "drifted from git tree",
        ):
            assert token in script, f"release receipt lacks tree evidence {token!r}"
        assert "source_sha256" not in script  # superseded by tree_sha256

    def test_source_installer_is_reproducible(self):
        raw = _read("install.py")
        # Both hash locks located with the same fail-closed source-root
        # behavior and installed with --require-hashes.
        assert 'RUNTIME_LOCK_FILE = "requirements.lock"' in raw
        assert 'BUILD_LOCK_FILE = "build-requirements.lock"' in raw
        assert "_locate(RUNTIME_LOCK_FILE)" in raw
        assert "_locate(BUILD_LOCK_FILE)" in raw
        assert '"--require-hashes", "-r", str(runtime_lock)' in raw
        assert '"--require-hashes", "-r", str(build_lock)' in raw
        # One temporary builder venv, exact wheel, --no-isolation.
        assert "upgrade_deps=False" in raw
        assert '"--no-isolation"' in raw
        assert "venv_python_path(builder_dir)" in raw
        assert "tempfile.TemporaryDirectory" in raw
        assert "len(wheels) != 1" in raw
        assert '"--no-deps", str(wheel_path)' in raw
        assert '"-m", "pip", "check"' in raw
        # No floating pip upgrade survives anywhere in the installer.
        assert '"--upgrade", "pip"' not in raw
        assert '"pip", "install", "--upgrade"' not in raw
        # PyPI branch warns it is NOT the admitted reproducible route.
        assert "NOT the admitted reproducible" in raw


class TestPackageDataIdentity:
    """Bundled package data must be byte-identical to canonical sources."""

    _PAIRS = {
        "config.example.yaml": "mcp_server/data/config.example.yaml",
        "presets/cybersecurity.yaml": "mcp_server/data/cybersecurity.yaml",
        "presets/developer.yaml": "mcp_server/data/developer.yaml",
        "presets/research.yaml": "mcp_server/data/research.yaml",
        "presets/general.yaml": "mcp_server/data/general.yaml",
        "presets/multilingual.yaml": "mcp_server/data/multilingual.yaml",
    }

    def test_all_five_presets_and_config_are_byte_identical(self):
        for source, bundled in self._PAIRS.items():
            src = _ROOT / source
            dst = _ROOT / bundled
            assert src.is_file(), f"canonical source missing: {source}"
            assert dst.is_file(), f"bundled package-data missing: {bundled}"
            assert dst.read_bytes() == src.read_bytes(), (
                f"package-data drift: {bundled} != {source} — source and "
                "wheel behavior diverged; re-sync mcp_server/data/"
            )
