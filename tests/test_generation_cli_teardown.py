"""Two-sided tests for the deterministic chroma teardown squeeze.

Criteria: kr-teardown-fix (docs/criteria-kr-teardown-fix.md, Критерий 2).

The squeeze is a child-scope function in ``mcp_server.generation_cli``:
``_teardown_chroma_sqlite_sidecars``. It runs in the population child AFTER
``orch.close(strict=True)`` and BEFORE the sidecar sweep, and must never
weaken that sweep: if the sidecars survive, the squeeze yields and the
sweep honestly fails the build with exit 14.

- GREEN: a WAL-mode sqlite db with committed rows, all connections closed,
  sidecars on disk -> the squeeze removes -wal/-shm.
- RED: a second connection is held open -> the squeeze does NOT remove the
  sidecars and reports them (the sweep would fail the build).

On the base commit (d3782c4) the function does not exist, so this module
fails at import — the test knows how to be red.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from mcp_server import generation_cli as gcli


def _make_wal_db_with_sidecars(root: Path) -> Path:
    """Create a WAL-mode sqlite db with committed rows and live sidecars.

    The fixture mirrors the production state ("sidecars on disk, db
    consistent, no writers"): a subprocess opens the WAL db, commits rows
    and dies via ``os._exit(0)`` WITHOUT closing the connection — the
    -wal/-shm files survive on disk while no connection remains. (On
    CPython/macOS closing the LAST connection checkpoints and deletes the
    sidecars, so an in-process create-then-close fixture cannot ever
    satisfy the GREEN precondition.)
    """
    db_dir = root / "chroma_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "chroma.sqlite3"

    code = textwrap.dedent(
        """
        import os
        import sqlite3

        conn = sqlite3.connect(os.environ["KR_TEARDOWN_TEST_DB"])
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, document TEXT)")
        conn.executemany(
            "INSERT INTO chunks (id, document) VALUES (?, ?)",
            [("doc-" + str(i), "row payload " + str(i)) for i in range(200)],
        )
        conn.commit()
        os._exit(0)
        """
    )
    env = dict(os.environ, KR_TEARDOWN_TEST_DB=str(db_path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    return db_path


def _sidecars(root: Path) -> list:
    return [p.name for p in (root / "chroma_db").iterdir() if p.name.endswith(("-wal", "-shm"))]


def test_squeeze_removes_sidecars_when_all_connections_closed(tmp_path: Path) -> None:
    """GREEN: closed connections -> the squeeze clears -wal/-shm."""
    db_path = _make_wal_db_with_sidecars(tmp_path)
    assert db_path.exists()

    # Wait briefly: sqlite may checkpoint sidecars away on its own; only
    # run the squeeze while the sidecars are actually present on disk.
    deadline = time.monotonic() + 2.0
    while not _sidecars(tmp_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _sidecars(tmp_path), (
        "precondition failed: WAL sidecars did not survive close — the GREEN case cannot be exercised on this platform"
    )

    result = gcli._teardown_chroma_sqlite_sidecars(tmp_path)

    assert result["cleared"] is True
    assert result["remaining"] == []
    assert _sidecars(tmp_path) == []


def test_squeeze_yields_and_reports_when_foreign_connection_open(tmp_path: Path) -> None:
    """RED: an open foreign connection -> the squeeze yields with a report."""
    db_path = _make_wal_db_with_sidecars(tmp_path)
    assert db_path.exists()

    holder = sqlite3.connect(str(db_path))
    try:
        # Keep a read transaction open so the TRUNCATE checkpoint stays
        # busy and the sidecars cannot be removed.
        holder.execute("BEGIN")
        holder.execute("SELECT COUNT(*) FROM chunks").fetchall()

        deadline = time.monotonic() + 2.0
        while not _sidecars(tmp_path) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _sidecars(tmp_path), (
            "precondition failed: WAL sidecars did not survive close — "
            "the RED case cannot be exercised on this platform"
        )

        # Shrink the timeout: the squeeze must yield within ~1s, not burn
        # the full 5s production ceiling.
        started = time.monotonic()
        result = gcli._teardown_chroma_sqlite_sidecars(tmp_path, timeout=1.0)
        waited = time.monotonic() - started

        assert result["cleared"] is False
        assert result["remaining"], "the squeeze must name the surviving sidecars"
        assert waited < 3.0
        # Sidecars still on disk — the sweep (byte-identical, exit 14)
        # would honestly fail the build here.
        assert _sidecars(tmp_path)
    finally:
        holder.rollback()
        holder.close()


def test_squeeze_source_wired_between_close_and_sweep() -> None:
    """The child script must run the squeeze BEFORE the byte-identical sweep."""
    script = gcli._CHILD_SCRIPT
    squeeze_call = "_teardown_chroma_sqlite_sidecars(STAGING)"
    # Anchor on a unique fragment of the ACTUAL call line, not the bare
    # signature: a bare ``orch.close(strict=True)`` anchor matches the
    # first occurrence, which may be a comment mentioning the call rather
    # than the call itself (the assertion must order code, not comments).
    close_call = "orch.close(strict=True)  # exactly ONE FTS writer"
    sweep_marker = '_SQLITE_SIDECAR_SUFFIXES = ("-", ".")'
    assert squeeze_call in script
    assert script.index(close_call) < script.index(squeeze_call) < script.index(sweep_marker)
