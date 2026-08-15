#!/usr/bin/env python3
"""Standalone rebuilder for the FTS5 lexical index (Task 05 / ADR-008).

Use when the daemon's lazy migration is inconvenient — e.g. suspected
corruption, migration stall, or a maintenance window when the operator
wants to block until the rebuild is done.

Repopulates ``<data_dir>/fts5_index.db`` through the same content-bound
Package-B primitive the daemon uses (capture + digest + staging + guarded
swap + schema-v2 marker). Offline-only until Package C closes CRUD parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow running directly from source checkout ``python scripts/build_fts5_index.py``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the FTS5 lexical index from ChromaDB.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to the knowledge-rag data directory (defaults to config.data_dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a staged rebuild (compatibility flag; never unlinks the prior DB/marker up front).",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Block until migration completes (default). Retained for symmetry.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Reserved for compatibility; no per-batch output.")
    return parser.parse_args(argv)


def _resolve_data_dir(cli_arg: Path | None) -> Path:
    if cli_arg is not None:
        if not cli_arg.is_dir():
            raise SystemExit(f"[BUILD-FTS5] data root does not exist: {cli_arg}")
        os.environ["KNOWLEDGE_RAG_DIR"] = str(cli_arg)  # D5: bind BEFORE any mcp_server config import
        return cli_arg
    from mcp_server.config import config

    return Path(config.data_dir)


def _load_fts5() -> Any:
    # The mcp_server package __init__ imports config, whose import CREATES the
    # configured directories. fts5_index.py is stdlib-only, so load it by file
    # path to keep --data-dir runs free of any config side effects (D5).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fts5_index_rootbound", _ROOT / "mcp_server" / "fts5_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yaml_collection_name(config_path: Path) -> "str | None":
    import yaml

    if not config_path.exists():
        return None
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    name = (loaded.get("search") or {}).get("collection_name")
    return str(name) if name else None


def _yaml_index_mode(config_path: Path) -> "str | None":
    import yaml

    if not config_path.exists():
        return None
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mode = (loaded.get("indexing") or {}).get("mode")
    return str(mode) if mode else None


def _refuse_versioned_mode(data_dir: Path, cli_bound: bool) -> None:
    """Versioned generations are sealed + immutable; this legacy rebuilder must not touch them."""
    # Refuse a generation-store ROOT even when no config.yaml says "versioned":
    # a ``current`` pointer or ``generations/`` tree means sealed artifacts.
    if (data_dir / "current").exists() or (data_dir / "generations").is_dir():
        raise SystemExit(
            "[BUILD-FTS5] data_dir is a versioned generation store (current/generations present) — "
            "the FTS index is a sealed generation artifact. "
            "Use `knowledge-rag-generation build` to build a new immutable generation instead."
        )
    # Refuse being INSIDE a sealed generation: equal to or nested under
    # ``<store>/generations/<id>``. Such a directory contains a real
    # ``chroma_db/`` and would otherwise be opened and rebuilt in place.
    # Walk the resolved ancestry: any node whose parent is a ``generations``
    # directory is a generation id (or something nested beneath one).
    _resolved = data_dir.resolve()
    _node = _resolved
    while True:
        _parent = _node.parent
        if _node != _parent and _parent.name == "generations" and _parent.is_dir():
            raise SystemExit(
                "[BUILD-FTS5] data_dir is inside a sealed versioned generation "
                f"({_parent.parent / 'generations' / _node.name}) — never open or mutate a "
                "sealed generation. Use `knowledge-rag-generation build` instead."
            )
        if _node == _parent:
            break
        _node = _parent
    for candidate in (data_dir.parent / "config.yaml", data_dir / "config.yaml"):
        if _yaml_index_mode(candidate) == "versioned":
            raise SystemExit(
                "[BUILD-FTS5] indexing.mode=versioned — the FTS index is a sealed generation artifact. "
                "Use `knowledge-rag-generation build` to build a new immutable generation instead."
            )
    if not cli_bound:
        from mcp_server.config import config

        if config.index_mode == "versioned":
            raise SystemExit(
                "[BUILD-FTS5] indexing.mode=versioned — the FTS index is a sealed generation artifact. "
                "Use `knowledge-rag-generation build` to build a new immutable generation instead."
            )


def _collection_name(data_dir: Path, client: Any) -> str:
    # D5-r5: --data-dir is the DATA directory. Canonical project config sits
    # beside it (data_dir.parent/config.yaml); a root-local config.yaml is
    # honored only when unambiguous; otherwise exactly one existing Chroma
    # collection resolves — anything else fails closed. No mcp_server import.
    parent_name = _yaml_collection_name(data_dir.parent / "config.yaml")
    local_name = _yaml_collection_name(data_dir / "config.yaml")
    if parent_name and local_name and parent_name != local_name:
        raise SystemExit(f"[BUILD-FTS5] ambiguous collection_name: project={parent_name!r} vs root-local={local_name!r}")
    if parent_name or local_name:
        return parent_name or local_name
    existing = [str(getattr(col, "name", col)) for col in client.list_collections()]
    if len(existing) != 1:
        raise SystemExit(f"[BUILD-FTS5] cannot resolve collection: no configured name and {len(existing)} existing collections {existing!r}")
    return existing[0]


def _open_collection(data_dir: Path, cli_bound: bool) -> Any:
    import chromadb

    chroma_dir = data_dir / "chroma_db"
    if not chroma_dir.is_dir():
        raise SystemExit(f"[BUILD-FTS5] chroma source does not exist: {chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if cli_bound:
        name = _collection_name(data_dir, client)
    else:
        from mcp_server.config import config

        name = config.collection_name
    # D5 fail-closed: get_collection only — a typo must not materialize an empty corpus.
    return client.get_collection(name=name)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = _resolve_data_dir(args.data_dir)
    _refuse_versioned_mode(data_dir, cli_bound=args.data_dir is not None)
    print(f"[BUILD-FTS5] data_dir={data_dir} force={args.force}")

    # D5: --force never unlinks the prior credible DB/marker; the swap stays atomic.
    start = time.time()
    fts5 = _load_fts5()
    collection = _open_collection(data_dir, cli_bound=args.data_dir is not None)
    rows = fts5.capture_chunk_rows(collection)
    index = fts5.Fts5LexicalIndex(db_path=data_dir / "fts5_index.db", state_path=data_dir / "fts5_migration.state")
    try:
        result = index.rebuild_content_bound(rows)  # shared primitive — no positional/zip path
    finally:
        index.close()
    print(f"[BUILD-FTS5] complete: {result['docs_indexed']} docs source={result['source_rows_sha256'][:12]} verified={result['verified_fts_rows_sha256'][:12]}")

    elapsed = time.time() - start
    print(f"[BUILD-FTS5] elapsed_seconds={elapsed:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entrypoint
    raise SystemExit(main())
