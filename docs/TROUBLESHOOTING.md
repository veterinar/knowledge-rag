# Troubleshooting Guide

> Common problems and their solutions when running knowledge-rag on Windows, Linux, or macOS.

**Not finding your issue?**
- Search existing issues → https://github.com/lyonzin/knowledge-rag/issues
- Open a new bug report → https://github.com/lyonzin/knowledge-rag/issues/new/choose
- Discussions → https://github.com/lyonzin/knowledge-rag/discussions

**Related docs:**
- [Installation guide →](INSTALLATION.md)
- [Configuration reference →](CONFIGURATION.md)
- [API reference →](API.md)

---

### Python version mismatch

Requires Python 3.11 or newer.

```bash
python --version    # Must be 3.11+
```

### FastEmbed model download fails

On first run, FastEmbed downloads models to `models_cache/`. If the download fails:

```bash
# Clear cache and retry
# Windows:
rmdir /s /q models_cache

# Linux/macOS:
rm -rf models_cache

# Then restart the MCP server
```

### Reranker model download fails

The reranker is lazy-loaded on the first query. If the model is not cached and the machine is offline, search continues without reranking and uses the RRF order from hybrid retrieval. To keep reranking enabled offline, run one query while online or pre-populate `models_cache/` on the target machine.

You can still disable reranking explicitly in `config.yaml`:

```yaml
models:
  reranker:
    enabled: false
```

Disabling reranking reduces memory use and avoids first-query model loading. The tradeoff is lower ranking precision, especially when several chunks match the same terms but only one is the best answer.

### ChromaDB index crashes on startup

Native ChromaDB failures can terminate Python before normal exception handling runs. Startup now probes ChromaDB in a child process before initializing the MCP server. If the probe crashes, the active `chroma_db/` and `index_metadata.json` are moved to `data/backups/auto-repair-*`, and the next startup can rebuild a clean index.

The same guarded behavior is available through either console script:

```bash
knowledge-rag
knowledge-rag-guarded
```

### Index is empty

```bash
# Check documents directory has files
ls documents/

# Force reindex via Claude Code:
# reindex_documents(force=True)

# Or nuclear rebuild if model changed:
# reindex_documents(full_rebuild=True)
```

### MCP server not loading

1. Check `~/.claude.json` exists and has valid JSON in the `mcpServers` section
2. Verify paths use double backslashes (`\\`) on Windows
3. Restart Claude Code completely
4. Run `claude mcp list` to check connection status

### "Failed to connect" error

The MCP server uses stdout for JSON-RPC communication. If a library prints to stdout during init, the stream gets corrupted. v3.4.3+ includes stdout protection that prevents this. If you're on an older version, upgrade:

```bash
pip install --upgrade knowledge-rag
```

### Slow first query

The cross-encoder reranker model is lazy-loaded on the first query. This adds a one-time ~2-3 second delay for model download and loading. Subsequent queries are fast. If the model cannot be loaded, search falls back to RRF ordering and does not retry loading the reranker until the server restarts.

### Memory usage

With ~200 documents, expect ~300-500MB RAM. The embedding model (~200MB ONNX runtime resident, lazy-loaded on first query since v3.8.0) and reranker (~25MB, lazy-loaded) are loaded into memory only when actually used. For very large knowledge bases (1000+ documents), consider enabling GPU acceleration and using exclude patterns to limit index scope.

### Multiple MCP clients spawn duplicate servers

MCP stdio is one process per client by protocol — multiple Claude Code windows, Claude Desktop + IDE, etc. each spawn their own `knowledge-rag` process. Since v3.8.0 idle processes are cheap (no embedding model loaded until first query). If you've measured and want a hard cap of one server per data directory, opt in:

```bash
export KNOWLEDGE_RAG_SINGLE_INSTANCE=1
```

A second instance exits immediately with code 75. Default is OFF (multi-client friendly). Full guide: [docs/single-instance.md](docs/single-instance.md). Sample MCP config: [examples/mcp-config-single-instance.json](examples/mcp-config-single-instance.json).

### SSE server won't start

```bash
# Check if port 8179 is already in use
# Windows:
netstat -aon | findstr :8179
# Linux/macOS:
lsof -i :8179
```

If `uvicorn` is not found, install the server extras: `pip install knowledge-rag[server]`

### Can't connect to SSE server

Verify the server is running and the URL is correct:

```bash
curl http://127.0.0.1:8179/sse
```

Common issues:
- Wrong URL: must end with `/sse` (not just the port)
- Firewall blocking the port
- Server started with a different host/port than configured in the MCP client

### Versioned build fails with exit 14 (sidecar sweep) right after a green population

Fixed as of `eb672a84` (2026-08-28). The cause was located in the pinned
chromadb bytes: `client.close()` only decrements the `SharedSystemClient`
refcount and can orphan the sqlite connection, so WAL/SHM sidecars under the
child's chroma directory survived the close nondeterministically and the
post-population sweep correctly refused to seal (exit 14). Before the fix the
build was 0/3 green on the same inputs; with it, 1/1.

Since that commit the child runs a deterministic teardown before the sweep
(`_teardown_chroma_sqlite_sidecars`: gc pass, `PRAGMA
wal_checkpoint(TRUNCATE)` with a 0.2 s connect timeout, bounded wait ≤5 s).
"Re-run the build" is no longer the remedy: on `eb672a84` or later an exit 14
means the sweep found a real holder — a live process keeping the sqlite file
open. Find it (`lsof <db path>`) instead of re-running blind.

The guard itself is unchanged and the old advice about it stands: a
generation must be sealed from quiescent bytes only. Do not weaken or skip
the sweep, and do not delete the sidecars by hand inside a candidate
generation — a hand-mutated candidate is a new candidate.

---
