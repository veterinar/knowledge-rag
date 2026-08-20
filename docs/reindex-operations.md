# Reindex Operations Guide (v4.8.0+)

## Sync vs. Async APIs

The MCP tool `reindex_documents(force=True, full_rebuild=True)` returns immediately after spawning a daemon thread that performs the rebuild in the background. This is intentional for the MCP protocol (the client shouldn't block for hours waiting on a large reindex), BUT it has a subtle failure mode:

**If the Python process running the MCP server dies mid-rebuild** (crash, OOM kill, `Ctrl+C`, remote SSH disconnect, laptop suspend, etc.), the daemon thread dies with it — leaving the ChromaDB collection **deleted but never repopulated**. Result: RAG returns zero results until the operator manually re-runs the reindex.

**Rule of thumb:**

- **MCP tool calls:** use `reindex_documents(...)` as documented — the async wrapper is correct. The whole point of MCP is that clients (Claude, Cursor, etc.) get a response inside the request budget.
- **CLI scripts / one-off jobs:** prefer `get_orchestrator().nuclear_rebuild()` (sync) directly. This guarantees the rebuild completes before Python exits. If the script gets killed, at least the collection is not in a half-destroyed state.
- **Long-running rebuilds on unstable connections:** use `reindex_documents(force=True)` (smart reindex, not full rebuild) and rely on the resume checkpoint (see below) to recover.

## Resume Kwarg (v4.8.0+)

`reindex_documents(resume=True)` recovers from `data/reindex_checkpoint.json` after an interrupted smart reindex.

**Rules:**

- `resume=True` is only valid when `full_rebuild=False`. The combination `full_rebuild=True + resume=True` is rejected up front with a structured error — nuclear rebuild has no meaningful checkpoint semantic because the entire collection is thrown away and rebuilt from scratch, so "resuming" would leave a partial collection with no coherent state.
- `resume=True` implicitly forces `mode='smart_reindex'`, even when `force=False` was passed. An interrupted smart reindex must be resumed with smart, not silently downgraded to `incremental` (which would ignore the checkpoint entirely).
- When no valid checkpoint exists, `resume=True` prints an informational log and starts a fresh smart reindex.

**Checkpoint cadence:** written every 500 docs OR every 30 seconds, whichever comes first.

- 500-doc cadence covers fast runs (small markdown docs mostly cached by mtime skip, thousands per minute).
- 30-second cadence covers slow runs (one PDF with 5000 chunks could take longer than 30s, leaving too long a gap between checkpoints).

**Checkpoint invalidation:** the checkpoint stores a `config_signature` — an SHA256 hash of the current `embedding_model | embedding_dim | chunk_size | chunk_overlap` tuple. If any of those config values change between checkpoint write and resume load, the checkpoint is discarded with a WARN and the reindex starts fresh. This prevents a mixed collection (partial old vectors with the previous model + partial new vectors with the current one) which would silently degrade retrieval quality.

**Checkpoint lifecycle:**

- Written every 500 docs / 30s during a smart reindex, alongside a metadata flush so `_indexed_docs` on disk stays in sync with the checkpoint's `indexed_doc_ids` list.
- Cleared automatically on successful reindex completion.
- Cleared automatically if `_load_checkpoint()` returns None on a `resume=True` attempt (missing/corrupt/version-mismatch/signature-mismatch).

## Progress Fields (v4.8.0+)

`get_reindex_status()` returns these fields while a reindex is active:

| Field                 | Type    | Meaning                                                                                       |
| --------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `chunks_processed`    | int     | Chunks committed to ChromaDB so far                                                           |
| `chunks_total`        | int     | Rolling estimate (0 during warmup, then running average from completed docs × total_files)    |
| `throughput_cps`      | float   | Chunks per second, sliding window (last 30s OR 100 samples, whichever smaller — for stability) |
| `eta_seconds`         | int     | Estimated seconds to completion, derived from throughput + remaining chunks                   |
| `checkpoint_saved_at` | str/None | ISO timestamp of the last checkpoint write (None until first checkpoint)                     |
| `resumed`             | bool    | True when this run recovered from a checkpoint via `resume=True`                              |

**Warmup behavior:** `chunks_total` starts at 0 (or an estimate from previously indexed docs if any exist) and is refined each iteration using the running average across docs already processed. `throughput_cps` stays at 0.0 until at least 2 samples land in the sliding window. `eta_seconds` stays at 0 during throughput warmup and near completion (when `chunks_processed` catches up to the estimate).

**Sliding window rationale:** using a fixed-size deque bounded at 100 entries AND pruning entries older than 30 seconds keeps the throughput number stable during transient stalls (e.g. one giant PDF that takes 60s to embed) without diluting current speed with ancient samples from the start of the run.

## Example Recovery Flow

```
# Terminal 1: start a smart reindex (say, 5000 docs)
mcp> reindex_documents(force=True)
{"status": "started", "operation": "smart_reindex", ...}

# ... 3 minutes in, laptop crashes at doc 2100 / 5000 ...

# Terminal 1 restart: check if a checkpoint survived
$ ls data/reindex_checkpoint.json
data/reindex_checkpoint.json

# Resume from where it stopped
mcp> reindex_documents(resume=True)
{"status": "started", "operation": "smart_reindex", ...}
[REINDEX] Resuming smart reindex from checkpoint (2100 docs already processed, 12345 chunks)

# Poll status — resumed=True marker + chunks_processed continues from checkpoint
mcp> get_reindex_status()
{
  "active": true,
  "operation": "smart_reindex",
  "progress": "2103/5000",
  "chunks_processed": 12360,
  "chunks_total": 29400,
  "throughput_cps": 45.2,
  "eta_seconds": 377,
  "resumed": true,
  ...
}
```

## Zero-downtime Rebuild (v4.8.0 Fase 5+)

`nuclear_rebuild(swap=True)` (the new default) uses a staging collection to eliminate the RAG downtime window that the destructive rebuild (`swap=False`) creates.

**Workflow:**

1. **Cleanup** — sweep staging collections older than 24h (crash-orphaned by a previous rebuild).
2. **Snapshot** — save `self.collection`, `self.bm25_index`, `self._bm25_initialized`, `self._indexed_docs`, `self._source_to_docid` so populate/validate/swap failures can rollback to exact pre-call state.
3. **Create staging** — `{collection_name}__staging_{unix_ts}` (timestamp avoids collisions with concurrent or previously-crashed rebuilds; same embedding function as prod so the swap is dimensionally compatible).
4. **Populate** — temporary orchestrator rebind: `self.collection` points at staging, `self.bm25_index` is a throwaway `BM25Index()` so production BM25 keeps serving queries throughout the rebuild window. Full `index_all(force=True)` path runs against staging with byte-identical logic (no code fork).
5. **Validate** (three gates, all must pass):
   - `staging.count() >= baseline_count * 0.9` (10% loss threshold accommodates a small number of parser-skipped docs; larger loss indicates regression → abort).
   - 4 of 5 canonical queries (`readme`, `function`, `import`, `return`, `class`) return at least one hit each. Threshold is 4/5 (not 5/5) so a genuinely small corpus that legitimately lacks e.g. `return` still passes.
   - No query() call raises (catches embedding dim mismatches and other backend corruption that a raw count check would miss).
6. **Atomic swap** — two-step `Collection.modify(name=...)`:
   - Step 1: prod → `{prod}__old_{ts}` (frees the production name).
   - Step 2: staging → `{prod}` (staging assumes the production name).
   - Step 3: delete `__old_{ts}` (non-fatal — cleanup helper ages it out later if it fails).
   - Race window between step 1 and step 2 is a single Python statement (~microseconds); if step 2 raises, rollback renames prod back so the previous state is still queryable.
7. **Post-swap BM25 rebuild** — reconnect `self.collection` to the new prod, clear + rebuild BM25 from the swapped-in ChromaDB contents, invalidate query cache.

**Failure modes:**

| Failure                  | Effect on production                                             |
| ------------------------ | ---------------------------------------------------------------- |
| Validate fails           | Staging kept for inspection; snapshot restored; prod untouched   |
| Swap step 1 fails        | Nothing renamed; snapshot restored; prod untouched               |
| Swap step 2 fails        | Rollback of step 1; snapshot restored; prod queryable at old name if inner rollback fails |
| Python crash mid-populate | Staging orphaned; next Orchestrator boot cleans it (24h TTL)     |

**Storage impact:**

- **During rebuild:** ~2x storage temporarily (both prod and staging on disk).
- **After swap:** back to 1x (`__old_{ts}` deleted at end of swap).
- **Stale cleanup:** 24h TTL — a staging orphan sits at most one day before automatic reclaim.

**Backwards compat:**

`nuclear_rebuild(swap=False)` preserves the legacy destructive behavior byte-for-byte (delete prod collection first, wipe SQLite files, rebuild from scratch, ~4min–40h window of empty queries). Preserved for tests, forced-cleanup edge cases, and situations where the 2x storage overhead of staging is unacceptable.

## Escalation Rules of Thumb

- Reindex started, hours later still `active: false`? Check `last_error` in status. If Python was killed, the collection may be half-populated — run `reindex_documents(resume=True)` if the operation was smart, or `reindex_documents(force=True, full_rebuild=True)` if it was nuclear.
- Getting `resume=True is only valid for smart reindex` error? Drop `full_rebuild=True`. Nuclear rebuild does not support resume by design (see above).
- Checkpoint keeps invalidating with "config_signature mismatch"? Something in `config.yaml` (or the effective merged config) changed embedding model / dim / chunk size / chunk overlap between runs. Either revert the config or accept that the next reindex will start fresh.

---

## Versioned Generations (Phase B) — `indexing.mode: "versioned"`

**Read this before switching indexing.mode to "versioned".** Everything above
describes the legacy in-place index. Versioned mode replaces *all* of it with
immutable sealed generations; the smart/nuclear reindex machinery, the watcher,
and the add/update/remove tools are **not used** in that mode.

### What changes

| Concern | legacy | versioned |
| --- | --- | --- |
| Index state | one mutable `data/` tree | sealed generations under `data/generations/<id>` + a `current` pointer |
| Mutations | live via MCP tools / watcher | offline only: `knowledge-rag-generation build` |
| Serving | reads + writes | **query-only**; the process pins ONE verified generation for its lifetime |
| Reindex recovery | smart/nuclear rebuild | never; build a new generation instead |
| Watcher | auto-reindex on change (`advanced.watch_for_changes`, `advanced.watch_debounce_seconds`; config.yaml is the sole default/validation owner — the server passes the debounce field directly) | **disabled** (no hot reload, no auto-index, no repair) |

### Acceptance contract

Versioned serving is admitted only when all of the following remain true:

- the controlling corpus identity is the SHA-256 of the sorted
  `relative POSIX path + file SHA-256` manifest selected with the same rules
  as ingestion; Vault `HEAD` is provenance only;
- the receipt binds the secret-free retrieval configuration, installed
  runtime RECORD aggregate, canonical dependency lock, model artifacts and
  effective model configuration, chunking, generation identity, and the
  Chroma/FTS artifact and row counts/digests;
- the declared embedding `runtime_version`, `dimensions`, and `pooling` are
  verified against the **installed FastEmbed registry** at startup
  (registry-only lookup — no model construction, artifact download, or
  network). Versioned mode admits EXACTLY three models:
  `BAAI/bge-small-en-v1.5` (384, `cls-or-prepooled` via the exact
  `fastembed.text.onnx_embedding.OnnxTextEmbedding` class),
  `BAAI/bge-large-en-v1.5` (1024, `cls-or-prepooled`, same exact class),
  and `intfloat/multilingual-e5-large` (1024, `mean`, via the exact
  `fastembed.text.pooled_embedding.PooledEmbedding` class). `cls-or-prepooled`
  is the exact FastEmbed 0.8.0 branch contract (CLS for rank-3 sequence
  output, pass-through for already-pooled rank-2 output) — not a claim about
  a given artifact's internals. Every other model or class — including other
  exact Onnx models, PooledNormalized, and Custom — fails closed. The runtime
  string must equal `fastembed <installed CPU version>` exactly, and a
  shadowing `fastembed-gpu` distribution is rejected. Unknown models,
  duplicate registrations, non-canonical spellings, registry dimension
  drift, and missing distribution metadata all fail closed. A mismatch
  degrades serving to stats-only (`restart_required_environment_changed`,
  retrieval blocked, no raw exception text) instead of aborting startup;
  on match, the canonical actuals are stored and sealed into the receipt's
  compatibility object. Generations sealed with the older generic `cls`
  pooling value must be rebuilt offline;
- every retrieval path checks freshness before cache/backend access and again
  before serving a materialized result; any mismatch returns
  `retrieval_blocked` and never serves a stale cache entry;
- `tools/list` in versioned mode contains no mutating tools. Direct in-process
  calls retain a defense-in-depth `offline_generation_required` refusal;
- `get_index_stats` remains available in healthy and degraded mode. The
  **degraded** process returns the full stable additive envelope
  (`index_mode`, `ready`, `reason`, `detail`, `checked_at`, `generation`,
  `restart_required`, `runtime_mutations`, `watcher`) with `ready: false`,
  `restart_required: true`, and sanitized `reason`/`detail` — no base
  orchestrator index stats are included (no orchestrator exists and no
  Chroma/FTS handle is opened). The **healthy** versioned process returns
  those additive fields plus the full orchestrator index stats;
- QMD refresh, deployment and live generation activation are outside this
  repository acceptance contract.

### Server startup (versioned)

1. The server verifies the `current` pointer and the full receipt of the
   generation it names before preflight and before any Chroma/SQLite handle
   opens. A missing, corrupt, incompatible, or stale generation starts a
   **degraded stats-only process**: retrieval is blocked, nothing is created,
   and `get_index_stats` reports the sanitized reason.
2. On success it binds the sealed corpus, FTS5 and metadata artifacts. FTS5 is
   opened through SQLite `mode=ro`; Chroma is opened only from a
   receipt-verified, process-local disposable copy because Chroma's
   `PersistentClient` requires a writable store. The published Chroma tree is
   never opened by the serving client.
3. Mutating MCP tools (`add_document`, `update_document`, `remove_document`,
   `add_from_url`, `reindex_documents`) are removed from the advertised
   versioned tool surface. Their Python callables still return the stable
   `offline_generation_required` envelope before filesystem or network work.
4. If `current`, the live corpus, config, code/runtime evidence, models or
   either backend changes underneath a running server, all retrieval is
   blocked with `restart_required`; the process never rebinds or serves the
   old cached result. `get_index_stats` remains available.

### Immutability boundary (honest scope)

Sealed Chroma/FTS/corpus artifacts are immutable **at the application
boundary**: the server refuses every write path, FTS5 uses a read-only handle,
and Chroma uses only a verified disposable copy. Receipt and backend evidence
are rechecked on every access, so published-byte drift fails closed. This is
not an OS-level enforcement claim — the deployment must still mount or
permission the published generation read-only against unrelated processes.

### Operations

```bash
# Inspect the current generation (fail-closed verification)
knowledge-rag-generation status

# Build + activate a new generation from the live corpus.
# Runs OFFLINE; never mutates sealed generations; reports restart_required.
knowledge-rag-generation build

# The build is transactional per generation:
#   - corpus drift during the copy, indexing errors, FTS/Chroma parity
#     failures, or receipt validation failures abort and clean ONLY the
#     .building-<id> staging tree — `current` stays byte-identical;
#   - a publish race (CAS conflict) preserves the sealed generation and
#     prints the exact `activate` recovery command;
#   - sealed generations are never deleted (no GC).

# Switch to an already-published generation (validates it fully first;
# atomic pointer swap; requires a server restart)
knowledge-rag-generation activate <generation_id>

# Roll back to a previously published generation (same semantics)
knowledge-rag-generation rollback <generation_id>
```

### Rollback decision table

| Situation | Command |
| --- | --- |
| Bad corpus content in the newest generation | `knowledge-rag-generation rollback <previous_id>` + restart |
| New generation fails to build | nothing to roll back — `current` never moved |
| Compatibility pins changed (new model) | build a NEW generation; activate only after verifying |
| Running server shows `restart_required` | restart the server; it will pin whatever `current` names |

### Compatibility pins

A generation records the exact 13-field compatibility object: collection;
embedding model, dimension, query/passage prefixes, artifact digest, runtime
version and actual pooling; chunk size/overlap; and reranker enabled state,
logical model and artifact digest. A server whose effective pins differ cannot
serve or activate that generation. Set the pins under `models.embedding` and,
when enabled, `models.reranker` in `config.yaml` (see `config.example.yaml`).

### Dependency evidence (requirements.lock)

Every generation build and every runtime drift recheck verifies the
environment against `requirements.lock` — the ONE canonical production
install input (pip-compile `--generate-hashes` output, shipped in the repo,
the sdist, and the wheel's `mcp_server/data/`):

- every active default `Requires-Dist` dependency of the installed
  `knowledge-rag` distribution must be pinned in the lock;
- every lock pin that is installed must match the lock version exactly
  (optional marker-inactive pins need not be installed);
- the installed `knowledge-rag` distribution version must equal
  `mcp_server.__version__` (the self package is not part of the pip-compile
  lock, so version parity is its identity);
- the installed RECORD digest and the lock digest stay DISTINCT evidence —
  one binds installed bytes, the other binds the resolved lock file.

Docker, CI, and the release workflow all install the same way:
`pip install --require-hashes -r requirements.lock`, then the wheel with
`--no-deps`, then `pip check`. Runtime always imports the installed package
bytes, never a source checkout shadow.

The legacy `scripts/build_fts5_index.py` refuses to run in versioned mode and
points at the generation CLI instead.
