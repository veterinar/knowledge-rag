# FTS5 Lexical Fast-Path — Migration Runbook

Operational guide for the FTS5 lexical fast-path (v4.8.2+, Task 05). Covers
enable, wait, verify, manual rebuild, and large-corpus caveats.

References: `_techspec.md` §Migration Plan · ADR-001 · ADR-008.

## 1. Enable the feature

Edit `config.yaml`:

```yaml
search:
  lexical_fast_path:
    enabled: true
```

Restart the daemon. Construction only creates handles (Package B): the single
startup dispatch in `main()` is a bounded admission — a background worker
captures the Chroma snapshot and either proves live FTS == that exact source
(publishing ready) or rebuilds from zero; hybrid serves in the meantime.

## 2. Wait for the migration to finish

While the rebuild runs, `ready=false` is observable immediately: automatic
lexical queries fall back to the hybrid path
(`fast_path_fallback_total{reason="disabled"}`) while explicit
`search_method="fts5"` raises `Fts5NotReadyError`. The daemon keeps serving
queries the entire time — the rebuild never blocks the request path.

Expected timings (SSD SATA):
- 3865 docs (canonical bench corpus): ~60 s
- 10 000 docs: ~2–3 min
- 100 000 docs: ~15–30 min

On `/metrics`, `…migration_docs_total` is set at snapshot capture, while
`…migration_docs_indexed` resets to 0 at each admitted start and on failure and
is set only after a verified complete (no incremental 10 % logs).

## 3. Verify the marker file

`<data_dir>/fts5_migration.state` — canonical schema v2 (Package B):

```json
{
  "schema_version": 2,
  "generation": 1,
  "status": "complete",
  "docs_total": 3865,
  "docs_indexed": 3865,
  "source_rows_sha256": "<64-hex canonical source digest>",
  "verified_fts_rows_sha256": "<64-hex read-back digest>",
  "started_at": "2026-08-07T12:00:00+00:00",
  "completed_at": "2026-08-07T12:01:02+00:00",
  "error": null
}
```

- Serving readiness requires full credibility: schema v2, strict integer
  generation/counts, matching 64-hex digests, exact counts, and a recomputed
  live read-back digest equal to the *currently captured* Chroma source —
  after a corpus swap the old index is never re-advertised (P1-2).
- `status: "in_progress"` / `"invalidated"` (durable reset) — or anything
  unversioned/malformed/inconsistent — rebuilds from zero; there is no resume.
- `status: "failed"` → `error` holds the sanitized exception class only.
  Queries fall back until a rebuild succeeds.

## 4. Manual rebuild

`scripts/build_fts5_index.py` runs the same content-bound primitive the
daemon uses (capture → digest → staging → read-back verify → guarded swap →
schema-v2 marker), offline-only until Package C closes direct-CRUD parity.
`--data-dir` binds the Chroma source (`<root>/chroma_db`) AND the FTS target:

```bash
# Staged, atomic rebuild bound to one data root.
python scripts/build_fts5_index.py --data-dir data/ --force --foreground --verbose
```

Flags:
- `--data-dir <path>` — data root for the Chroma source and FTS target (defaults to `config.data_dir`).
- `--force` — force a staged rebuild; the prior credible DB/marker are never
  unlinked before source capture (the swap is atomic).
- `--foreground` — block until complete (default; kept for parity).
- `--verbose` / `-v` — reserved for compatibility; the rebuild emits no per-batch output.

The script exits `0` on success, prints an elapsed-time banner, and leaves
the marker file at `status: "complete"`.

## 5. Large corpora — dont interrupt the first rebuild

For corpora over ~10 k docs, the initial rebuild takes minutes. Best
practices:

- Kick the migration off intentionally (edit config, restart daemon)
  during a low-traffic window so the fallback logs and metric spikes are
  expected.
- Prefer `scripts/build_fts5_index.py --foreground` in ops runbooks: the
  operator sees progress synchronously and cannot accidentally reboot the
  daemon mid-rebuild.
- If the daemon is killed mid-rebuild, the marker stays non-credible and the
  next startup dispatch rebuilds from zero — nothing resumes positionally.
- Direct CRUD during a rebuild lands outside the captured snapshot; that
  parity boundary belongs to Package C, which is also why the builder stays
  offline-only until Package C is accepted.

## Related

- `_techspec.md` §Migration Plan — full lifecycle description.
- ADR-008 — CRUD sync incremental (why FTS5 diverges from BM25 full-rebuild).
- ADR-001 — SQLite dedicated storage + WAL + `busy_timeout=5000ms`.
