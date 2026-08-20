---
name: rag-evaluate-quality
description: Offline smoke check that expected documents are still reachable in top-5 results, using evaluate_retrieval (evaluation_mode offline_smoke, MRR@5/Recall@5 diagnostics only) plus get_index_stats for health metrics. Run after significant reindex activity, model changes, or when the user reports search problems. Not a benchmark and not a retrieval-quality measurement.
metadata:
  type: rag-workflow
  kind: maintenance
  target: any-mcp-client
---

# rag-evaluate-quality — reachability smoke check, not a benchmark

## When to use this skill

Trigger this skill:

- **After significant reindex activity** — new content added, models changed, presets swapped
- **After a version upgrade** — `pip install -U knowledge-rag` bump, worth confirming expected docs are still reachable
- **When the user reports search problems** — "I can't find X anymore" translates directly into a reachability case

**Do NOT run:**

- Every session (waste of cycles)
- On brand-new empty corpora (nothing to reach)

---

## What this skill commits to

The agent produces a **reachability report over curated queries**, not a quality measurement. Outputs:

- **Index health** — chunks, cache hit rate, embedding model, dimensions (from `get_index_stats`)
- **Per-query reachability** — for each hand-picked query, did the expected document appear in the top-5, and at what rank
- **Aggregate diagnostics** — MRR@5 and Recall@5 over that curated set only (`evaluation_mode: "offline_smoke"`)

These numbers describe THIS curated set on THIS corpus. They carry no statistical weight: do not quote them as retrieval-quality measurements, do not compare them to universal thresholds (none exist), and do not derive tuning decisions from them.

---

## Steps

1. **Snapshot index health:**
   ```
   get_index_stats()
   ```
   Capture:
   - `documents_count`, `chunks_count`
   - `cache_hit_rate` (higher after warmup = healthy)
   - `embedding_model`, `embedding_dim`

2. **Prepare or reuse a curated case set.** Each `evaluate_retrieval` test case carries exactly `query` (non-empty string) and `expected_filepath` (non-empty path of the document that should appear in top-5):

   - **Reuse a canonical set** — if the project already has `tests/evaluation-queries.json` or similar, load it.
   - **Build a quick set inline** — 5-10 queries where a domain expert (or the user) knows the expected document.

   Format (per the API contract):
   ```json
   [
     {"query": "authentication design", "expected_filepath": "docs/adr/0018-auth.md"},
     {"query": "retry policy", "expected_filepath": "docs/adr/0031-retries.md"}
   ]
   ```

3. **Run the smoke check:**
   ```
   evaluate_retrieval(test_cases=<the-json-array>)
   ```
   Response includes:
   - `evaluation_mode` — `"offline_smoke"`
   - `mrr_at_5` — mean reciprocal rank of expected docs across the curated set
   - `recall_at_5` — fraction of curated queries whose expected doc appeared in top-5
   - `per_query` — which expected doc was found and at what rank

4. **Read the per-query breakdown, not the aggregates.** The actionable output is the list of queries whose expected document was NOT reached (`found_at_rank: null`). Those are concrete reachability failures worth investigating; the aggregate numbers are context, not a verdict.

5. **Investigate unreached documents concretely:**

   | Symptom | Likely cause | Next step |
   |---|---|---|
   | Expected doc absent from index | Ingestion skipped it | Check `list_documents` / reindex |
   | Doc indexed but unreached | Chunking or embedding changed how it matches | Inspect the doc with `get_document`; try its exact vocabulary as a query |
   | Many queries unreached after a model/preset change | Distribution shifted | Review the change; reindex if the model moved |
   | `cache_hit_rate` = 0% after 100+ queries | Queries not repeating (fine) OR cache misconfigured | Only worry if you expected repeats |

6. **Report honestly.** State that this was an offline smoke check over N curated queries, list unreached docs, and describe what you inspected. Do not present MRR/Recall as quality scores, do not claim regressions or improvements from them, and do not recommend tuning values derived from them.

---

## Examples

**Example 1 — Post-reindex confirmation**

User: *"I just did nuclear_rebuild after switching to bge-large. Did anything break?"*

Agent:
```
get_index_stats() → 3865 docs / 39241 chunks, embedding: bge-large-en-v1.5 (1024D)
Loaded tests/evaluation-queries.json (12 cases)
evaluate_retrieval(test_cases=<the 12>) → evaluation_mode: offline_smoke
  recall_at_5: 12/12 reached; one doc moved from rank 1 to rank 4 (per_query)

Reply: "Smoke check green: all 12 expected documents are still reachable in
top-5 after the rebuild. One (docs/adr/0031-retries.md) moved from rank 1 to
rank 4 — inspected it and it still matches on its own vocabulary, so no
action needed. This was a reachability check, not a quality measurement."
```

**Example 2 — User reports a search problem**

User: *"I can't find the refund policy doc anymore."*

Agent:
```
evaluate_retrieval(test_cases=[{"query": "refund policy",
  "expected_filepath": "docs/policies/refunds.md"}])
→ per_query: found_at_rank: null, top_result: none

Reply: "Confirmed: the refund policy doc did not come back in top-5 for its
own title query. list_documents() shows it is no longer in the index — the
last reindex ran before that file was added. Reindexing now."
```

---

## Edge cases

- **No curated set exists** — offer to help build one. Ask the user for 5-10 queries where they know the expected doc, keyed as `query` + `expected_filepath`. Bootstrap that as `tests/evaluation-queries.json`.
- **Corpus < 50 docs** — reachability checks still work (each case is concrete), but say so: tiny sets say nothing about the corpus in general.
- **Cache hit rate always 0%** — either no repeat queries (fine, expected for one-shot use) OR `query_cache` is disabled in config. Not necessarily a problem.
- **Prometheus available** — chain to `/metrics` scrape for latency percentiles: `knowledge_rag_search_latency_seconds` histogram. Latency and reachability are separate signals; keep them separate in the report.

---

## Related skills

- **[`rag-onboard-context`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-onboard-context/SKILL.md)** — the light-touch version (get_index_stats only, no evaluation).
- **[`rag-index-decisions`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/maintenance/rag-index-decisions/SKILL.md)** — after fixing an indexing problem found here, index the decision so next reader knows what changed and why.
- **[`rag-check-first`](https://github.com/lyonzin/knowledge-rag/blob/master/skills/foundation/rag-check-first/SKILL.md)** — the workhorse whose reachability this skill verifies.
