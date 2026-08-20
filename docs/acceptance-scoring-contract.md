# Scoring contract acceptance

These criteria are written before the point-8 corrective and define its release boundary.

## Requirements

1. Every served FTS5 or hybrid search result exposes an unrounded effective `raw_score`, a
   `query_relative_score` normalized only within that query's returned cohort, and a
   `score_source` naming the scorer. The legacy `score` field remains an alias of
   `query_relative_score`.
2. `min_score` filters `query_relative_score`. Public API text must not describe it as an
   absolute quality threshold, recommend universal cut-offs, or compare values across queries.
3. `evaluate_retrieval` is labelled and documented as an `offline_smoke` reachability check.
   It must not be presented as a benchmark, quality measurement, or source of universal MRR /
   Recall / Precision thresholds.
4. Existing MCP tool names and parameter lists remain unchanged.
5. The unfixed exact base must fail at least one focused assertion for the stale public contract;
   the candidate must pass the focused scoring/API tests.

## Focused verification

- `tests/test_search.py`
- `tests/test_tools.py`
- `tests/test_backwards_compat.py`

## Out of scope

- Changing ranking, RRF, reranker, MMR, embedding, chunking, or index-generation algorithms.
- Claiming benchmark-quality retrieval metrics from the offline smoke tool.
- Removing the legacy `score` or `filtered_by_score` compatibility aliases.
