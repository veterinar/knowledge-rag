# Criteria: offline query-expansion and reranker boundary

SDD: class HIGH, mode integration. The original P0-3 hypothesis is Linear
SEM-6 comment `e7976d63-df09-4e50-8062-80074339de8d`; the current owner
instruction is `делай`. This slice proves the already implemented boundary;
it does not change retrieval ranking or enable a production reranker.

## Outcome

Versioned retrieval cannot fetch a reranker model or load an unpinned artifact,
and query expansion remains a pure in-process transformation over frozen config.

## Acceptance

- **OB-1:** with versioned mode and an admitted local artifact, the production
  `TextCrossEncoder` constructor receives the exact resolved
  `specific_model_path` and `local_files_only=True`.
- **OB-2:** with versioned mode, reranker enabled, and no admitted artifact,
  no model constructor is invoked and retrieval fails closed or keeps the
  explicit unavailable state already defined by the production contract.
- **OB-3:** a load failure in enabled versioned mode raises the typed
  `RerankerUnavailableError`; it never silently becomes RRF order.
- **OB-4:** `expand_query` performs no file, subprocess, socket, HTTP or model
  call; its output is derived only from the input and frozen
  `config.query_expansions`.
- **OB-5:** tests use spies/mocks and temporary local artifacts only. They do
  not download models, open network connections, mutate a live generation or
  run a production reseal.

## Preserved behavior

- Legacy-mode fallback remains unchanged.
- A disabled versioned reranker remains an explicit identity state.
- Query-expansion semantics and result ordering do not change.

## Out of scope

- Enabling a reranker in the live generation.
- Model download, benchmark, reseal, rollout or network policy mutation.
- Query augmentation modes beyond the existing synonym expansion.
