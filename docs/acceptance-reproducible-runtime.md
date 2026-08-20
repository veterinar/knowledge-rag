# Acceptance: reproducible build and pinned runtime

## Goal

Make the production runtime and release wheel reproducible without changing
retrieval semantics or the model's pooling strategy.

## Requirements

1. `requirements.lock` is the canonical production dependency input.  For
   each supported Python/platform environment it resolves exactly one active,
   hash-bearing pin for FastEmbed, ONNX Runtime, ChromaDB, MCP, and every
   mandatory transitive dependency.  A missing, duplicate-active, uninstalled,
   or version-mismatched active pin fails closed.
2. A versioned generation binds the exact embedding and enabled-reranker
   artifact tree digests, logical model names, embedding dimension, query and
   passage prefixes, declared runtime version, effective pooling, and loader
   provider configuration.  Serving loads only those local artifact
   directories with network fetching disabled.  Pooling remains unchanged.
3. `advanced.watch_for_changes` and
   `advanced.watch_debounce_seconds` control the legacy watcher.  The server
   does not contain an independent hardcoded debounce value; versioned mode
   keeps the watcher disabled.
4. A clean Python 3.11--3.13 environment installs the hash-locked build inputs
   and produces exactly one wheel with `python -m build --no-isolation`.
   Verification compares the wheel package-file set and bytes, including every
   force-included package-data file and embedded runtime lock, with the exact
   Git tree.
5. The deployment/release receipt records the exact Git commit, wheel filename,
   wheel SHA-256, archive manifest, build/runtime lock digests, build-tool
   versions, and package-data comparison.  The versioned runtime receipt keeps
   installed distribution `RECORD` evidence and never substitutes a wheel SHA
   for it.

## Verification

- Existing coverage first: `tests/test_generations.py`,
  `tests/test_generation_integration.py`, `tests/test_embedding_profile.py`,
  and `tests/test_vetclub_runtime_regressions.py`.
- A focused check must be observed RED on the unfixed/mutated base for every
  new or corrected guard, then GREEN on the candidate.
- Build TWO wheels independently in clean environments (two fresh
  ``python -m build --no-isolation`` runs). Their wheel SHA-256, archive
  manifest, and package bytes must be byte-identical; run
  ``scripts/release_receipt.py`` against the exact candidate commit/tree for
  each and inspect both JSON receipts.
- The full suite runs once in GitHub CI on the exact PR head, not locally.

## Out of scope

- Changing pooling, embedding vectors, model choice, chunking, or retrieval
  scoring without a separately reproduced defect.
- Publishing a release, deploying a wheel, rebuilding a production generation,
  or mutating production runtime state.
- Adding a second lock format, build backend, model downloader, or RAG feature.
