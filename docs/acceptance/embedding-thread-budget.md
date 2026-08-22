# Embedding thread budget

## Requirement

- `models.embedding.threads` is optional; omission preserves FastEmbed's existing default.
- An explicit value is an integer from 1 through 64. Booleans and values outside that range fail configuration loading.
- The configured value is passed unchanged to FastEmbed `TextEmbedding`, which binds the ONNX intra-op and inter-op session thread counts.
- The value participates in the fresh retrieval-config digest and model-config identity, so post-build drift blocks retrieval.
- Model name, artifact, dimensions, prefixes, pooling, vectors, ranking, and the public MCP tool surface do not change.
- The production runtime sets `threads: 2`. A five-document offline generation canary must remain below the existing host resource stop limits before a full generation is attempted.

## Proof

- RED on the unfixed base: a focused loader assertion shows the configured thread value is absent from `TextEmbedding` kwargs.
- GREEN on the candidate: the same assertion observes the exact value, configuration boundary cases pass, and identity changes when only `threads` changes.
- Run only the focused tests and changed-file lint locally. The repository's complete suite runs once in GitHub CI on the exact PR head.

## Out of scope

- Changing model artifacts, pooling, vector semantics, batch parallelism, GPU routing, or the FastEmbed/ONNX dependency versions.
- Raising host resource limits or treating environment variables as the thread control.
