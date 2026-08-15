FROM python:3.13-slim

LABEL maintainer="Lyon. <lyonzin@users.noreply.github.com>"
LABEL description="Local RAG System for Claude Code — Hybrid search + Cross-encoder Reranking + 12 MCP Tools + 20 Format Parsers"
LABEL org.opencontainers.image.source="https://github.com/lyonzin/knowledge-rag"

WORKDIR /app

# Reproducible production install (v4.9.0):
#   1. hash-locked third-party deps from the CANONICAL requirements.lock
#      (requirements.txt is the loose dev/contribution list, never the
#      production install input);
#   2. the package itself from the EXACT wheel built by the release
#      workflow (downloaded python-dist artifact), --no-deps (the lock
#      already resolved every dependency — pip must not second-guess it);
#   3. `pip check` proves the resulting environment is consistent.
# The mcp_server source tree is deliberately NOT copied into /app: the
# runtime imports the INSTALLED package bytes, never a source shadow.
COPY requirements.lock ./requirements.lock
COPY dist/*.whl ./dist/

RUN pip install --no-cache-dir --require-hashes -r requirements.lock && \
    pip install --no-cache-dir --no-deps dist/*.whl && \
    pip check && \
    rm -rf dist

# Model artifacts are NOT pre-downloaded into this image. The embedding and
# reranker model directories are DEPLOYMENT INPUTS: the exact materialized
# artifact directories are pinned (models.embedding.artifact_path et al.) and
# their recursive SHA-256 is verified against the generation receipt at
# runtime. Nothing about unrelated generic bytes implies air-gapped model
# availability — mount the exact artifacts your generation was built with.

VOLUME ["/app/documents", "/app/data"]

ENTRYPOINT ["python", "-m", "mcp_server.server"]
