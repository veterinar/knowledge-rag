"""Focused regression for the offline retrieval boundary.

Proves the already implemented behavior described by
docs/criteria-offline-retrieval-boundary.md, acceptance OB-1..OB-5:

- OB-1: admission seam only — with an admitted versioned artifact the
  TextCrossEncoder constructor receives the exact resolved
  specific_model_path and local_files_only=True (digest-gated via the real
  gate inputs; a digest mismatch blocks the constructor). This does NOT
  prove the complete sealed-generation chain.
- OB-2: versioned enabled reranker without an admitted artifact never calls
  TextCrossEncoder and fails closed (RerankerUnavailableError, never silent RRF).
- OB-3: constructor failure in enabled versioned mode raises the typed
  RerankerUnavailableError on every subsequent rerank call of the same
  instance (constructor not retried) instead of silently degrading to RRF.
- OB-4: expand_query is a pure in-process transformation: socket, subprocess,
  file and model entry points stay untouched; output derives only from the
  input query and the frozen config.query_expansions mapping.
- OB-5: only spies/mocks and temporary local directories; no downloads, no
  network, no real model instantiation, no live-generation mutation.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def _versioned(monkeypatch, tmp_path=None):
    """Put the module config into versioned mode with the reranker enabled."""
    from mcp_server import server as srv

    monkeypatch.setattr(srv.config, "index_mode", "versioned")
    monkeypatch.setattr(srv.config, "reranker_enabled", True)
    monkeypatch.setattr(
        srv.config, "reranker_model", "Xenova/ms-marco-MiniLM-L-6-v2"
    )
    monkeypatch.setattr(srv.config, "models_cache_dir", str(tmp_path or "."))
    return srv


def _admit_artifact(monkeypatch, srv, tmp_path):
    """Create a local artifact directory and pin it via the real gate inputs."""
    from mcp_server import generations as gens

    artifact = tmp_path / "reranker-artifact"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"\x00\x01offline-reranker")
    digest, _ = gens.digest_tree(artifact)
    monkeypatch.setattr(srv.config, "reranker_local_artifact", str(artifact))
    monkeypatch.setattr(
        srv.config,
        "generation_compatibility",
        lambda: {"reranker_artifact_sha256": digest},
    )
    return artifact


def test_ob1_admitted_artifact_pins_exact_path_and_local_only(
    monkeypatch, tmp_path
):
    """OB-1 (admission seam only): the production TextCrossEncoder
    constructor receives the exact resolved specific_model_path and
    local_files_only=True for an admitted artifact whose tree digest
    matches the pinned identity. This proves the admission seam (path,
    digest gate inputs, offline constructor flags) — NOT the full sealed
    generation chain, which is out of scope here."""
    srv = _versioned(monkeypatch, tmp_path)
    artifact = _admit_artifact(monkeypatch, srv, tmp_path)

    docs = [{"document": "first"}, {"document": "second"}]
    reranker = srv.CrossEncoderReranker()

    with patch("mcp_server.server.TextCrossEncoder") as constructor:
        reranker.rerank("query", docs, top_k=2)

    assert constructor.call_count == 1
    kwargs = constructor.call_args.kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["specific_model_path"] == str(Path(str(artifact)).resolve())


def test_ob1_digest_mismatch_blocks_constructor(monkeypatch, tmp_path):
    """OB-1 (negative, real gate): an artifact whose tree digest does NOT
    match the pinned reranker_artifact_sha256 is not admitted — the gate
    returns False and the constructor is never invoked; enabled versioned
    retrieval fails closed with the typed error instead."""
    srv = _versioned(monkeypatch, tmp_path)
    _admit_artifact(monkeypatch, srv, tmp_path)
    # Pin a digest of different content: same gate, mismatching identity.
    other = tmp_path / "other-artifact"
    other.mkdir()
    (other / "model.onnx").write_bytes(b"\xffdifferent-bytes")
    from mcp_server import generations as gens

    wrong_digest, _ = gens.digest_tree(other)
    monkeypatch.setattr(
        srv.config,
        "generation_compatibility",
        lambda: {"reranker_artifact_sha256": wrong_digest},
    )

    docs = [{"document": "first", "rrf_score": 0.5}]
    reranker = srv.CrossEncoderReranker()

    with patch("mcp_server.server.TextCrossEncoder") as constructor:
        with pytest.raises(srv.RerankerUnavailableError):
            reranker.rerank("query", docs, top_k=1)

    constructor.assert_not_called()


def test_ob2_enabled_without_admitted_artifact_never_loads_model(monkeypatch, tmp_path):
    """OB-2: enabled but no admitted artifact -> no constructor call, fail closed."""
    srv = _versioned(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.config, "reranker_local_artifact", None)

    docs = [{"document": "first", "rrf_score": 0.5}]
    reranker = srv.CrossEncoderReranker()

    with patch("mcp_server.server.TextCrossEncoder") as constructor:
        with pytest.raises(srv.RerankerUnavailableError):
            reranker.rerank("query", docs, top_k=1)

    constructor.assert_not_called()


def test_ob3_load_failure_raises_typed_error_not_rrf(monkeypatch, tmp_path):
    """OB-3: constructor failure in enabled versioned mode fails closed,
    typed, on EVERY subsequent rerank call of the same instance — the
    constructor itself is never retried after the first failure."""
    srv = _versioned(monkeypatch, tmp_path)
    _admit_artifact(monkeypatch, srv, tmp_path)

    docs = [{"document": "first", "rrf_score": 0.5}]
    reranker = srv.CrossEncoderReranker()

    with patch(
        "mcp_server.server.TextCrossEncoder", side_effect=RuntimeError("offline")
    ) as constructor:
        with pytest.raises(srv.RerankerUnavailableError, match="load failed"):
            reranker.rerank("query", docs, top_k=1)
        # Same instance, second call: still the typed fail-closed error,
        # never silent RRF order via the sticky flag.
        with pytest.raises(srv.RerankerUnavailableError, match="load failed"):
            reranker.rerank("query", docs, top_k=1)

    assert reranker._load_failed is True
    # The constructor ran exactly once (first call); it was not retried.
    assert constructor.call_count == 1


def test_ob4_expand_query_is_pure_and_local(monkeypatch):
    """OB-4: expand_query touches no socket/subprocess/file/model and is deterministic."""
    from mcp_server.server import BM25Index

    frozen = {
        "sqli": ["sql injection"],
        "kerberos ticket": ["kerberoasting"],
    }
    monkeypatch.setattr(
        "mcp_server.server.config.query_expansions", frozen, raising=False
    )

    index = BM25Index()
    with patch("socket.socket") as sock, patch(
        "socket.create_connection"
    ) as connect, patch("subprocess.Popen") as popen, patch(
        "subprocess.run"
    ) as run, patch("builtins.open") as fopen, patch(
        "mcp_server.server.TextEmbedding"
    ) as embed, patch(
        "mcp_server.server.TextCrossEncoder"
    ) as rerank_model:
        result = index.expand_query("SQLI Kerberos Ticket")
        # Second call inside the spy window proves determinism with every
        # forbidden I/O entry point still patched.
        again = index.expand_query("SQLI Kerberos Ticket")

    sock.assert_not_called()
    connect.assert_not_called()
    popen.assert_not_called()
    run.assert_not_called()
    fopen.assert_not_called()
    embed.assert_not_called()
    rerank_model.assert_not_called()

    # Deterministic derivation from input + frozen mapping only: the token
    # "sqli" expands via token lookup, the bigram "kerberos ticket" via
    # bigram lookup; order and dedup are fixed.
    assert result == "sqli kerberos ticket sql injection kerberoasting"
    assert again == result
