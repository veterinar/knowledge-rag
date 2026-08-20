"""Tests for configuration integrity."""

import importlib.metadata
import sys
from types import SimpleNamespace

import pytest

import mcp_server.config as config_module
from mcp_server.config import _merge_query_expansion_sources, config


def test_no_ollama_references():
    """Config must not reference Ollama (removed in v3.0)."""
    assert not hasattr(config, "ollama_model")
    assert not hasattr(config, "ollama_base_url")


def test_embedding_model():
    """FastEmbed model must be configured."""
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.embedding_dim == 384


def test_reranker_config():
    """Reranker must be configured and enabled."""
    assert "ms-marco" in config.reranker_model
    assert config.reranker_enabled is True
    assert config.reranker_top_k_multiplier >= 2


def test_supported_formats():
    """Core formats must be present in supported_formats."""
    core = {".md", ".txt", ".pdf"}
    assert core.issubset(set(config.supported_formats))


def test_query_expansions_count():
    """Must have 50+ query expansion terms."""
    assert len(config.query_expansions) >= 50


def test_query_expansion_security_terms():
    """Key security terms must have expansions."""
    must_have = ["sqli", "xss", "privesc", "amsi", "suid", "kerberoast"]
    for term in must_have:
        assert term in config.query_expansions, f"Missing expansion for: {term}"


def test_cve_aliases():
    """CVE aliases must be present."""
    must_have = ["printnightmare", "eternalblue", "pwnkit", "log4shell", "zerologon"]
    for term in must_have:
        assert term in config.query_expansions, f"Missing CVE alias for: {term}"


def test_category_mappings():
    """Essential categories must exist."""
    must_have = ["security", "ctf", "logscale", "development", "general", "aar"]
    for cat in must_have:
        found = any(cat in v for v in config.category_mappings.values())
        assert found, f"Missing category mapping for: {cat}"


def test_chunk_settings():
    """Chunk settings must be reasonable."""
    assert 500 <= config.chunk_size <= 2000
    assert 100 <= config.chunk_overlap <= 500
    assert config.chunk_overlap < config.chunk_size


# ── v3.4.0 Features ──


def test_models_cache_dir_exists():
    """models_cache_dir must be a Path and directory must be created."""
    from pathlib import Path

    assert hasattr(config, "models_cache_dir")
    assert isinstance(config.models_cache_dir, Path)
    assert config.models_cache_dir.exists()


def test_venv_project_dir_uses_unresolved_executable(monkeypatch):
    """Venv detection must survive python symlinks that resolve to system Python."""
    from pathlib import Path

    import mcp_server.config as config_module

    monkeypatch.setattr(config_module.sys, "prefix", "/usr")
    monkeypatch.setattr(config_module.sys, "executable", "/opt/knowledge-rag/venv/bin/python")

    assert config_module._venv_project_dir() == Path("/opt/knowledge-rag")


def test_exclude_patterns_default_empty():
    """Default exclude_patterns must be an empty list."""
    assert hasattr(config, "exclude_patterns")
    assert isinstance(config.exclude_patterns, list)


def test_ipynb_in_supported_suffixes():
    """.ipynb must be in the internal supported suffixes set."""
    from mcp_server.config import _SUPPORTED_SUFFIXES

    assert ".ipynb" in _SUPPORTED_SUFFIXES


def test_new_code_formats_in_supported_suffixes():
    """New code formats must be in _SUPPORTED_SUFFIXES for directory detection."""
    from mcp_server.config import _SUPPORTED_SUFFIXES

    for ext in [".c", ".h", ".cpp", ".js", ".jsx", ".ts", ".tsx", ".xml"]:
        assert ext in _SUPPORTED_SUFFIXES, f"{ext} missing from _SUPPORTED_SUFFIXES"


def test_new_code_formats_default_enabled():
    """New code formats must be in default supported_formats (not opt-in)."""
    for ext in [".c", ".h", ".cpp", ".js", ".jsx", ".ts", ".tsx", ".xml"]:
        assert ext in config.supported_formats, f"{ext} missing from supported_formats defaults"


def test_query_expansion_groups_are_symmetric():
    """A synonym group must generate reciprocal expansions for every member."""
    merged = _merge_query_expansion_sources({}, [["metatrader 4", "mt4", "mql4"]])

    assert merged["metatrader 4"] == ["mt4", "mql4"]
    assert merged["mt4"] == ["metatrader 4", "mql4"]
    assert merged["mql4"] == ["metatrader 4", "mt4"]


def test_query_expansion_groups_extend_legacy_entries():
    """Grouped synonyms must extend, not replace, legacy directional mappings."""
    merged = _merge_query_expansion_sources(
        {"tb": ["triple barrier", "trip_barr", "legacy_alias"]},
        [["triple barrier", "tb", "trip_barr"]],
    )

    assert merged["tb"] == ["triple barrier", "trip_barr", "legacy_alias"]
    assert "tb" in merged["triple barrier"]
    assert "trip_barr" in merged["triple barrier"]


# ── Versioned-mode registry-verified embedding runtime contract ──────────────
# Deterministic fakes: the resolver imports ``fastembed`` lazily, so stub
# modules in ``sys.modules`` (the registry plus the two admitted exact
# implementation submodules, exposing the SAME class objects the registry
# lists) are inspected instead of the real ones. No model is ever
# constructed, downloaded, or contacted.


class OnnxTextEmbedding:
    @staticmethod
    def _list_supported_models():
        return [
            SimpleNamespace(model="BAAI/bge-small-en-v1.5", dim=384),
            SimpleNamespace(model="BAAI/bge-large-en-v1.5", dim=1024),
        ]


class PooledEmbedding:
    @staticmethod
    def _list_supported_models():
        return [SimpleNamespace(model="intfloat/multilingual-e5-large", dim=1024)]


def _install_fake_fastembed(monkeypatch, registry):
    for name, module in {
        "fastembed": SimpleNamespace(TextEmbedding=SimpleNamespace(EMBEDDINGS_REGISTRY=registry)),
        "fastembed.text.onnx_embedding": SimpleNamespace(OnnxTextEmbedding=OnnxTextEmbedding),
        "fastembed.text.pooled_embedding": SimpleNamespace(PooledEmbedding=PooledEmbedding),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    real_version = importlib.metadata.version

    def _fake_version(name):
        if name == "fastembed":
            return "0.8.0"
        if name == "fastembed-gpu":
            raise importlib.metadata.PackageNotFoundError("fastembed-gpu")
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", _fake_version)


def _versioned_yaml(tmp_path, *, model, dimensions, runtime_version, pooling):
    artifact = tmp_path / "artifact"
    artifact.mkdir(exist_ok=True)
    return {
        "indexing": {"mode": "versioned"},
        "models": {
            "embedding": {
                "model": model,
                "dimensions": dimensions,
                "runtime_version": runtime_version,
                "pooling": pooling,
                "artifact_path": str(artifact),
            }
        },
    }


def _build_config(monkeypatch, tmp_path, yaml_payload, registry=None):
    _install_fake_fastembed(monkeypatch, registry if registry is not None else [OnnxTextEmbedding, PooledEmbedding])
    monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config_module, "_yaml", yaml_payload, raising=False)
    return config_module.Config()


@pytest.mark.parametrize(
    "model,dim,pooling",
    [
        ("BAAI/bge-small-en-v1.5", 384, "cls-or-prepooled"),
        ("BAAI/bge-large-en-v1.5", 1024, "cls-or-prepooled"),
        ("intfloat/multilingual-e5-large", 1024, "mean"),
    ],
)
def test_versioned_matching_contract_passes_and_canonicalizes(monkeypatch, tmp_path, model, dim, pooling):
    """Matching versioned config is admitted; canonical actuals are stored."""
    cfg = _build_config(
        monkeypatch,
        tmp_path,
        _versioned_yaml(
            tmp_path,
            model=model,
            dimensions=dim,
            runtime_version="fastembed 0.8.0",
            pooling=f"  {pooling.upper()}  ",  # case/whitespace tolerated, canonicalized
        ),
    )
    assert cfg.index_mode == "versioned"
    assert cfg._embedding_runtime_contract_error is None
    assert cfg.embedding_model == model
    assert cfg.embedding_runtime_version == "fastembed 0.8.0"
    assert cfg.embedding_dim == dim
    assert cfg.embedding_pooling == pooling
    # Compatibility is built from the freshly returned canonical actuals.
    compat = cfg.generation_compatibility()
    assert compat["embedding_model"] == model
    assert compat["embedding_dimension"] == dim
    assert compat["runtime_version"] == "fastembed 0.8.0"
    assert compat["pooling"] == pooling


def test_versioned_wrong_runtime_version_survives_but_compatibility_fails(monkeypatch, tmp_path):
    payload = _versioned_yaml(
        tmp_path,
        model="BAAI/bge-small-en-v1.5",
        dimensions=384,
        runtime_version="fastembed 0.3.6",
        pooling="cls-or-prepooled",
    )
    cfg = _build_config(monkeypatch, tmp_path, payload)
    # Config construction (and module import) survives the drift...
    assert cfg.index_mode == "versioned"
    assert isinstance(cfg._embedding_runtime_contract_error, str)
    assert "runtime_version" in cfg._embedding_runtime_contract_error
    # ...while generation compatibility hard-fails with the typed error.
    with pytest.raises(config_module.EmbeddingRuntimeContractError, match="runtime_version"):
        cfg.generation_compatibility()


def test_versioned_wrong_dimension_survives_but_compatibility_fails(monkeypatch, tmp_path):
    payload = _versioned_yaml(
        tmp_path,
        model="BAAI/bge-small-en-v1.5",
        dimensions=768,
        runtime_version="fastembed 0.8.0",
        pooling="cls-or-prepooled",
    )
    cfg = _build_config(monkeypatch, tmp_path, payload)
    assert cfg._embedding_runtime_contract_error is not None
    with pytest.raises(config_module.EmbeddingRuntimeContractError, match="dimensions"):
        cfg.generation_compatibility()


def test_versioned_wrong_pooling_survives_but_compatibility_fails(monkeypatch, tmp_path):
    payload = _versioned_yaml(
        tmp_path,
        model="intfloat/multilingual-e5-large",
        dimensions=1024,
        runtime_version="fastembed 0.8.0",
        pooling="cls",
    )
    cfg = _build_config(monkeypatch, tmp_path, payload)
    assert cfg._embedding_runtime_contract_error is not None
    with pytest.raises(config_module.EmbeddingRuntimeContractError, match="pooling"):
        cfg.generation_compatibility()


def test_versioned_unregistered_model_survives_but_compatibility_fails(monkeypatch, tmp_path):
    payload = _versioned_yaml(
        tmp_path,
        model="org/never-registered",
        dimensions=384,
        runtime_version="fastembed 0.8.0",
        pooling="cls",
    )
    cfg = _build_config(monkeypatch, tmp_path, payload)
    assert cfg._embedding_runtime_contract_error is not None
    with pytest.raises(config_module.EmbeddingRuntimeContractError):
        cfg.generation_compatibility()


def test_legacy_mode_never_calls_the_resolver(monkeypatch, tmp_path):
    """Legacy config never resolves the runtime contract (imports unchanged)."""
    calls: list = []

    def _explode(self, *args, **kwargs):
        calls.append(1)
        raise AssertionError("resolver must never be called in legacy mode")

    monkeypatch.setattr(config_module.Config, "_resolve_embedding_runtime_contract", _explode, raising=True)
    monkeypatch.setattr(config_module.Config, "require_embedding_runtime_contract", _explode, raising=True)
    monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config_module, "_yaml", {"indexing": {"mode": "legacy"}}, raising=False)
    cfg = config_module.Config()
    assert cfg.index_mode == "legacy"
    assert cfg._embedding_runtime_contract_error is None
    assert cfg.embedding_runtime_version == ""
    assert calls == []


def test_static_versioned_misconfiguration_still_hard_fails(monkeypatch, tmp_path):
    """Missing declared runtime (a STATIC gap) still aborts Config()."""
    _install_fake_fastembed(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])
    artifact = tmp_path / "artifact"
    artifact.mkdir(exist_ok=True)
    monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(
        config_module,
        "_yaml",
        {
            "indexing": {"mode": "versioned"},
            "models": {
                "embedding": {
                    "model": "BAAI/bge-small-en-v1.5",
                    "dimensions": 384,
                    "pooling": "cls-or-prepooled",
                    "artifact_path": str(artifact),
                }
            },
        },
        raising=False,
    )
    with pytest.raises(ValueError, match="runtime_version"):
        config_module.Config()
