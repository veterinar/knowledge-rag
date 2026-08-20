"""Coverage for embedding profile + prefix support (v4.8.0 Fase 1).

Two concerns exercised here:

* **TestProfileResolution** — the profile resolver in
  ``Config.__post_init__`` correctly maps ``profile: "<name>"`` onto the
  matching ``_EMBEDDING_PROFILES`` entry, respects user overrides for
  prefix, warns and yields to profile when ``models.embedding.model`` is
  redeclared, and falls back to ``custom`` for unknown / non-string
  profile names.
* **TestPrefixApplication** — the ``FastEmbedEmbeddings`` pipeline
  prepends the right prefix per scope: ``passage_prefix`` for
  ``__call__`` and ``embed_documents``, ``query_prefix`` for
  ``embed_query``. Empty prefixes are pass-through (the input list is
  returned unchanged) so the default profile allocates zero new strings.

The Config tests rebuild the dataclass under a monkey-patched ``_yaml``
dict rather than mutating the process-global ``config`` instance so we
never leak state into other test modules.
"""

from __future__ import annotations

import importlib.metadata
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from mcp_server import config as cfg_module
from mcp_server.server import FastEmbedEmbeddings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rebuild_config(monkeypatch: pytest.MonkeyPatch, yaml_dict: dict) -> cfg_module.Config:
    """Return a fresh Config resolved against ``yaml_dict``.

    Monkey-patches ``mcp_server.config._yaml`` for the duration of the
    test — factory lambdas read the module-level name dynamically, so
    swapping the dict here reroutes all defaults.
    """
    monkeypatch.setattr(cfg_module, "_yaml", yaml_dict)
    return cfg_module.Config()


# ---------------------------------------------------------------------------
# TestProfileResolution
# ---------------------------------------------------------------------------


class TestProfileResolution:
    """Config.__post_init__ profile resolver — 5 cases."""

    def test_profile_custom_preserves_explicit_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: user opts out of profile shorthand and declares model + dim.
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "custom",
                    "model": "sentence-transformers/all-MiniLM-L12-v2",
                    "dimensions": 384,
                    "query_prefix": "",
                    "passage_prefix": "",
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)

        # Assert
        assert cfg.embedding_profile == "custom"
        assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L12-v2"
        assert cfg.embedding_dim == 384
        assert cfg.query_prefix == ""
        assert cfg.passage_prefix == ""

    def test_profile_multilingual_populates_e5_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: YAML only declares the profile — resolver fills the rest.
        yaml_dict = {"models": {"embedding": {"profile": "multilingual"}}}

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)

        # Assert
        assert cfg.embedding_profile == "multilingual"
        assert cfg.embedding_model == "intfloat/multilingual-e5-large"
        assert cfg.embedding_dim == 1024
        assert cfg.query_prefix == "query: "
        assert cfg.passage_prefix == "passage: "

    def test_profile_overrides_model_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange: user set BOTH a named profile AND an explicit model.
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "quality",
                    "model": "BAAI/bge-small-en-v1.5",  # would-be override, loses
                    "dimensions": 384,  # ditto
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)
        captured = capsys.readouterr()

        # Assert: profile wins, WARN was emitted, dim promoted to 1024
        assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
        assert cfg.embedding_dim == 1024
        assert "profile takes precedence" in captured.out

    def test_invalid_profile_falls_back_to_custom(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange: bogus profile name + explicit model that must be honored
        # after the fallback (custom leaves models.embedding.* alone).
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "does-not-exist",
                    "model": "BAAI/bge-small-en-v1.5",
                    "dimensions": 384,
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)
        captured = capsys.readouterr()

        # Assert: profile normalized to custom, WARN emitted, explicit model kept
        assert cfg.embedding_profile == "custom"
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.embedding_dim == 384
        assert "Invalid embedding profile" in captured.out
        assert "does-not-exist" in captured.out

    def test_user_prefix_overrides_profile_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: profile=multilingual ships "query: "/"passage: " but the
        # user prefers alternative sentinels for a fine-tuned e5 checkpoint.
        yaml_dict = {
            "models": {
                "embedding": {
                    "profile": "multilingual",
                    "query_prefix": "search_query: ",
                    "passage_prefix": "search_document: ",
                }
            }
        }

        # Act
        cfg = _rebuild_config(monkeypatch, yaml_dict)

        # Assert: model + dim still come from profile; prefixes stay user's
        assert cfg.embedding_model == "intfloat/multilingual-e5-large"
        assert cfg.embedding_dim == 1024
        assert cfg.query_prefix == "search_query: "
        assert cfg.passage_prefix == "search_document: "


# ---------------------------------------------------------------------------
# TestPrefixApplication
# ---------------------------------------------------------------------------


def _fake_embed_output(texts):
    """Yield deterministic 4D vectors so length + dim invariants hold."""
    for i, _ in enumerate(texts):
        yield np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32)


def _prepared_embedder(monkeypatch: pytest.MonkeyPatch) -> tuple[FastEmbedEmbeddings, MagicMock]:
    """Return a FastEmbedEmbeddings with load short-circuited and a spy model.

    The spy captures the ``texts`` argument passed to ``model.embed`` so
    the test can assert what actually reached the ONNX layer after any
    prefix massaging.
    """
    embedder = FastEmbedEmbeddings()
    embedder._dim = 4  # match the fake output
    spy = MagicMock()
    spy.embed.side_effect = lambda texts: _fake_embed_output(texts)
    embedder._model = spy
    # Bypass the lazy loader (model is already "loaded")
    monkeypatch.setattr(embedder, "_load_model", lambda: None)
    return embedder, spy


class TestPrefixApplication:
    """FastEmbedEmbeddings prefix wiring — 3 cases."""

    def test_passage_prefix_prepended_in_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(cfg_module.config, "passage_prefix", "passage: ")
        monkeypatch.setattr(cfg_module.config, "query_prefix", "query: ")
        embedder, spy = _prepared_embedder(monkeypatch)

        # Act — __call__ is the ChromaDB embedding_function entrypoint (passage)
        embedder(["mitre att&ck", "cve-2024-1234"])

        # Assert: model saw the passage-prefixed strings, not the raw ones
        spy.embed.assert_called_once()
        (called_texts,) = spy.embed.call_args.args
        assert called_texts == ["passage: mitre att&ck", "passage: cve-2024-1234"]

    def test_query_prefix_prepended_in_embed_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr(cfg_module.config, "passage_prefix", "passage: ")
        monkeypatch.setattr(cfg_module.config, "query_prefix", "query: ")
        embedder, spy = _prepared_embedder(monkeypatch)

        # Act
        embedder.embed_query("what is kerberoasting")

        # Assert: query prefix wins on this path, passage prefix never leaks
        spy.embed.assert_called_once()
        (called_texts,) = spy.embed.call_args.args
        assert called_texts == ["query: what is kerberoasting"]

    def test_empty_prefix_does_not_reallocate_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange: empty prefix must be a no-op — inputs pass through as-is.
        # This matters on the hot ingest path where the default "compact"
        # profile allocates zero new strings.
        monkeypatch.setattr(cfg_module.config, "passage_prefix", "")
        monkeypatch.setattr(cfg_module.config, "query_prefix", "")
        embedder, spy = _prepared_embedder(monkeypatch)
        raw = ["alpha", "beta", "gamma"]

        # Act
        embedder(raw)

        # Assert: model got the same list object (identity) because
        # ``_apply_prefix`` returns the input unchanged on empty prefix.
        spy.embed.assert_called_once()
        (called_texts,) = spy.embed.call_args.args
        assert called_texts is raw, "empty prefix must be identity, not a copy"


# ---------------------------------------------------------------------------
# Versioned runtime-contract resolver (registry-only; no model/network)
# Deterministic fakes: stub modules in ``sys.modules`` (the registry plus the
# two admitted exact implementation submodules, exposing the SAME class
# objects the registry lists) are inspected instead of the real ones —
# class-identity checks stay hermetic and no model is ever constructed or
# contacted.
# ---------------------------------------------------------------------------


class OnnxTextEmbedding:
    @staticmethod
    def _list_supported_models():
        return [
            SimpleNamespace(model="BAAI/bge-small-en-v1.5", dim=384),
            SimpleNamespace(model="BAAI/bge-large-en-v1.5", dim=1024),
            # Non-project exact Onnx model: registered but NOT admitted.
            SimpleNamespace(model="BAAI/bge-base-en-v1.5", dim=768),
        ]


class PooledEmbedding:
    @staticmethod
    def _list_supported_models():
        return [SimpleNamespace(model="intfloat/multilingual-e5-large", dim=1024)]


class PooledNormalizedEmbedding:
    @staticmethod
    def _list_supported_models():
        return [SimpleNamespace(model="intfloat/multilingual-e5-base", dim=768)]


class CustomTextEmbedding:
    @staticmethod
    def _list_supported_models():
        return [SimpleNamespace(model="org/custom-model", dim=64)]


def _install_fake_fastembed(monkeypatch: pytest.MonkeyPatch, registry) -> None:
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


def _resolver(monkeypatch: pytest.MonkeyPatch, registry):
    _install_fake_fastembed(monkeypatch, registry)
    return cfg_module.Config.__new__(cfg_module.Config)


def _resolve_contract(resolver, model: str) -> dict:
    # Declared values match the allowlist expectation so only the registry
    # side determines admission in rejection tests.
    if model == "BAAI/bge-small-en-v1.5":
        resolver.embedding_model, resolver.embedding_dim = model, 384
        resolver.embedding_runtime_version, resolver.embedding_pooling = "fastembed 0.8.0", "cls-or-prepooled"
    elif model == "BAAI/bge-large-en-v1.5":
        resolver.embedding_model, resolver.embedding_dim = model, 1024
        resolver.embedding_runtime_version, resolver.embedding_pooling = "fastembed 0.8.0", "cls-or-prepooled"
    elif model == "intfloat/multilingual-e5-large":
        resolver.embedding_model, resolver.embedding_dim = model, 1024
        resolver.embedding_runtime_version, resolver.embedding_pooling = "fastembed 0.8.0", "mean"
    else:
        resolver.embedding_model = model
        resolver.embedding_dim, resolver.embedding_runtime_version = 384, "fastembed 0.8.0"
        resolver.embedding_pooling = "cls-or-prepooled"
    return resolver._resolve_embedding_runtime_contract(model)


@pytest.mark.parametrize(
    "model,dim,pooling",
    [
        ("BAAI/bge-small-en-v1.5", 384, "cls-or-prepooled"),
        ("BAAI/bge-large-en-v1.5", 1024, "cls-or-prepooled"),
        ("intfloat/multilingual-e5-large", 1024, "mean"),
    ],
)
def test_builtin_actual_contracts_from_registry(monkeypatch: pytest.MonkeyPatch, model, dim, pooling) -> None:
    """The three built-ins resolve to their exact allowlist contracts."""
    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])
    contract = _resolve_contract(resolver, model)
    assert contract == {
        "embedding_model": model,
        "runtime_version": "fastembed 0.8.0",
        "embedding_dim": dim,
        "pooling": pooling,
    }


def test_model_match_is_case_insensitive_returning_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])
    contract = _resolve_contract(resolver, "baai/bge-small-en-v1.5")
    assert contract["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert contract["embedding_dim"] == 384
    assert contract["pooling"] == "cls-or-prepooled"


def test_non_project_onnx_model_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact OnnxTextEmbedding model outside the allowlist fails closed."""
    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="not admitted"):
        _resolve_contract(resolver, "BAAI/bge-base-en-v1.5")


def test_custom_text_embedding_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """CustomTextEmbedding is not admitted even for its own registered model."""
    resolver = _resolver(monkeypatch, [CustomTextEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="not admitted"):
        _resolve_contract(resolver, "org/custom-model")


def test_pooled_normalized_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """PooledNormalizedEmbedding models are not admitted in versioned mode."""
    resolver = _resolver(monkeypatch, [PooledNormalizedEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="not admitted"):
        _resolve_contract(resolver, "intfloat/multilingual-e5-base")


def test_wrong_exact_class_for_allowed_name_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowed name registered by the WRONG exact class fails closed."""

    class OtherOnnxTextEmbedding:  # exact name, different class object
        @staticmethod
        def _list_supported_models():
            return [SimpleNamespace(model="BAAI/bge-small-en-v1.5", dim=384)]

    resolver = _resolver(monkeypatch, [OtherOnnxTextEmbedding, PooledEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="exact.*OnnxTextEmbedding"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_same_name_lookalike_for_allowed_name_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-name lookalike class is not admitted for an ALLOWED model."""

    class OnnxTextEmbedding:  # noqa: A001 - deliberately shadows for identity proof
        @staticmethod
        def _list_supported_models():
            return [SimpleNamespace(model="BAAI/bge-small-en-v1.5", dim=384)]

    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="exact-class identity"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_registry_dimension_drift_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowed model whose REGISTERED dim drifted fails closed."""

    class DriftedOnnx:
        __name__ = "OnnxTextEmbedding"

        @staticmethod
        def _list_supported_models():
            return [SimpleNamespace(model="BAAI/bge-small-en-v1.5", dim=999)]

    resolver = _resolver(monkeypatch, [DriftedOnnx, PooledEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="requires exactly 384"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_duplicate_registration_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class PooledEmbeddingToo:
        @staticmethod
        def _list_supported_models():
            return [SimpleNamespace(model="BAAI/bge-small-en-v1.5", dim=384)]

    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbeddingToo])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="multiple fastembed implementations"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_non_canonical_registered_spelling_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class RenamedOnnx:
        __name__ = "OnnxTextEmbedding"

        @staticmethod
        def _list_supported_models():
            return [SimpleNamespace(model="baai/bge-small-en-v1.5", dim=384)]  # non-canonical case

    resolver = _resolver(monkeypatch, [RenamedOnnx, PooledEmbedding])
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="non-canonical spelling"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_mapping_registry_shape_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = _resolver(monkeypatch, {"OnnxTextEmbedding": OnnxTextEmbedding})
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="dispatch-ordered list"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_missing_distribution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])
    _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")  # baseline resolves

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    with pytest.raises(
        cfg_module.EmbeddingRuntimeContractError, match=r"fastembed \(CPU\) distribution metadata unavailable"
    ):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_both_fastembed_and_fastembed_gpu_installed_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shadowing GPU distribution must never be reported as CPU fastembed."""
    resolver = _resolver(monkeypatch, [OnnxTextEmbedding, PooledEmbedding])

    def _both_installed(name):
        if name in ("fastembed", "fastembed-gpu"):
            return "0.8.0"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _both_installed)
    with pytest.raises(cfg_module.EmbeddingRuntimeContractError, match="fastembed-gpu"):
        _resolve_contract(resolver, "BAAI/bge-small-en-v1.5")


def test_support_allowlist_agrees_with_embedding_profiles() -> None:
    """The runtime-contract allowlist must mirror _EMBEDDING_PROFILES exactly."""
    profiles = {
        str(profile["model"]): profile["dimensions"]
        for name, profile in cfg_module._EMBEDDING_PROFILES.items()
        if name != "custom"
    }
    allowlist = {str(entry["model"]): entry["dim"] for entry in cfg_module._EMBEDDING_RUNTIME_CONTRACTS.values()}
    assert allowlist == profiles


def test_real_installed_registry_canary_for_three_builtins() -> None:
    """Registry-only canary against the ACTUAL installed FastEmbed.

    Walks the real ``TextEmbedding.EMBEDDINGS_REGISTRY`` metadata only —
    no TextEmbedding construction, no model download, no network — and
    asserts each admitted built-in is registered by the expected exact
    implementation class with the expected dimension.
    """
    from fastembed import TextEmbedding
    from fastembed.text.onnx_embedding import OnnxTextEmbedding
    from fastembed.text.pooled_embedding import PooledEmbedding

    expected_classes = {
        "BAAI/bge-small-en-v1.5": (OnnxTextEmbedding, 384),
        "BAAI/bge-large-en-v1.5": (OnnxTextEmbedding, 1024),
        "intfloat/multilingual-e5-large": (PooledEmbedding, 1024),
    }
    found: dict = {}
    for impl in TextEmbedding.EMBEDDINGS_REGISTRY:
        for description in impl._list_supported_models():
            name = str(getattr(description, "model", ""))
            if name in expected_classes:
                found[name] = (impl, getattr(description, "dim", None))
    for name, (cls, dim) in expected_classes.items():
        assert name in found, f"{name} missing from the installed fastembed registry"
        assert found[name][0] is cls, f"{name} registered by unexpected implementation {found[name][0]!r}"
        assert found[name][1] == dim, f"{name} registered dimension drifted: {found[name][1]!r}"
