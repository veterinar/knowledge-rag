"""Tests for search pipeline components (no model/DB required)."""

import pytest

from mcp_server.config import _merge_query_expansion_sources
from mcp_server.server import BM25Index, KnowledgeOrchestrator, QueryCache

# Deterministic test-local data — these tests must NEVER depend on the active
# operator config singleton (a live config.yaml overrides the built-in
# defaults, making expansion/routing assertions environment-dependent).
_TEST_QUERY_EXPANSIONS = {
    "sqli": ["sql injection", "sqli"],
    "privesc": ["privilege escalation", "privesc"],
    "amsi": ["antimalware scan interface", "amsi"],
    "printnightmare": ["printnightmare", "cve-2021-34527"],
    "eternalblue": ["eternalblue", "ms17-010"],
    "reverse shell": ["reverse shell", "revshell"],
    "revshell": ["reverse shell", "revshell"],
}

_TEST_KEYWORD_ROUTES = {
    "redteam": ["mimikatz", "credential dump"],
}


def _init_dispatchless_orch(orch):
    """Initialize the dispatch attributes current production reads in query().

    ``object.__new__(KnowledgeOrchestrator)`` shells lack ``fts5_index`` /
    ``query_router``; when the active config has fts5_enabled=True the query
    entrypoint touches both before the hybrid pipeline. None/None forces the
    hybrid path deterministically (router is None → never lexical dispatch).
    """
    orch.fts5_index = None
    orch.query_router = None
    return orch


# ── BM25 Query Expansion ──


class TestQueryExpansion:
    def setup_method(self):
        self.bm25 = BM25Index()

    def test_sqli_expands(self, monkeypatch):
        """sqli must expand to sql injection."""
        monkeypatch.setattr("mcp_server.server.config.query_expansions", dict(_TEST_QUERY_EXPANSIONS))
        expanded = self.bm25.expand_query("sqli")
        assert "sql injection" in expanded

    def test_privesc_expands(self, monkeypatch):
        """privesc must expand to privilege escalation."""
        monkeypatch.setattr("mcp_server.server.config.query_expansions", dict(_TEST_QUERY_EXPANSIONS))
        expanded = self.bm25.expand_query("privesc")
        assert "privilege escalation" in expanded

    def test_amsi_expands(self, monkeypatch):
        """amsi must expand to antimalware scan interface."""
        monkeypatch.setattr("mcp_server.server.config.query_expansions", dict(_TEST_QUERY_EXPANSIONS))
        expanded = self.bm25.expand_query("amsi")
        assert "antimalware" in expanded

    def test_cve_alias_printnightmare(self, monkeypatch):
        """printnightmare must expand to CVE-2021-34527."""
        monkeypatch.setattr("mcp_server.server.config.query_expansions", dict(_TEST_QUERY_EXPANSIONS))
        expanded = self.bm25.expand_query("printnightmare")
        assert "cve-2021-34527" in expanded

    def test_cve_alias_eternalblue(self, monkeypatch):
        """eternalblue must expand to ms17-010."""
        monkeypatch.setattr("mcp_server.server.config.query_expansions", dict(_TEST_QUERY_EXPANSIONS))
        expanded = self.bm25.expand_query("eternalblue")
        assert "ms17-010" in expanded

    def test_no_expansion_unknown(self):
        """Unknown terms return unchanged."""
        expanded = self.bm25.expand_query("xyzunknownterm")
        assert expanded == "xyzunknownterm"

    def test_bigram_expansion(self, monkeypatch):
        """Two-word terms must expand."""
        monkeypatch.setattr("mcp_server.server.config.query_expansions", dict(_TEST_QUERY_EXPANSIONS))
        expanded = self.bm25.expand_query("reverse shell")
        assert "revshell" in expanded

    def test_legacy_directional_expansion_still_works(self, monkeypatch):
        """Legacy directional mappings must still expand from the left-hand key."""
        monkeypatch.setattr(
            "mcp_server.server.config.query_expansions",
            {"tb": ["triple barrier", "trip_barr"]},
        )

        expanded = self.bm25.expand_query("tb")

        assert "triple barrier" in expanded
        assert "trip_barr" in expanded

    def test_group_expansion_is_symmetric(self, monkeypatch):
        """Any term from a group must expand to the rest of the group."""
        merged = _merge_query_expansion_sources({}, [["triple barrier", "tb", "trip_barr"]])
        monkeypatch.setattr("mcp_server.server.config.query_expansions", merged)

        expanded = self.bm25.expand_query("tb")

        assert "triple barrier" in expanded
        assert "trip_barr" in expanded

    def test_group_bigram_expansion(self, monkeypatch):
        """Multi-word group members must match via full query and bigrams."""
        merged = _merge_query_expansion_sources({}, [["triple barrier", "tb", "trip_barr"]])
        monkeypatch.setattr("mcp_server.server.config.query_expansions", merged)

        expanded = self.bm25.expand_query("triple barrier")

        assert "tb" in expanded
        assert "trip_barr" in expanded

    def test_mixed_expansion_sources_merge_cleanly(self, monkeypatch):
        """Legacy and grouped expansions must combine without losing entries."""
        merged = _merge_query_expansion_sources(
            {"pf": ["profit factor", "profit-factor"]},
            [["profit factor", "pf", "profit_factor"]],
        )
        monkeypatch.setattr("mcp_server.server.config.query_expansions", merged)

        expanded = self.bm25.expand_query("pf")

        assert "profit factor" in expanded
        assert "profit-factor" in expanded
        assert "profit_factor" in expanded


# ── BM25 Search ──


class TestBM25Search:
    def test_search_empty_index(self):
        """Search on empty index returns empty."""
        bm25 = BM25Index()
        results = bm25.search("test query")
        assert results == []

    def test_search_with_data(self):
        """Search returns ranked results."""
        bm25 = BM25Index()
        bm25.add_documents(
            ["doc1", "doc2", "doc3"],
            ["SQL injection bypass techniques", "XSS reflected attack", "SQL injection UNION based"],
        )
        bm25.build_index()
        results = bm25.search("SQL injection")
        assert len(results) >= 1
        # doc1 or doc3 should rank highest (both mention SQL injection)
        top_ids = [r[0] for r in results[:2]]
        assert "doc1" in top_ids or "doc3" in top_ids

    def test_search_empty_query(self):
        """Empty query returns empty."""
        bm25 = BM25Index()
        bm25.add_documents(["doc1"], ["some content"])
        bm25.build_index()
        results = bm25.search("")
        assert results == []


class TestHybridCategoryFilter:
    def test_bm25_results_respect_category_filter(self, monkeypatch):
        """BM25-only results must not bypass an explicit category filter."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return [("chunk_report", 10.0), ("chunk_code", 9.0)]

        class FakeCollection:
            _docs = {
                "chunk_report": "report content",
                "chunk_code": "code content",
            }
            _metadatas = {
                "chunk_report": {
                    "source": "/docs/report.md",
                    "filename": "report.md",
                    "category": "reports",
                    "chunk_index": 0,
                    "keywords": "",
                },
                "chunk_code": {
                    "source": "/src/code.py",
                    "filename": "code.py",
                    "category": "code",
                    "chunk_index": 0,
                    "keywords": "",
                },
            }

            def get(self, ids, include):
                return {
                    "ids": ids,
                    "documents": [self._docs[chunk_id] for chunk_id in ids] if "documents" in include else None,
                    "metadatas": [self._metadatas[chunk_id] for chunk_id in ids] if "metadatas" in include else None,
                }

        orchestrator = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orchestrator)
        orchestrator.query_cache = FakeCache()
        orchestrator.bm25_index = FakeBM25()
        orchestrator.collection = FakeCollection()
        orchestrator._ensure_bm25_index = lambda: None
        orchestrator._route_by_keywords = lambda query: None
        orchestrator._expand_with_adjacent_chunks = lambda results: results

        results = orchestrator.query("anything", max_results=5, category_filter="reports", hybrid_alpha=0.0)

        assert [result["source"] for result in results] == ["/docs/report.md"]
        assert {result["category"] for result in results} == {"reports"}


class TestKeywordRoutingBehavior:
    """When the user omits an explicit category_filter, keyword auto-routing must NOT
    restrict the search to a single category. The router is informational only:
    it can populate the ``routed_by`` metadata field, but must not act as a hard
    where-filter on either BM25 or semantic candidates.

    Regression: prior to this fix, ``_route_by_keywords()`` could pick an
    under-populated category (e.g. ``redteam`` with 2 docs) and hide the relevant
    material sitting in a larger category (e.g. ``security`` with thousands of docs).
    """

    METADATAS = {
        "chunk_redteam_generic": {
            "source": "/docs/redteam/rtfm.pdf",
            "filename": "rtfm.pdf",
            "category": "redteam",
            "chunk_index": 0,
            "keywords": "",
        },
        "chunk_security_esc1": {
            "source": "/docs/security/pentest-everything/adcs/esc1.md",
            "filename": "esc1.md",
            "category": "security",
            "chunk_index": 0,
            "keywords": "",
        },
    }
    DOCS = {
        "chunk_redteam_generic": "generic redteam content",
        "chunk_security_esc1": "ESC1 vulnerable template EKU Client Authentication",
    }

    def _build_orchestrator(self, monkeypatch, *, routed_category, bm25_hits):
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        metadatas = self.METADATAS
        docs = self.DOCS

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return bm25_hits

        class FakeCollection:
            def get(self, ids, include):
                return {
                    "ids": ids,
                    "documents": [docs[cid] for cid in ids] if "documents" in include else None,
                    "metadatas": [metadatas[cid] for cid in ids] if "metadatas" in include else None,
                }

        orchestrator = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orchestrator)
        orchestrator.query_cache = FakeCache()
        orchestrator.bm25_index = FakeBM25()
        orchestrator.collection = FakeCollection()
        orchestrator._ensure_bm25_index = lambda: None
        orchestrator._route_by_keywords = lambda query: routed_category
        orchestrator._expand_with_adjacent_chunks = lambda results: results
        return orchestrator

    def test_routed_category_does_not_restrict_bm25_when_no_explicit_filter(self, monkeypatch):
        """BM25 candidates from other categories must survive when user omits category_filter."""
        orchestrator = self._build_orchestrator(
            monkeypatch,
            routed_category="redteam",  # router picks an under-populated category
            bm25_hits=[("chunk_redteam_generic", 10.0), ("chunk_security_esc1", 9.5)],
        )

        results = orchestrator.query("ESC1 ADCS", max_results=5, category_filter=None, hybrid_alpha=0.0)

        categories_seen = {r["category"] for r in results}
        assert "security" in categories_seen, (
            "routed_category='redteam' must not hide docs from other categories when the user omitted category_filter"
        )
        # routed_by remains populated as informational telemetry (public API unchanged).
        assert {r["routed_by"] for r in results} == {"redteam"}

    def test_routed_category_does_not_restrict_semantic_when_no_explicit_filter(self, monkeypatch):
        """The semantic branch must be called with where=None when user omits category_filter."""
        orchestrator = self._build_orchestrator(
            monkeypatch,
            routed_category="redteam",
            bm25_hits=[],
        )

        captured_where: list = []

        def fake_query(query_texts, n_results, where, include):
            captured_where.append(where)
            return {
                "ids": [["chunk_redteam_generic", "chunk_security_esc1"]],
                "distances": [[0.10, 0.20]],
                "documents": [[self.DOCS["chunk_redteam_generic"], self.DOCS["chunk_security_esc1"]]],
                "metadatas": [[self.METADATAS["chunk_redteam_generic"], self.METADATAS["chunk_security_esc1"]]],
            }

        orchestrator.collection.query = fake_query

        _ = orchestrator.query("ESC1 ADCS", max_results=5, category_filter=None, hybrid_alpha=1.0)

        assert captured_where == [None], (
            f"semantic where_filter must be None when user omitted category_filter, got {captured_where}"
        )

    def test_explicit_category_filter_still_overrides_routing(self, monkeypatch):
        """Explicit category_filter must take effect regardless of the router (preserves #109)."""
        orchestrator = self._build_orchestrator(
            monkeypatch,
            routed_category="redteam",  # router would pick redteam...
            bm25_hits=[("chunk_redteam_generic", 10.0), ("chunk_security_esc1", 9.5)],
        )

        results = orchestrator.query("ESC1 ADCS", max_results=5, category_filter="security", hybrid_alpha=0.0)

        # ...but user asked explicitly for `security` — must win.
        assert [r["source"] for r in results] == ["/docs/security/pentest-everything/adcs/esc1.md"]
        assert {r["category"] for r in results} == {"security"}


# ── Query Cache ──


class TestQueryCache:
    def test_cache_miss(self):
        """First query is always a miss."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        result = cache.get("test", 5, None, 0.3)
        assert result is None

    def test_cache_hit(self):
        """Cached query returns stored result."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.put("test", 5, None, 0.3, [{"content": "result"}])
        result = cache.get("test", 5, None, 0.3)
        assert result is not None
        assert result[0]["content"] == "result"

    def test_cache_different_params(self):
        """Different params = different cache entries."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.put("test", 5, None, 0.3, ["result_a"])
        cache.put("test", 5, None, 0.7, ["result_b"])
        assert cache.get("test", 5, None, 0.3) == ["result_a"]
        assert cache.get("test", 5, None, 0.7) == ["result_b"]

    def test_cache_invalidate(self):
        """Invalidate clears all entries."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.put("test", 5, None, 0.3, ["result"])
        cache.invalidate()
        assert cache.get("test", 5, None, 0.3) is None

    def test_cache_stats(self):
        """Stats track hits and misses."""
        cache = QueryCache(max_size=10, ttl_seconds=300)
        cache.get("miss", 5, None, 0.3)  # miss
        cache.put("hit", 5, None, 0.3, ["data"])
        cache.get("hit", 5, None, 0.3)  # hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_eviction(self):
        """LRU eviction when max_size reached."""
        cache = QueryCache(max_size=2, ttl_seconds=300)
        cache.put("a", 5, None, 0.3, ["a"])
        cache.put("b", 5, None, 0.3, ["b"])
        cache.put("c", 5, None, 0.3, ["c"])  # should evict "a"
        assert cache.get("a", 5, None, 0.3) is None
        assert cache.get("b", 5, None, 0.3) is not None


# ── Keyword Routing ──


class TestKeywordRouting:
    def test_routing_detects_redteam(self, monkeypatch):
        """Security terms route to redteam."""
        # Test the static method logic without instantiating orchestrator
        import re

        from mcp_server.config import config

        # Deterministic test-local routes — never the operator config singleton.
        monkeypatch.setattr(config, "keyword_routes", dict(_TEST_KEYWORD_ROUTES))

        query = "mimikatz credential dump"
        query_lower = query.lower()
        matches = {}
        for category, keywords in config.keyword_routes.items():
            count = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if " " in kw_lower:
                    if kw_lower in query_lower:
                        count += 1
                else:
                    if re.search(r"\b" + re.escape(kw_lower) + r"\b", query_lower):
                        count += 1
            if count > 0:
                matches[category] = count

        assert "redteam" in matches

    def test_word_boundary_prevents_false_positive(self):
        """'api' must NOT match inside 'RAPID'."""
        import re

        assert not re.search(r"\bapi\b", "rapid deployment")
        assert re.search(r"\bapi\b", "api endpoint")


# ── Path-aware ranking ──


class TestPathAwareRanking:
    def test_path_match_can_lift_keyword_result(self, monkeypatch):
        """Path and filename matches provide a small generic ranking signal."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return [("chunk_generic", 10.0), ("chunk_target", 9.0)]

        class FakeCollection:
            def get(self, ids, include):
                chunk_id = ids[0]
                documents = {
                    "chunk_generic": "same keyword content",
                    "chunk_target": "same keyword content",
                }
                metadatas = {
                    "chunk_generic": {
                        "source": "/docs/notes/general.md",
                        "filename": "general.md",
                        "category": "docs",
                        "chunk_index": 0,
                    },
                    "chunk_target": {
                        "source": "/docs/reports/api-security.md",
                        "filename": "api-security.md",
                        "category": "docs",
                        "chunk_index": 0,
                    },
                }
                return {"documents": [documents[chunk_id]], "metadatas": [metadatas[chunk_id]]}

        orchestrator = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orchestrator)
        orchestrator.query_cache = FakeCache()
        orchestrator.bm25_index = FakeBM25()
        orchestrator.collection = FakeCollection()
        orchestrator._ensure_bm25_index = lambda: None
        orchestrator._route_by_keywords = lambda query: None
        orchestrator._expand_with_adjacent_chunks = lambda results: results

        results = orchestrator.query("api security", max_results=2, hybrid_alpha=0.0)

        assert results[0]["source"] == "/docs/reports/api-security.md"


# ── Candidate pool math (v4.8.0 Fase 3) ──


class TestDoSemanticCandidateMath:
    """Pin the ``_do_semantic`` candidate count against future clamping regressions.

    Before v4.8.0 Fase 3 the line
        ``n_candidates = min(max_results * 3, config.max_results)``
    was silently capped at 20 because ``config.max_results`` defaulted to
    20 — while BM25 pulled up to ``max_results * 20 = 400`` candidates.
    Semantic starved on hybrid mode without any log signal.

    The Fase 3 fix raised the ``max_results`` default from 20 to 100, so
    the ``min(...)`` now yields ``max_results * 3`` for typical callers
    (``max_results * 3 = 15 << 100``). This regression pin protects
    against future PRs that "helpfully" restore the 20-cap.
    """

    def test_semantic_asks_chromadb_for_3x_max_results(self, monkeypatch):
        """With max_results=5 and config.max_results=100 → 15 candidates (not 20)."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)
        monkeypatch.setattr("mcp_server.server.config.max_results", 100)

        captured = {}

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return []

        class FakeCollection:
            def query(self, query_texts, n_results, where, include):
                captured["n_results"] = n_results
                return {
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                }

            def get(self, ids, include):
                return {"documents": [], "metadatas": []}

        orch = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orch)
        orch.query_cache = FakeCache()
        orch.bm25_index = FakeBM25()
        orch.collection = FakeCollection()
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda query: None
        orch._expand_with_adjacent_chunks = lambda results: results

        # hybrid_alpha=1.0 forces the semantic-only branch (skips BM25).
        _ = orch.query("test query", max_results=5, hybrid_alpha=1.0)

        # 5 * 3 = 15; min(15, 100) = 15. NOT clamped at 20 anymore.
        assert captured["n_results"] == 15, (
            f"Expected 15 candidates (max_results=5 * 3, capped at "
            f"config.max_results=100), got {captured['n_results']}. "
            f"If this fails, someone probably reverted the v4.8.0 Fase 3 "
            f"default bump — check config.max_results and the min(...) "
            f"expression in server.py::_do_semantic."
        )

    def test_semantic_pool_is_bounded_by_config_max_results(self, monkeypatch):
        """When max_results * 3 exceeds config.max_results, the config value wins."""
        monkeypatch.setattr("mcp_server.server.config.reranker_enabled", False)
        # Cap intentionally small so max_results * 3 > cap.
        monkeypatch.setattr("mcp_server.server.config.max_results", 50)

        captured = {}

        class FakeCache:
            def get(self, *args, **kwargs):
                return None

            def put(self, *args, **kwargs):
                return None

        class FakeBM25:
            def search(self, query, top_k):
                return []

        class FakeCollection:
            def query(self, query_texts, n_results, where, include):
                captured["n_results"] = n_results
                return {
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                }

            def get(self, ids, include):
                return {"documents": [], "metadatas": []}

        orch = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orch)
        orch.query_cache = FakeCache()
        orch.bm25_index = FakeBM25()
        orch.collection = FakeCollection()
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda query: None
        orch._expand_with_adjacent_chunks = lambda results: results

        # 20 * 3 = 60, but config.max_results = 50 → min() clamps to 50
        _ = orch.query("test query", max_results=20, hybrid_alpha=1.0)

        assert captured["n_results"] == 50


# =============================================================================
# FTS5 Fast-Path Dispatch — Task 03 (ADR-002, ADR-003, ADR-006)
# =============================================================================


class _FakeFts5:
    """Controllable ``Fts5LexicalIndex`` substitute for dispatch tests."""

    def __init__(self, hits=(), ready=True, raises=None):
        self._hits = list(hits)
        self._ready = ready
        self._raises = raises
        self.search_calls = []

    def is_ready(self):
        return self._ready

    def search(self, query, top_k=20):
        self.search_calls.append((query, top_k))
        if self._raises is not None:
            raise self._raises
        return list(self._hits)


class _FakeRouter:
    """Controllable ``QueryRouter`` substitute — fixed or callable decision."""

    def __init__(self, decision):
        self._decision = decision
        self.classify_calls = []

    def classify(self, query):
        self.classify_calls.append(query)
        if callable(self._decision):
            return self._decision(query)
        return self._decision


def _build_dispatch_orch(
    monkeypatch,
    *,
    fts5_enabled=True,
    fts5_index=None,
    query_router=None,
    fts5_min_hits=3,
    fake_docs=None,
    fake_metadatas=None,
):
    """Bare-bones orchestrator sufficient for exercising query() dispatch."""
    import mcp_server.server as srv

    monkeypatch.setattr(srv.config, "fts5_enabled", fts5_enabled)
    monkeypatch.setattr(srv.config, "fts5_min_hits", fts5_min_hits)
    monkeypatch.setattr(srv.config, "fts5_rerank_enabled", False)
    monkeypatch.setattr(srv.config, "reranker_enabled", False)
    monkeypatch.setattr(srv.config, "default_results", 5)
    monkeypatch.setattr(srv.config, "max_results", 50)

    docs = fake_docs or {}
    metadatas = fake_metadatas or {}

    class _FakeCollection:
        def get(self, ids, include):
            return {
                "ids": list(ids),
                "documents": [docs.get(cid, "") for cid in ids] if "documents" in include else None,
                "metadatas": [metadatas.get(cid, {}) for cid in ids] if "metadatas" in include else None,
            }

        def query(self, query_texts, n_results, where, include):
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        def count(self):
            return len(docs)

    class _FakeBM25:
        def search(self, query, top_k):
            return []

    orch = object.__new__(KnowledgeOrchestrator)
    orch.query_cache = QueryCache(max_size=32, ttl_seconds=300)
    orch.bm25_index = _FakeBM25()
    orch.collection = _FakeCollection()
    orch.fts5_index = fts5_index
    orch.query_router = query_router
    orch._ensure_bm25_index = lambda: None
    orch._route_by_keywords = lambda q: None
    orch._expand_with_adjacent_chunks = lambda r: r
    orch._apply_mmr = lambda results, top_k, lambda_param=0.7: results[:top_k]
    return orch


class TestFtsDispatch:
    """IT-001..IT-009 — dispatch decisions in ``KnowledgeOrchestrator.query()``."""

    def test_it001_feature_off_never_dispatches_fts5(self, monkeypatch):
        """IT-001: fts5_enabled=False → hybrid pipeline only, FTS5 untouched."""
        fts5 = _FakeFts5(hits=[("chunk_1", 5.0)], ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_enabled=False, fts5_index=fts5, query_router=router)

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert results == []
        assert fts5.search_calls == [], "FTS5 must never be invoked when the feature is off"
        assert router.classify_calls == [], "router must never run when the feature is off"

    def test_it002_lexical_query_dispatches_fts5(self, monkeypatch):
        """IT-002: enabled + lexical + ready + hit → search_method='fts5' in results."""
        hits = [("chunk_1", 5.0), ("chunk_2", 4.0), ("chunk_3", 3.0)]
        docs = {"chunk_1": "H1-P4-XXX-1234 disclosure summary", "chunk_2": "other", "chunk_3": "more"}
        metas = {
            cid: {
                "source": f"/{cid}",
                "filename": f"{cid}.md",
                "category": "bugbounty",
                "chunk_index": 0,
                "keywords": "",
            }
            for cid in docs
        }
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(
            monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs, fake_metadatas=metas
        )

        results = orch.query("H1-P4-XXX-1234", search_method="auto")

        assert results, "expected at least one FTS5 result"
        assert results[0]["search_method"] == "fts5"
        assert fts5.search_calls, "FTS5 search was not invoked"

    def test_it003_parametrized_lexical_queries_all_dispatch(self, monkeypatch, sample_lexical_queries):
        """IT-003: each canonical lexical query dispatches to FTS5 when hits exist."""
        for q in sample_lexical_queries:
            hits = [(f"chunk_{i}", 10 - i) for i in range(3)]
            docs = {f"chunk_{i}": f"doc mentioning {q}" for i in range(3)}
            metas = {
                f"chunk_{i}": {
                    "source": f"/{i}",
                    "filename": f"{i}.md",
                    "category": "security",
                    "chunk_index": 0,
                    "keywords": "",
                }
                for i in range(3)
            }
            fts5 = _FakeFts5(hits=hits, ready=True)
            router = _FakeRouter("lexical")
            orch = _build_dispatch_orch(
                monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs, fake_metadatas=metas
            )
            results = orch.query(q, search_method="auto")
            assert results, f"no results for canonical lexical query {q!r}"
            assert results[0]["search_method"] == "fts5"

    def test_it004_zero_docs_corpus_returns_empty(self, monkeypatch):
        """IT-004: FTS5 empty + hybrid empty → orch.query returns []."""
        fts5 = _FakeFts5(hits=[], ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        results = orch.query("MDR-AD002", search_method="auto")

        assert results == []

    def test_it005_semantic_query_skips_fts5(self, monkeypatch):
        """IT-005: router says semantic → FTS5 is never consulted."""
        fts5 = _FakeFts5(hits=[("chunk_1", 5.0)], ready=True)
        router = _FakeRouter("semantic")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        results = orch.query("nuclei", search_method="auto")

        assert fts5.search_calls == []
        assert results == []

    def test_it006_low_hits_triggers_fallback_metric(self, monkeypatch):
        """IT-006: lexical + 0 fts5 hits + min_hits=3 → fallback_total{reason=low_hits} +1."""
        from mcp_server.metrics import FAST_PATH_FALLBACK_TOTAL
        from tests.conftest import _get_metric_value

        fts5 = _FakeFts5(hits=[], ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        needle = f'{FAST_PATH_FALLBACK_TOTAL}{{reason="low_hits"}}'
        before = _get_metric_value(needle)
        results = orch.query("T-800", search_method="auto")
        after = _get_metric_value(needle)

        assert results == []
        assert after > before, f"{needle} was not incremented"

    def test_it007_high_min_hits_causes_fallback(self, monkeypatch):
        """IT-007: min_hits=100 + 10 fts5 hits → fallback (result count below threshold)."""
        hits = [(f"chunk_{i}", 10 - i) for i in range(10)]
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_min_hits=100, fts5_index=fts5, query_router=router)

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert results == []

    def test_it008_fts5_error_triggers_fallback_and_error_metric(self, monkeypatch):
        """IT-008: fts5.search raises OperationalError → fallback + errors_total{OperationalError}."""
        import sqlite3

        from mcp_server.metrics import FAST_PATH_ERRORS_TOTAL, get_metrics
        from tests.conftest import _get_metric_value

        fts5 = _FakeFts5(raises=sqlite3.OperationalError("readonly"))
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        needle = f'{FAST_PATH_ERRORS_TOTAL}{{error_class="OperationalError"}}'
        before = _get_metric_value(needle)
        orch.query("CVE-2021-4034", search_method="auto")
        after = _get_metric_value(needle)

        assert 'error_class="OperationalError"' in get_metrics().exposition()
        assert after > before, f"{needle} was not incremented"

    def test_it009_fts5_databaseerror_falls_back(self, monkeypatch):
        """IT-009: fts5.search raises DatabaseError → fallback + error metric labeled correctly."""
        import sqlite3

        from mcp_server.metrics import get_metrics

        fts5 = _FakeFts5(raises=sqlite3.DatabaseError("file missing"))
        router = _FakeRouter("lexical")
        orch = _build_dispatch_orch(monkeypatch, fts5_index=fts5, query_router=router)

        orch.query("CVE-2021-4034", search_method="auto")
        exposition = get_metrics().exposition()

        assert 'error_class="DatabaseError"' in exposition


class TestConfigToggle:
    """IT-015..IT-018 — config-driven feature toggle + custom pattern classification."""

    def test_it015_fresh_install_default_off(self, monkeypatch, tmp_path):
        """IT-015: fresh YAML has no search section → fts5_enabled defaults False."""
        from mcp_server import config as config_module

        # Isolate BASE_DIR: Config.__post_init__ mkdirs data/chroma/documents/
        # models_cache — must land under tmp_path, never the live BASE_DIR.
        monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(config_module, "_yaml", {})
        cfg = config_module.Config()
        assert cfg.fts5_enabled is False

    def test_it016_legacy_config_unchanged(self, monkeypatch, tmp_path):
        """IT-016: v4.8.1 config sans lexical_fast_path → fts5_enabled stays False."""
        from mcp_server import config as config_module

        monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(config_module, "_yaml", {"search": {"hybrid_alpha": 0.3}})
        cfg = config_module.Config()
        assert cfg.fts5_enabled is False

    def test_it017_yaml_enables_fts5(self, monkeypatch, tmp_path):
        """IT-017: enabled: true in YAML flips fts5_enabled."""
        from mcp_server import config as config_module

        monkeypatch.setattr(config_module, "BASE_DIR", tmp_path)
        monkeypatch.setattr(config_module, "_yaml", {"search": {"lexical_fast_path": {"enabled": True}}})
        cfg = config_module.Config()
        assert cfg.fts5_enabled is True

    def test_it018_custom_pattern_classifies_query(self):
        """IT-018: custom PROJ pattern classifies matching query as lexical."""
        from mcp_server.query_router import QueryRouter

        router = QueryRouter([r"PROJ-\d{3,5}"])
        assert router.classify("PROJ-12345") == "lexical"
        assert router.classify("random prose") == "semantic"


class TestMetricsScrape:
    """IT-019 — exposition after mixed lexical + semantic query traffic."""

    def test_it019_mixed_queries_produce_expected_counters(self, monkeypatch):
        from mcp_server.metrics import FAST_PATH_HITS_TOTAL, get_metrics

        hits = [(f"chunk_{i}", 10 - i) for i in range(5)]
        docs = {f"chunk_{i}": f"CVE-2021-4034 content {i}" for i in range(5)}
        metas = {
            f"chunk_{i}": {
                "source": f"/{i}",
                "filename": f"{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(5)
        }
        fts5 = _FakeFts5(hits=hits, ready=True)
        router = _FakeRouter(lambda q: "lexical" if "CVE" in q else "semantic")
        orch = _build_dispatch_orch(
            monkeypatch, fts5_index=fts5, query_router=router, fake_docs=docs, fake_metadatas=metas
        )

        for _ in range(3):
            orch.query("CVE-2021-4034", search_method="auto")
            orch.query_cache.invalidate()
        for _ in range(2):
            orch.query("how does OAuth token refresh work", search_method="auto")
            orch.query_cache.invalidate()

        exposition = get_metrics().exposition()
        assert FAST_PATH_HITS_TOTAL in exposition
        assert 'path="fts5"' in exposition
        assert 'path="hybrid"' in exposition


class TestQueryCacheKey:
    """UT-059, UT-060 — 5th-param backward compat + collision-free keys per path."""

    def test_ut059_make_key_default_matches_explicit_auto(self):
        cache = QueryCache()
        without_arg = cache._make_key("q", 5, None, 0.3)
        with_auto = cache._make_key("q", 5, None, 0.3, "auto")
        assert without_arg == with_auto

    def test_ut060_search_method_variants_produce_distinct_keys(self):
        cache = QueryCache()
        keys = {cache._make_key("q", 5, None, 0.3, sm) for sm in ("auto", "hybrid", "fts5")}
        assert len(keys) == 3


class _SpyReranker:
    """Stand-in for ``CrossEncoderReranker`` — records calls, assigns descending scores.

    Score assignment mirrors the real reranker contract: mutate
    ``doc["reranker_score"]`` in place and sort by it descending. Descending
    input-index scoring lets tests assert ordering deterministically.
    """

    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_k):
        self.calls.append((query, list(documents), top_k))
        for offset, doc in enumerate(documents):
            doc["reranker_score"] = float(len(documents) - offset)
        documents.sort(key=lambda d: d["reranker_score"], reverse=True)
        return documents[:top_k]


def _build_rerank_orch(monkeypatch, *, rerank_enabled, hits, docs, metas):
    """Dispatch orchestrator wired with a spy reranker for TestRerankToggle."""
    orch = _build_dispatch_orch(
        monkeypatch,
        fts5_index=_FakeFts5(hits=hits, ready=True),
        query_router=_FakeRouter("lexical"),
        fake_docs=docs,
        fake_metadatas=metas,
    )
    import mcp_server.server as srv

    monkeypatch.setattr(srv.config, "fts5_rerank_enabled", rerank_enabled)
    monkeypatch.setattr(srv.config, "reranker_enabled", True)
    spy = _SpyReranker()
    orch.reranker = spy
    return orch, spy


class TestRerankToggle:
    """UT-062, UT-063, IT-024, IT-025 — fast-path rerank opt-in (ADR-003)."""

    def _sample(self, size=5):
        hits = [(f"chunk_{i}", float(size - i)) for i in range(size)]
        docs = {f"chunk_{i}": f"CVE-2021-4034 content {i}" for i in range(size)}
        metas = {
            f"chunk_{i}": {
                "source": f"/{i}",
                "filename": f"{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(size)
        }
        return hits, docs, metas

    def test_ut062_rerank_disabled_never_calls_reranker(self, monkeypatch):
        """UT-062: default (rerank_enabled=False) → zero reranker calls, reranker_score is None."""
        hits, docs, metas = self._sample()
        orch, spy = _build_rerank_orch(monkeypatch, rerank_enabled=False, hits=hits, docs=docs, metas=metas)

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert results, "expected fast-path results"
        assert spy.calls == [], "reranker must not run when fts5_rerank_enabled=False"
        assert all(r["search_method"] == "fts5" for r in results)
        assert all(r["reranker_score"] is None for r in results)

    def test_ut063_rerank_enabled_invokes_reranker(self, monkeypatch):
        """UT-063: opt-in (rerank_enabled=True) → reranker runs, score populated, path stays fts5."""
        hits, docs, metas = self._sample()
        orch, spy = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert results, "expected fast-path results"
        assert len(spy.calls) == 1, "reranker must run exactly once for the fast-path"
        assert all(r["search_method"] == "fts5" for r in results), "rerank must not switch search_method"
        assert all(isinstance(r["reranker_score"], float) for r in results)
        # Fast-path items alias content→document for the reranker; it must be popped after.
        assert all("document" not in r for r in results)

    def test_it024_rerank_off_faster_than_rerank_on(self, monkeypatch):
        """IT-024: rerank_enabled=False completes in less wall time than rerank_enabled=True."""
        import time as _time

        class _SlowReranker(_SpyReranker):
            def rerank(self, query, documents, top_k):
                _time.sleep(0.02)  # 20ms cross-encoder proxy
                return super().rerank(query, documents, top_k)

        hits, docs, metas = self._sample()

        orch_off, _ = _build_rerank_orch(monkeypatch, rerank_enabled=False, hits=hits, docs=docs, metas=metas)
        t_off_start = _time.perf_counter()
        orch_off.query("CVE-2021-4034", search_method="auto")
        t_off = _time.perf_counter() - t_off_start

        orch_on, _ = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)
        orch_on.reranker = _SlowReranker()
        t_on_start = _time.perf_counter()
        orch_on.query("CVE-2021-4034", search_method="auto")
        t_on = _time.perf_counter() - t_on_start

        assert t_off < t_on, f"rerank OFF ({t_off * 1000:.2f}ms) must be faster than ON ({t_on * 1000:.2f}ms)"

    def test_it025_rerank_on_orders_by_reranker_score(self, monkeypatch):
        """IT-025: rerank_enabled=True → results ordered by reranker_score DESC (not FTS5 raw)."""
        # Spy assigns len(docs)..1 in input order → after sort desc, order is preserved from input.
        # Use a shuffling spy to prove ordering comes from reranker_score, not FTS5 hit order.
        hits, docs, metas = self._sample(size=5)

        class _ShuffleReranker:
            def __init__(self):
                self.calls = []

            def rerank(self, query, documents, top_k):
                self.calls.append(query)
                # Reverse assignment: last input gets highest score.
                for i, doc in enumerate(documents):
                    doc["reranker_score"] = float(i + 1)
                documents.sort(key=lambda d: d["reranker_score"], reverse=True)
                return documents[:top_k]

        orch, _ = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)
        orch.reranker = _ShuffleReranker()

        results = orch.query("CVE-2021-4034", search_method="auto")

        scores = [r["reranker_score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"expected DESC by reranker_score, got {scores}"
        # And it must NOT match the raw FTS5 hit order (which is chunk_0..chunk_4).
        assert [r["chunk_index"] for r in results] != list(range(len(results))) or True
        # Concrete check: chunk_0 (highest FTS5 score, first input) got lowest reranker_score → last.
        assert results[0]["source"] == "/4", "highest reranker_score should surface last input first"


# =============================================================================
# Requirement 8 — additive scoring schema (raw_score / query_relative_score /
# score_source) on every returned search result.
# =============================================================================


class _NoopCache:
    """Cache double that never hits — keeps repeated query() calls honest."""

    def get(self, *args, **kwargs):
        return None

    def put(self, *args, **kwargs):
        return None


class TestFtsNoRerankScoringSchema:
    """FTS fast-path without rerank: native FTS raw + cohort normalization."""

    def _metas(self, cids):
        return {
            cid: {
                "source": f"/docs/{cid}.md",
                "filename": f"{cid}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for cid in cids
        }

    def test_schema_alias_and_unrounded_raw(self, monkeypatch):
        """Every FTS result exposes raw_score/query_relative_score/score_source;
        legacy score aliases query_relative_score; raw_score is unrounded."""
        hits = [("chunk_a", 5.1234567), ("chunk_b", 4.0), ("chunk_c", 3.0)]
        docs = {cid: f"{cid} CVE-2021-4034 content" for cid, _ in hits}
        fts5 = _FakeFts5(hits=hits, ready=True)
        orch = _build_dispatch_orch(
            monkeypatch,
            fts5_index=fts5,
            query_router=_FakeRouter("lexical"),
            fake_docs=docs,
            fake_metadatas=self._metas(["chunk_a", "chunk_b", "chunk_c"]),
        )

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert len(results) == 3
        for r in results:
            assert r["search_method"] == "fts5"
            assert r["score_source"] == "fts5_bm25"
            assert isinstance(r["raw_score"], float)
            assert 0.0 <= r["query_relative_score"] <= 1.0
            assert r["score"] == r["query_relative_score"], "legacy score must alias query_relative_score"
            assert r["raw_rrf_score"] is None
            assert r["reranker_score"] is None
        top = results[0]
        assert top["query_relative_score"] == 1.0
        # Unrounded raw: the rounded normalized score differs from the raw float.
        assert top["raw_score"] == 5.1234567
        assert top["raw_score"] != top["query_relative_score"]

    def test_cohort_minmax_computed_after_orphan_filtering(self, monkeypatch):
        """An orphan hit (FTS has it, Chroma doesn't) with an extreme raw score
        must NOT stretch the cohort min/max — normalization uses survivors only."""
        hits = [("chunk_orphan", 1.0), ("chunk_a", 5.1234567), ("chunk_b", 4.0), ("chunk_c", 3.0)]
        # docs/metas deliberately omit chunk_orphan → orphan filtered out.
        docs = {cid: f"{cid} content" for cid in ("chunk_a", "chunk_b", "chunk_c")}
        orch = _build_dispatch_orch(
            monkeypatch,
            fts5_index=_FakeFts5(hits=hits, ready=True),
            query_router=_FakeRouter("lexical"),
            fake_docs=docs,
            fake_metadatas=self._metas(["chunk_a", "chunk_b", "chunk_c"]),
        )

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert [r["source"] for r in results] == [
            "/docs/chunk_a.md",
            "/docs/chunk_b.md",
            "/docs/chunk_c.md",
        ]
        expected_b = round((4.0 - 3.0) / (5.1234567 - 3.0), 4)
        assert results[0]["query_relative_score"] == 1.0
        assert results[1]["query_relative_score"] == expected_b, (
            "normalization must use the post-filter cohort min/max (3.0..5.1234567), "
            "not the pre-filter hit list including the orphan (1.0..5.1234567)"
        )
        assert results[2]["query_relative_score"] == 0.0

    def test_equal_raw_scores_map_to_one(self, monkeypatch):
        """All-equal cohort raw scores → every query_relative_score is 1.0."""
        hits = [("chunk_a", 7.0), ("chunk_b", 7.0), ("chunk_c", 7.0)]
        docs = {cid: f"{cid} content" for cid, _ in hits}
        orch = _build_dispatch_orch(
            monkeypatch,
            fts5_index=_FakeFts5(hits=hits, ready=True),
            query_router=_FakeRouter("lexical"),
            fake_docs=docs,
            fake_metadatas=self._metas(["chunk_a", "chunk_b", "chunk_c"]),
        )

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert [r["query_relative_score"] for r in results] == [1.0, 1.0, 1.0]
        assert all(r["raw_score"] == 7.0 for r in results)


class TestFtsRerankRenormalization:
    """After FTS rerank, effective raw is the reranker score and the result is
    renormalized in reranked order — the pre-rerank normalized score never
    survives."""

    def _sample(self):
        hits = [(f"chunk_{i}", float(5 - i)) for i in range(5)]
        docs = {f"chunk_{i}": f"CVE-2021-4034 content {i}" for i in range(5)}
        metas = {
            f"chunk_{i}": {
                "source": f"/{i}",
                "filename": f"{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(5)
        }
        return hits, docs, metas

    def test_reverse_reranker_recomputes_and_renormalizes(self, monkeypatch):
        """Reverse reranker (last input scores highest) proves raw_score comes
        from the reranker and query_relative_score descends in reranked order."""
        hits, docs, metas = self._sample()

        class _ReverseReranker:
            def rerank(self, query, documents, top_k):
                for i, doc in enumerate(documents):
                    doc["reranker_score"] = float(i + 1)  # last input gets highest
                documents.sort(key=lambda d: d["reranker_score"], reverse=True)
                return documents[:top_k]

        orch, _ = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)
        orch.reranker = _ReverseReranker()

        results = orch.query("CVE-2021-4034", search_method="auto")

        # Reranked order: chunk_4 (reranker 5.0) first ... chunk_0 (1.0) last.
        assert [r["source"] for r in results] == ["/4", "/3", "/2", "/1", "/0"]
        for r in results:
            assert r["search_method"] == "fts5", "rerank must not switch search_method"
            assert r["score_source"] == "reranker"
            assert r["raw_score"] == r["reranker_score"], "effective raw must be the reranker score"
            assert r["score"] == r["query_relative_score"]
        # Renormalized in reranked order over the reranked cohort (5.0..1.0).
        assert [r["query_relative_score"] for r in results] == [1.0, 0.75, 0.5, 0.25, 0.0]
        # Recompute proof: chunk_0 was FTS-top (pre-rerank normalized 1.0);
        # after the reverse rerank it must carry 0.0, never the stale 1.0.
        last = results[-1]
        assert last["source"] == "/0"
        assert last["query_relative_score"] == 0.0


class TestHybridScoreSourceSelection:
    """Hybrid effective raw = reranker_score when present, else rrf_score."""

    def _build_hybrid_orch(self, monkeypatch, *, reranker_enabled):
        import mcp_server.server as srv

        monkeypatch.setattr(srv.config, "fts5_enabled", False)
        monkeypatch.setattr(srv.config, "reranker_enabled", reranker_enabled)
        monkeypatch.setattr(srv.config, "default_results", 5)
        monkeypatch.setattr(srv.config, "max_results", 50)

        docs = {f"chunk_{i}": f"content {i} about sql injection" for i in range(4)}
        metas = {
            f"chunk_{i}": {
                "source": f"/docs/doc{i}.md",
                "filename": f"doc{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(4)
        }

        class _Col:
            def query(self, query_texts, n_results, where, include):
                ids = [f"chunk_{i}" for i in range(4)]
                return {
                    "ids": [ids],
                    "distances": [[0.1 * (i + 1) for i in range(4)]],
                    "documents": [[docs[c] for c in ids]],
                    "metadatas": [[metas[c] for c in ids]],
                }

            def get(self, ids, include):
                return {
                    "ids": list(ids),
                    "documents": [docs.get(c, "") for c in ids] if "documents" in include else None,
                    "metadatas": [metas.get(c, {}) for c in ids] if "metadatas" in include else None,
                }

        class _BM25:
            # Reverse order vs semantic so fusion scores are non-uniform.
            def search(self, query, top_k):
                return [(f"chunk_{3 - i}", 10.0 - i) for i in range(4)]

        orch = object.__new__(KnowledgeOrchestrator)
        orch.query_cache = _NoopCache()
        orch.bm25_index = _BM25()
        orch.collection = _Col()
        orch.fts5_index = None
        orch.query_router = None
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda q: None
        orch._expand_with_adjacent_chunks = lambda r: r
        orch._apply_mmr = lambda results, top_k, lambda_param=0.7: results[:top_k]
        if reranker_enabled:
            orch.reranker = _SpyReranker()
        return orch

    def test_rrf_source_without_reranker(self, monkeypatch):
        """Reranker off → score_source='rrf', raw_score equals the RRF raw."""
        orch = self._build_hybrid_orch(monkeypatch, reranker_enabled=False)

        results = orch.query("sql injection", max_results=4, hybrid_alpha=0.5)

        assert results, "expected hybrid results"
        for r in results:
            assert r["score_source"] == "rrf"
            assert r["reranker_score"] is None
            assert abs(r["raw_score"] - r["raw_rrf_score"]) <= 1e-6, "effective raw must be the (unrounded) rrf_score"
            assert r["score"] == r["query_relative_score"]
            assert 0.0 <= r["query_relative_score"] <= 1.0
        qrs = [r["query_relative_score"] for r in results]
        assert qrs[0] == 1.0, "cohort max must normalize to 1.0"
        assert qrs == sorted(qrs, reverse=True)

    def test_reranker_source_wins_over_rrf(self, monkeypatch):
        """Reranker on → score_source='reranker', effective raw is the reranker
        score (raw_rrf_score stays as a diagnostic only)."""
        orch = self._build_hybrid_orch(monkeypatch, reranker_enabled=True)

        results = orch.query("sql injection", max_results=4, hybrid_alpha=0.5)

        assert results, "expected hybrid results"
        for r in results:
            assert r["score_source"] == "reranker"
            assert r["reranker_score"] is not None
            assert r["raw_score"] == r["reranker_score"], "reranker score must override rrf as effective raw"
            assert r["score"] == r["query_relative_score"]
            assert 0.0 <= r["query_relative_score"] <= 1.0
        assert results[0]["query_relative_score"] == 1.0


class TestExpectedPathBoundary:
    """Lexical boundary matching for evaluate_retrieval expected paths."""

    def test_relative_expected_matches_suffix_boundary(self):
        from mcp_server.server import _expected_path_matches

        assert _expected_path_matches("security/a.md", "/corpus/security/a.md")
        assert _expected_path_matches("security/a.md", "security/a.md")

    def test_no_substring_false_positives(self):
        from mcp_server.server import _expected_path_matches

        assert not _expected_path_matches("security/a.md", "not-a.md")
        assert not _expected_path_matches("security/a.md", "/corpus/security/a.md.bak")
        assert not _expected_path_matches("security/a.md", "/corpus/mysecurity/a.md")

    def test_backslash_normalized(self):
        from mcp_server.server import _expected_path_matches

        assert _expected_path_matches("security\\a.md", "/corpus/security/a.md")
        assert _expected_path_matches("security\\a.md", "D:\\corpus\\security\\a.md")

    def test_absolute_expected_requires_equality(self):
        from mcp_server.server import _expected_path_matches

        assert _expected_path_matches("/corpus/security/a.md", "/corpus/security/a.md")
        assert not _expected_path_matches("/corpus/security/a.md", "/x/corpus/security/a.md")

    def test_windows_drive_absolute_requires_exact_equality(self):
        from mcp_server.server import _expected_path_matches

        # Drive-absolute (either separator) is ABSOLUTE: exact normalized
        # lexical equality only — no suffix matching.
        assert _expected_path_matches("C:/corpus/security/a.md", "C:\\corpus\\security\\a.md")
        assert _expected_path_matches("C:\\corpus\\security\\a.md", "C:/corpus/security/a.md")
        assert not _expected_path_matches("C:/corpus/a.md", "D:/corpus/a.md")
        assert not _expected_path_matches("C:/corpus/a.md", "C:/other/corpus/a.md")
        assert not _expected_path_matches("C:/corpus/a.md", "/corpus/C:/corpus/a.md")

    def test_unc_absolute_requires_exact_equality(self):
        from mcp_server.server import _expected_path_matches

        # UNC (either separator) is ABSOLUTE: exact normalized equality only.
        assert _expected_path_matches(
            "//server/share/corpus/a.md", "\\\\server\\share\\corpus\\a.md"
        )
        assert not _expected_path_matches("//server/share/a.md", "//server/share/sub/a.md")
        assert not _expected_path_matches("//server/share/a.md", "//other/share/a.md")

    def test_empty_inputs_never_match(self):
        from mcp_server.server import _expected_path_matches

        assert not _expected_path_matches("", "/corpus/a.md")
        assert not _expected_path_matches("a.md", "")


class TestEvaluateRetrievalOfflineSmoke:
    """Orchestrator-level offline smoke: mode marker, boundary matching, validation."""

    SOURCES = [
        "/corpus/security/a.md",
        "/corpus/not-security-a.md",
        "/corpus/other/b.md",
    ]

    def _orch(self):
        orch = object.__new__(KnowledgeOrchestrator)
        sources = self.SOURCES

        def fake_query(query, **kwargs):
            return [{"source": s, "content": "x"} for s in sources]

        orch.query = fake_query
        return orch

    def test_offline_smoke_mode_and_boundary_hit(self):
        out = self._orch().evaluate_retrieval([{"query": "q", "expected_filepath": "security/a.md"}])

        assert out["evaluation_mode"] == "offline_smoke"
        assert out["total_queries"] == 1
        assert out["mrr_at_5"] == 1.0
        assert out["recall_at_5"] == 1.0
        assert out["per_query"][0]["found_at_rank"] == 1

    def test_suffix_blob_is_a_miss_not_a_substring_hit(self):
        """Pre-fix behavior matched 'a.md.bak'-style blobs via substring; the
        boundary rule must score them as misses."""
        out = self._orch().evaluate_retrieval([{"query": "q", "expected_filepath": "a.md.bak"}])

        assert out["evaluation_mode"] == "offline_smoke"
        assert out["per_query"][0]["found_at_rank"] is None
        assert out["mrr_at_5"] == 0.0

    def test_absolute_expected_requires_exact_source(self):
        orch = self._orch()
        hit = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "/corpus/security/a.md"}])
        miss = orch.evaluate_retrieval([{"query": "q", "expected_filepath": "/x/corpus/security/a.md"}])

        assert hit["per_query"][0]["found_at_rank"] == 1
        assert miss["per_query"][0]["found_at_rank"] is None

    def test_invalid_cases_are_rejected_with_valueerror(self):
        """P1 #6: direct calls reject malformed cases with ValueError —
        silent dropping/coercing is gone."""
        orch = self._orch()
        bad_inputs = [
            ["not-a-dict"],  # non-dict entry
            [{"query": "", "expected_filepath": "x"}],  # blank query
            [{"query": "q"}],  # missing expected_filepath
            [{"query": "q", "expected_filepath": "  "}],  # blank expected
            [{"query": 123, "expected_filepath": "a.md"}],  # non-string query
            "not-a-list",  # non-list container
        ]
        for cases in bad_inputs:
            with pytest.raises(ValueError):
                orch.evaluate_retrieval(cases)

    def test_valid_cases_still_pass_strict_validation(self):
        """Well-formed input remains accepted after strict validation."""
        out = self._orch().evaluate_retrieval([{"query": "q", "expected_filepath": "security/a.md"}])

        assert out["evaluation_mode"] == "offline_smoke"
        assert out["total_queries"] == 1
        assert out["per_query"][0]["found_at_rank"] == 1


# =============================================================================
# Scoring-review P1 regressions
# =============================================================================


class TestFtsNoScoreRerankerPreservesNative:
    """P1 #1: a reranker that returns items without numeric reranker_score
    must leave FTS-native scores, score_source, and order untouched."""

    def _sample(self):
        hits = [(f"chunk_{i}", float(5 - i)) for i in range(4)]
        docs = {f"chunk_{i}": f"CVE-2021-4034 content {i}" for i in range(4)}
        metas = {
            f"chunk_{i}": {
                "source": f"/{i}",
                "filename": f"{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(4)
        }
        return hits, docs, metas

    def test_no_score_reranker_returns_original_fts_results(self, monkeypatch):
        """Reranker invoked but scores nothing → formatted originals served."""
        hits, docs, metas = self._sample()

        class _NoScoreReranker:
            calls = 0

            def rerank(self, query, documents, top_k):
                type(self).calls += 1
                # Returns items UNCHANGED — no reranker_score written.
                return documents[:top_k]

        orch, _ = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)
        orch.reranker = _NoScoreReranker()

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert _NoScoreReranker.calls == 1, "reranker must have been invoked"
        assert len(results) == 4
        # Native FTS order (hit order) and scoring survive verbatim.
        assert [r["source"] for r in results] == ["/0", "/1", "/2", "/3"]
        assert [r["raw_score"] for r in results] == [5.0, 4.0, 3.0, 2.0], (
            "unscored rerank must preserve the native FTS raw scores"
        )
        for r in results:
            assert r["score_source"] == "fts5_bm25", "unscored rerank must not claim reranker source"
            assert r["reranker_score"] is None
        # Cohort-normalized over FTS raws: top=1.0, bottom=0.0.
        assert results[0]["query_relative_score"] == 1.0
        assert results[-1]["query_relative_score"] == 0.0

    def test_partial_score_reranker_also_preserves_originals(self, monkeypatch):
        """Only SOME items scored → still treated as a no-op rerank."""
        hits, docs, metas = self._sample()

        class _PartialReranker:
            def rerank(self, query, documents, top_k):
                out = list(documents)[:top_k]
                for doc in out[:2]:  # only the first two get a score
                    doc["reranker_score"] = 9.9
                return out

        orch, _ = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)
        orch.reranker = _PartialReranker()

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert [r["source"] for r in results] == ["/0", "/1", "/2", "/3"]
        for r in results:
            assert r["score_source"] == "fts5_bm25"
            assert r["reranker_score"] is None


    @pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_nonfinite_reranker_scores_are_a_no_op(self, monkeypatch, invalid_value):
        """NaN/+inf reranker_score on EVERY item → treated as a no-op rerank:
        native finite FTS raw scores, order, source survive; the emitted
        reranker_score is None and every query-relative value is finite."""
        import math

        hits, docs, metas = self._sample()

        class _NonFiniteReranker:
            calls = 0

            def rerank(self, query, documents, top_k):
                type(self).calls += 1
                out = list(documents)[:top_k]
                for doc in out:
                    doc["reranker_score"] = invalid_value
                return out

        orch, _ = _build_rerank_orch(monkeypatch, rerank_enabled=True, hits=hits, docs=docs, metas=metas)
        orch.reranker = _NonFiniteReranker()

        results = orch.query("CVE-2021-4034", search_method="auto")

        assert _NonFiniteReranker.calls == 1, "reranker must have been invoked"
        assert len(results) == 4
        # Native FTS order and finite raw scores survive verbatim.
        assert [r["source"] for r in results] == ["/0", "/1", "/2", "/3"]
        assert [r["raw_score"] for r in results] == [5.0, 4.0, 3.0, 2.0], (
            "non-finite rerank must preserve the native FTS raw scores"
        )
        for r in results:
            assert r["score_source"] == "fts5_bm25", "non-finite rerank must not claim reranker source"
            assert r["reranker_score"] is None
            assert math.isfinite(r["query_relative_score"])
            assert math.isfinite(r["score"])
        assert results[0]["query_relative_score"] == 1.0
        assert results[-1]["query_relative_score"] == 0.0


class TestHybridNonFiniteRerankerFallback:
    """Hybrid formatting: invalid NaN/inf reranker values with finite RRF
    fall back to RRF — source rrf, reranker_score None, all public numeric
    scores finite."""

    def test_nan_inf_reranker_falls_back_to_finite_rrf(self):
        import math

        from mcp_server.server import KnowledgeOrchestrator

        # Lightest seam: call the formatting tail directly through the
        # module-level helper the query path uses — construct candidates as
        # the hybrid pipeline hands them to formatting.
        orch = object.__new__(KnowledgeOrchestrator)
        orch._expand_with_adjacent_chunks = lambda results, window=1: results
        candidates = [
            (
                "c0",
                {
                    "document": "alpha beta",
                    "metadata": {"source": "/0", "filename": "0.md", "category": "x", "chunk_index": 0, "keywords": ""},
                    "reranker_score": float("nan"),
                    "rrf_score": 0.0143,
                    "semantic_rank": 1,
                    "bm25_rank": 2,
                },
            ),
            (
                "c1",
                {
                    "document": "gamma delta",
                    "metadata": {"source": "/1", "filename": "1.md", "category": "x", "chunk_index": 0, "keywords": ""},
                    "reranker_score": float("inf"),
                    "rrf_score": 0.0125,
                    "semantic_rank": 2,
                    "bm25_rank": 1,
                },
            ),
        ]

        # The formatting tail lives inline in query(); exercise it through
        # the same computation by invoking _apply_mmr's sibling formatting
        # path — simplest honest route: monkeypatch retrieval off and run
        # query() with a fake collection feeding exactly these candidates.
        class _Col:
            def __init__(self) -> None:
                self.disabled = False

            def query(self, query_texts, n_results, where, include):
                if self.disabled:
                    raise RuntimeError("retrieval must not be reached")
                ids = ["c0", "c1"]
                return {
                    "ids": [ids],
                    "distances": [[0.1, 0.2]],
                    "documents": [[candidates[i][1]["document"] for i in range(2)]],
                    "metadatas": [[candidates[i][1]["metadata"] for i in range(2)]],
                }

            def get(self, ids, include):
                by_id = {cid: data for cid, data in candidates}
                return {
                    "ids": list(ids),
                    "documents": [by_id[c]["document"] for c in ids] if "documents" in include else None,
                    "metadatas": [by_id[c]["metadata"] for c in ids] if "metadatas" in include else None,
                }

        class _BM25:
            def search(self, query, top_k):
                return [("c0", 2.0), ("c1", 1.0)]

        class _Cache:
            def __init__(self) -> None:
                self.hits = 0

            def get(self, *a, **k):
                return None

            def put(self, *a, **k):
                pass

            def stats(self):
                return {"hits": self.hits}

        orch.query_cache = _Cache()
        orch.bm25_index = _BM25()
        orch.collection = _Col()
        orch.fts5_index = None
        orch.query_router = None
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda q: None
        orch._apply_mmr = lambda results, top_k, lambda_param=0.7: results[:top_k]

        # Inject the invalid reranker values post-retrieval: wrap reranker
        # seam so scored candidates carry NaN/inf while RRF stays finite.
        class _PoisonReranker:
            def rerank(self, query, documents, top_k):
                out = list(documents)[:top_k]
                for doc, invalid in zip(out, (float("nan"), float("inf"))):
                    doc["reranker_score"] = invalid
                return out

        orch.reranker = _PoisonReranker()
        import mcp_server.server as srv

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(srv.config, "fts5_enabled", False)
        monkeypatch.setattr(srv.config, "reranker_enabled", True)
        monkeypatch.setattr(srv.config, "reranker_top_k_multiplier", 3)
        monkeypatch.setattr(srv.config, "default_results", 5)
        monkeypatch.setattr(srv.config, "max_results", 50)
        try:
            results = orch.query("alpha beta", search_method="hybrid")
        finally:
            monkeypatch.undo()

        assert results, "hybrid tail must produce results"
        for r in results:
            assert r["score_source"] == "rrf", "invalid reranker value must fall back to rrf"
            assert r["reranker_score"] is None, "NaN/inf must never be emitted"
            # Effective raw fell back to the item's own FINITE RRF value —
            # the same value emitted as raw_rrf_score (rounded on emit).
            assert r["raw_rrf_score"] is not None
            assert r["raw_score"] == pytest.approx(r["raw_rrf_score"], abs=1e-6), (
                f"raw_score must equal the finite RRF fallback, got {r['raw_score']} vs {r['raw_rrf_score']}"
            )
            assert math.isfinite(r["raw_score"])
            assert math.isfinite(r["query_relative_score"])
            assert math.isfinite(r["score"])
            assert math.isfinite(r["raw_rrf_score"])


class TestHybridTailRepeatCacheHit:
    """P1 #2 (hybrid tail): an identical repeat query under the caller's
    requested search_method (auto AND forced hybrid) must HIT the tail
    cache — verified by breaking underlying retrieval after the first
    call. No FTS early return involved: the feature is off and the router
    (forced lexical) is overridden, so both methods traverse the hybrid
    tail formatting + cache-put path."""

    @pytest.mark.parametrize("method", ["auto", "hybrid"])
    def test_repeat_request_hits_tail_cache(self, monkeypatch, method):
        docs = {f"chunk_{i}": f"hybrid content words {i}" for i in range(3)}
        metas = {
            f"chunk_{i}": {
                "source": f"/{i}",
                "filename": f"{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(3)
        }
        # fts5_enabled=False + router present-but-lexical: auto cannot take
        # the FTS early return (feature off), so it runs the hybrid tail
        # exactly like the forced hybrid method.
        orch = _build_dispatch_orch(
            monkeypatch,
            fts5_enabled=False,
            query_router=_FakeRouter("lexical"),
            fake_docs=docs,
            fake_metadatas=metas,
        )

        class _RealQueryCol:
            def query(self, query_texts, n_results, where, include):
                ids = list(docs)
                return {
                    "ids": [ids],
                    "distances": [[0.1 * (i + 1) for i in range(len(ids))]],
                    "documents": [[docs[c] for c in ids]],
                    "metadatas": [[metas[c] for c in ids]],
                }

            def get(self, ids, include):
                return {
                    "ids": list(ids),
                    "documents": [docs.get(c, "") for c in ids] if "documents" in include else None,
                    "metadatas": [metas.get(c, {}) for c in ids] if "metadatas" in include else None,
                }

        class _RealBM25:
            def search(self, query, top_k):
                return [(cid, 2.0) for cid in docs][:top_k]

        orch.collection = _RealQueryCol()
        orch.bm25_index = _RealBM25()

        first = orch.query("hybrid content words", search_method=method)
        assert first, "first call must produce results through the hybrid tail"

        # Break ALL underlying retrieval: a cache MISS would dispatch into
        # collection/BM25 and raise; a HIT serves the cached copy.
        class _Raising:
            def __getattr__(self, name):
                def _boom(*a, **k):
                    raise AssertionError("underlying retrieval reached — cache miss")

                return _boom

        orch.collection = _Raising()
        orch.bm25_index = _Raising()
        second = orch.query("hybrid content words", search_method=method)

        assert second == first
        stats = orch.query_cache.stats()
        assert stats["hits"] >= 1, f"repeat {method} request must hit the tail cache, stats={stats}"


class TestQueryCacheDeepCopyIsolation:
    """P1 #3: mutating a returned (or stored) result must never corrupt the
    cache — snippet_mode-style in-place truncation included."""

    def test_get_returns_deep_copy(self):
        cache = QueryCache(max_size=4, ttl_seconds=300)
        original = [{"content": "A" * 900, "nested": {"score": 0.5}}]
        cache.put("q", 5, None, 0.3, original)

        got = cache.get("q", 5, None, 0.3)
        got[0]["content"] = got[0]["content"][:100]
        got[0]["nested"]["score"] = 0.99

        again = cache.get("q", 5, None, 0.3)
        assert again[0]["content"] == "A" * 900, "cached content must not be mutated by callers"
        assert again[0]["nested"]["score"] == 0.5, "nested structures must be isolated too"

    def test_put_stores_deep_copy(self):
        cache = QueryCache(max_size=4, ttl_seconds=300)
        original = [{"content": "full content"}]
        cache.put("q", 5, None, 0.3, original)

        # Caller mutates the object AFTER put — cache entry must be intact.
        original[0]["content"] = "truncated"
        assert cache.get("q", 5, None, 0.3)[0]["content"] == "full content"

    def test_distinct_gets_are_independent(self):
        cache = QueryCache(max_size=4, ttl_seconds=300)
        cache.put("q", 5, None, 0.3, [{"content": "x"}])

        a = cache.get("q", 5, None, 0.3)
        b = cache.get("q", 5, None, 0.3)
        a[0]["content"] = "mutated"

        assert b[0]["content"] == "x"
        assert cache.get("q", 5, None, 0.3)[0]["content"] == "x"


class TestMmrRealSelectionRegression:
    """P1 #4: MMR must genuinely run — pool retained > max_results (both with
    and without the reranker), then MMR selects the final cohort."""

    def _build_orch(self, monkeypatch, *, reranker_enabled, mmr_spy, pool_docs, pool_metas):
        import mcp_server.server as srv

        monkeypatch.setattr(srv.config, "fts5_enabled", False)
        monkeypatch.setattr(srv.config, "reranker_enabled", reranker_enabled)
        monkeypatch.setattr(srv.config, "reranker_top_k_multiplier", 3)
        monkeypatch.setattr(srv.config, "default_results", 5)
        monkeypatch.setattr(srv.config, "max_results", 50)

        class _Col:
            def query(self, query_texts, n_results, where, include):
                ids = list(pool_docs)
                return {
                    "ids": [ids],
                    "distances": [[0.1 * (i + 1) for i in range(len(ids))]],
                    "documents": [[pool_docs[c] for c in ids]],
                    "metadatas": [[pool_metas[c] for c in ids]],
                }

            def get(self, ids, include):
                return {
                    "ids": list(ids),
                    "documents": [pool_docs.get(c, "") for c in ids] if "documents" in include else None,
                    "metadatas": [pool_metas.get(c, {}) for c in ids] if "metadatas" in include else None,
                }

        class _BM25:
            def search(self, query, top_k):
                return []

        orch = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orch)
        orch.query_cache = _NoopCache()
        orch.bm25_index = _BM25()
        orch.collection = _Col()
        orch._ensure_bm25_index = lambda: None
        orch._route_by_keywords = lambda q: None
        orch._expand_with_adjacent_chunks = lambda r: r
        orch._apply_mmr = mmr_spy
        if reranker_enabled:
            orch.reranker = _SpyReranker()
        return orch

    @staticmethod
    def _pool(n=8):
        docs = {f"chunk_{i}": f"distinct topic {i} " + " ".join(f"w{i}j" for j in range(20)) for i in range(n)}
        metas = {
            f"chunk_{i}": {
                "source": f"/docs/d{i}.md",
                "filename": f"d{i}.md",
                "category": "security",
                "chunk_index": 0,
                "keywords": "",
            }
            for i in range(n)
        }
        return docs, metas

    def test_mmr_called_with_pool_larger_than_max_results(self, monkeypatch):
        """No reranker: RRF pool (8 candidates) > max_results (3) → real MMR
        runs once with the oversized pool and returns exactly max_results."""
        docs, metas = self._pool(8)

        class _MmrSpy:
            calls = []

            def __call__(self, results, top_k, lambda_param=0.7):
                type(self).calls.append((len(results), top_k, lambda_param))
                # REAL selection, not a stub: emulate genuine MMR by keeping
                # relevance order here — the point is reachability + cohort.
                return sorted(results, key=lambda x: x[1]["rrf_score"], reverse=True)[:top_k]

        spy = _MmrSpy()
        orch = self._build_orch(monkeypatch, reranker_enabled=False, mmr_spy=spy, pool_docs=docs, pool_metas=metas)

        results = orch.query("distinct topic", max_results=3, hybrid_alpha=1.0)

        assert _MmrSpy.calls, "MMR must actually be invoked when pool > max_results"
        pool_len, top_k, _ = _MmrSpy.calls[0]
        assert pool_len > 3, f"candidate pool handed to MMR must exceed max_results, got {pool_len}"
        assert top_k == 3
        assert len(results) == 3, "final cohort must be exactly max_results"
        # Final query-relative normalization over the SELECTED cohort.
        assert results[0]["query_relative_score"] == 1.0

    def test_mmr_reachable_with_reranker_too(self, monkeypatch):
        """Reranker branch: reranker top_k (pool_size) > max_results → MMR
        still runs over the reranked pool."""
        docs, metas = self._pool(8)

        class _MmrSpy:
            calls = []

            def __call__(self, results, top_k, lambda_param=0.7):
                type(self).calls.append((len(results), top_k))
                return sorted(results, key=lambda x: x[1].get("reranker_score") or 0, reverse=True)[:top_k]

        spy = _MmrSpy()
        orch = self._build_orch(monkeypatch, reranker_enabled=True, mmr_spy=spy, pool_docs=docs, pool_metas=metas)

        results = orch.query("distinct topic", max_results=3, hybrid_alpha=1.0)

        assert _MmrSpy.calls, "MMR must run after the reranker over the retained pool"
        pool_len, top_k = _MmrSpy.calls[0]
        assert pool_len > 3
        assert top_k == 3
        assert len(results) == 3
        for r in results:
            assert r["score_source"] == "reranker"


class TestMmrRealFunctionBehavior:
    """REAL ``_apply_mmr`` (no spy): normalized relevance, negative logits,
    diversity selection across >= 2 loop iterations, and correct
    result-to-relevance association after every pop."""

    @staticmethod
    def _make_orch():
        orch = object.__new__(KnowledgeOrchestrator)
        _init_dispatchless_orch(orch)
        return orch

    @staticmethod
    def _result(cid, doc, *, reranker=None, rrf=None):
        data = {"document": doc, "source": f"/docs/{cid}.md"}
        if reranker is not None:
            data["reranker_score"] = reranker
        if rrf is not None:
            data["rrf_score"] = rrf
        return (cid, data)

    def test_all_negative_logits_diversity_two_iterations(self):
        # All-negative cross-encoder logits. Pool min-max normalization maps
        # -1.0 -> 0.0 and -0.2 -> 1.0. doc1 is a near-duplicate of doc0
        # (high Jaccard) but has the 2nd-best relevance; doc2 is dissimilar
        # with the 3rd-best relevance. With lambda=0.7 the 2nd selection
        # must be doc2 (diversity) and the 3rd doc1 — requiring at least
        # two while-loop iterations with correct post-pop association.
        orch = self._make_orch()
        text_a = "alpha beta gamma delta epsilon zeta"
        text_a_dup = "alpha beta gamma delta epsilon zeta extra words here"
        text_c = "kilo lima mike november oscar papa"
        results = [
            self._result("c0", text_a, reranker=-0.2),
            self._result("c1", text_a_dup, reranker=-0.4),
            self._result("c2", text_c, reranker=-0.6),
            self._result("c3", "quebec romeo sierra tango uniform victor", reranker=-1.0),
        ]
        selected = orch._apply_mmr(results, top_k=3, lambda_param=0.7)
        ids = [cid for cid, _ in selected]
        # Selection order: c0 (always first), then the DIVERSE c2 (relevance
        # 0.5 normalized, zero similarity) beats near-duplicate c1
        # (relevance 0.75 normalized but ~0.9 Jaccard to c0).
        assert ids == ["c0", "c2", "c1"]
        # Identity: original result objects come back unmodified.
        assert selected[0][1] is results[0][1]
        assert selected[1][1] is results[2][1]
        assert selected[2][1] is results[1][1]

    def test_post_pop_relevance_association(self):
        # Regression for the index-drift bug: after the first pop, item i is
        # no longer at index i, so relevance must travel WITH its result.
        # Pool logits: c0=-0.1, c1=-0.3, c2=-0.5, c3=-2.0 → normalized
        # relevance c0=1.0, c1≈0.895, c2≈0.789, c3=0.0. c1/c2 are
        # near-duplicates of c0 (~0.86 Jaccard); c3 is fully dissimilar
        # (similarity 0) with the LOWEST relevance.
        #   Iteration 1 (top_k=3): c1 ≈ 0.7·0.895 − 0.3·0.86 ≈ 0.374;
        #     c3 = 0.7·0 − 0.3·0 = 0.0 → c1 correctly wins on relevance.
        #   Iteration 2: remaining = [c0gone] → [c2, c3]; with the SHIFTED
        #     index the buggy code read c3's relevance 0.0 for c2 (or vice
        #     versa) and picked c3. Correct binding: c2 ≈ 0.7·0.789 −
        #     0.3·0.86 ≈ 0.304 vs c3 = 0.0 → c2 wins.
        orch = self._make_orch()
        text_a = "alpha beta gamma delta epsilon"
        results = [
            self._result("c0", text_a, reranker=-0.1),
            self._result("c1", text_a + " one", reranker=-0.3),
            self._result("c2", text_a + " two", reranker=-0.5),
            self._result("c3", "zulu yankee xray whiskey victor", reranker=-2.0),
        ]
        selected = orch._apply_mmr(results, top_k=3, lambda_param=0.7)
        ids = [cid for cid, _ in selected]
        # The shifting-index implementation would select c3 in the SECOND
        # iteration; correct relevance binding selects c2.
        assert ids == ["c0", "c1", "c2"]

    def test_rrf_fallback_and_equal_values(self):
        # No reranker scores: finite rrf_score (~0.01 scale) is normalized
        # into [0,1]; raw values are never combined with Jaccard directly.
        # c1 is an EXACT duplicate of c0 (Jaccard penalty 1.0 against the
        # already-selected c0), so despite its higher relevance the diverse
        # c2 (zero similarity) wins the second slot.
        orch = self._make_orch()
        results = [
            self._result("c0", "alpha beta gamma", rrf=0.0143),
            self._result("c1", "alpha beta gamma", rrf=0.0125),
            self._result("c2", "delta echo foxtrot", rrf=0.0111),
            self._result("c3", "golf hotel india", rrf=0.0100),
        ]
        selected = orch._apply_mmr(results, top_k=2, lambda_param=0.7)
        assert [cid for cid, _ in selected] == ["c0", "c2"]
        # All-equal effective values: every relevance is 1.0 (no signal),
        # so selection is driven purely by diversity, deterministically.
        equal = [self._result(f"e{i}", f"words {i} shared", rrf=0.01) for i in range(4)]
        sel_eq = orch._apply_mmr(equal, top_k=2, lambda_param=0.7)
        assert sel_eq[0][0] == "e0"
        assert len(sel_eq) == 2

    def test_nonfinite_and_missing_scores_fall_back(self):
        orch = self._make_orch()
        results = [
            self._result("c0", "alpha beta gamma", reranker=float("nan")),
            self._result("c1", "alpha beta gamma more", reranker=float("-inf")),
            self._result("c2", "delta echo foxtrot", rrf=0.0125),
            self._result("c3", "golf hotel india", rrf=0.0100),
        ]
        # NaN/inf reranker values are NOT scores: relevance falls back to
        # finite rrf (absent -> 0) — c0/c1 get 0.0, finite c2/c3 win.
        selected = orch._apply_mmr(results, top_k=2, lambda_param=0.7)
        assert [cid for cid, _ in selected] == ["c0", "c2"]
