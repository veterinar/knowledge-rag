"""Tests for MCP tool input validation and error handling.

Tests tool wrapper functions WITHOUT requiring ChromaDB or embeddings.
Validates input sanitization and error responses.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_orchestrator():
    """Create a mock orchestrator that returns predictable results."""
    mock = MagicMock()
    mock.query.return_value = [
        {
            "content": "test",
            "source": "test.md",
            "filename": "test.md",
            "category": "general",
            "chunk_index": 0,
            "score": 1.0,
            "raw_rrf_score": 0.016,
            "reranker_score": None,
            "semantic_rank": 1,
            "bm25_rank": 1,
            "search_method": "hybrid",
            "keywords": ["test"],
            "routed_by": "none",
        }
    ]
    mock.query_cache.stats.return_value = {"hit_rate": "0%"}
    mock.list_categories.return_value = {"general": 5}
    mock.list_documents.return_value = [{"id": "abc", "source": "test.md"}]
    mock.get_stats.return_value = {"total_documents": 5, "total_chunks": 50}
    mock.get_document.return_value = {"content": "doc content", "source": "test.md"}
    return mock


@pytest.fixture
def mock_orch():
    mock = _mock_orchestrator()
    with patch("mcp_server.server.get_orchestrator", return_value=mock):
        yield mock


class TestSearchKnowledge:
    def test_empty_query_error(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge(""))
        assert r["status"] == "error"

    def test_whitespace_query_error(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge("   "))
        assert r["status"] == "error"

    def test_invalid_category_error(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge("test", category="NONEXISTENT"))
        assert r["status"] == "error"

    def test_valid_query_success(self, mock_orch):
        from mcp_server.server import search_knowledge

        r = json.loads(search_knowledge("test query", snippet_mode=False))
        assert r["status"] == "success"
        assert r["result_count"] == 1

    def test_min_score_filters_low_results(self, mock_orch):
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "high",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.9,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": ["test"],
                "routed_by": "none",
            },
            {
                "content": "low",
                "source": "b.md",
                "filename": "b.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.1,
                "raw_rrf_score": 0.001,
                "reranker_score": None,
                "semantic_rank": None,
                "bm25_rank": 5,
                "search_method": "keyword",
                "keywords": ["test"],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.5, snippet_mode=False))
        assert r["result_count"] == 1
        assert r["filtered_by_score"] == 1
        assert r["results"][0]["content"] == "high"

    def test_min_score_zero_returns_all(self, mock_orch):
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "a",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.9,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
            {
                "content": "b",
                "source": "b.md",
                "filename": "b.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.0,
                "raw_rrf_score": 0.001,
                "reranker_score": None,
                "semantic_rank": None,
                "bm25_rank": 5,
                "search_method": "keyword",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.0, snippet_mode=False))
        assert r["result_count"] == 2
        assert r["filtered_by_score"] == 0

    def test_snippet_mode_truncates_long_content(self, mock_orch):
        from mcp_server.server import search_knowledge

        long_content = "A" * 300 + ". " + "B" * 300 + ". " + "C" * 300
        mock_orch.query.return_value = [
            {
                "content": long_content,
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", snippet_mode=True))
        result = r["results"][0]
        assert len(result["content"]) <= 510
        assert result["content_length"] == len(long_content)

    def test_snippet_mode_preserves_short_content(self, mock_orch):
        from mcp_server.server import search_knowledge

        short = "Short content here."
        mock_orch.query.return_value = [
            {
                "content": short,
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", snippet_mode=True))
        assert r["results"][0]["content"] == short
        assert r["results"][0]["content_length"] == len(short)

    def test_snippet_mode_false_returns_full(self, mock_orch):
        from mcp_server.server import search_knowledge

        long_content = "X" * 1000
        mock_orch.query.return_value = [
            {
                "content": long_content,
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", snippet_mode=False))
        assert r["results"][0]["content"] == long_content
        assert "content_length" not in r["results"][0]


class TestAddDocument:
    def test_empty_content_error(self, mock_orch):
        from mcp_server.server import add_document

        r = json.loads(add_document("", "test.md", "general"))
        assert r["status"] == "error"

    def test_empty_filepath_error(self, mock_orch):
        from mcp_server.server import add_document

        r = json.loads(add_document("content", "", "general"))
        assert r["status"] == "error"


class TestUpdateDocument:
    def test_empty_content_error(self, mock_orch):
        from mcp_server.server import update_document

        r = json.loads(update_document("somefile.md", ""))
        assert r["status"] == "error"

    def test_missing_filepath_error(self, mock_orch):
        from mcp_server.server import update_document

        r = json.loads(update_document("", "content"))
        assert r["status"] == "error"


class TestRemoveDocument:
    def test_empty_filepath_error(self, mock_orch):
        from mcp_server.server import remove_document

        r = json.loads(remove_document(""))
        assert r["status"] == "error"


class TestAddFromUrl:
    def test_empty_url_error(self, mock_orch):
        from mcp_server.server import add_from_url

        r = json.loads(add_from_url(""))
        assert r["status"] == "error"

    def test_file_scheme_blocked(self, mock_orch):
        from mcp_server.server import add_from_url

        mock_orch.add_from_url.return_value = {"error": "Only http:// and https:// URLs are supported"}
        r = json.loads(add_from_url("file:///etc/passwd"))
        assert r["status"] == "error"


class TestSearchSimilar:
    def test_empty_filepath_error(self, mock_orch):
        from mcp_server.server import search_similar

        r = json.loads(search_similar(""))
        assert r["status"] == "error"


class TestEvaluateRetrieval:
    def test_invalid_json_error(self, mock_orch):
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval("not json"))
        assert r["status"] == "error"

    def test_empty_array_error(self, mock_orch):
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval("[]"))
        assert r["status"] == "error"

    def test_case_missing_expected_filepath_error(self, mock_orch):
        """Requirement 8: a case without expected_filepath is a validation error."""
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval(json.dumps([{"query": "suid exploit"}])))
        assert r["status"] == "error"
        assert "expected_filepath" in r["message"]

    def test_case_empty_query_error(self, mock_orch):
        """Requirement 8: a case with an empty query is a validation error."""
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval(json.dumps([{"query": "  ", "expected_filepath": "a.md"}])))
        assert r["status"] == "error"

    def test_non_dict_case_error(self, mock_orch):
        """Requirement 8: a non-object entry is a validation error."""
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval(json.dumps(["just-a-string"])))
        assert r["status"] == "error"

    def test_valid_cases_offline_smoke_success(self, mock_orch):
        """Valid cases run the offline smoke and surface evaluation_mode."""
        from mcp_server.server import evaluate_retrieval

        mock_orch.evaluate_retrieval.return_value = {
            "evaluation_mode": "offline_smoke",
            "total_queries": 1,
            "mrr_at_5": 1.0,
            "recall_at_5": 1.0,
            "per_query": [{"query": "q", "expected": "security/a.md", "found_at_rank": 1}],
        }
        r = json.loads(evaluate_retrieval(json.dumps([{"query": "q", "expected_filepath": "security/a.md"}])))
        assert r["status"] == "success"
        assert r["evaluation_mode"] == "offline_smoke"
        assert r["mrr_at_5"] == 1.0

    @pytest.mark.parametrize(
        "case",
        [
            {"query": 123, "expected_filepath": "a.md"},        # numeric query
            {"query": None, "expected_filepath": "a.md"},      # null query
            {"query": "q", "expected_filepath": 12.5},         # numeric expected
            {"query": "q", "expected_filepath": ["a.md"]},     # list expected
        ],
        ids=["query-number", "query-null", "expected-number", "expected-list"],
    )
    def test_non_string_values_rejected_before_orchestrator(self, mock_orch, case):
        """Actual non-string query/expected_filepath values (no str()
        coercion) are a structured validation error, and the orchestrator
        is NEVER reached."""
        from mcp_server.server import evaluate_retrieval

        r = json.loads(evaluate_retrieval(json.dumps([case])))
        assert r["status"] == "error"
        assert "query" in r["message"] or "expected_filepath" in r["message"]
        mock_orch.evaluate_retrieval.assert_not_called()

    def test_direct_orchestrator_empty_cases_raise_valueerror_before_query(self):
        """Direct orchestrator call with [] raises ValueError BEFORE any
        query runs (zero-query success is forbidden). No model/DB needed."""
        from mcp_server.server import KnowledgeOrchestrator

        orch = object.__new__(KnowledgeOrchestrator)
        orch.query = MagicMock()

        with pytest.raises(ValueError):
            KnowledgeOrchestrator.evaluate_retrieval(orch, [])
        orch.query.assert_not_called()


# ---------------------------------------------------------------------------
# Requirement 8 — min_score filters on query_relative_score (legacy fallback)
# ---------------------------------------------------------------------------


class TestMinScoreQueryRelative:
    def test_min_score_uses_query_relative_not_legacy_score(self, mock_orch):
        """When a result carries both keys and they disagree (contrived legacy
        score), the filter must honor query_relative_score."""
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "high-rel",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.99,  # contrived legacy score — must be IGNORED
                "query_relative_score": 0.9,
                "raw_score": 12.5,
                "score_source": "rrf",
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
            {
                "content": "low-rel",
                "source": "b.md",
                "filename": "b.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.5,  # legacy score ABOVE threshold...
                "query_relative_score": 0.1,  # ...but relative score BELOW
                "raw_score": 1.0,
                "score_source": "rrf",
                "raw_rrf_score": 0.001,
                "reranker_score": None,
                "semantic_rank": None,
                "bm25_rank": 5,
                "search_method": "keyword",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.5, snippet_mode=False))

        assert r["result_count"] == 1
        assert r["filtered_by_score"] == 1
        assert r["filtered_by_query_relative_score"] == 1
        assert r["results"][0]["content"] == "high-rel"

    def test_min_score_falls_back_to_legacy_score_when_key_absent(self, mock_orch):
        """Legacy results lacking query_relative_score still filter on score."""
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "kept",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.9,  # legacy-only result, no query_relative_score key
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
            {
                "content": "dropped",
                "source": "b.md",
                "filename": "b.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.1,
                "raw_rrf_score": 0.001,
                "reranker_score": None,
                "semantic_rank": None,
                "bm25_rank": 5,
                "search_method": "keyword",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.5, snippet_mode=False))

        assert r["result_count"] == 1
        assert r["filtered_by_score"] == 1
        assert r["filtered_by_query_relative_score"] == 1
        assert r["results"][0]["content"] == "kept"

    def test_query_relative_score_zero_filters_out_high_legacy_score(self, mock_orch):
        """A zero query_relative_score must be dropped even if score says 1.0."""
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "conflicting",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 1.0,
                "query_relative_score": 0.0,
                "raw_score": 3.0,
                "score_source": "reranker",
                "raw_rrf_score": 0.02,
                "reranker_score": 3.0,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.5, snippet_mode=False))

        assert r["result_count"] == 0
        assert r["filtered_by_score"] == 1
        assert r["filtered_by_query_relative_score"] == 1

    def test_filtered_by_query_relative_score_alias_matches_legacy_field(self, mock_orch):
        """Both envelope counters report the same number in every filter regime."""
        from mcp_server.server import search_knowledge

        mock_orch.query.return_value = [
            {
                "content": "edge",
                "source": "a.md",
                "filename": "a.md",
                "category": "general",
                "chunk_index": 0,
                "score": 0.5,
                "query_relative_score": 0.5,  # exactly at threshold → kept
                "raw_score": 5.0,
                "score_source": "rrf",
                "raw_rrf_score": 0.02,
                "reranker_score": None,
                "semantic_rank": 1,
                "bm25_rank": 1,
                "search_method": "hybrid",
                "keywords": [],
                "routed_by": "none",
            },
        ]
        r = json.loads(search_knowledge("test", min_score=0.5, snippet_mode=False))

        assert r["result_count"] == 1, "threshold comparison must be inclusive (>=)"
        assert r["filtered_by_score"] == 0
        assert r["filtered_by_query_relative_score"] == 0
