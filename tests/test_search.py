"""Tests for semantic search interface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from docfinder.index.reranker import Reranker
from docfinder.index.search import Searcher, SearchResult


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_create_search_result(self) -> None:
        """Should create SearchResult with all fields."""
        result = SearchResult(
            path=Path("/test.pdf"),
            title="Test Document",
            chunk_index=5,
            score=0.85,
            text="This is the chunk text",
            metadata={"page": 1},
        )

        assert result.path == Path("/test.pdf")
        assert result.title == "Test Document"
        assert result.chunk_index == 5
        assert result.score == 0.85
        assert result.text == "This is the chunk text"
        assert result.metadata == {"page": 1}


class TestSearcher:
    """Test Searcher class."""

    def test_search_simple(self) -> None:
        """Should perform search and return results."""
        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2, 0.3])

        # Mock store
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "path": "/doc1.pdf",
                "title": "Document 1",
                "chunk_index": 0,
                "score": 0.95,
                "text": "Relevant text",
                "metadata": json.dumps({"page": 1}),
            }
        ]

        searcher = Searcher(mock_embedder, mock_store)
        results = searcher.search("test query", top_k=10)

        assert len(results) == 1
        assert results[0].path == Path("/doc1.pdf")
        assert results[0].title == "Document 1"
        assert results[0].score == 0.95
        assert results[0].metadata == {"page": 1}

        # Verify embedder was called
        mock_embedder.embed_query.assert_called_once_with("test query")
        # Verify store was called with embedding and top_k
        mock_store.search.assert_called_once()

    def test_search_multiple_results(self) -> None:
        """Should return multiple search results."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2])

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "path": "/doc1.pdf",
                "title": "Doc 1",
                "chunk_index": 0,
                "score": 0.95,
                "text": "Text 1",
                "metadata": "{}",
            },
            {
                "path": "/doc2.pdf",
                "title": "Doc 2",
                "chunk_index": 5,
                "score": 0.85,
                "text": "Text 2",
                "metadata": "{}",
            },
            {
                "path": "/doc3.pdf",
                "title": "Doc 3",
                "chunk_index": 2,
                "score": 0.75,
                "text": "Text 3",
                "metadata": "{}",
            },
        ]

        searcher = Searcher(mock_embedder, mock_store)
        results = searcher.search("query", top_k=3)

        assert len(results) == 3
        assert results[0].score == 0.95
        assert results[1].score == 0.85
        assert results[2].score == 0.75

    def test_search_empty_results(self) -> None:
        """Should handle empty search results."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        mock_store = MagicMock()
        mock_store.search.return_value = []

        searcher = Searcher(mock_embedder, mock_store)
        results = searcher.search("no results query")

        assert len(results) == 0

    def test_search_no_metadata(self) -> None:
        """Should handle results without metadata."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "path": "/doc.pdf",
                "title": "Doc",
                "chunk_index": 0,
                "score": 0.9,
                "text": "Text",
                "metadata": None,
            }
        ]

        searcher = Searcher(mock_embedder, mock_store)
        results = searcher.search("query")

        assert len(results) == 1
        assert results[0].metadata == {}

    def test_search_custom_top_k(self) -> None:
        """Should respect custom top_k parameter."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        mock_store = MagicMock()
        mock_store.search.return_value = []

        searcher = Searcher(mock_embedder, mock_store)
        searcher.search("query", top_k=25)

        # Verify top_k was passed to store
        call_args = mock_store.search.call_args
        assert call_args[1]["top_k"] == 25

    def test_search_passes_folder_filters(self) -> None:
        """Should pass selected folders down to storage search."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        mock_store = MagicMock()
        mock_store.search.return_value = []

        searcher = Searcher(mock_embedder, mock_store)
        folders = ["/Users/test/articles", "/Users/test/posters"]
        searcher.search("query", top_k=10, folders=folders)

        call_args = mock_store.search.call_args
        assert call_args[1]["folders"] == folders

    def test_search_metadata_parsing(self) -> None:
        """Should parse JSON metadata correctly."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        complex_metadata = {"page": 5, "section": "intro", "tags": ["important", "review"]}

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "path": "/doc.pdf",
                "title": "Doc",
                "chunk_index": 0,
                "score": 0.9,
                "text": "Text",
                "metadata": json.dumps(complex_metadata),
            }
        ]

        searcher = Searcher(mock_embedder, mock_store)
        results = searcher.search("query")

        assert results[0].metadata == complex_metadata

    def test_search_without_reranker(self) -> None:
        """Should work identically when reranker is None."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2])

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "path": "/doc.pdf",
                "title": "Doc",
                "chunk_index": 0,
                "score": 0.9,
                "text": "Text",
                "metadata": "{}",
            }
        ]

        searcher = Searcher(mock_embedder, mock_store, reranker=None)
        results = searcher.search("query", top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.9
        # Without reranker, top_k is passed directly to store
        call_args = mock_store.search.call_args
        assert call_args[1]["top_k"] == 5

    def test_search_with_reranker_fetches_more_candidates(self) -> None:
        """With reranker, should fetch more candidates from the store."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "path": f"/doc{i}.pdf",
                "title": f"Doc {i}",
                "chunk_index": 0,
                "score": 0.9 - i * 0.01,
                "text": f"Text {i}",
                "metadata": "{}",
            }
            for i in range(30)
        ]

        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = [
            {
                "path": "/doc5.pdf",
                "title": "Doc 5",
                "chunk_index": 0,
                "score": 0.99,
                "text": "Text 5",
                "metadata": "{}",
            }
        ]

        searcher = Searcher(mock_embedder, mock_store, reranker=mock_reranker)
        results = searcher.search("query", top_k=5)

        # Should fetch max(5*3, 30) = 30 candidates
        call_args = mock_store.search.call_args
        assert call_args[1]["top_k"] == 30

        # Should have called reranker with the candidates
        mock_reranker.rerank.assert_called_once()
        rerank_args = mock_reranker.rerank.call_args
        assert rerank_args[1]["top_k"] == 5

        # Should return the reranked result
        assert len(results) == 1
        assert results[0].path == Path("/doc5.pdf")
        assert results[0].score == 0.99

    def test_search_with_reranker_empty_results(self) -> None:
        """With reranker, should handle empty store results gracefully."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([0.1])

        mock_store = MagicMock()
        mock_store.search.return_value = []

        mock_reranker = MagicMock(spec=Reranker)

        searcher = Searcher(mock_embedder, mock_store, reranker=mock_reranker)
        results = searcher.search("query", top_k=5)

        assert len(results) == 0
        # Reranker should not be called for empty results
        mock_reranker.rerank.assert_not_called()
