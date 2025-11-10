"""Tests for semantic search interface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from docfinder.index.search import SearchResult, Searcher


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
