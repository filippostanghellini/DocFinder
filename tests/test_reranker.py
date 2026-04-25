"""Tests for cross-encoder reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from docfinder.index.reranker import Reranker, _sigmoid


class TestReranker:
    """Test Reranker class."""

    def test_lazy_model_loading(self) -> None:
        """Model should not be loaded until rerank() is called."""
        reranker = Reranker()
        assert reranker._model is None

    @patch("docfinder.index.reranker.CrossEncoder", create=True)
    def test_model_loaded_on_first_rerank(self, mock_ce_cls: MagicMock) -> None:
        """Model should be loaded on first rerank() call."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        mock_ce_cls.return_value = mock_model

        reranker = Reranker()
        with patch("sentence_transformers.CrossEncoder", mock_ce_cls):
            reranker.rerank("query", [{"text": "doc", "score": 0.5}])

        assert reranker._model is mock_model

    def test_rerank_empty_results(self) -> None:
        """Should return empty list for empty input without loading model."""
        reranker = Reranker()
        result = reranker.rerank("query", [])
        assert result == []
        assert reranker._model is None

    def test_rerank_preserves_embedding_score(self) -> None:
        """Original score should be preserved as embedding_score."""
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.6])
        reranker._model = mock_model

        results = [
            {"text": "doc1", "score": 0.95},
            {"text": "doc2", "score": 0.90},
        ]
        reranked = reranker.rerank("query", results)

        assert reranked[0]["embedding_score"] == 0.95
        assert reranked[1]["embedding_score"] == 0.90

    def test_rerank_sorts_by_rerank_score(self) -> None:
        """Results should be sorted by rerank_score descending."""
        reranker = Reranker()
        mock_model = MagicMock()
        # Second result gets higher rerank score
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])
        reranker._model = mock_model

        results = [
            {"text": "doc1", "score": 0.95},
            {"text": "doc2", "score": 0.85},
            {"text": "doc3", "score": 0.75},
        ]
        reranked = reranker.rerank("query", results)

        assert reranked[0]["rerank_score"] == 0.9
        assert reranked[1]["rerank_score"] == 0.6
        assert reranked[2]["rerank_score"] == 0.3

    def test_rerank_score_becomes_final_score(self) -> None:
        """The final 'score' key should be sigmoid(rerank_score)."""
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.42])
        reranker._model = mock_model

        results = [{"text": "doc", "score": 0.99}]
        reranked = reranker.rerank("query", results)

        assert reranked[0]["score"] == _sigmoid(0.42)
        assert reranked[0]["rerank_score"] == 0.42
        assert reranked[0]["embedding_score"] == 0.99

    def test_rerank_top_k_trimming(self) -> None:
        """Should return at most top_k results."""
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        reranker._model = mock_model

        results = [{"text": f"doc{i}", "score": 0.5} for i in range(5)]
        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0]["rerank_score"] == 0.9
        assert reranked[1]["rerank_score"] == 0.8

    def test_rerank_passes_correct_pairs(self) -> None:
        """Should pass (query, text) pairs to the cross-encoder."""
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.6])
        reranker._model = mock_model

        results = [
            {"text": "first document", "score": 0.9},
            {"text": "second document", "score": 0.8},
        ]
        reranker.rerank("my query", results)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [("my query", "first document"), ("my query", "second document")]

    def test_default_model_name(self) -> None:
        """Should use the default model name."""
        reranker = Reranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-2-v2"

    def test_custom_model_name(self) -> None:
        """Should accept a custom model name."""
        reranker = Reranker(model_name="custom/model")
        assert reranker.model_name == "custom/model"

    def test_rerank_preserves_extra_keys(self) -> None:
        """Should preserve extra keys in result dicts."""
        reranker = Reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.7])
        reranker._model = mock_model

        results = [{"text": "doc", "score": 0.5, "path": "/test.pdf", "chunk_index": 3}]
        reranked = reranker.rerank("query", results)

        assert reranked[0]["path"] == "/test.pdf"
        assert reranked[0]["chunk_index"] == 3
