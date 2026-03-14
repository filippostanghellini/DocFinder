"""Cross-encoder reranker for improving search precision."""

from __future__ import annotations

import logging
import math
from typing import List

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Convert a raw logit to a 0-1 probability."""
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    """Re-scores search results using a cross-encoder model.

    Cross-encoders evaluate (query, document) pairs jointly, capturing
    semantic relationships that bi-encoder cosine similarity misses.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded reranker model: %s", self.model_name)

    def rerank(self, query: str, results: List[dict], *, top_k: int = 10) -> List[dict]:
        """Re-rank results using the cross-encoder.

        Args:
            query: The search query.
            results: List of dicts with at least "text" and "score" keys.
            top_k: Number of top results to return after reranking.

        Returns:
            Re-ranked and trimmed list of result dicts, with updated scores.
        """
        if not results:
            return results

        self._ensure_model()

        pairs = [(query, r["text"]) for r in results]
        scores = self._model.predict(pairs)

        for r, rerank_score in zip(results, scores):
            r["embedding_score"] = r["score"]
            r["rerank_score"] = float(rerank_score)
            r["score"] = _sigmoid(float(rerank_score))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
