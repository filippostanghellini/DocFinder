"""Cross-encoder reranker for improving search precision (with multimodal passthrough)."""

from __future__ import annotations

import json
import logging
import math
from typing import List

from docfinder.embedding.encoder import is_hf_offline

logger = logging.getLogger(__name__)

_NON_TEXT_MODALITIES = frozenset({"image", "pdf_page_image", "audio", "video"})


def _sigmoid(x: float) -> float:
    """Convert a raw logit to a 0-1 probability."""
    return 1.0 / (1.0 + math.exp(-x))


def _is_non_text_candidate(candidate: dict) -> bool:
    """Return True if the candidate represents a visual/audio item without prose text."""
    raw_meta = candidate.get("metadata")
    if isinstance(raw_meta, str) and raw_meta:
        try:
            meta = json.loads(raw_meta)
        except Exception:
            meta = {}
    elif isinstance(raw_meta, dict):
        meta = raw_meta
    else:
        meta = {}
    return meta.get("modality") in _NON_TEXT_MODALITIES


class Reranker:
    """Re-scores search results using a cross-encoder model.

    Cross-encoders evaluate (query, document) pairs jointly, capturing
    semantic relationships that bi-encoder cosine similarity misses.
    Non-text multimodal items (images, scanned PDF pages, audio/video) preserve
    their EmbeddingGemma 2 cosine similarity score so the text-only cross-encoder
    does not penalize them.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, local_files_only=is_hf_offline())
            logger.info("Loaded reranker model: %s", self.model_name)

    def rerank(self, query: str, results: List[dict], *, top_k: int = 10) -> List[dict]:
        """Re-rank results using the cross-encoder while preserving multimodal scores."""
        if not results:
            return results

        text_indices: list[int] = []
        pairs: list[tuple[str, str]] = []
        for idx, r in enumerate(results):
            if _is_non_text_candidate(r):
                r["embedding_score"] = r["score"]
                r["rerank_score"] = float(r["score"])
                # Map cosine similarity [-1, 1] onto a comparable [0, 1] confidence scale
                r["score"] = max(0.0, min(1.0, float(r["score"])))
            else:
                text_indices.append(idx)
                pairs.append((query, r["text"]))

        if pairs:
            self._ensure_model()
            scores = self._model.predict(pairs)
            for idx, rerank_score in zip(text_indices, scores):
                r = results[idx]
                r["embedding_score"] = r["score"]
                r["rerank_score"] = float(rerank_score)
                r["score"] = _sigmoid(float(rerank_score))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
