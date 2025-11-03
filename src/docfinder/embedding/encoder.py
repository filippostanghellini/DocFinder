"""Embedding model management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 16
    normalize: bool = True


class EmbeddingModel:
    """Thin wrapper around `SentenceTransformer` for query and document embeddings."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = SentenceTransformer(self.config.model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        """Return float32 embeddings for input texts."""
        sentences = list(texts)
        embeddings = self._model.encode(
            sentences,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        return embeddings.astype("float32", copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience wrapper for single-query embedding."""
        return self.embed([text])[0]
