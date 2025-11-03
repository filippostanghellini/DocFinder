"""Semantic search interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from docfinder.embedding.encoder import EmbeddingModel
from docfinder.index.storage import SQLiteVectorStore


@dataclass(slots=True)
class SearchResult:
    path: Path
    title: str
    chunk_index: int
    score: float
    text: str
    metadata: dict


class Searcher:
    """High-level API to query the vector store."""

    def __init__(self, embedder: EmbeddingModel, store: SQLiteVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, query: str, *, top_k: int = 10) -> List[SearchResult]:
        embedding = self.embedder.embed_query(query)
        rows = self.store.search(embedding, top_k=top_k)
        results: List[SearchResult] = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
            results.append(
                SearchResult(
                    path=Path(row["path"]),
                    title=row["title"],
                    chunk_index=row["chunk_index"],
                    score=float(row["score"]),
                    text=row["text"],
                    metadata=metadata,
                )
            )
        return results
