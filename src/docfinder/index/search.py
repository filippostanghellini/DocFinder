"""Semantic search interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from docfinder.embedding.encoder import EmbeddingModel
from docfinder.index.reranker import Reranker
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

    def __init__(
        self,
        embedder: EmbeddingModel,
        store: SQLiteVectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

    def search(self, query: str, *, top_k: int = 10) -> List[SearchResult]:
        # When reranking, fetch more candidates for the cross-encoder to evaluate
        fetch_k = max(top_k * 3, 30) if self.reranker is not None else top_k

        embedding = self.embedder.embed_query(query)
        rows = self.store.search(embedding, top_k=fetch_k)

        if self.reranker is not None and rows:
            # Build dicts for the reranker
            candidates = []
            for row in rows:
                candidates.append(
                    {
                        "path": row["path"],
                        "title": row["title"],
                        "chunk_index": row["chunk_index"],
                        "score": float(row["score"]),
                        "text": row["text"],
                        "metadata": row.get("metadata"),
                    }
                )
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            results: List[SearchResult] = []
            for r in reranked:
                metadata = json.loads(r["metadata"]) if r.get("metadata") else {}
                results.append(
                    SearchResult(
                        path=Path(r["path"]),
                        title=r["title"],
                        chunk_index=r["chunk_index"],
                        score=float(r["score"]),
                        text=r["text"],
                        metadata=metadata,
                    )
                )
            return results

        results = []
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
