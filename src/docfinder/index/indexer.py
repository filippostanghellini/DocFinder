"""Document indexing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from docfinder.embedding.encoder import EmbeddingModel
from docfinder.index.storage import SQLiteVectorStore
from docfinder.ingestion.pdf_loader import build_chunks
from docfinder.models import ChunkRecord, DocumentMetadata
from docfinder.utils.files import compute_sha256, iter_pdf_paths

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexStats:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    processed_files: List[Path] = field(default_factory=list)

    def increment(self, status: str, path: Path) -> None:
        if status == "inserted":
            self.inserted += 1
        elif status == "updated":
            self.updated += 1
        elif status == "skipped":
            self.skipped += 1
        else:
            self.failed += 1
        self.processed_files.append(path)


class Indexer:
    """Coordinates document ingestion and persistence."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        store: SQLiteVectorStore,
        *,
        chunk_chars: int = 1200,
        overlap: int = 200,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    def index(self, inputs: Iterable[Path]) -> IndexStats:
        stats = IndexStats()
        for path in iter_pdf_paths(inputs):
            try:
                status = self._index_single(path)
                stats.increment(status, path)
                LOGGER.info("%s -> %s", path, status)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.exception("Failed to index %s: %s", path, exc)
                stats.increment("failed", path)
        return stats

    def _index_single(self, path: Path) -> str:
        chunk_records = list(
            build_chunks(path, max_chars=self.chunk_chars, overlap=self.overlap)
        )
        if not chunk_records:
            LOGGER.warning("No text extracted from %s", path)
            return "skipped"

        sha256 = compute_sha256(path)
        stat = path.stat()
        document = DocumentMetadata(
            path=path,
            title=chunk_records[0].metadata.get("title", path.stem),
            sha256=sha256,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )

        embeddings = self.embedder.embed([chunk.text for chunk in chunk_records])
        return self.store.upsert_document(document, chunk_records, embeddings)
