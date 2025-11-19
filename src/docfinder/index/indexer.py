"""Document indexing pipeline."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from docfinder.embedding.encoder import EmbeddingModel
from docfinder.index.storage import SQLiteVectorStore
from docfinder.ingestion.pdf_loader import build_chunks
from docfinder.models import DocumentMetadata
from docfinder.utils.files import compute_sha256, iter_pdf_paths

LOGGER = logging.getLogger(__name__)


def find_pdfs(paths: Sequence[Path]) -> list[Path]:
    """Find all PDF files under the given paths."""
    return list(iter_pdf_paths(paths))


@dataclass(slots=True)
class IndexStats:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    processed_files: list[Path] = field(default_factory=list)

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

    def index(self, paths: Sequence[Path]) -> IndexStats:
        """Index all PDFs found under the given paths."""
        pdf_files = find_pdfs(paths)
        if not pdf_files:
            LOGGER.warning("No PDF files found")
            return IndexStats()

        stats = IndexStats()

        # Processa solo 2 file alla volta per ridurre memoria
        batch_size = 2

        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i : i + batch_size]

            for path in batch:
                try:
                    LOGGER.info(f"Processing: {path}")
                    status = self._index_single(path)
                    stats.increment(status, path)

                except Exception as e:
                    LOGGER.error(f"Failed to process {path}: {e}")
                    stats.failed += 1
                    stats.processed_files.append(path)

            # Libera memoria dopo ogni batch
            gc.collect()

        return stats

    def _index_single(self, path: Path) -> str:
        """Index a single PDF file."""
        import itertools
        
        # Create generator
        chunk_gen = build_chunks(path, max_chars=self.chunk_chars, overlap=self.overlap)
        
        # Peek at first chunk to get metadata and ensure we have content
        try:
            first_chunk = next(chunk_gen)
        except StopIteration:
            LOGGER.warning("No text extracted from %s", path)
            return "skipped"

        sha256 = compute_sha256(path)
        stat = path.stat()
        document = DocumentMetadata(
            path=path,
            title=first_chunk.metadata.get("title", path.stem),
            sha256=sha256,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )

        # Use a transaction for the whole document update
        with self.store.transaction():
            doc_id, status = self.store.init_document(document)
            if status == "skipped":
                return status

            # Process chunks in batches
            batch_size = 32
            current_batch = []
            
            # Chain the first chunk back with the rest
            for chunk in itertools.chain([first_chunk], chunk_gen):
                current_batch.append(chunk)
                
                if len(current_batch) >= batch_size:
                    embeddings = self.embedder.embed([c.text for c in current_batch])
                    self.store.insert_chunks(doc_id, current_batch, embeddings)
                    current_batch = []
                    gc.collect()
            
            # Process remaining chunks
            if current_batch:
                embeddings = self.embedder.embed([c.text for c in current_batch])
                self.store.insert_chunks(doc_id, current_batch, embeddings)
                gc.collect()

        return status
