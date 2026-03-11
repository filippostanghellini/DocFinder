"""Document indexing pipeline."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from docfinder.embedding.encoder import EmbeddingModel
from docfinder.index.storage import SQLiteVectorStore
from docfinder.ingestion.pdf_loader import build_chunks
from docfinder.models import DocumentMetadata
from docfinder.utils.files import compute_sha256, iter_document_paths

LOGGER = logging.getLogger(__name__)


def find_documents(
    paths: Sequence[Path],
    exclude: frozenset[str] | None = None,
) -> list[Path]:
    """Find all supported documents under the given paths, honouring exclusions."""
    docs = list(iter_document_paths(paths))
    if exclude:
        docs = [p for p in docs if str(p) not in exclude]
    return docs


# Keep old name as alias so existing tests don't break
def find_pdfs(paths: Sequence[Path]) -> list[Path]:
    return [p for p in find_documents(paths) if p.suffix.lower() == ".pdf"]


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
        embed_batch_size: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.embed_batch_size = embed_batch_size
        self.progress_callback = progress_callback

    def index(
        self,
        paths: Sequence[Path],
        *,
        exclude_paths: frozenset[str] | None = None,
    ) -> IndexStats:
        """Index all supported documents found under the given paths."""
        doc_files = find_documents(paths, exclude_paths)
        if not doc_files:
            LOGGER.warning("No supported documents found")
            return IndexStats()

        total = len(doc_files)
        stats = IndexStats()

        for i, path in enumerate(doc_files):
            if self.progress_callback:
                self.progress_callback(i, total, str(path))
            try:
                LOGGER.info(f"Processing: {path}")
                status = self._index_single(path)
                stats.increment(status, path)
            except Exception as e:
                LOGGER.error(f"Failed to process {path}: {e}")
                stats.failed += 1
                stats.processed_files.append(path)
            gc.collect()

        if self.progress_callback:
            self.progress_callback(total, total, "")

        return stats

    def _index_single(self, path: Path) -> str:
        """Index a single document file."""
        import itertools

        chunk_gen = build_chunks(path, max_chars=self.chunk_chars, overlap=self.overlap)

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

        with self.store.transaction():
            doc_id, status = self.store.init_document(document)
            if status == "skipped":
                return status

            batch_size = 64
            current_batch = []

            for chunk in itertools.chain([first_chunk], chunk_gen):
                current_batch.append(chunk)
                if len(current_batch) >= batch_size:
                    embeddings = self.embedder.embed(
                        [c.text for c in current_batch],
                        batch_size=self.embed_batch_size,
                    )
                    self.store.insert_chunks(doc_id, current_batch, embeddings)
                    current_batch = []

            if current_batch:
                embeddings = self.embedder.embed(
                    [c.text for c in current_batch],
                    batch_size=self.embed_batch_size,
                )
                self.store.insert_chunks(doc_id, current_batch, embeddings)

        return status
