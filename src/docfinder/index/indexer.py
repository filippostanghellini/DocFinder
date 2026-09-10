"""Document indexing pipeline."""

from __future__ import annotations

import gc
import logging
import os
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from docfinder.embedding.encoder import EmbeddingModel
from docfinder.index.storage import SQLiteVectorStore
from docfinder.ingestion.pdf_loader import build_chunks
from docfinder.models import ChunkRecord, DocumentMetadata
from docfinder.utils.files import compute_sha256, iter_document_paths
from docfinder.utils.memory import get_memory_info

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


def _parse_document(args: tuple) -> dict | None:
    """Parse a single document into chunks (runs in worker process).

    This is a top-level function so it can be pickled by multiprocessing.
    Returns a dict with path, metadata, and chunk texts, or None on failure.
    """
    import sys
    import traceback

    path_str, chunk_chars, overlap = args
    path = Path(path_str)

    # Worker-local logger to ensure output is captured
    worker_logger = logging.getLogger(f"{__name__}.worker[{os.getpid()}]")
    worker_logger.info("Worker starting to parse: %s", path_str)

    try:
        chunks = list(build_chunks(path, max_chars=chunk_chars, overlap=overlap))
        worker_logger.info("Worker extracted %d chunks from: %s", len(chunks), path_str)

        if not chunks:
            worker_logger.warning("Worker found no chunks in: %s", path_str)
            return {"path": path_str, "status": "empty"}

        sha256 = compute_sha256(path)
        stat = path.stat()
        worker_logger.info(
            "Worker done: %s (size=%d, chunks=%d)",
            path_str,
            stat.st_size,
            len(chunks),
        )

        return {
            "path": path_str,
            "status": "ok",
            "sha256": sha256,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "title": chunks[0].metadata.get("title", path.stem),
            "chunks": [{"index": c.index, "text": c.text, "metadata": c.metadata} for c in chunks],
        }
    except Exception as e:
        exc_details = "".join(traceback.format_exception(*sys.exc_info()))
        worker_logger.error("Worker failed for %s: %s\n%s", path_str, e, exc_details)
        return {"path": path_str, "status": "error", "error": f"{type(e).__name__}: {e}"}


def _parse_document_safe(args: tuple) -> dict | None:
    """Safe wrapper for _parse_document with enhanced error reporting."""
    path_str, chunk_chars, overlap = args
    try:
        return _parse_document(args)
    except Exception as e:
        import sys
        import traceback

        exc_info = "".join(traceback.format_exception(*sys.exc_info()))
        LOGGER.error("Worker exception for %s: %s\n%s", path_str, e, exc_info)
        return {"path": path_str, "status": "error", "error": f"Worker exception: {e}"}


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
        self.last_num_workers: int = 1

    def index(
        self,
        paths: Sequence[Path],
        *,
        exclude_paths: frozenset[str] | None = None,
    ) -> IndexStats:
        """Index documents with parallel parsing when beneficial."""
        if self.store.ensure_embedding_model(self.embedder.config.model_name):
            LOGGER.warning("Embedding model changed — the existing index was cleared")

        doc_files = find_documents(paths, exclude_paths)
        if not doc_files:
            LOGGER.warning("No supported documents found")
            return IndexStats()

        total = len(doc_files)
        stats = IndexStats()

        if total >= 4 and self._should_parallelize():
            self._index_parallel(doc_files, stats, total)
        else:
            self._index_sequential(doc_files, stats, total)

        if self.progress_callback:
            self.progress_callback(total, total, "")

        return stats

    def _should_parallelize(self) -> bool:
        """Return True if parallel parsing should be used.

        Returns False when a fixed embed_batch_size is set, which typically
        indicates a test environment where multiprocessing adds unnecessary
        complexity.
        """
        return self.embed_batch_size is None

    def _index_sequential(
        self,
        doc_files: list[Path],
        stats: IndexStats,
        total: int,
    ) -> None:
        """Index files sequentially (original path)."""
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
        self.last_num_workers = 1

    def _index_parallel(
        self,
        doc_files: list[Path],
        stats: IndexStats,
        total: int,
    ) -> None:
        """Parse documents in parallel, then embed and store sequentially."""
        num_workers = self._compute_parallel_workers(len(doc_files))
        self.last_num_workers = num_workers
        LOGGER.info(
            "Parallel parsing %d documents with %d workers: %s",
            len(doc_files),
            num_workers,
            [str(p) for p in doc_files],
        )

        args = [(str(p), self.chunk_chars, self.overlap) for p in doc_files]

        results: list | None = None
        parsing_mode = "parallel"
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                LOGGER.info("ProcessPoolExecutor started with %d workers", num_workers)
                results = list(executor.map(_parse_document, args))
                LOGGER.info("ProcessPoolExecutor completed, got %d results", len(results))
        except (BrokenExecutor, OSError) as exc:
            parsing_mode = "sequential"
            LOGGER.warning(
                "Process pool crashed during parallel parsing: %s. "
                "Falling back to sequential processing for %d documents.",
                exc,
                len(doc_files),
            )
            self.last_num_workers = 1
            results = None
            try:
                LOGGER.info("Starting sequential parsing of %d documents", len(doc_files))
                results = []
                for i, arg in enumerate(args):
                    LOGGER.info(
                        "Sequential: processing document %d/%d: %s",
                        i + 1,
                        len(args),
                        arg[0],
                    )
                    try:
                        result = _parse_document(arg)
                        results.append(result)
                        status = result.get("status") if result else "None"
                        LOGGER.info(
                            "Sequential: completed document %d: status=%s",
                            i + 1,
                            status,
                        )
                    except Exception as doc_err:
                        LOGGER.error("Sequential parsing failed for %s: %s", arg[0], doc_err)
                        results.append({"path": arg[0], "status": "error", "error": str(doc_err)})
                LOGGER.info("Sequential parsing completed, got %d results", len(results))
            except Exception as seq_err:
                LOGGER.error("Sequential fallback also failed: %s", seq_err)
                results = []

        if results is None:
            LOGGER.error("Parallel parsing returned no results")
            for path in doc_files:
                stats.failed += 1
                stats.processed_files.append(path)
            return

        LOGGER.info(
            "Starting embedding/storage for %d docs (parsing: %s)",
            len(results),
            parsing_mode,
        )

        # Use conservative batch size to avoid memory issues on macOS
        mem_info = get_memory_info()
        available_mb = mem_info.get("available_mb")

        batch_size = 16
        if available_mb is not None:
            if available_mb >= 16384:
                batch_size = 64
            elif available_mb >= 8192:
                batch_size = 32
            elif available_mb >= 4096:
                batch_size = 16
            elif available_mb >= 2048:
                batch_size = 8
            else:
                batch_size = 4

        LOGGER.info(
            "Using embedding batch_size=%d (platform=%s, available RAM: %s MB)",
            batch_size,
            sys.platform,
            available_mb,
        )

        for i, result in enumerate(results):
            path = doc_files[i]
            if self.progress_callback:
                self.progress_callback(i, total, str(path))
            try:
                if result is None:
                    LOGGER.error("Worker returned None for %s", path)
                    stats.failed += 1
                    stats.processed_files.append(path)
                    continue

                if result["status"] == "error":
                    LOGGER.error("Failed to parse %s: %s", path, result.get("error", "unknown"))
                    stats.failed += 1
                    stats.processed_files.append(path)
                    continue

                if result["status"] == "empty":
                    LOGGER.warning("No text extracted from %s", path)
                    stats.increment("skipped", path)
                    continue

                status = self._embed_and_store(path, result, batch_size)
                LOGGER.info("Embedded/stored document %d/%d: %s", i + 1, len(results), status)
                stats.increment(status, path)
            except Exception as e:
                LOGGER.error("Failed to process %s: %s", path, e)
                stats.failed += 1
                stats.processed_files.append(path)
            gc.collect()

    def _embed_and_store(self, path: Path, parsed: dict, batch_size: int = 32) -> str:
        """Embed pre-parsed chunks and store them in the database."""
        num_chunks = len(parsed["chunks"])
        LOGGER.info(
            "Embedding %d chunks from %s (batch_size=%d)",
            num_chunks,
            path.name,
            batch_size,
        )

        document = DocumentMetadata(
            path=path,
            title=parsed["title"],
            sha256=parsed["sha256"],
            mtime=parsed["mtime"],
            size=parsed["size"],
        )

        with self.store.transaction():
            doc_id, status = self.store.init_document(document)
            if status == "skipped":
                return status

            chunk_dicts = parsed["chunks"]
            chunks = [
                ChunkRecord(
                    document_path=path,
                    index=cd["index"],
                    text=cd["text"],
                    metadata=cd["metadata"],
                )
                for cd in chunk_dicts
            ]

            embed_batch_size = min(batch_size, len(chunks))
            for start in range(0, len(chunks), embed_batch_size):
                batch = chunks[start : start + embed_batch_size]
                LOGGER.debug(
                    "Embedding batch %d-%d of %d chunks",
                    start,
                    min(start + embed_batch_size, len(chunks)),
                    len(chunks),
                )
                embeddings = self.embedder.embed(
                    [c.text for c in batch],
                    batch_size=self.embed_batch_size,
                )
                self.store.insert_chunks(doc_id, batch, embeddings)
                gc.collect()  # Collect after each batch to free memory

        return status

    def _compute_parallel_workers(self, doc_count: int) -> int:
        """Compute balanced worker count for parallel document parsing."""
        # Use 1 worker on macOS to avoid ProcessPoolExecutor crashes
        # TODO: investigate why workers die on macOS
        if sys.platform == "darwin":
            base_workers = 1
        else:
            cpu_count = os.cpu_count() or 1
            base_workers = min(cpu_count, max(1, doc_count))

            # Balanced mode: keep one CPU for responsiveness when possible.
            if base_workers > 2:
                base_workers -= 1

        mem_info = get_memory_info()
        available_mb = mem_info.get("available_mb")
        if isinstance(available_mb, int):
            if available_mb < 2048:
                base_workers = min(base_workers, 2)
            elif available_mb < 4096:
                base_workers = min(base_workers, 3)
            elif available_mb < 8192:
                base_workers = min(base_workers, 6)
            else:
                base_workers = min(base_workers, 8)

        return max(1, min(base_workers, doc_count))

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
