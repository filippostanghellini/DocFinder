"""PDF loading and chunking utilities.

Uses PyMuPDF (fitz) for fast PDF text extraction.
PyMuPDF is typically 2-10x faster than pypdf for text extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator

import fitz  # PyMuPDF

from docfinder.models import ChunkRecord
from docfinder.utils.text import normalize_whitespace

LOGGER = logging.getLogger(__name__)


def iter_text_parts(path: Path) -> Iterator[str]:
    """Yield text content from a PDF file page by page.
    
    Uses PyMuPDF for faster text extraction compared to pypdf.
    """
    try:
        doc = fitz.open(path)
    except Exception as exc:
        LOGGER.error("Failed to open PDF %s: %s", path, exc)
        return
    
    try:
        for index in range(len(doc)):
            try:
                page = doc[index]
                text = page.get_text() or ""
                # Normalize whitespace per page to keep it consistent with previous behavior
                # We add a newline to separate pages.
                normalized = normalize_whitespace([text])
                if normalized:
                    yield normalized + "\n"
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.warning("Failed to read page %s in %s: %s", index, path, exc)
    finally:
        doc.close()


def get_pdf_metadata(path: Path) -> Dict[str, str]:
    """Extract metadata from a PDF file using PyMuPDF."""
    doc = fitz.open(path)
    try:
        metadata = doc.metadata or {}
        title = metadata.get("title") or path.stem
        page_count = len(doc)
        return {
            "title": title,
            "page_count": str(page_count),
        }
    finally:
        doc.close()


def build_chunks(path: Path, *, max_chars: int = 1200, overlap: int = 200) -> Iterable[ChunkRecord]:
    """Produce chunk records for a PDF path lazily."""
    # We need metadata first for the chunks
    try:
        meta = get_pdf_metadata(path)
    except Exception as e:
        LOGGER.error(f"Failed to read metadata for {path}: {e}")
        return

    text_stream = iter_text_parts(path)

    # Use the streaming chunker
    # Note: chunk_text_stream needs to be imported or available
    from docfinder.utils.text import chunk_text_stream

    for idx, chunk in enumerate(
        chunk_text_stream(text_stream, max_chars=max_chars, overlap=overlap)
    ):
        yield ChunkRecord(
            document_path=path,
            index=idx,
            text=chunk,
            metadata={
                "page_span": meta.get("page_count", ""),
                "title": meta.get("title", path.stem),
            },
        )
