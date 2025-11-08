"""PDF loading and chunking utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

from pypdf import PdfReader

from docfinder.models import ChunkRecord
from docfinder.utils.text import chunk_text, normalize_whitespace

LOGGER = logging.getLogger(__name__)


def extract_text(path: Path) -> tuple[str, Dict[str, str]]:
    """Extract text content and metadata from a PDF file."""
    reader = PdfReader(path)
    page_text: List[str] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Failed to read page %s in %s: %s", index, path, exc)
            text = ""
        page_text.append(text)
    raw_text = normalize_whitespace(page_text)
    metadata = {
        "title": reader.metadata.title if reader.metadata and reader.metadata.title else path.stem,
        "page_count": str(len(reader.pages)),
    }
    return raw_text, metadata


def build_chunks(path: Path, *, max_chars: int = 1200, overlap: int = 200) -> Iterable[ChunkRecord]:
    """Produce chunk records for a PDF path."""
    text, meta = extract_text(path)
    for idx, chunk in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap)):
        yield ChunkRecord(
            document_path=path,
            index=idx,
            text=chunk,
            metadata={
                "page_span": meta.get("page_count", ""),
                "title": meta.get("title", path.stem),
            },
        )
