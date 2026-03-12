"""Document loading and chunking utilities.

Supports PDF (via PyMuPDF), plain text (.txt), Markdown (.md),
and Word documents (.docx via python-docx).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator

import fitz  # PyMuPDF

from docfinder.models import ChunkRecord
from docfinder.utils.text import normalize_whitespace

LOGGER = logging.getLogger(__name__)


# ── PDF ───────────────────────────────────────────────────────────────────────

def iter_text_parts(path: Path) -> Iterator[str]:
    """Yield text content from a PDF file, page by page."""
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
                normalized = normalize_whitespace([text])
                if normalized:
                    yield normalized + "\n"
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to read page %s in %s: %s", index, path, exc)
    finally:
        doc.close()


def get_pdf_metadata(path: Path) -> Dict[str, str]:
    """Extract title and page count from a PDF."""
    doc = fitz.open(path)
    try:
        metadata = doc.metadata or {}
        title = metadata.get("title") or path.stem
        return {"title": title, "page_count": str(len(doc))}
    finally:
        doc.close()


# ── Plain text ────────────────────────────────────────────────────────────────

def iter_text_parts_txt(path: Path) -> Iterator[str]:
    """Yield the full content of a .txt file as a single string."""
    try:
        yield path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── Markdown ──────────────────────────────────────────────────────────────────

_MD_HEADING   = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD_ITAL = re.compile(r"\*{1,3}([^*\n]+)\*{1,3}")
_MD_CODE      = re.compile(r"`{1,3}[^`]*`{1,3}", re.DOTALL)
_MD_IMAGE     = re.compile(r"!\[.*?\]\([^)]*\)")
_MD_LINK      = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HR        = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)


def iter_text_parts_md(path: Path) -> Iterator[str]:
    """Yield plain text extracted from a Markdown file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = _MD_HEADING.sub("", text)
        text = _MD_CODE.sub("", text)
        text = _MD_IMAGE.sub("", text)
        text = _MD_LINK.sub(r"\1", text)
        text = _MD_BOLD_ITAL.sub(r"\1", text)
        text = _MD_HR.sub("", text)
        yield text
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── Word (.docx) ──────────────────────────────────────────────────────────────

def iter_text_parts_docx(path: Path) -> Iterator[str]:
    """Yield paragraph text from a .docx file."""
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError:
        LOGGER.warning(
            "python-docx not installed — cannot index .docx files. "
            "Install with: pip install python-docx"
        )
        return
    try:
        doc = Document(str(path))
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if stripped:
                yield stripped + "\n"
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _iter_text_by_suffix(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from iter_text_parts(path)
    elif suffix == ".txt":
        yield from iter_text_parts_txt(path)
    elif suffix == ".md":
        yield from iter_text_parts_md(path)
    elif suffix == ".docx":
        yield from iter_text_parts_docx(path)
    else:
        LOGGER.warning("Unsupported file type: %s", path.suffix)


def _get_title(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            return get_pdf_metadata(path).get("title", path.stem)
        except Exception:
            pass
    return path.stem


def build_chunks(path: Path, *, max_chars: int = 1200, overlap: int = 200) -> Iterable[ChunkRecord]:
    """Produce overlapping chunk records for any supported document type."""
    from docfinder.utils.text import chunk_text_stream

    title = _get_title(path)
    text_stream = _iter_text_by_suffix(path)

    for idx, chunk in enumerate(
        chunk_text_stream(text_stream, max_chars=max_chars, overlap=overlap)
    ):
        yield ChunkRecord(
            document_path=path,
            index=idx,
            text=chunk,
            metadata={"title": title},
        )
