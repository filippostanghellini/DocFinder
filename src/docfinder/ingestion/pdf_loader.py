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


def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF Table object to a Markdown table string."""
    rows = table.extract()
    if not rows:
        return ""

    lines: list[str] = []
    # Header row
    header = [str(cell or "").strip().replace("\n", " ") for cell in rows[0]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    # Data rows
    for row in rows[1:]:
        cells = [str(cell or "").strip().replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _extract_page_text(page) -> str:
    """Extract text from a PDF page, converting tables to Markdown.

    If tables are detected, they are rendered as Markdown and inserted
    in place of the raw table text.  Non-table text is extracted normally.
    """
    try:
        tables = page.find_tables()
    except Exception:
        tables = None

    if not tables or not tables.tables:
        # No tables — fast path, same as before
        return page.get_text() or ""

    # Collect table bounding boxes and their Markdown representations
    table_items: list[tuple[float, str]] = []  # (top_y, markdown)
    table_rects: list[fitz.Rect] = []
    for tab in tables.tables:
        md = _table_to_markdown(tab)
        if md:
            rect = fitz.Rect(tab.bbox)
            table_items.append((rect.y0, md))
            table_rects.append(rect)

    # Extract text blocks excluding table areas
    text_blocks: list[tuple[float, str]] = []  # (top_y, text)
    for block in page.get_text("blocks") or []:
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        if block[6] != 0:  # skip image blocks
            continue
        block_rect = fitz.Rect(block[:4])
        # Skip blocks that overlap significantly with any table
        overlaps_table = False
        for tr in table_rects:
            intersection = block_rect & tr
            if not intersection.is_empty:
                block_area = block_rect.width * block_rect.height
                if block_area > 0:
                    overlap_ratio = (intersection.width * intersection.height) / block_area
                    if overlap_ratio > 0.5:
                        overlaps_table = True
                        break
        if not overlaps_table:
            text = block[4].strip()
            if text:
                text_blocks.append((block[1], text))

    # Merge text blocks and tables, sorted by vertical position
    all_parts: list[tuple[float, str]] = text_blocks + table_items
    all_parts.sort(key=lambda x: x[0])

    return "\n\n".join(part[1] for part in all_parts)


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
                text = _extract_page_text(page)
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

# Virtual page size for plain text files (characters per virtual page).
_TXT_VIRTUAL_PAGE_CHARS = 3000


def iter_text_parts_txt(path: Path) -> Iterator[str]:
    """Yield the full content of a .txt file as a single string."""
    try:
        yield path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_txt_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, text)`` for plain text files.

    Splits the file into virtual pages of ~3000 characters each,
    breaking at the nearest newline to avoid cutting mid-sentence.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return

    if not content.strip():
        return

    page = 1
    start = 0
    while start < len(content):
        end = start + _TXT_VIRTUAL_PAGE_CHARS
        if end < len(content):
            # Try to break at a newline within the last 20% of the slice
            search_start = max(start, end - _TXT_VIRTUAL_PAGE_CHARS // 5)
            nl = content.rfind("\n", search_start, end)
            if nl > start:
                end = nl + 1
        chunk = content[start:end]
        if chunk.strip():
            yield page, chunk
        page += 1
        start = end


# ── Markdown ──────────────────────────────────────────────────────────────────

_MD_HEADING_LINE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_HEADING_SPLIT = re.compile(r"(?=^#{1,6}\s)", re.MULTILINE)
_MD_BOLD_ITAL = re.compile(r"\*{1,3}([^*\n]+)\*{1,3}")
_MD_CODE = re.compile(r"`{1,3}[^`]*`{1,3}", re.DOTALL)
_MD_IMAGE = re.compile(r"!\[.*?\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HR = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)


def _clean_md(text: str) -> str:
    """Strip Markdown formatting from a text block."""
    text = _MD_CODE.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BOLD_ITAL.sub(r"\1", text)
    text = _MD_HR.sub("", text)
    text = _MD_HEADING_LINE.sub("", text)
    return text


def iter_text_parts_md(path: Path) -> Iterator[str]:
    """Yield plain text extracted from a Markdown file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        yield _clean_md(text)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_md_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(section_number, text)`` for Markdown files.

    Each top-level heading (``# …``, ``## …``, etc.) starts a new
    virtual page.  Content before the first heading is section 1.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return

    sections = _MD_HEADING_SPLIT.split(raw)
    page = 1
    for section in sections:
        cleaned = _clean_md(section).strip()
        if cleaned:
            yield page, cleaned + "\n"
            page += 1


# ── Word (.docx) ──────────────────────────────────────────────────────────────

# Number of paragraphs per virtual page for .docx files.
_DOCX_PARAS_PER_PAGE = 10


def _import_docx_document():
    """Import and return the ``Document`` class from python-docx, or *None*."""
    try:
        from docx import Document  # type: ignore[import-untyped]

        return Document
    except ImportError:
        LOGGER.warning(
            "python-docx not installed — cannot index .docx files. "
            "Install with: pip install python-docx"
        )
        return None


def iter_text_parts_docx(path: Path) -> Iterator[str]:
    """Yield paragraph text from a .docx file."""
    Document = _import_docx_document()
    if Document is None:
        return
    try:
        doc = Document(str(path))
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if stripped:
                yield stripped + "\n"
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_docx_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, text)`` for Word documents.

    Groups every 10 consecutive non-empty paragraphs into one virtual page.
    """
    Document = _import_docx_document()
    if Document is None:
        return
    try:
        doc = Document(str(path))
        buf: list[str] = []
        page = 1
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if not stripped:
                continue
            buf.append(stripped)
            if len(buf) >= _DOCX_PARAS_PER_PAGE:
                yield page, "\n".join(buf) + "\n"
                buf = []
                page += 1
        if buf:
            yield page, "\n".join(buf) + "\n"
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


def iter_text_parts_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(page_number, text)`` for PDF files (1-based page numbers).

    Uses PyMuPDF to extract text page by page.  Tables are detected
    automatically and rendered as Markdown to preserve structure.
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
                text = _extract_page_text(page)
                normalized = normalize_whitespace([text])
                if normalized:
                    yield index + 1, normalized + "\n"
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to read page %s in %s: %s", index, path, exc)
    finally:
        doc.close()


def _iter_paged_text(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(page_number, text)`` for any supported format.

    * PDF  → real page numbers (1-based)
    * Markdown → section numbers (split on headings)
    * Word → virtual pages (every 10 paragraphs)
    * Plain text → virtual pages (every ~3000 characters)
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from iter_text_parts_paged(path)
    elif suffix == ".md":
        yield from iter_text_parts_md_paged(path)
    elif suffix == ".docx":
        yield from iter_text_parts_docx_paged(path)
    elif suffix == ".txt":
        yield from iter_text_parts_txt_paged(path)
    else:
        LOGGER.warning("Unsupported file type: %s", path.suffix)


def build_chunks(path: Path, *, max_chars: int = 1200, overlap: int = 200) -> Iterable[ChunkRecord]:
    """Produce overlapping chunk records for any supported document type."""
    from docfinder.utils.text import chunk_text_stream_paged

    title = _get_title(path)
    pages = _iter_paged_text(path)

    for idx, (chunk, page_num) in enumerate(
        chunk_text_stream_paged(pages, max_chars=max_chars, overlap=overlap)
    ):
        yield ChunkRecord(
            document_path=path,
            index=idx,
            text=chunk,
            metadata={"title": title, "page": page_num},
        )
