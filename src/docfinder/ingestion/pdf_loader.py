"""Document loading and chunking utilities.

Supports PDF (via PyMuPDF), plain text (.txt), Markdown (.md),
Word documents (.docx via python-docx, .doc via olefile),
OpenDocument formats (.odt/.odp/.odg via odfpy),
PowerPoint (.pptx via python-pptx, .ppt via olefile),
HTML (.html/.htm via beautifulsoup4),
and EPUB (.epub via zipfile + beautifulsoup4).
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
        # No tables - fast path, same as before
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
            "python-docx not installed - cannot index .docx files. "
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


# ── ODF (ODT / ODP / ODG) ─────────────────────────────────────────────────────

_ODF_PARAS_PER_PAGE = 10


def _import_odf():
    """Import odfpy, return (load, P_element_class, extractText) or (None, None, None)."""
    try:
        from odf import teletype
        from odf.opendocument import load
        from odf.text import P

        return load, P, teletype.extractText
    except ImportError:
        LOGGER.warning(
            "odfpy not installed - cannot index ODF files (.odt/.odp/.odg). "
            "Install with: pip install odfpy"
        )
        return None, None, None


def iter_text_parts_odf(path: Path) -> Iterator[str]:
    """Yield text content from an OpenDocument file."""
    load, P, extract_text = _import_odf()
    if load is None:
        return
    try:
        doc = load(str(path))
        for p_elem in doc.getElementsByType(P):
            text = extract_text(p_elem)
            if text.strip():
                yield text.strip() + "\n"
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_odf_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, text)`` grouping ~10 paragraphs."""
    load, P, extract_text = _import_odf()
    if load is None:
        return
    try:
        doc = load(str(path))
        buf: list[str] = []
        page = 1
        for p_elem in doc.getElementsByType(P):
            text = extract_text(p_elem).strip()
            if not text:
                continue
            buf.append(text)
            if len(buf) >= _ODF_PARAS_PER_PAGE:
                yield page, "\n".join(buf) + "\n"
                buf = []
                page += 1
        if buf:
            yield page, "\n".join(buf) + "\n"
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── PPTX ──────────────────────────────────────────────────────────────────────


def _import_pptx():
    """Import and return the ``Presentation`` class from python-pptx, or None."""
    try:
        from pptx import Presentation  # type: ignore[import-untyped]

        return Presentation
    except ImportError:
        LOGGER.warning(
            "python-pptx not installed - cannot index .pptx files. "
            "Install with: pip install python-pptx"
        )
        return None


def _extract_pptx_slide_text(slide) -> str:
    """Extract all text from a single pptx Slide object."""
    parts: list[str] = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                t = para.text.strip()
                if t:
                    parts.append(t)
    return "\n".join(parts)


def iter_text_parts_pptx(path: Path) -> Iterator[str]:
    """Yield text from every slide in a .pptx file."""
    Presentation = _import_pptx()
    if Presentation is None:
        return
    try:
        prs = Presentation(str(path))
        for slide in prs.slides:
            text = _extract_pptx_slide_text(slide)
            if text:
                yield text + "\n"
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_pptx_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(slide_number, text)`` - one page per slide."""
    Presentation = _import_pptx()
    if Presentation is None:
        return
    try:
        prs = Presentation(str(path))
        for page_num, slide in enumerate(prs.slides, 1):
            text = _extract_pptx_slide_text(slide)
            if text:
                yield page_num, text + "\n"
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML_VIRTUAL_PAGE_CHARS = 3000


def _import_beautifulsoup4():
    """Import and return ``BeautifulSoup``, or None."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        return BeautifulSoup
    except ImportError:
        LOGGER.warning(
            "beautifulsoup4 not installed - cannot index .html/.epub files. "
            "Install with: pip install beautifulsoup4"
        )
        return None


def iter_text_parts_html(path: Path) -> Iterator[str]:
    """Yield cleaned plain text from an HTML file."""
    BeautifulSoup = _import_beautifulsoup4()
    if BeautifulSoup is None:
        return
    try:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if text:
            yield text
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_html_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, text)`` for HTML files (~3000 chars per page)."""
    BeautifulSoup = _import_beautifulsoup4()
    if BeautifulSoup is None:
        return
    try:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            return

        page = 1
        start = 0
        while start < len(text):
            end = start + _HTML_VIRTUAL_PAGE_CHARS
            if end < len(text):
                search_start = max(start, end - _HTML_VIRTUAL_PAGE_CHARS // 5)
                nl = text.rfind("\n", search_start, end)
                if nl > start:
                    end = nl + 1
            chunk = text[start:end]
            if chunk.strip():
                yield page, chunk + "\n"
            page += 1
            start = end
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── EPUB ──────────────────────────────────────────────────────────────────────


def _parse_epub(path: Path, BeautifulSoup) -> Iterator[tuple[int, str]]:
    """Parse an EPUB and yield ``(chapter_number, text)`` from spine items."""
    import zipfile
    from xml.etree import ElementTree

    with zipfile.ZipFile(path) as zf:
        try:
            container_data = zf.read("META-INF/container.xml")
        except KeyError:
            return

        ns_ct = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        container = ElementTree.fromstring(container_data)
        rootfile = container.find(".//c:rootfile", ns_ct)
        if rootfile is None:
            return
        opf_rel = rootfile.get("full-path", "")
        if not opf_rel:
            return

        base_dir = opf_rel.rsplit("/", 1)[0] if "/" in opf_rel else ""

        opf_data = zf.read(opf_rel)
        opf = ElementTree.fromstring(opf_data)

        ns_opf = {"opf": "http://www.idpf.org/2007/opf"}
        manifest: dict[str, str] = {}
        for item in opf.findall(".//opf:manifest/opf:item", ns_opf):
            item_id = item.get("id", "")
            href = item.get("href", "")
            if href and base_dir:
                href = base_dir + "/" + href
            manifest[item_id] = href

        spine = opf.find(".//opf:spine", ns_opf)
        if spine is None:
            return

        page_num = 0
        for ref in spine.findall("opf:itemref", ns_opf):
            idref = ref.get("idref", "")
            href = manifest.get(idref, "")
            if not href:
                continue
            try:
                content = zf.read(href)
            except KeyError:
                continue

            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if text:
                page_num += 1
                yield page_num, text + "\n"


def iter_text_parts_epub(path: Path) -> Iterator[str]:
    """Yield chapter text from an EPUB file."""
    BeautifulSoup = _import_beautifulsoup4()
    if BeautifulSoup is None:
        return
    try:
        for _, text in _parse_epub(path, BeautifulSoup):
            yield text
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_epub_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(chapter_number, text)`` - one page per spine item."""
    BeautifulSoup = _import_beautifulsoup4()
    if BeautifulSoup is None:
        return
    try:
        yield from _parse_epub(path, BeautifulSoup)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


# ── DOC (Word 97‑2003 binary) ────────────────────────────────────────────────

_DOC_VIRTUAL_PAGE_CHARS = 3000


def _import_olefile():
    """Import and return the ``olefile`` module, or None."""
    try:
        import olefile  # type: ignore[import-untyped]

        return olefile
    except ImportError:
        LOGGER.warning(
            "olefile not installed - cannot index .doc/.ppt files. "
            "Install with: pip install olefile"
        )
        return None


def _extract_text_from_doc_binary(path: Path) -> Iterator[str]:
    """Extract text from a Word 97-2003 file using its piece table."""
    olefile = _import_olefile()
    if olefile is None:
        return
    try:
        ole = olefile.OleFileIO(path)
        try:
            word_data = ole.openstream("WordDocument").read()
            if len(word_data) < 32 or word_data[:2] not in (
                b"\xec\xa5",  # Word 97-2003
                b"\xdc\xa5",  # Word 6/95
                b"\xdb\xa5",  # Word 95
            ):
                LOGGER.warning("Unsupported or invalid Word binary file: %s", path)
                return

            csw = int.from_bytes(word_data[32:34], "little")
            offset = 34 + csw * 2
            cslw = int.from_bytes(word_data[offset : offset + 2], "little")
            offset += 2 + cslw * 4
            cb_rg_fc_lcb = int.from_bytes(word_data[offset : offset + 2], "little")
            offset += 2
            fc_lcb = word_data[offset : offset + cb_rg_fc_lcb * 8]
            # fcClx is entry 33 in FibRgFcLcb97 (fc followed by lcb).
            clx_entry = 33 * 8
            if len(fc_lcb) < clx_entry + 8:
                LOGGER.warning("Word binary file has no CLX reference: %s", path)
                return
            fc_clx = int.from_bytes(fc_lcb[clx_entry : clx_entry + 4], "little")
            lcb_clx = int.from_bytes(fc_lcb[clx_entry + 4 : clx_entry + 8], "little")
            if not lcb_clx:
                return

            flags = int.from_bytes(word_data[10:12], "little")
            table_name = "1Table" if flags & 0x0200 else "0Table"
            table_data = ole.openstream(table_name).read()
            clx = table_data[fc_clx : fc_clx + lcb_clx]
        finally:
            ole.close()
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return

    # CLX may begin with formatting records before the piece table.
    pos = 0
    while pos < len(clx) and clx[pos] == 0x01:
        if pos + 5 > len(clx):
            return
        pos += 5 + int.from_bytes(clx[pos + 1 : pos + 5], "little")
    if pos >= len(clx) or clx[pos] != 0x02 or pos + 5 > len(clx):
        LOGGER.warning("Word binary file has an invalid piece table: %s", path)
        return

    piece_table_size = int.from_bytes(clx[pos + 1 : pos + 5], "little")
    piece_table = clx[pos + 5 : pos + 5 + piece_table_size]
    if len(piece_table) < 4 or (len(piece_table) - 4) % 12:
        LOGGER.warning("Word binary file has a truncated piece table: %s", path)
        return

    piece_count = (len(piece_table) - 4) // 12
    cps = [
        int.from_bytes(piece_table[index * 4 : index * 4 + 4], "little")
        for index in range(piece_count + 1)
    ]
    text_parts: list[str] = []
    for index in range(piece_count):
        pcd_offset = 4 * (piece_count + 1) + index * 8
        fc = int.from_bytes(piece_table[pcd_offset + 2 : pcd_offset + 6], "little")
        cp_length = cps[index + 1] - cps[index]
        if cp_length <= 0:
            continue
        compressed = bool(fc & 0x40000000)
        if compressed:
            # Compressed pieces store the byte offset multiplied by two.
            start = (fc & 0x3FFFFFFF) // 2
            raw = word_data[start : start + cp_length]
            text_parts.append(raw.decode("cp1252", errors="replace"))
        else:
            start = fc
            raw = word_data[start : start + cp_length * 2]
            text_parts.append(raw.decode("utf-16-le", errors="replace"))

    text = "".join(text_parts)
    text = text.replace("\x07", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
    text = "".join(char for char in text if char in "\n\r\t" or char.isprintable())
    text = re.sub(r"[^\S\n\r\t]+", " ", text).strip()
    if text:
        yield text


def iter_text_parts_doc(path: Path) -> Iterator[str]:
    """Yield text extracted from a .doc file."""
    yield from _extract_text_from_doc_binary(path)


def iter_text_parts_doc_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, text)`` for Word binary documents."""
    text = "\n".join(_extract_text_from_doc_binary(path))
    if not text.strip():
        return

    page = 1
    start = 0
    while start < len(text):
        end = start + _DOC_VIRTUAL_PAGE_CHARS
        if end < len(text):
            search_start = max(start, end - _DOC_VIRTUAL_PAGE_CHARS // 5)
            nl = text.rfind("\n", search_start, end)
            if nl > start:
                end = nl + 1
        chunk = text[start:end]
        if chunk.strip():
            yield page, chunk + "\n"
        page += 1
        start = end


# ── PPT (PowerPoint 97‑2003 binary) ───────────────────────────────────────────

_PPT_VIRTUAL_PAGE_CHARS = 3000

# Record types for the PowerPoint binary format
_PPT_TEXT_CHARS_ATOM = 0x0FA0
_PPT_TEXT_BYTES_ATOM = 0x0FA8


def _walk_ppt_records(data: bytes, offset: int, end: int, texts: list[str]) -> None:
    """Walk a PowerPoint binary record tree looking for text atoms.

    Record headers are little-endian ([MS-PPT]): ver+instance (u16), type
    (u16), length (u32). A record is a container when its version nibble
    is ``0xF``.
    """
    import struct

    while offset + 8 <= end:
        # rec_ver = data[offset] & 0x0F  (used below for container detection)
        # rec_instance = ((data[offset] >> 4) & 0x0F) << 8 | data[offset + 1]  (not needed)
        rec_type = struct.unpack_from("<H", data, offset + 2)[0]
        rec_len = struct.unpack_from("<I", data, offset + 4)[0]

        data_start = offset + 8
        data_end = data_start + rec_len

        if data_end > end:
            break

        if rec_type == _PPT_TEXT_CHARS_ATOM:
            raw = data[data_start:data_end].decode("utf-16-le", errors="replace")
            clean = "".join(c if c.isprintable() or c in "\n\r\t" else " " for c in raw)
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean:
                texts.append(clean)
        elif rec_type == _PPT_TEXT_BYTES_ATOM:
            raw = data[data_start:data_end].decode("utf-8", errors="replace")
            clean = "".join(c if c.isprintable() or c in "\n\r\t" else " " for c in raw)
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean:
                texts.append(clean)

        if (data[offset] & 0x0F) == 0x0F:
            _walk_ppt_records(data, data_start, data_end, texts)

        offset = data_end


def _extract_text_from_ppt_binary(path: Path) -> Iterator[str]:
    """Extract text from PowerPoint 97‑2003 (.ppt) files via olefile."""
    olefile = _import_olefile()
    if olefile is None:
        return
    try:
        ole = olefile.OleFileIO(path)
        try:
            stream = ole.openstream("PowerPoint Document")
            data = stream.read()
        finally:
            ole.close()
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return

    texts: list[str] = []
    _walk_ppt_records(data, 0, len(data), texts)

    for text in texts:
        if text.strip():
            yield text.strip() + "\n"


def iter_text_parts_ppt(path: Path) -> Iterator[str]:
    """Yield text from a .ppt file."""
    yield from _extract_text_from_ppt_binary(path)


def iter_text_parts_ppt_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, text)`` for PowerPoint binary documents."""
    text = "\n".join(_extract_text_from_ppt_binary(path))
    if not text.strip():
        return

    page = 1
    start = 0
    while start < len(text):
        end = start + _PPT_VIRTUAL_PAGE_CHARS
        if end < len(text):
            search_start = max(start, end - _PPT_VIRTUAL_PAGE_CHARS // 5)
            nl = text.rfind("\n", search_start, end)
            if nl > start:
                end = nl + 1
        chunk = text[start:end]
        if chunk.strip():
            yield page, chunk + "\n"
        page += 1
        start = end


# ── CSV / TSV / XLSX / Structured text / Email / RTF / Code ──────────────────

_CSV_ROWS_PER_PAGE = 40


def iter_text_parts_csv_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(virtual_page, markdown_table)`` from a .csv or .tsv file."""
    import csv

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return

    reader = csv.reader(raw.splitlines(), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return

    header = [c.strip().replace("\n", " ") for c in rows[0]]
    body = rows[1:] or [rows[0]]
    page = 1
    for i in range(0, len(body), _CSV_ROWS_PER_PAGE):
        batch = body[i : i + _CSV_ROWS_PER_PAGE]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in batch:
            padded = (row + [""] * len(header))[: len(header)]
            cells = [c.strip().replace("\n", " ") for c in padded]
            lines.append("| " + " | ".join(cells) + " |")
        yield page, "\n".join(lines) + "\n"
        page += 1


def iter_text_parts_csv(path: Path) -> Iterator[str]:
    for _, text in iter_text_parts_csv_paged(path):
        yield text


def iter_text_parts_xlsx_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Extract worksheet text from a .xlsx workbook using Python's stdlib zipfile + XML."""
    import zipfile
    from xml.etree import ElementTree

    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with zipfile.ZipFile(path) as zf:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                ss_root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in ss_root.findall("s:si", ns):
                    texts = [t.text or "" for t in si.findall(".//s:t", ns)]
                    shared.append("".join(texts))

            sheet_files = sorted(
                name
                for name in zf.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            page = 1
            for sheet_name in sheet_files:
                root = ElementTree.fromstring(zf.read(sheet_name))
                rows_text: list[str] = []
                for row_el in root.findall(".//s:sheetData/s:row", ns):
                    cells: list[str] = []
                    for c_el in row_el.findall("s:c", ns):
                        c_type = c_el.get("t", "")
                        v_el = c_el.find("s:v", ns)
                        is_el = c_el.find("s:is", ns)
                        val = ""
                        if c_type == "s" and v_el is not None and v_el.text:
                            idx = int(v_el.text)
                            if 0 <= idx < len(shared):
                                val = shared[idx]
                        elif c_type == "inlineStr" and is_el is not None:
                            val = "".join(t.text or "" for t in is_el.findall(".//s:t", ns))
                        elif v_el is not None and v_el.text:
                            val = v_el.text
                        val = val.strip().replace("\n", " ")
                        if val:
                            cells.append(val)
                    if cells:
                        rows_text.append(" | ".join(cells))
                if rows_text:
                    yield page, "\n".join(rows_text) + "\n"
                    page += 1
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_xlsx(path: Path) -> Iterator[str]:
    for _, text in iter_text_parts_xlsx_paged(path):
        yield text


def _strip_rtf(raw: str) -> str:
    """Convert basic RTF markup to plain text using stdlib regex."""
    # Decode hex escapes \'xx
    text = re.sub(
        r"\\'([0-9a-fA-F]{2})",
        lambda m: bytes.fromhex(m.group(1)).decode("cp1252", errors="replace"),
        raw,
    )
    # Remove RTF destination groups like {\fonttbl...}, {\colortbl...}, {\stylesheet...}
    text = re.sub(r"\{\\\*\\[a-z]+[^}]*\}", "", text)
    text = re.sub(r"\{\\(?:fonttbl|colortbl|stylesheet|info)[^}]*\}", "", text)
    text = re.sub(r"\\(?:par|line|row)\b", "\n", text)
    text = re.sub(r"\\tab\b", "\t", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return normalize_whitespace([text])


def iter_text_parts_rtf_paged(path: Path) -> Iterator[tuple[int, str]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return
    cleaned = _strip_rtf(raw)
    if not cleaned:
        return
    page = 1
    for start in range(0, len(cleaned), _DOC_VIRTUAL_PAGE_CHARS):
        chunk = cleaned[start : start + _DOC_VIRTUAL_PAGE_CHARS].strip()
        if chunk:
            yield page, chunk + "\n"
            page += 1


def iter_text_parts_rtf(path: Path) -> Iterator[str]:
    for _, text in iter_text_parts_rtf_paged(path):
        yield text


def _extract_email_message_text(msg) -> str:
    """Format an email.message.Message into searchable header + body text."""
    headers = []
    for key in ("Subject", "From", "To", "Date"):
        val = msg.get(key)
        if val:
            headers.append(f"{key}: {val}")
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body_parts.append(payload.decode("utf-8", errors="replace"))
    combined = "\n".join(headers + [""] + body_parts).strip()
    return combined


def iter_text_parts_email_paged(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(message_number, text)`` from .eml or .mbox files."""
    import email
    import mailbox

    suffix = path.suffix.lower()
    try:
        if suffix == ".eml":
            msg = email.message_from_bytes(path.read_bytes())
            text = _extract_email_message_text(msg)
            if text:
                yield 1, text + "\n"
        elif suffix == ".mbox":
            box = mailbox.mbox(str(path))
            try:
                for idx, msg in enumerate(box, 1):
                    text = _extract_email_message_text(msg)
                    if text:
                        yield idx, text + "\n"
            finally:
                box.close()
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)


def iter_text_parts_email(path: Path) -> Iterator[str]:
    for _, text in iter_text_parts_email_paged(path):
        yield text


# ── Dispatcher ────────────────────────────────────────────────────────────────


def _iter_text_by_suffix(path: Path) -> Iterator[str]:
    from docfinder.utils.files import CODE_EXTENSIONS

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from iter_text_parts(path)
    elif suffix in (".txt", ".rst", ".tex", ".log", ".json", ".yaml", ".yml", ".xml") or (
        suffix in CODE_EXTENSIONS
    ):
        yield from iter_text_parts_txt(path)
    elif suffix == ".md":
        yield from iter_text_parts_md(path)
    elif suffix == ".docx":
        yield from iter_text_parts_docx(path)
    elif suffix in (".odt", ".odp", ".odg", ".ods"):
        yield from iter_text_parts_odf(path)
    elif suffix == ".pptx":
        yield from iter_text_parts_pptx(path)
    elif suffix in (".html", ".htm"):
        yield from iter_text_parts_html(path)
    elif suffix == ".epub":
        yield from iter_text_parts_epub(path)
    elif suffix == ".doc":
        yield from iter_text_parts_doc(path)
    elif suffix == ".ppt":
        yield from iter_text_parts_ppt(path)
    elif suffix in (".csv", ".tsv"):
        yield from iter_text_parts_csv(path)
    elif suffix == ".xlsx":
        yield from iter_text_parts_xlsx(path)
    elif suffix == ".rtf":
        yield from iter_text_parts_rtf(path)
    elif suffix in (".eml", ".mbox"):
        yield from iter_text_parts_email(path)
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
    """Yield ``(page_number, text)`` for any supported text/document format."""
    from docfinder.utils.files import CODE_EXTENSIONS

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from iter_text_parts_paged(path)
    elif suffix == ".md":
        yield from iter_text_parts_md_paged(path)
    elif suffix == ".docx":
        yield from iter_text_parts_docx_paged(path)
    elif suffix in (".txt", ".rst", ".tex", ".log", ".json", ".yaml", ".yml", ".xml") or (
        suffix in CODE_EXTENSIONS
    ):
        yield from iter_text_parts_txt_paged(path)
    elif suffix in (".odt", ".odp", ".odg", ".ods"):
        yield from iter_text_parts_odf_paged(path)
    elif suffix == ".pptx":
        yield from iter_text_parts_pptx_paged(path)
    elif suffix in (".html", ".htm"):
        yield from iter_text_parts_html_paged(path)
    elif suffix == ".epub":
        yield from iter_text_parts_epub_paged(path)
    elif suffix == ".doc":
        yield from iter_text_parts_doc_paged(path)
    elif suffix == ".ppt":
        yield from iter_text_parts_ppt_paged(path)
    elif suffix in (".csv", ".tsv"):
        yield from iter_text_parts_csv_paged(path)
    elif suffix == ".xlsx":
        yield from iter_text_parts_xlsx_paged(path)
    elif suffix == ".rtf":
        yield from iter_text_parts_rtf_paged(path)
    elif suffix in (".eml", ".mbox"):
        yield from iter_text_parts_email_paged(path)
    else:
        LOGGER.warning("Unsupported file type: %s", path.suffix)


def _find_scanned_pdf_pages(path: Path, text_pages: set[int]) -> list[int]:
    """Return 1-based page numbers in a PDF that have no extracted text but contain images."""
    scanned: list[int] = []
    try:
        doc = fitz.open(path)
    except Exception:
        return scanned
    try:
        for index in range(len(doc)):
            page_num = index + 1
            if page_num in text_pages:
                continue
            try:
                page = doc[index]
                if page.get_images():
                    scanned.append(page_num)
            except Exception:
                pass
    finally:
        doc.close()
    return scanned


def render_pdf_page_image(path: Path, page_num: int, dpi: int = 150):
    """Render a 1-based PDF page into a PIL RGB Image using PyMuPDF."""
    from PIL import Image

    doc = fitz.open(path)
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


_AUDIO_WINDOW_SEC = 30.0
_AUDIO_STRIDE_SEC = 25.0
_AUDIO_SAMPLE_RATE = 16000


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _probe_audio_duration_sec(path: Path) -> float | None:
    """Return duration in seconds for a .wav file (via stdlib wave) or media file (via ffprobe)."""
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0 and frames > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            dur = float(proc.stdout.strip())
            if dur > 0:
                return dur
    except Exception:
        pass
    return None


def _load_audio_window_array(path: Path, start_sec: float = 0.0, duration_sec: float = 30.0):
    """Decode an audio segment into a 16kHz mono float32 numpy array, or None on fallback."""
    import numpy as np

    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as wf:
                rate = wf.getframerate()
                n_channels = max(1, wf.getnchannels())
                sampwidth = wf.getsampwidth()
                start_frame = min(wf.getnframes(), max(0, int(start_sec * rate)))
                read_frames = int(duration_sec * rate)
                wf.setpos(start_frame)
                raw = wf.readframes(read_frames)
                if raw and sampwidth in (1, 2, 4):
                    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
                    scale_map = {1: 128.0, 2: 32768.0, 4: 2147483648.0}
                    samples = np.frombuffer(raw, dtype=dtype_map[sampwidth]).astype("float32")
                    if sampwidth == 1:
                        samples = (samples - 128.0) / 128.0
                    else:
                        samples = samples / scale_map[sampwidth]
                    if n_channels > 1:
                        samples = samples.reshape(-1, n_channels).mean(axis=1)
                    if samples.size > 0:
                        return samples
        except Exception:
            pass

    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{max(0.0, start_sec):.2f}",
                "-t",
                f"{duration_sec:.2f}",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(_AUDIO_SAMPLE_RATE),
                "pipe:1",
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            samples = np.frombuffer(proc.stdout, dtype="float32")
            if samples.size > 0:
                return samples.copy()
    except Exception:
        pass
    return None


def _build_code_chunks(
    path: Path,
    *,
    title: str,
    max_chars: int = 1200,
    overlap: int = 200,
) -> Iterator[ChunkRecord]:
    """Build line-tracked chunks for a source code file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return

    lines = content.splitlines(keepends=True)
    if not lines or not content.strip():
        return

    idx = 0
    start_line_idx = 0
    total_lines = len(lines)

    while start_line_idx < total_lines:
        char_count = 0
        end_line_idx = start_line_idx
        while end_line_idx < total_lines and (
            char_count + len(lines[end_line_idx]) <= max_chars or end_line_idx == start_line_idx
        ):
            char_count += len(lines[end_line_idx])
            end_line_idx += 1

        chunk_text = "".join(lines[start_line_idx:end_line_idx]).strip()
        if chunk_text:
            line_start = start_line_idx + 1
            line_end = end_line_idx
            yield ChunkRecord(
                document_path=path,
                index=idx,
                text=chunk_text,
                metadata={
                    "title": title,
                    "page": idx + 1,
                    "modality": "code",
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
            idx += 1

        if end_line_idx >= total_lines:
            break

        # Step back by overlap characters for the next chunk
        overlap_chars = 0
        next_start = end_line_idx
        while next_start > start_line_idx + 1 and overlap_chars < overlap:
            next_start -= 1
            overlap_chars += len(lines[next_start])
        start_line_idx = max(start_line_idx + 1, next_start)


def prepare_chunk_embedding_input(path: Path, chunk: dict) -> object:
    """Prepare the input object passed to ``EmbeddingModel.embed()`` for a chunk.

    For text/code chunks, returns the plain text string.
    For visual/audio chunks, returns the PIL.Image, audio waveform, or multimodal
    dict supported by EmbeddingGemma 2, falling back cleanly to text on error.
    """
    meta = chunk.get("metadata") or {}
    modality = meta.get("modality", "text")
    text = chunk.get("text", "")

    try:
        if modality == "image":
            from PIL import Image

            with Image.open(path) as img:
                return img.convert("RGB")
        if modality == "pdf_page_image":
            page_num = int(meta.get("page", 1))
            return render_pdf_page_image(path, page_num)
        if modality in ("audio", "video") and "start_sec" in meta:
            start_sec = float(meta.get("start_sec", 0.0))
            end_sec = float(meta.get("end_sec", start_sec + _AUDIO_WINDOW_SEC))
            samples = _load_audio_window_array(
                path, start_sec=start_sec, duration_sec=max(1.0, end_sec - start_sec)
            )
            if samples is not None:
                return samples
    except Exception as exc:
        LOGGER.debug("Multimodal load fallback to text for %s: %s", path, exc)

    return text


def build_chunks(
    path: Path,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
    modalities: Iterable[str] | None = None,
) -> Iterable[ChunkRecord]:
    """Produce overlapping chunk records for any supported document, image, or media type."""
    from docfinder.utils.files import (
        AUDIO_VIDEO_EXTENSIONS,
        CODE_EXTENSIONS,
        IMAGE_EXTENSIONS,
    )
    from docfinder.utils.text import chunk_text_stream_paged

    active_mods = set(modalities) if modalities is not None else {"text", "image", "audio"}
    suffix = path.suffix.lower()
    title = _get_title(path)

    if suffix in IMAGE_EXTENSIONS:
        if "image" not in active_mods:
            return
        yield ChunkRecord(
            document_path=path,
            index=0,
            text=f"[Image: {path.name}]",
            metadata={"title": title, "page": 1, "modality": "image"},
        )
        return

    if suffix in AUDIO_VIDEO_EXTENSIONS:
        if "audio" not in active_mods:
            return
        media_kind = "video" if suffix in {".mp4", ".mov", ".mkv", ".webm"} else "audio"
        duration = _probe_audio_duration_sec(path)
        if duration is not None and duration > _AUDIO_WINDOW_SEC:
            idx = 0
            start_sec = 0.0
            while start_sec < duration:
                end_sec = min(duration, start_sec + _AUDIO_WINDOW_SEC)
                ts_label = _format_timestamp(start_sec)
                end_label = _format_timestamp(end_sec)
                yield ChunkRecord(
                    document_path=path,
                    index=idx,
                    text=f"[{media_kind.capitalize()}: {path.name} @ {ts_label}–{end_label}]",
                    metadata={
                        "title": title,
                        "page": idx + 1,
                        "modality": media_kind,
                        "start_sec": round(start_sec, 1),
                        "end_sec": round(end_sec, 1),
                        "timestamp": ts_label,
                    },
                )
                idx += 1
                if end_sec >= duration:
                    break
                start_sec += _AUDIO_STRIDE_SEC
            return

        meta_media: Dict[str, object] = {
            "title": title,
            "page": 1,
            "modality": media_kind,
        }
        if duration is not None:
            meta_media["start_sec"] = 0.0
            meta_media["end_sec"] = round(duration, 1)
            meta_media["timestamp"] = "0:00"
        yield ChunkRecord(
            document_path=path,
            index=0,
            text=f"[{media_kind.capitalize()}: {path.name}]",
            metadata=meta_media,
        )
        return

    if suffix in CODE_EXTENSIONS:
        yield from _build_code_chunks(path, title=title, max_chars=max_chars, overlap=overlap)
        return

    pages = list(_iter_paged_text(path))
    text_page_nums = {page_num for page_num, _ in pages}

    idx = 0
    for chunk, page_num in chunk_text_stream_paged(
        iter(pages), max_chars=max_chars, overlap=overlap
    ):
        meta: Dict[str, object] = {"title": title, "page": page_num}
        yield ChunkRecord(
            document_path=path,
            index=idx,
            text=chunk,
            metadata=meta,
        )
        idx += 1

    # For PDFs with scanned/image-only pages when the Vision module is enabled
    if suffix == ".pdf" and "image" in active_mods:
        for scanned_page in _find_scanned_pdf_pages(path, text_page_nums):
            yield ChunkRecord(
                document_path=path,
                index=idx,
                text=f"[Scanned PDF page {scanned_page} - {title}]",
                metadata={
                    "title": title,
                    "page": scanned_page,
                    "modality": "pdf_page_image",
                },
            )
            idx += 1
