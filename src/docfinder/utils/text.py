"""Text helpers including simple token-aware chunking."""

from __future__ import annotations

from typing import Iterable, Iterator


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> Iterator[str]:
    """Split text into overlapping character chunks.

    This coarse chunker keeps things simple while preserving context overlap.
    """
    if not text:
        return iter(())

    step = max(max_chars - overlap, 1)
    for start in range(0, len(text), step):
        yield text[start : start + max_chars]


def chunk_text_stream(
    text_stream: Iterable[str], *, max_chars: int = 1200, overlap: int = 200
) -> Iterator[str]:
    """Split a stream of text parts into overlapping character chunks.

    Buffers incoming text just enough to produce chunks of `max_chars`.
    """
    buffer = ""
    step = max(max_chars - overlap, 1)

    for part in text_stream:
        buffer += part
        while len(buffer) >= max_chars:
            yield buffer[:max_chars]
            buffer = buffer[step:]

    # Process remaining buffer
    if buffer:
        # If buffer is smaller than max_chars but we have content, yield it
        # Note: standard chunk_text yields the last part even if short
        yield buffer


def chunk_text_stream_paged(
    pages: Iterable[tuple[int, str]], *, max_chars: int = 1200, overlap: int = 200
) -> Iterator[tuple[str, int]]:
    """Split a stream of (page_number, text) into overlapping chunks.

    Yields ``(chunk_text, page_number)`` where ``page_number`` is the page
    that contributed the **start** of the chunk.

    This preserves page provenance so that downstream code can store the
    originating page in chunk metadata.
    """
    buffer = ""
    # Track which page the start of the buffer came from
    buf_page = 0
    step = max(max_chars - overlap, 1)

    for page_num, part in pages:
        if not buffer:
            buf_page = page_num
        buffer += part
        while len(buffer) >= max_chars:
            yield buffer[:max_chars], buf_page
            buffer = buffer[step:]
            # After slicing, the start of the buffer has shifted.
            # The overlap region still belongs to the old page, so
            # we keep buf_page unchanged — it's the page of the
            # beginning of the chunk.  It will be updated when new
            # text is appended from the next page.
            if not buffer:
                buf_page = page_num

    if buffer:
        yield buffer, buf_page


def normalize_whitespace(lines: Iterable[str]) -> str:
    """Collapse whitespace and join lines."""
    return "\n".join(line.strip() for line in lines if line.strip())
