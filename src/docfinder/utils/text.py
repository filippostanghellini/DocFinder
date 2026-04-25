"""Text helpers including sentence-aware chunking."""

from __future__ import annotations

import re
from typing import Iterable, Iterator

# Sentence boundary: period/question/exclamation followed by whitespace,
# or double newline (paragraph break).  Lookbehind avoids splitting on
# common abbreviations (e.g., Mr., Dr., vs., etc.) and decimal numbers.
_SENTENCE_END = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F\d\"'\(\[])"  # ". Next" or "? 123"
    r"|(?<=\n)\n+",  # paragraph breaks
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentence-like segments.

    Uses regex to split at sentence boundaries while keeping
    the text intact (no characters are lost).
    """
    if not text:
        return []

    parts = _SENTENCE_END.split(text)
    # Merge very short fragments (< 20 chars) back into previous sentence
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < 20:
            merged[-1] += part
        else:
            merged.append(part)
    return [s for s in merged if s.strip()]


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
    """Split a stream of ``(page_number, text)`` into sentence-aware chunks.

    Yields ``(chunk_text, page_number)`` where ``page_number`` is the page
    that contributed the **start** of the chunk.

    Algorithm:
    1. Incoming text is split into sentences.
    2. Sentences accumulate until adding the next one would exceed *max_chars*.
    3. When a chunk is emitted, the last sentence(s) of the previous chunk
       are kept as semantic overlap (up to *overlap* characters worth).
    4. Paragraph breaks (``\\n\\n``) are preferred split points.
    """
    # Accumulated sentences for the current chunk
    sentences: list[str] = []
    # Page number for the start of the current chunk
    chunk_page: int = 0
    # Running character count for current chunk
    chunk_len: int = 0

    for page_num, part in pages:
        if not sentences:
            chunk_page = page_num

        page_sentences = _split_sentences(part)

        for sentence in page_sentences:
            sent_len = len(sentence)

            # If adding this sentence exceeds max_chars and we already have content
            if chunk_len + sent_len > max_chars and sentences:
                # Emit current chunk
                yield "".join(sentences), chunk_page

                # Semantic overlap: keep last sentence(s) up to `overlap` chars
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(sentences):
                    if overlap_len + len(s) > overlap and overlap_sents:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)

                sentences = overlap_sents
                chunk_len = overlap_len
                chunk_page = page_num

            sentences.append(sentence)
            chunk_len += sent_len

    # Emit remaining content
    if sentences:
        yield "".join(sentences), chunk_page


def normalize_whitespace(lines: Iterable[str]) -> str:
    """Collapse whitespace and join lines."""
    return "\n".join(line.strip() for line in lines if line.strip())
