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


def normalize_whitespace(lines: Iterable[str]) -> str:
    """Collapse whitespace and join lines."""
    return "\n".join(line.strip() for line in lines if line.strip())
