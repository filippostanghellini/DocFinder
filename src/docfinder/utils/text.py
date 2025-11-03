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


def normalize_whitespace(lines: Iterable[str]) -> str:
    """Collapse whitespace and join lines."""
    return "\n".join(line.strip() for line in lines if line.strip())
