"""Utility helpers for working with files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".docx"})


def iter_document_paths(inputs: Iterable[Path]) -> Iterator[Path]:
    """Yield supported document paths from input paths, descending into directories."""
    for item in inputs:
        if item.is_dir():
            yield from sorted(
                p for p in item.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield item


def iter_pdf_paths(inputs: Iterable[Path]) -> Iterator[Path]:
    """Yield PDF paths from input paths (kept for backward compatibility)."""
    for item in inputs:
        if item.is_dir():
            yield from sorted(p for p in item.rglob("*.pdf") if p.is_file())
        elif item.is_file() and item.suffix.lower() == ".pdf":
            yield item


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()
