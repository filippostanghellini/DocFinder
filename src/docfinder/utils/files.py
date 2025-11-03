"""Utility helpers for working with files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator


def iter_pdf_paths(inputs: Iterable[Path]) -> Iterator[Path]:
    """Yield PDF paths from input paths, descending into directories."""
    for item in inputs:
        if item.is_dir():
            yield from iter_pdf_paths(sorted(child for child in item.rglob("*.pdf")))
        elif item.is_file() and item.suffix.lower() == ".pdf":
            yield item


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()
