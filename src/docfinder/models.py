"""Core DocFinder data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class DocumentMetadata:
    """Minimal metadata describing a document."""

    path: Path
    title: str
    sha256: str
    mtime: float
    size: int


@dataclass(slots=True)
class ChunkRecord:
    """Chunk of document text paired with metadata."""

    document_path: Path
    index: int
    text: str
    metadata: Dict[str, Any]
