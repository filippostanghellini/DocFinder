"""Application configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docfinder.embedding.encoder import DEFAULT_MODEL


@dataclass(slots=True)
class AppConfig:
    db_path: Path = Path("data/docfinder.db")
    model_name: str = DEFAULT_MODEL
    chunk_chars: int = 1200
    overlap: int = 200

    def resolve_db_path(self, base_dir: Path | None = None) -> Path:
        if Path(self.db_path).is_absolute() or base_dir is None:
            return Path(self.db_path)
        return base_dir / self.db_path
