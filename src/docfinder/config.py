"""Application configuration defaults."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from docfinder.embedding.encoder import DEFAULT_MODEL


def _get_default_db_path() -> Path:
    """Get the default database path based on platform and execution context."""
    # User's Documents folder path (primary for frozen apps)
    user_db = Path.home() / "Documents" / "DocFinder" / "docfinder.db"
    
    # When running as a frozen app (PyInstaller bundle)
    if getattr(sys, "frozen", False):
        # Use Documents folder for frozen apps
        return user_db
    
    # When running from source, prefer local data/ if it exists
    local_db = Path("data/docfinder.db")
    if local_db.exists():
        return local_db
    
    # Fall back to Documents folder for consistency
    return user_db


@dataclass(slots=True)
class AppConfig:
    db_path: Path | None = None
    model_name: str = DEFAULT_MODEL
    chunk_chars: int = 500
    overlap: int = 50

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = _get_default_db_path()

    def resolve_db_path(self, base_dir: Path | None = None) -> Path:
        if self.db_path is None:
            self.db_path = _get_default_db_path()
        if Path(self.db_path).is_absolute() or base_dir is None:
            return Path(self.db_path)
        return base_dir / self.db_path
