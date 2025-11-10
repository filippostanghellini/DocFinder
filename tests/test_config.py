"""Tests for application configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from docfinder.config import AppConfig


class TestAppConfig:
    """Test AppConfig dataclass."""

    def test_default_config(self) -> None:
        """Should create config with default values."""
        config = AppConfig()
        
        assert config.db_path == Path("data/docfinder.db")
        assert config.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert config.chunk_chars == 1200
        assert config.overlap == 200

    def test_custom_config(self) -> None:
        """Should create config with custom values."""
        config = AppConfig(
            db_path=Path("/custom/path.db"),
            model_name="custom-model",
            chunk_chars=800,
            overlap=100,
        )
        
        assert config.db_path == Path("/custom/path.db")
        assert config.model_name == "custom-model"
        assert config.chunk_chars == 800
        assert config.overlap == 100

    def test_resolve_db_path_absolute(self) -> None:
        """Should return absolute path as-is."""
        config = AppConfig(db_path=Path("/absolute/path/db.db"))
        
        resolved = config.resolve_db_path()
        
        assert resolved == Path("/absolute/path/db.db")

    def test_resolve_db_path_relative_no_base(self) -> None:
        """Should return relative path when no base_dir provided."""
        config = AppConfig(db_path=Path("relative/db.db"))
        
        resolved = config.resolve_db_path(base_dir=None)
        
        assert resolved == Path("relative/db.db")

    def test_resolve_db_path_relative_with_base(self) -> None:
        """Should resolve relative path against base_dir."""
        config = AppConfig(db_path=Path("relative/db.db"))
        base = Path("/base/directory")
        
        resolved = config.resolve_db_path(base_dir=base)
        
        assert resolved == Path("/base/directory/relative/db.db")

    def test_resolve_db_path_default(self) -> None:
        """Should resolve default path against base_dir."""
        config = AppConfig()
        base = Path("/project")
        
        resolved = config.resolve_db_path(base_dir=base)
        
        assert resolved == Path("/project/data/docfinder.db")
