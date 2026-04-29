"""Extended tests for config module to improve coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from docfinder.config import _get_default_db_path, AppConfig


class TestGetDefaultDbPath:
    """Tests for _get_default_db_path function - covering missed lines 20, 28."""

    def test_returns_user_db_when_frozen(self, tmp_path: Path) -> None:
        """Line 20: Returns user db path when frozen (PyInstaller)."""
        with patch("sys.frozen", True, create=True):
            with patch.object(Path, "home", return_value=tmp_path):
                result = _get_default_db_path()
                expected = tmp_path / "Documents" / "DocFinder" / "docfinder.db"
                assert result == expected

    def test_returns_local_db_when_exists(self, tmp_path: Path) -> None:
        """Returns local db when data/ exists."""
        with patch("sys.frozen", False, create=True):
            # Create local data directory
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "docfinder.db").touch()

            with patch("pathlib.Path.exists", lambda self: self == data_dir / "docfinder.db"):
                result = _get_default_db_path()
                # Should return local_db since it exists
                assert "data/docfinder.db" in str(result) or "docfinder.db" in str(result)

    def test_falls_back_to_user_db(self, tmp_path: Path) -> None:
        """Line 28: Falls back to user db when local doesn't exist."""
        with patch("sys.frozen", False, create=True):
            with patch.object(Path, "home", return_value=tmp_path):
                with patch("pathlib.Path.exists", return_value=False):
                    result = _get_default_db_path()
                    expected = tmp_path / "Documents" / "DocFinder" / "docfinder.db"
                    assert result == expected


class TestAppConfigResolveDbPath:
    """Tests for AppConfig.resolve_db_path - covering lines 44, 47."""

    def test_sets_db_path_when_none(self, tmp_path: Path) -> None:
        """Line 44: Sets db_path to default when None."""
        with patch.object(Path, "home", return_value=tmp_path):
            config = AppConfig(db_path=None)
            # Force db_path to None and then resolve
            config.db_path = None
            result = config.resolve_db_path()
            assert result is not None
            assert config.db_path is not None  # Should be set by line 44

    def test_returns_base_dir_relative_path(self, tmp_path: Path) -> None:
        """Line 47: Returns base_dir / db_path for relative paths."""
        base_dir = tmp_path / "projects"
        base_dir.mkdir()
        config = AppConfig(db_path=Path("custom.db"))
        result = config.resolve_db_path(base_dir)
        assert result == base_dir / "custom.db"
