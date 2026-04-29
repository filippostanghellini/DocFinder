"""Tests for settings module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from docfinder import settings
from docfinder.settings import (
    _default_hotkey,
    _settings_dir,
    get_settings_path,
    load_settings,
    save_settings,
)


class TestSettingsDir:
    """Tests for _settings_dir function."""

    def test_settings_dir_windows(self) -> None:
        """Return correct path on Windows."""
        with patch("sys.platform", "win32"):
            with patch.dict("os.environ", {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}):
                result = _settings_dir()
                # On non-Windows platforms Path converts backslashes, so check ends
                assert "DocFinder" in str(result)
                assert "AppData" in str(result) or "AppData\\Roaming" in str(result)

    def test_settings_dir_darwin(self) -> None:
        """Return correct path on macOS."""
        with patch("sys.platform", "darwin"):
            with patch.object(Path, "home", return_value=Path("/Users/test")):
                result = _settings_dir()
                assert result == Path("/Users/test/Library/Application Support/DocFinder")

    def test_settings_dir_linux_with_xdg(self) -> None:
        """Return XDG config path on Linux when set."""
        with patch("sys.platform", "linux"):
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": "/home/test/.config"}):
                result = _settings_dir()
                assert result == Path("/home/test/.config/docfinder")

    def test_settings_dir_linux_without_xdg(self) -> None:
        """Return default config path on Linux when XDG not set."""
        with patch("sys.platform", "linux"):
            with patch.dict("os.environ", {}, clear=True):
                with patch.object(Path, "home", return_value=Path("/home/test")):
                    result = _settings_dir()
                    assert result == Path("/home/test/.config/docfinder")


class TestDefaultHotkey:
    """Tests for _default_hotkey function."""

    def test_returns_alt_d(self) -> None:
        """Returns Alt+D as default hotkey."""
        result = _default_hotkey()
        assert result == "<alt>+d"


class TestGetSettingsPath:
    """Tests for get_settings_path function."""

    def test_returns_settings_json(self) -> None:
        """Returns path to settings.json."""
        path = get_settings_path()
        assert path.name == "settings.json"
        assert "DocFinder" in str(path) or "docfinder" in str(path)


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_returns_defaults_when_file_missing(self, tmp_path: Path) -> None:
        """Returns default settings when file doesn't exist."""
        with patch.object(settings, "get_settings_path", return_value=tmp_path / "nonexistent.json"):
            result = load_settings()
            assert result["hotkey_enabled"] is True
            assert result["hotkey"] == "<alt>+d"

    def test_loads_existing_settings(self, tmp_path: Path) -> None:
        """Loads settings from existing file."""
        settings_file = tmp_path / "settings.json"
        custom_data = {"hotkey": "<cmd>+space", "hotkey_enabled": False}
        settings_file.write_text(json.dumps(custom_data))

        with patch.object(settings, "get_settings_path", return_value=settings_file):
            result = load_settings()
            assert result["hotkey"] == "<cmd>+space"
            assert result["hotkey_enabled"] is False

    def test_merges_defaults_with_custom(self, tmp_path: Path) -> None:
        """Merges custom settings with defaults."""
        settings_file = tmp_path / "settings.json"
        custom_data = {"hotkey_enabled": False}  # Only set one field
        settings_file.write_text(json.dumps(custom_data))

        with patch.object(settings, "get_settings_path", return_value=settings_file):
            result = load_settings()
            assert result["hotkey_enabled"] is False
            # hotkey should still have default value
            assert result["hotkey"] == "<alt>+d"

    def test_returns_defaults_on_invalid_json(self, tmp_path: Path) -> None:
        """Returns defaults when JSON is invalid."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("not valid json")

        with patch.object(settings, "get_settings_path", return_value=settings_file):
            with patch("docfinder.settings.logger") as mock_logger:
                result = load_settings()
                assert result["hotkey_enabled"] is True
                assert result["hotkey"] == "<alt>+d"
                mock_logger.warning.assert_called_once()


class TestSaveSettings:
    """Tests for save_settings function."""

    def test_creates_settings_file(self, tmp_path: Path) -> None:
        """Creates settings file when saving."""
        settings_file = tmp_path / "settings.json"

        with patch.object(settings, "get_settings_path", return_value=settings_file):
            save_settings({"hotkey": "<cmd>+f", "hotkey_enabled": False})

        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert data["hotkey"] == "<cmd>+f"
        assert data["hotkey_enabled"] is False

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Creates parent directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "deep"
        settings_file = nested_dir / "settings.json"

        with patch.object(settings, "get_settings_path", return_value=settings_file):
            save_settings({"hotkey": "<ctrl>+space"})

        assert nested_dir.exists()
        assert settings_file.exists()

    def test_saves_unicode_content(self, tmp_path: Path) -> None:
        """Handles unicode content in settings."""
        settings_file = tmp_path / "settings.json"

        with patch.object(settings, "get_settings_path", return_value=settings_file):
            save_settings({"hotkey": "<cmd>+ß", "custom_text": "日本語"})

        data = json.loads(settings_file.read_text())
        assert data["custom_text"] == "日本語"
