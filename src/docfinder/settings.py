"""Persistent user settings for DocFinder."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _settings_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home())))
        return base / "DocFinder"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "DocFinder"
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "docfinder"


def get_settings_path() -> Path:
    return _settings_dir() / "settings.json"


def _default_hotkey() -> str:
    """Return a platform-appropriate default global hotkey string (pynput format).

    Uses Option+D (⌥D) which is free on all platforms and easy to press.
    """
    return "<alt>+d"


_DEFAULTS: dict = {
    "hotkey_enabled": True,
}


def load_settings() -> dict:
    defaults = {**_DEFAULTS, "hotkey": _default_hotkey()}
    path = get_settings_path()
    if not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {**defaults, **data}
    except Exception as exc:
        logger.warning("Failed to read settings from %s: %s", path, exc)
        return defaults


def save_settings(data: dict) -> None:
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Settings saved to %s", path)
