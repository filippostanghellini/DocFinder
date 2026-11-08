"""Tests for the in-app updater helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

import docfinder
from docfinder.web import updater


class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data

    def getheaders(self):
        return []

    @property
    def headers(self):
        return {}


def test_parse_version():
    assert updater._parse_version("v2.4.0") == (2, 4, 0)
    assert updater._parse_version("release 1.2.3.4") == (1, 2, 3, 4)
    assert updater._parse_version("no version here") is None


def test_version_key_padding():
    assert updater._version_key((2, 4)) > updater._version_key((2, 3, 9))


def test_pick_asset_darwin():
    assets = [
        {
            "name": "DocFinder-Linux-x86_64.AppImage",
            "browser_download_url": "https://x/y",
            "size": 1,
        },
        {
            "name": "DocFinder-macOS.dmg",
            "browser_download_url": "https://github.com/filippostanghellini/DocFinder/releases/download/v2/x.dmg",
            "size": 42,
        },
    ]
    picked = updater._pick_asset(assets)
    assert picked is not None
    assert picked["name"] == "DocFinder-macOS.dmg"
    assert picked["size"] == 42


def test_check_for_update_available(monkeypatch):
    monkeypatch.setattr(docfinder, "__version__", "2.4.0")
    payload = {
        "tag_name": "v2.5.0",
        "html_url": "https://github.com/filippostanghellini/DocFinder/releases/tag/v2.5.0",
        "body": "notes",
        "assets": [
            {
                "name": "DocFinder-macOS.dmg",
                "browser_download_url": "https://github.com/filippostanghellini/DocFinder/releases/download/v2.5.0/DocFinder-macOS.dmg",
                "size": 10,
            }
        ],
    }
    with patch("docfinder.web.updater.urllib.request.urlopen", return_value=_FakeResponse(payload)):
        info = updater.check_for_update()
    assert info["available"] is True
    assert info["latest_version"] == "2.5.0"
    assert info["asset"]["name"] == "DocFinder-macOS.dmg"


def test_check_for_update_up_to_date(monkeypatch):
    monkeypatch.setattr(docfinder, "__version__", "2.5.0")
    payload = {
        "tag_name": "v2.5.0",
        "html_url": "https://example.com",
        "body": "",
        "assets": [],
    }
    with patch("docfinder.web.updater.urllib.request.urlopen", return_value=_FakeResponse(payload)):
        info = updater.check_for_update()
    assert info["available"] is False
