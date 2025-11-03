"""Smoke tests for package imports."""

from docfinder import __version__


def test_version_present() -> None:
    assert __version__ == "0.1.0"
