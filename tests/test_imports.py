"""Smoke tests for package imports."""

from docfinder import __version__


def test_version_present() -> None:
    assert __version__ == "1.1.2"
