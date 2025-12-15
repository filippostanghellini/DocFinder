"""Tests for the frontend module."""

from __future__ import annotations

from docfinder.web.frontend import _load_template, router


class TestLoadTemplate:
    """Tests for _load_template function."""

    def test_load_template_returns_string(self) -> None:
        """Template is loaded as a string."""
        result = _load_template()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_template_contains_html(self) -> None:
        """Template contains valid HTML."""
        result = _load_template()
        assert "<html" in result.lower() or "<!doctype" in result.lower()
        assert "</html>" in result.lower()

    def test_load_template_contains_docfinder(self) -> None:
        """Template contains DocFinder branding."""
        result = _load_template()
        assert "DocFinder" in result


class TestRouter:
    """Tests for the frontend router."""

    def test_router_has_index_route(self) -> None:
        """Router has the index route registered."""
        routes = [route.path for route in router.routes]
        assert "/" in routes
