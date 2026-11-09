"""Tests for the frontend module."""

from __future__ import annotations

from docfinder.web.frontend import _load_template, router


class TestLoadTemplate:
    """Tests for _load_template function."""

    def test_load_template_returns_string(self) -> None:
        """Template is loaded as a string."""
        result = _load_template("index.html")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_template_contains_html(self) -> None:
        """Template contains valid HTML."""
        result = _load_template("index.html")
        assert "<html" in result.lower() or "<!doctype" in result.lower()
        assert "</html>" in result.lower()

    def test_load_template_contains_docfinder(self) -> None:
        """Template contains DocFinder branding."""
        result = _load_template("index.html")
        assert "DocFinder" in result


class TestRouter:
    """Tests for the frontend router."""

    def test_router_has_index_route(self) -> None:
        """Router has the index route registered."""
        routes = [route.path for route in router.routes]
        assert "/" in routes


class TestRagSettingsUiRegression:
    """Regression tests for the AI Chat (RAG) settings section behavior.

    Guards against the bugs where enabling AI Chat stopped suggesting the
    recommended model and blocked model selection while a model loads.
    """

    @property
    def template(self) -> str:
        return _load_template("index.html")

    def test_model_cards_clickable_while_busy(self) -> None:
        """Cards must attach click handlers on canRun alone, not canRun && !isDownloading."""
        t = self.template
        assert "canRun && !isDownloading" not in t
        assert (
            "if (canRun) {\n              div.addEventListener('click', () => selectRagModel(m.name));"
            in t
        )

    def test_poll_starts_for_late_auto_loads(self) -> None:
        """A load triggered after render must be picked up by startRagPollIfBusy."""
        t = self.template
        assert "function startRagPollIfBusy()" in t
        # Called right after enabling the toggle…
        assert "await loadRagModels();\n          startRagPollIfBusy();" in t
        # …and after the auto-load of an already-downloaded model.
        assert "startRagPollIfBusy();\n          });" in t

    def test_toggle_off_stops_polling(self) -> None:
        """Disabling AI Chat stops the download/load poll."""
        t = self.template
        assert "ragModelsSection.style.display = 'none';\n          stopRagPoll();" in t

    def test_recommended_fallback_when_saved_model_invalid(self) -> None:
        """A stale or unrunnable saved model falls back to the recommended tier."""
        t = self.template
        assert "selectedStillValid" in t
        assert "x.recommended && data.total_ram_mb >= x.ram_min_mb" in t

    def test_recommended_badge_uses_backend_flag(self) -> None:
        """The Recommended badge renders from the backend flag regardless of selection."""
        t = self.template
        assert (
            "if (m.recommended) badges += '<span class=\"model-badge recommended\">Recommended</span>';"
            in t
        )
        assert ".model-badge.recommended" in t
