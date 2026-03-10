.PHONY: setup run run-web lint format format-check test check-all install install-gui clean build-macos build-windows build-linux

# ── First-time setup ──────────────────────────────────────────────────────────
# Creates a virtual environment and installs all dependencies in one command.
# Run this once after cloning the repository.
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip --quiet
	.venv/bin/pip install -e ".[dev,web,gui]"
	@echo ""
	@echo "✅ Setup complete!"
	@echo "   Launch the desktop app : make run"
	@echo "   Launch the web UI      : make run-web"
	@echo "   Run tests              : make test"

# ── Run ───────────────────────────────────────────────────────────────────────

# Launch the native desktop GUI
run:
	.venv/bin/docfinder-gui

# Launch the web interface (opens in browser at http://127.0.0.1:8000)
run-web:
	.venv/bin/docfinder web

# ── Install (legacy targets, prefer 'make setup') ─────────────────────────────

# Install dependencies (no GUI)
install:
	.venv/bin/pip install -e ".[dev,web]"

# Install with GUI support
install-gui:
	.venv/bin/pip install -e ".[dev,web,gui]"

# Run linter
lint:
	.venv/bin/ruff check src/ tests/

# Check code formatting
format-check:
	.venv/bin/ruff format --check src/ tests/

# Auto-format code
format:
	.venv/bin/ruff format src/ tests/

# Run tests
test:
	.venv/bin/pytest -v --cov=docfinder --cov-report=term

# Run all CI checks locally
check-all: lint format-check test
	@echo "✅ All checks passed! Ready to push."

# Build macOS app (DMG)
build-macos:
	./scripts/build-macos.sh

# Build Windows app (run on Windows)
build-windows:
	powershell -ExecutionPolicy Bypass -File scripts/build-windows.ps1

# Build Linux app (AppImage)
build-linux:
	./scripts/build-linux.sh

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage coverage.xml .pytest_cache/
	rm -rf resources/DocFinder.iconset resources/DocFinder.icns resources/DocFinder.ico
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
