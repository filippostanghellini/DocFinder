.PHONY: lint format test check-all install clean build-macos build-windows build-linux

# Install dependencies
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
