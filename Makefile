.PHONY: lint format test check-all install clean

# Install dependencies
install:
	pip install -e ".[dev,web]"

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

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage coverage.xml .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
