#!/bin/bash
# Script to run the same checks as CI locally

set -e  # Exit on error

echo "🔍 Running lint checks..."
.venv/bin/ruff check src/ tests/

echo "✨ Running format checks..."
.venv/bin/ruff format --check src/ tests/

echo "🧪 Running tests..."
.venv/bin/pytest -v --cov=docfinder --cov-report=term

echo "✅ All checks passed! Ready to push."
