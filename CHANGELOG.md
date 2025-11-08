# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-08

### Added
- Initial release of DocFinder
- PDF text extraction and chunking with pypdf
- Local semantic embeddings using sentence-transformers and onnxruntime
- SQLite-based vector storage with BLOB storage
- Command-line interface for indexing, searching, and pruning
- Top-k semantic search using cosine similarity
- Optional FastAPI web interface for interactive document search
- Multi-platform support (macOS, Linux, Windows)
- Python 3.10+ compatibility
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- Code quality checks with ruff
- Security scanning with CodeQL and dependency review
- Automated coverage reporting

### Changed
- Implemented lazy imports for web dependencies to allow basic CLI usage without web extras

### Fixed
- Fixed import errors when using CLI without web extras installed
- Fixed linting issues for consistent code style
- Updated ruff configuration to use non-deprecated settings

[Unreleased]: https://github.com/filippostanghellini/DocFinder/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/filippostanghellini/DocFinder/releases/tag/v0.1.0
