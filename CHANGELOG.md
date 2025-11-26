# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2025-11-26

### Fixed
- Fixed Linux desktop build failing due to disk space issues on CI runner
- Excluded unnecessary CUDA/GPU libraries from PyInstaller bundle (reduces size ~2GB)
- Optimized CI workflow with disk cleanup and CPU-only PyTorch for builds

### Notes
- GPU acceleration (NVIDIA CUDA, AMD ROCm, Apple Metal) still fully supported via system libraries at runtime

## [1.0.0] - 2025-11-24

### Added
- **🖥️ Native Desktop Application**: Cross-platform GUI using pywebview
  - macOS: `.app` bundle with DMG installer
  - Windows: NSIS installer (`.exe`)
  - Linux: AppImage portable executable
- **Desktop GUI launcher** (`docfinder-gui` command) for running from source
- **PyInstaller build configuration** (`DocFinder.spec`) for standalone packaging
- **Build scripts** for all platforms:
  - `scripts/build-macos.sh` - Creates `.app` and `.dmg`
  - `scripts/build-windows.ps1` - Creates NSIS installer
  - `scripts/build-linux.sh` - Creates AppImage
- **GitHub Actions workflow** (`build-desktop.yml`) for automated multi-platform builds on release
- **New `[gui]` optional dependency group** for pywebview
- macOS app icon (`resources/DocFinder.icns`)

### Changed
- **Database location for desktop app**: Now stored in `~/Documents/DocFinder/docfinder.db`
- Improved error messages in web API for missing database
- Updated README with desktop installation instructions and security bypass guides
- Updated Makefile with `install-gui`, `build-macos`, `build-windows`, `build-linux` targets

### Fixed
- Database path resolution for frozen (packaged) applications
- Config tests updated for dynamic database path behavior

## [0.2.0] - 2025-11-11

### Added
- **ONNX Runtime Backend Support**: Automatic backend detection with ONNX optimization
  - 1.5x-1.7x performance improvement on Apple Silicon with quantized ONNX models
  - Full GPU support: NVIDIA CUDA, AMD ROCm, Apple Silicon (CoreML/MPS)
  - Automatic platform and GPU detection with intelligent fallback
  - Configurable backend selection via `EmbeddingConfig`
- GPU detection functions: `_check_gpu_availability()`, `_check_onnx_providers()`
- Added `optimum` dependency for ONNX model support
- **New optional dependency group `[gpu]`** for easy NVIDIA GPU support installation
- Comprehensive test suite for backend detection and GPU support
- **Mini-batch processing in embeddings** (4 texts at a time) to reduce memory peaks
- **Batch file processing during indexing** (2 files at a time) for better memory management
- **Strategic garbage collection** after each processing batch to prevent memory leaks

### Changed
- `EmbeddingModel` auto-detects optimal backend based on platform and GPU availability
- `detect_optimal_backend()` extended with CUDA and ROCm GPU support
- `EmbeddingConfig` extended with `backend`, `onnx_model_file`, and `device` parameters
- Improved logging for backend and execution provider information
- **Reduced default `batch_size` from 16 to 8** in `EmbeddingConfig` for better memory efficiency
- **Reduced default `chunk_chars` from 1000 to 500** to lower memory footprint during chunking
- **Reduced default `overlap` from 100 to 50** for more efficient text processing
- **`IndexPayload` now accepts `List[str]` instead of `List[Path]`** for better web form compatibility
- Embedding process now uses mini-batching with memory cleanup between batches

### Fixed
- **Fixed path handling with spaces in directory names** in web interface
- **Fixed memory exhaustion (OOM) on 16GB RAM systems** during large indexing operations
- **Fixed web form validation** to properly accept file paths with spaces and special characters
- Improved memory cleanup to prevent process kills on memory-constrained systems

### Performance
- Apple Silicon (M4): ~1.7x faster with optimized memory usage
- **Memory usage optimized**: Reduced peak RAM consumption by ~30% during indexing
- **Prevented OOM crashes**: Successfully indexes large document collections on 16GB RAM systems
- NVIDIA GPU: 2-5x faster for large batches
- AMD GPU: Optimized inference with ROCm provider
- Intel Mac: ~1.5x faster with ONNX

## Security
- **🔒 CRITICAL: Fixed path traversal vulnerabilities** ([CWE-22](https://cwe.mitre.org/data/definitions/22.html))
  - Implemented canonical path validation with `os.path.realpath()`
  - Added string prefix check to ensure paths are within safe base directory
  - Restricted file access to user's home directory by default
  - Added null byte validation to prevent directory traversal attacks
  - Protection against symlink attacks
- **Fixed log injection vulnerability** by using safe format strings (`%s`) instead of f-strings with user input
- All paths now validated before filesystem operations
- Comprehensive security validation as suggested by GitHub CodeQL

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

[Unreleased]: https://github.com/filippostanghellini/DocFinder/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/filippostanghellini/DocFinder/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/filippostanghellini/DocFinder/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/filippostanghellini/DocFinder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/filippostanghellini/DocFinder/releases/tag/v0.1.0
