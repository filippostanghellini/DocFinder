# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cross-encoder reranking** — search results are re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-2-v2`) for significantly improved precision; the bi-encoder fetches 3× candidates and the cross-encoder selects the best matches
- **Parallel document ingestion** — files are parsed in parallel via `ProcessPoolExecutor` (up to 4 workers) when indexing ≥ 4 documents; embedding and storage remain on the main process for thread-safety
- **Hybrid RAG context strategy** — small documents (≤ 20 chunks) are loaded entirely into the RAG context, while large documents use a sliding window around the matched chunk
- **Table extraction from PDFs** — PyMuPDF `find_tables()` extracts structured tables and converts them to Markdown for better embedding quality
- New `Reranker` class with lazy model loading (`index/reranker.py`)
- New storage methods `get_document_chunk_count()` and `get_all_chunks()` for hybrid context
- Reranker model pre-loaded at startup alongside the embedding model

### Changed
- Spotlight panel (`spotlight.html`) redesigned to match the main app's visual style (indigo accent, smaller search bar, rounded result items)


## [2.0.0] - 2026-03-12

### Added
- **Local RAG (AI Chat)** — ask questions about your documents using a fully local LLM. No data leaves your machine.
  - Automatic model selection based on system RAM: Qwen2.5-7B (16 GB+), Qwen2.5-3B (8-16 GB), or Qwen2.5-1.5B (any)
  - GPU acceleration: Apple Metal on Apple Silicon, CUDA on NVIDIA, CPU fallback
  - Models downloaded once to `~/.cache/docfinder/models/` via Hugging Face Hub
  - Chat button appears on search results when RAG is enabled in Settings
  - Chat panel slides up from bottom-right with full conversation history per session
- **Page-aware context retrieval** — RAG uses document structure for smarter context:
  - PDF: real page boundaries
  - Markdown: heading-based sections
  - Word (.docx): groups of 10 paragraphs
  - Plain text: virtual pages of ~3000 characters
  - Context expands symmetrically to adjacent pages/sections until the token budget is filled
- **RAG Settings UI** — new "AI Chat (RAG)" section in Settings:
  - Toggle to enable/disable AI Chat
  - Hardware detection shows available RAM
  - Three model cards with size, RAM requirements, and "Recommended" badge
  - Download progress bar with real-time bytes/percentage tracking
  - Model status persisted across sessions
- **Multi-format document support** — DocFinder now indexes PDF, plain text (`.txt`), Markdown (`.md`), and Word (`.docx`) files
- **Spotlight-style quick-search panel** *(experimental)* — a floating `NSPanel` + `WKWebView` can be summoned via the global hotkey to search documents without switching to the main window
- New `[rag]` optional dependency group (`pip install docfinder[rag]`)
- New storage method `get_context_window()` for fixed-window chunk retrieval
- New storage method `get_context_by_page()` for page-aware chunk retrieval
- Search results now include `document_id` for downstream RAG integration

### Changed
- **Redesigned UI theme**
- `EmbeddingModel.embed()` accepts an optional `batch_size` override for low-RAM scenarios
- `build_chunks()` now stores `page` number in chunk metadata for all document formats
- Chunking pipeline uses page-aware `chunk_text_stream_paged()` to preserve page provenance

### New dependencies
- `llama-cpp-python >= 0.3.0` (optional, in `[rag]` extra)

## [1.2.0] - 2026-03-10

### Fixed
- **Windows: Fixed crash on startup** — `create_window()` raised `TypeError: unexpected keyword argument 'icon'` on pywebview builds that don't expose the `icon` parameter. The app now checks for parameter support at runtime via `inspect.signature` and falls back gracefully, so the window opens without an icon instead of crashing entirely.
- **macOS: Fixed dock icon showing Python logo** — When running from source the dock now displays the DocFinder logo instead of the generic Python 3.x icon, via AppKit `NSApplication.setApplicationIconImage_()`.

### Added
- **Global hotkey** — bring DocFinder to the front from any app with a configurable system-wide keyboard shortcut (default: `⌘+Shift+F` on macOS, `Ctrl+Shift+F` on Windows/Linux); implemented via pynput `GlobalHotKeys`
- **Settings tab** — new gear-icon tab lets you enable/disable the global hotkey and change the key combination via an interactive capture modal (press the desired keys, confirm)
- **Native folder picker** — "Browse…" button in the Index tab opens the system file dialog (Finder on macOS, Explorer on Windows) via `window.pywebview.api.pick_folder()`; button is shown only when running inside the desktop app

### Performance 
- **Indexing 2–4× faster** through several compounding improvements:
  - `insert_chunks()` now uses `executemany()` for batch SQLite inserts (was one `execute()` per row)
  - `EmbeddingModel.embed()` uses SentenceTransformer's native batching directly (batch size 32, up from 8); removed the artificial inner mini-batch loop of 4
  - Chunk batch size per document increased from 32 to 64
  - Removed `gc.collect()` calls from inside the per-chunk loop; one call per document is sufficient
  - Removed the artificial 2-files-at-a-time outer loop during indexing
- **First request instant** — `EmbeddingModel` is now a singleton loaded once at startup; previously a new model instance was created for every `/search`, `/documents`, `/index`, and `/cleanup` request

### UI
- **Real-time indexing progress** — animated progress bar with file counter and current filename, updated every 600 ms via polling
- **macOS-native design** — header uses `backdrop-filter: saturate(180%) blur(20px)` for the system frosted-glass effect; improved shadows and depth
- **⌘K / Ctrl+K** shortcut to jump to search from any tab; search input auto-focused on load
- **Drag & drop** — drag a folder from Finder/Explorer directly onto the path input in the Index tab
- Relevance score shown as a **percentage** (e.g. `87%`) instead of a raw float
- Search result **count** displayed above the results list

## [1.1.2] - 2025-12-15

### Fixed
- **Windows: Fixed silent crash on startup** - Application would terminate immediately without any visible window
  - Added `multiprocessing.freeze_support()` required for PyInstaller-bundled apps on Windows
  - Added `torch.cuda` hidden imports to PyInstaller spec for proper bundling
  - Added persistent file logging to `%LOCALAPPDATA%\DocFinder\logs\docfinder.log` for debugging
  - Added native Windows error dialog on startup failures to inform users of issues
  - Improved startup logging with platform and Python version information
- **UI: Fixed Search button jumping on hover** - Button no longer moves when hovered

### Added
- **CI: Smoke tests for all platforms** - Automated verification that bundled apps start correctly
  - macOS: Tests `.app` bundle launches and stays running
  - Windows: Tests `.exe` launches and stays running  
  - Linux: Tests AppImage launches with virtual display (Xvfb)
  - All tests verify log file creation and check for startup errors

### Changed
- Improved multiprocessing handling for PyInstaller frozen apps with early child process detection

## [1.1.1] - 2025-12-12

### Added
- **Improved User Interface**: Enhanced web UI with better design and user experience

### Changed
- **License changed from MIT to AGPL-3.0** to comply with PyMuPDF licensing requirements
- **Switched PDF parsing from pypdf to PyMuPDF** for faster and more reliable text extraction and chunking

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

[Unreleased]: https://github.com/filippostanghellini/DocFinder/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/filippostanghellini/DocFinder/compare/v1.2.0...v2.0.0
[1.2.0]: https://github.com/filippostanghellini/DocFinder/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/filippostanghellini/DocFinder/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/filippostanghellini/DocFinder/compare/v1.0.1...v1.1.1
[1.0.1]: https://github.com/filippostanghellini/DocFinder/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/filippostanghellini/DocFinder/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/filippostanghellini/DocFinder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/filippostanghellini/DocFinder/releases/tag/v0.1.0
