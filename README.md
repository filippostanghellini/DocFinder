# DocFinder

[![CI](https://img.shields.io/github/actions/workflow/status/filippostanghellini/DocFinder/ci.yml?branch=main&label=CI&logo=github)](https://github.com/filippostanghellini/DocFinder/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/filippostanghellini/DocFinder/codeql.yml?branch=main&label=CodeQL&logo=github)](https://github.com/filippostanghellini/DocFinder/actions/workflows/codeql.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Stars](https://img.shields.io/github/stars/filippostanghellini/DocFinder?style=social)](https://github.com/filippostanghellini/DocFinder/stargazers)
[![Release](https://img.shields.io/github/v/release/filippostanghellini/DocFinder?logo=github)](https://github.com/filippostanghellini/DocFinder/releases)
[![Downloads](https://img.shields.io/github/downloads/filippostanghellini/DocFinder/total?logo=github)](https://github.com/filippostanghellini/DocFinder/releases)

![Logo](Logo.png)

DocFinder is a local-first CLI for indexing and searching PDF documents using semantic embeddings stored in SQLite. Everything runs on your machine—no external services required.

## Features
- Extracts text from PDFs with configurable chunking powered by `pypdf`.
- Generates local embeddings via `sentence-transformers` and `onnxruntime`.
- Performs top-k semantic search backed by SQLite BLOB storage and cosine similarity.
- Ships with an optional FastAPI web interface that lets you trigger indexing and open PDFs with one click.
- **Auto-detects hardware** (Apple Silicon, NVIDIA GPU, AMD GPU, CPU) and optimizes performance automatically.

## Requirements
- Python 3.10 or newer.
- macOS, Linux, or Windows (tested on Apple Silicon).
- No native SQLite extensions needed—vector data is stored as plain BLOBs.

## Desktop App Installation

Download the latest release for your operating system from [GitHub Releases](https://github.com/filippostanghellini/DocFinder/releases):

| Platform | Download | Notes |
|----------|----------|-------|
| **macOS** | `DocFinder-macOS.dmg` | Drag to Applications folder |
| **Windows** | `DocFinder-Windows-Setup.exe` | Run the installer |
| **Linux** | `DocFinder-Linux-x86_64.AppImage` | Make executable and run |

### macOS Installation

1. Download `DocFinder-macOS.dmg`
2. Open the DMG file
3. Drag **DocFinder** to the **Applications** folder
4. **First launch**: Right-click (or Control-click) on DocFinder and select **Open**
5. Click **Open** in the dialog to bypass Gatekeeper

> ⚠️ **Security Note**: Since DocFinder is not signed with an Apple Developer ID, macOS will show a warning on first launch. This is normal for open-source software. The "Open" option appears only when you right-click the app.

### Windows Installation

1. Download `DocFinder-Windows-Setup.exe`
2. Run the installer
3. **SmartScreen warning**: Click **More info** → **Run anyway**
4. Follow the installation wizard
5. Launch from Start Menu or Desktop shortcut

> ⚠️ **Security Note**: Windows SmartScreen may warn about an unrecognized app. This is expected for unsigned software. Click "More info" to reveal the "Run anyway" button.

### Linux Installation

```bash
# Download the AppImage
wget https://github.com/filippostanghellini/DocFinder/releases/latest/download/DocFinder-Linux-x86_64.AppImage

# Make it executable
chmod +x DocFinder-Linux-x86_64.AppImage

# Run
./DocFinder-Linux-x86_64.AppImage
```

> **Tip**: You can integrate the AppImage with your desktop environment using [AppImageLauncher](https://github.com/TheAssassin/AppImageLauncher).

## Installation (Python Package)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

## Build from Source

If you clone this repository and want to build the desktop app yourself:

### Quick Start (Development)

```bash
# Clone the repository
git clone https://github.com/filippostanghellini/DocFinder.git
cd DocFinder

# Create virtual environment and install dependencies
make install-gui

# Run the desktop GUI
docfinder-gui
```

### Build the Desktop App

```bash
# macOS - creates DocFinder.app in dist/
make build-macos

# Windows (PowerShell)
.\scripts\build-windows.ps1

# Linux
./scripts/build-linux.sh
```

The built app will be in the `dist/` folder.

### Data Storage

- **Desktop App**: Database stored in `~/Documents/DocFinder/docfinder.db`
- **Development**: Database stored in `data/docfinder.db` (project folder)

---

For **NVIDIA GPU acceleration** (Linux/Windows):

```bash
pip install '.[gpu]'
```

Development extras:

```bash
pip install '.[dev]'
```

Web interface extras:

```bash
pip install '.[web]'
```

All extras combined:

```bash
pip install '.[dev,web,gpu]'
```

Desktop GUI extras (for running the native window interface from source):

```bash
pip install '.[gui]'
```

> **Note**: DocFinder automatically detects your hardware (Apple Silicon, NVIDIA GPU, AMD GPU, or CPU) and uses the optimal backend. GPU support on NVIDIA requires the `gpu` extra above.

## Usage

### Desktop App

If you installed the desktop app, simply launch **DocFinder** from your Applications folder (macOS), Start Menu (Windows), or run the AppImage (Linux). The app opens a native window with the full web interface.

### Command Line

#### Index a folder

```bash
docfinder index ~/Documents --db data/docfinder.db
```

#### Run a semantic search

```bash
docfinder search "contract of sale" --db data/docfinder.db --top-k 10
```

#### Launch the web interface

```bash
docfinder web --db data/docfinder.db --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser to search and open PDFs.

#### Launch the desktop GUI (from source)

```bash
docfinder-gui
```

This opens a native window with the web interface embedded.

### Screenshots

**Search**
![Search](images/search.png)

**Index Documents**
![Index Documents](images/index-documents.png)

**Database**
![Database](images/database-documents.png)

To index through the UI:

1. Start the server with the command above.
2. Enter the absolute path of the folder (or single PDF) in the **Index** panel.
3. Click **Index** and wait for the completion summary.

Search results list the file path, similarity score, and an excerpt from the matching chunk.

## Project structure
- `src/docfinder/ingestion`: PDF parsing and chunking.
- `src/docfinder/embedding`: embedding model wrappers.
- `src/docfinder/index`: SQLite vector storage and search.
- `src/docfinder/utils`: hashing, chunking, and file helpers.
- `tests`: automated checks.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

> **Note**: DocFinder was originally released under the MIT License. Starting from version 0.2.0, the license has been changed to AGPL-3.0 to comply with the licensing requirements of [PyMuPDF](https://pymupdf.readthedocs.io/), which is distributed under AGPL-3.0. This ensures full license compatibility and keeps DocFinder and its ecosystem fully open source.
