# DocFinder

[![CI](https://img.shields.io/github/actions/workflow/status/filippostanghellini/DocFinder/ci.yml?branch=main&label=CI&logo=github)](https://github.com/filippostanghellini/DocFinder/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/filippostanghellini/DocFinder/codeql.yml?branch=main&label=CodeQL&logo=github)](https://github.com/filippostanghellini/DocFinder/actions/workflows/codeql.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/filippostanghellini/DocFinder?logo=github)](https://github.com/filippostanghellini/DocFinder/releases)
[![Downloads](https://img.shields.io/github/downloads/filippostanghellini/DocFinder/total?logo=github)](https://github.com/filippostanghellini/DocFinder/releases)

<p align="center">
  <img src="Logo.png" alt="DocFinder Logo" width="160">
</p>

<p align="center">
  <strong>Local-first semantic search for your documents.</strong><br>
  Supports PDF, Word, PowerPoint, OpenDocument, HTML, EPUB, Markdown, plain text, and more.<br>
  Everything runs on your machine — no cloud, no accounts, complete privacy.
</p>

<p align="center">
  <img src="images/demo.gif" alt="DocFinder Demo" width="700">
</p>                                                                        

## Features

- **Semantic search** — find documents by meaning, not just keywords (PDF, DOCX, PPTX, ODT, HTML, EPUB + others)
- **AI chat** — ask questions about any document and get precise answers, powered by local Qwen3.5 models (automatically selects the best model for your hardware)
- **Bring your own models (Ollama)** — connect to a local or remote Ollama server and pick any embedding or chat model it serves
- **100% privacy mode** — per-indexing-run checkbox: files marked private never leave your machine (local embedder required, remote LLMs blocked for chat)
- **100% local** — your files never leave your machine
- **GPU accelerated** — auto-detects Apple Silicon (Metal), NVIDIA (CUDA), AMD (ROCm)
- **Cross-platform** — native apps for macOS, Windows, and Linux
- **Global shortcut** (Experimental) — bring DocFinder to front from anywhere with a configurable hotkey

<p align="center">
  <img src="images/chat.png" alt="DocFinder Chat" width="700">
</p>

## Download

| Platform | Installer |
|----------|-----------|
| **macOS** | [DocFinder-macOS.dmg](https://github.com/filippostanghellini/DocFinder/releases/latest) |
| **Windows** | [DocFinder-Windows-Setup.exe](https://github.com/filippostanghellini/DocFinder/releases/latest) |
| **Linux** | [DocFinder-Linux-x86_64.AppImage](https://github.com/filippostanghellini/DocFinder/releases/latest) |

**macOS** — open the DMG, drag DocFinder to Applications, then right-click → **Open** on first launch (Gatekeeper warning — normal for unsigned open-source apps).

**Windows** — run the installer; if SmartScreen appears choose **More info → Run anyway**.

**Linux**
```bash
chmod +x DocFinder-Linux-x86_64.AppImage && ./DocFinder-Linux-x86_64.AppImage
```

## Run from Source

Requires Python 3.10+ and `make`.

```bash
git clone https://github.com/filippostanghellini/DocFinder.git
cd DocFinder
make setup   # create .venv and install all dependencies
make run     # desktop GUI
make run-web # web interface at http://127.0.0.1:8000
```

### Runtime acceleration (auto)

DocFinder automatically selects the best available runtime on your machine:

- **NVIDIA**: ONNX CUDA provider when available, otherwise PyTorch CUDA
- **AMD**: ONNX ROCm provider when available, otherwise PyTorch fallback
- **Apple Silicon**: PyTorch MPS when available (fastest), ONNX ARM64 fallback otherwise
- **Intel Mac / CPU-only**: ONNX or PyTorch CPU fallback

Indexing uses an adaptive parallel parser strategy by default, selected automatically based on your
machine resources.

### Custom models via Ollama

DocFinder is not locked to its built-in models. In **Settings → Models & Ollama** you can connect
to any Ollama server and choose which models to use:

- **Server URL** (e.g. `http://127.0.0.1:11434`) plus an optional API key for remote Ollama
  providers, with a one-click **Test connection** that lists the installed models
- **Embedding model** — pick any embedding model served by Ollama; note that changing the
  embedding model invalidates the existing index, so use the **Re-index now** button afterwards
- **Chat model** — pick any chat model served by Ollama for the AI Chat; applies immediately,
  no reindex needed

Both choices are independent, so you can run a small embedding model and a larger chat LLM side
by side on the same server. The server can be local or a VPS/remote Ollama provider — embeddings
and chat requests go over HTTP.

<p align="center">
  <img src="images/ollama_models.png" alt="Models & Ollama settings card" width="700">
</p>

### 100% privacy mode

The **100% privacy** checkbox in the Index tab marks an indexing run as strictly local:

- **Indexing** requires a local embedding model — the checkbox is disabled while a remote
  Ollama server is configured
- **AI chat** on privacy-marked documents refuses remote LLMs (Ollama on `localhost` is fine —
  data never leaves your machine); the built-in GGUF models are always allowed
- Privacy-marked documents show a shield badge in the Documents tab, and re-indexing preserves
  the flag

Documents indexed without the flag keep working with any model, including remote Ollama servers.

<p align="center">
  <img src="images/100_privacy.png" alt="100% privacy checkbox in the Index tab" width="700">
</p>

## Contributing

Contributions are welcome, feel free to open an issue or submit a pull request.

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

> DocFinder was originally released under the MIT License. Starting from version 1.1.1 the license was changed to AGPL-3.0 to comply with the [PyMuPDF](https://pymupdf.readthedocs.io/) licensing requirements, as PyMuPDF itself is AGPL-3.0 licensed.
