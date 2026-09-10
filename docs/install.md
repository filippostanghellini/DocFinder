# Installation

## Download Prebuilt Binaries

| Platform | Installer | Notes |
|----------|-----------|-------|
| **macOS** | [DocFinder-macOS.dmg](https://github.com/filippostanghellini/DocFinder/releases/latest) | Open the DMG, drag to Applications, right-click → **Open** on first launch |
| **Windows** | [DocFinder-Windows-Setup.exe](https://github.com/filippostanghellini/DocFinder/releases/latest) | Run the installer; if SmartScreen appears choose **More info → Run anyway** |
| **Linux** | [DocFinder-Linux-x86_64.AppImage](https://github.com/filippostanghellini/DocFinder/releases/latest) | `chmod +x DocFinder-Linux-x86_64.AppImage && ./DocFinder-Linux-x86_64.AppImage` |

## Run from Source

### Prerequisites

- Python 3.10 or later
- `make` (on Windows: install via [Chocolatey](https://chocolatey.org/) or use WSL)

### Setup

```bash
git clone https://github.com/filippostanghellini/DocFinder.git
cd DocFinder
make setup
```

This creates a virtual environment (`.venv/`) and installs all dependencies.

### Launch

```bash
make run       # desktop GUI
make run-web   # web interface at http://127.0.0.1:8000
```

### Manual Installation

If you prefer not to use `make`:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -e ".[dev,web,gui]"
```

## Available Extras

| Extra | Packages | When to use |
|-------|----------|-------------|
| `dev` | pytest, pytest-cov, ruff, safety, bandit, pyinstaller | Development & testing |
| `web` | fastapi, uvicorn, pydantic | Web interface only |
| `gui` | pywebview, fastapi, uvicorn, pydantic, pynput | Desktop GUI |
| `rag` | llama-cpp-python | AI chat / RAG features |
| `gpu` | onnxruntime-gpu | GPU acceleration (NVIDIA) |
| `docs` | mkdocs, mkdocs-material | Documentation site (`make serve-docs` / `build-docs`) |

Install a specific set:

```bash
pip install -e ".[dev,web,gui,rag]"
```

## Runtime Acceleration

DocFinder automatically selects the best available backend:

- **NVIDIA**: ONNX CUDA provider → PyTorch CUDA
- **AMD**: ONNX ROCm provider → PyTorch fallback
- **Apple Silicon**: PyTorch MPS → ONNX ARM64
- **Intel Mac / CPU-only**: ONNX or PyTorch CPU
