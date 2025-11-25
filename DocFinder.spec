# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for DocFinder desktop application.

This spec file configures PyInstaller to bundle DocFinder as a standalone
desktop application with all dependencies included.

Usage:
    pyinstaller DocFinder.spec

The output will be in the dist/ directory.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Determine the project root
SPEC_ROOT = Path(SPECPATH)

# Collect all necessary data files
datas = []

# Include HTML templates
datas += [(str(SPEC_ROOT / "src" / "docfinder" / "web" / "templates"), "docfinder/web/templates")]

# Include Logo
if (SPEC_ROOT / "Logo.png").exists():
    datas += [(str(SPEC_ROOT / "Logo.png"), ".")]

# Collect sentence-transformers model data
datas += collect_data_files("sentence_transformers")

# Collect ONNX runtime data
datas += collect_data_files("onnxruntime")

# Collect optimum data
datas += collect_data_files("optimum")

# Hidden imports for dynamic imports
hiddenimports = [
    # Sentence transformers and dependencies
    "sentence_transformers",
    "sentence_transformers.models",
    "transformers",
    "transformers.models",
    "tokenizers",
    "huggingface_hub",
    # ONNX runtime
    "onnxruntime",
    "onnxruntime.transformers",
    # Web framework
    "fastapi",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    "pydantic",
    # GUI
    "webview",
    # PDF processing
    "pypdf",
    # Rich console
    "rich",
    "rich.console",
    "rich.table",
    # Typer CLI
    "typer",
    "click",
    # Numpy and scipy
    "numpy",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.distance",
    # Torch (for sentence-transformers fallback)
    "torch",
    "torch.nn",
    "torch.nn.functional",
    # Tqdm progress bars
    "tqdm",
    "tqdm.auto",
    # Additional dependencies
    "mpmath",
    "sympy",
    "safetensors",
    "filelock",
    "packaging",
    "regex",
    "requests",
    "urllib3",
    "certifi",
    "charset_normalizer",
    "idna",
]

# Collect all submodules for complex packages
hiddenimports += collect_submodules("sentence_transformers")
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fastapi")
hiddenimports += collect_submodules("starlette")

# Platform-specific configurations
if sys.platform == "darwin":
    icon_file = str(SPEC_ROOT / "resources" / "DocFinder.icns")
    if not Path(icon_file).exists():
        icon_file = None
elif sys.platform == "win32":
    icon_file = str(SPEC_ROOT / "resources" / "DocFinder.ico")
    if not Path(icon_file).exists():
        icon_file = None
else:
    icon_file = None

# Analysis
a = Analysis(
    [str(SPEC_ROOT / "src" / "docfinder" / "gui.py")],
    pathex=[str(SPEC_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        "matplotlib",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "wx",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
        "docutils",
        # Exclude CUDA/GPU libraries from bundle (user's system provides these)
        "nvidia",
        "nvidia.cublas",
        "nvidia.cuda_runtime",
        "nvidia.cuda_nvrtc",
        "nvidia.cudnn",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.nccl",
        "nvidia.nvtx",
        "nvidia.nvjitlink",
        "triton",
        "torch.cuda",
        "torch.distributed",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DocFinder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed mode, no console
    disable_windowed_traceback=False,
    argv_emulation=True if sys.platform == "darwin" else False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DocFinder",
)

# macOS app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="DocFinder.app",
        icon=icon_file,
        bundle_identifier="com.docfinder.app",
        info_plist={
            "CFBundleName": "DocFinder",
            "CFBundleDisplayName": "DocFinder",
            "CFBundleVersion": "0.1.0",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleIdentifier": "com.docfinder.app",
            "CFBundlePackageType": "APPL",
            "CFBundleSignature": "DCFN",
            "LSMinimumSystemVersion": "10.15.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,  # Support dark mode
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "PDF Document",
                    "CFBundleTypeExtensions": ["pdf"],
                    "CFBundleTypeRole": "Viewer",
                }
            ],
        },
    )
