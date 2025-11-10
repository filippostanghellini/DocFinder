"""Embedding model management."""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

logger = logging.getLogger(__name__)


def _check_gpu_availability() -> tuple[bool, str | None]:
    """Check if GPU is available and return GPU type.
    
    Returns:
        (has_gpu, gpu_type) where gpu_type is one of:
        - "cuda" for NVIDIA GPU
        - "rocm" for AMD GPU
        - "mps" for Apple Silicon GPU
        - None if no GPU available
    """
    try:
        import torch
        
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.debug(f"CUDA GPU detected: {device_name}")
            return (True, "cuda")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("Apple MPS GPU detected")
            return (True, "mps")
        
        # ROCm detection is trickier - check if CUDA backend is actually ROCm
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            logger.debug("AMD ROCm GPU detected")
            return (True, "rocm")
        
        logger.debug("No GPU detected, will use CPU")
        return (False, None)
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
        return (False, None)
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
        return (False, None)


def _check_onnx_providers() -> list[str]:
    """Check which ONNX Runtime execution providers are available.
    
    Returns:
        List of available provider names.
    """
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        return []


def detect_optimal_backend() -> tuple[Literal["torch", "onnx"], str | None]:
    """Auto-detect the optimal backend based on hardware and platform.
    
    Returns:
        A tuple of (backend_name, onnx_model_file).
        - backend_name: "torch" or "onnx"
        - onnx_model_file: ONNX model file name (if backend is "onnx"), or None
    
    Strategy:
        - Apple Silicon (M1/M2/M3): Use ONNX with ARM64 quantized model + CoreML
        - NVIDIA GPU (CUDA): Use ONNX with CUDAExecutionProvider if available
        - AMD GPU (ROCm): Use ONNX with ROCMExecutionProvider if available
        - Intel Mac: Use ONNX with standard model
        - CPU only: Use ONNX with standard model
    """
    try:
        has_gpu, gpu_type = _check_gpu_availability()
        onnx_providers = _check_onnx_providers()
        
        # Apple Silicon - use quantized ONNX + CoreML
        if sys.platform == "darwin" and (
            platform.processor() == "arm" or platform.machine() == "arm64"
        ):
            logger.info("Detected Apple Silicon - using ONNX with ARM64 quantized model + CoreML")
            return ("onnx", "onnx/model_qint8_arm64.onnx")
        
        # NVIDIA GPU - use ONNX with CUDA if provider available
        if gpu_type == "cuda" and "CUDAExecutionProvider" in onnx_providers:
            logger.info("Detected NVIDIA GPU with CUDA - using ONNX with CUDA acceleration")
            return ("onnx", None)
        
        # AMD GPU - use ONNX with ROCm if provider available
        if gpu_type == "rocm" and "ROCMExecutionProvider" in onnx_providers:
            logger.info("Detected AMD GPU with ROCm - using ONNX with ROCm acceleration")
            return ("onnx", None)
        
        # Intel Mac or generic x86_64 with ONNX
        if sys.platform == "darwin":
            logger.info("Detected Intel Mac - using ONNX with standard model")
            return ("onnx", None)
        
        # Linux/Windows with ONNX (CPU)
        if onnx_providers:
            logger.info(
                f"Detected platform {sys.platform} - using ONNX backend "
                f"(providers: {', '.join(onnx_providers)})"
            )
            return ("onnx", None)
        
        # Fallback to PyTorch if ONNX not available
        logger.info(f"ONNX not available, using PyTorch backend on {sys.platform}")
        return ("torch", None)
        
    except Exception as e:
        logger.warning(f"Failed to detect optimal backend: {e}, falling back to PyTorch")
        return ("torch", None)


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 16
    normalize: bool = True
    backend: Literal["torch", "onnx", "openvino"] | None = None
    onnx_model_file: str | None = None
    device: str | None = None


class EmbeddingModel:
    """Thin wrapper around `SentenceTransformer` for query and document embeddings.
    
    Features:
    - Automatic backend detection (ONNX on macOS, PyTorch elsewhere)
    - Support for quantized ONNX models on Apple Silicon
    - Configurable device selection
    - Fallback to PyTorch if ONNX fails
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        
        # Auto-detect backend if not specified
        if self.config.backend is None:
            self.config.backend, self.config.onnx_model_file = detect_optimal_backend()
        
        # Try to load model with specified backend, fallback to PyTorch on error
        try:
            self._model = self._load_model()
            logger.info(f"Successfully loaded model with backend: {self.config.backend}")
        except Exception as e:
            if self.config.backend != "torch":
                logger.warning(
                    f"Failed to load model with backend '{self.config.backend}': {e}. "
                    "Falling back to PyTorch."
                )
                self.config.backend = "torch"
                self.config.onnx_model_file = None
                self._model = self._load_model()
            else:
                raise
        
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._log_backend_info()
    
    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model with appropriate backend settings."""
        model_kwargs = {}
        
        # Add ONNX-specific model kwargs
        if self.config.backend == "onnx" and self.config.onnx_model_file:
            model_kwargs["file_name"] = self.config.onnx_model_file
        
        # Load model
        return SentenceTransformer(
            self.config.model_name,
            backend=self.config.backend,
            device=self.config.device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
    
    def _log_backend_info(self) -> None:
        """Log information about the backend being used."""
        info_parts = [f"Backend: {self.config.backend}"]
        
        if self.config.backend == "onnx":
            # Get ONNX Runtime providers if available
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                info_parts.append(f"ONNX Providers: {', '.join(providers)}")
            except ImportError:
                pass
            
            if self.config.onnx_model_file:
                info_parts.append(f"ONNX Model: {self.config.onnx_model_file}")
        
        if self.config.device:
            info_parts.append(f"Device: {self.config.device}")
        
        logger.info(" | ".join(info_parts))

    def embed(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        """Return float32 embeddings for input texts."""
        sentences = list(texts)
        embeddings = self._model.encode(
            sentences,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        return embeddings.astype("float32", copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience wrapper for single-query embedding."""
        return self.embed([text])[0]
