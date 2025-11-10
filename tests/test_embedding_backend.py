"""Tests for embedding backend detection and ONNX optimization."""

from __future__ import annotations

import platform
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docfinder.embedding.encoder import (
    EmbeddingConfig,
    EmbeddingModel,
    _check_gpu_availability,
    _check_onnx_providers,
    detect_optimal_backend,
)


class TestGPUDetection:
    """Test GPU availability detection."""

    def test_check_gpu_availability_no_torch(self) -> None:
        """Should return (False, None) if torch is not available."""
        with patch.dict("sys.modules", {"torch": None}):
            has_gpu, gpu_type = _check_gpu_availability()
            assert has_gpu is False
            assert gpu_type is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 3090")
    def test_check_gpu_availability_cuda(
        self, mock_device_name: MagicMock, mock_cuda_available: MagicMock
    ) -> None:
        """Should detect CUDA GPU."""
        has_gpu, gpu_type = _check_gpu_availability()
        assert has_gpu is True
        assert gpu_type == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_check_gpu_availability_mps(
        self, mock_mps_available: MagicMock, mock_cuda_available: MagicMock
    ) -> None:
        """Should detect Apple MPS GPU."""
        has_gpu, gpu_type = _check_gpu_availability()
        assert has_gpu is True
        assert gpu_type == "mps"

    def test_check_onnx_providers(self) -> None:
        """Should return list of ONNX providers."""
        providers = _check_onnx_providers()
        assert isinstance(providers, list)
        # On macOS we should have at least CPU provider
        assert "CPUExecutionProvider" in providers


class TestBackendDetection:
    """Test auto-detection of optimal backend based on platform."""

    def test_detect_apple_silicon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should detect Apple Silicon and return ONNX with ARM64 quantized model."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(platform, "processor", lambda: "arm")
        
        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file == "onnx/model_qint8_arm64.onnx"

    def test_detect_intel_mac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should detect Intel Mac and return ONNX with standard model."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(platform, "processor", lambda: "i386")
        
        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file is None

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(True, "cuda"))
    @patch("docfinder.embedding.encoder._check_onnx_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"])
    def test_detect_cuda_gpu(
        self,
        mock_onnx_providers: MagicMock,
        mock_gpu_check: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should detect NVIDIA CUDA GPU and use ONNX with CUDA."""
        monkeypatch.setattr(sys, "platform", "linux")
        
        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file is None

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(True, "rocm"))
    @patch("docfinder.embedding.encoder._check_onnx_providers", return_value=["ROCMExecutionProvider", "CPUExecutionProvider"])
    def test_detect_rocm_gpu(
        self,
        mock_onnx_providers: MagicMock,
        mock_gpu_check: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should detect AMD ROCm GPU and use ONNX with ROCm."""
        monkeypatch.setattr(sys, "platform", "linux")
        
        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file is None

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(False, None))
    @patch("docfinder.embedding.encoder._check_onnx_providers", return_value=["CPUExecutionProvider"])
    def test_detect_linux_cpu(
        self,
        mock_onnx_providers: MagicMock,
        mock_gpu_check: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should detect Linux CPU and use ONNX."""
        monkeypatch.setattr(sys, "platform", "linux")
        
        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file is None

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(False, None))
    @patch("docfinder.embedding.encoder._check_onnx_providers", return_value=["CPUExecutionProvider"])
    def test_detect_windows(
        self,
        mock_onnx_providers: MagicMock,
        mock_gpu_check: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should detect Windows and use ONNX."""
        monkeypatch.setattr(sys, "platform", "win32")
        
        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file is None


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""

    def test_default_config(self) -> None:
        """Should create config with default values."""
        config = EmbeddingConfig()
        assert config.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert config.batch_size == 16
        assert config.normalize is True
        assert config.backend is None
        assert config.onnx_model_file is None
        assert config.device is None

    def test_custom_config(self) -> None:
        """Should create config with custom values."""
        config = EmbeddingConfig(
            model_name="test-model",
            batch_size=32,
            normalize=False,
            backend="onnx",
            onnx_model_file="test.onnx",
            device="cpu",
        )
        assert config.model_name == "test-model"
        assert config.batch_size == 32
        assert config.normalize is False
        assert config.backend == "onnx"
        assert config.onnx_model_file == "test.onnx"
        assert config.device == "cpu"


class TestEmbeddingModel:
    """Test EmbeddingModel with different backends."""

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for testing."""
        return ["Hello world", "Test document", "Another example"]

    def test_auto_detection(self, sample_texts: list[str]) -> None:
        """Should auto-detect backend and work correctly."""
        model = EmbeddingModel()
        assert model.config.backend in ["torch", "onnx"]
        assert model.dimension > 0

        # Test embedding generation
        embeddings = model.embed(sample_texts)
        assert embeddings.shape == (len(sample_texts), model.dimension)
        assert embeddings.dtype == np.float32

    def test_pytorch_backend(self, sample_texts: list[str]) -> None:
        """Should work with PyTorch backend."""
        config = EmbeddingConfig(backend="torch")
        model = EmbeddingModel(config)
        assert model.config.backend == "torch"
        assert model.dimension > 0

        embeddings = model.embed(sample_texts)
        assert embeddings.shape == (len(sample_texts), model.dimension)
        assert embeddings.dtype == np.float32

    @pytest.mark.skipif(
        not (sys.platform == "darwin" or sys.platform == "linux"),
        reason="ONNX backend might not be available on this platform",
    )
    def test_onnx_backend(self, sample_texts: list[str]) -> None:
        """Should work with ONNX backend."""
        config = EmbeddingConfig(backend="onnx")
        model = EmbeddingModel(config)
        # Backend might fallback to torch if ONNX fails
        assert model.config.backend in ["onnx", "torch"]
        assert model.dimension > 0

        embeddings = model.embed(sample_texts)
        assert embeddings.shape == (len(sample_texts), model.dimension)
        assert embeddings.dtype == np.float32

    def test_embed_query(self) -> None:
        """Should generate single query embedding."""
        model = EmbeddingModel()
        query_emb = model.embed_query("test query")
        assert query_emb.shape == (model.dimension,)
        assert query_emb.dtype == np.float32

    def test_embedding_normalization(self) -> None:
        """Should normalize embeddings when configured."""
        config_normalized = EmbeddingConfig(normalize=True)
        model_normalized = EmbeddingModel(config_normalized)
        
        embeddings = model_normalized.embed(["test"])
        # Normalized embeddings should have L2 norm close to 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_embedding_consistency(self) -> None:
        """Should produce consistent embeddings for same input."""
        model = EmbeddingModel()
        text = "consistency test"
        
        emb1 = model.embed_query(text)
        emb2 = model.embed_query(text)
        
        # Should be identical (or extremely close)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    def test_batch_size_effect(self) -> None:
        """Should handle different batch sizes."""
        texts = ["text"] * 100
        
        # Small batch
        config_small = EmbeddingConfig(batch_size=8)
        model_small = EmbeddingModel(config_small)
        emb_small = model_small.embed(texts)
        
        # Large batch
        config_large = EmbeddingConfig(batch_size=32)
        model_large = EmbeddingModel(config_large)
        emb_large = model_large.embed(texts)
        
        # Results should be the same regardless of batch size
        np.testing.assert_array_almost_equal(emb_small, emb_large, decimal=5)
