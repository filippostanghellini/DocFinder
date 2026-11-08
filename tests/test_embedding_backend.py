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
    _preferred_torch_device,
    detect_optimal_backend,
    detect_optimal_backend_config,
)

# Small model for integration tests: keeps them fast and independent of DEFAULT_MODEL
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(True, "mps"))
    def test_preferred_torch_device_mps(self, mock_gpu_check: MagicMock) -> None:
        """Should prefer mps torch device when MPS is available."""
        assert _preferred_torch_device() == "mps"


class TestBackendDetection:
    """Test auto-detection of optimal backend based on platform."""

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(True, "mps"))
    def test_detect_apple_silicon(self, mock_gpu_check: MagicMock) -> None:
        """Should detect Apple Silicon MPS GPU and use PyTorch MPS backend."""
        backend, model_file = detect_optimal_backend()
        assert backend == "torch"
        assert model_file is None

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(False, None))
    def test_detect_intel_mac(
        self, mock_gpu_check: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should detect Intel Mac and return ONNX with standard model."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(platform, "processor", lambda: "i386")

        backend, model_file = detect_optimal_backend()
        assert backend == "onnx"
        assert model_file is None

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(True, "cuda"))
    @patch(
        "docfinder.embedding.encoder._check_onnx_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
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
    @patch(
        "docfinder.embedding.encoder._check_onnx_providers",
        return_value=["ROCMExecutionProvider", "CPUExecutionProvider"],
    )
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
    @patch(
        "docfinder.embedding.encoder._check_onnx_providers", return_value=["CPUExecutionProvider"]
    )
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
    @patch(
        "docfinder.embedding.encoder._check_onnx_providers", return_value=["CPUExecutionProvider"]
    )
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

    @patch("docfinder.embedding.encoder._check_gpu_availability", return_value=(True, "cuda"))
    @patch(
        "docfinder.embedding.encoder._check_onnx_providers",
        return_value=["CPUExecutionProvider"],
    )
    def test_detect_cuda_without_onnx_cuda_provider_uses_torch_cuda(
        self,
        mock_onnx_providers: MagicMock,
        mock_gpu_check: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should use torch/cuda if GPU exists but ONNX CUDA provider is missing."""
        monkeypatch.setattr(sys, "platform", "linux")
        backend, model_file, device = detect_optimal_backend_config()
        assert backend == "torch"
        assert model_file is None
        assert device == "cuda"


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""

    def test_default_config(self) -> None:
        """Should create config with default values."""
        config = EmbeddingConfig()
        assert config.model_name == "google/embeddinggemma-2"
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.backend is None
        assert config.onnx_model_file is None
        assert config.device is None
        assert config.modalities == ("text",)
        assert config.truncate_dim is None

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
        model = EmbeddingModel(EmbeddingConfig(model_name=TEST_MODEL))
        assert model.config.backend in ["torch", "onnx"]
        assert model.dimension > 0

        # Test embedding generation
        embeddings = model.embed(sample_texts)
        assert embeddings.shape == (len(sample_texts), model.dimension)
        assert embeddings.dtype == np.float32

    def test_pytorch_backend(self, sample_texts: list[str]) -> None:
        """Should work with PyTorch backend."""
        config = EmbeddingConfig(model_name=TEST_MODEL, backend="torch")
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
        config = EmbeddingConfig(model_name=TEST_MODEL, backend="onnx")
        model = EmbeddingModel(config)
        # Backend might fallback to torch if ONNX fails
        assert model.config.backend in ["onnx", "torch"]
        assert model.dimension > 0

        embeddings = model.embed(sample_texts)
        assert embeddings.shape == (len(sample_texts), model.dimension)
        assert embeddings.dtype == np.float32

    def test_embed_query(self) -> None:
        """Should generate single query embedding."""
        model = EmbeddingModel(EmbeddingConfig(model_name=TEST_MODEL))
        query_emb = model.embed_query("test query")
        assert query_emb.shape == (model.dimension,)
        assert query_emb.dtype == np.float32

    def test_embedding_normalization(self) -> None:
        """Should normalize embeddings when configured."""
        config_normalized = EmbeddingConfig(model_name=TEST_MODEL, normalize=True)
        model_normalized = EmbeddingModel(config_normalized)

        embeddings = model_normalized.embed(["test"])
        # Normalized embeddings should have L2 norm close to 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_embedding_consistency(self) -> None:
        """Should produce consistent embeddings for same input."""
        model = EmbeddingModel(EmbeddingConfig(model_name=TEST_MODEL))
        text = "consistency test"

        emb1 = model.embed_query(text)
        emb2 = model.embed_query(text)

        # Should be identical (or extremely close)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    def test_batch_size_effect(self) -> None:
        """Should handle different batch sizes."""
        texts = ["text"] * 100

        # Small batch
        config_small = EmbeddingConfig(model_name=TEST_MODEL, batch_size=8)
        model_small = EmbeddingModel(config_small)
        emb_small = model_small.embed(texts)

        # Large batch
        config_large = EmbeddingConfig(model_name=TEST_MODEL, batch_size=32)
        model_large = EmbeddingModel(config_large)
        emb_large = model_large.embed(texts)

        # Results should be the same regardless of batch size
        np.testing.assert_array_almost_equal(emb_small, emb_large, decimal=5)


class TestEmbeddingGemma2Features:
    """Tests for EmbeddingGemma 2 modular loading, precision, prefixes, and legacy cleanup."""

    def test_selective_modality_config_kwargs(self) -> None:
        from docfinder.embedding.encoder import build_modality_config_kwargs

        assert build_modality_config_kwargs("google/embeddinggemma-2", ("text",)) == {
            "vision_config": None,
            "audio_config": None,
        }
        assert build_modality_config_kwargs("google/embeddinggemma-2", ("text", "image")) == {
            "audio_config": None,
        }
        assert (
            build_modality_config_kwargs("google/embeddinggemma-2", ("text", "image", "audio"))
            is None
        )
        assert build_modality_config_kwargs(TEST_MODEL, ("text",)) is None

    def test_safe_torch_dtype_never_float16(self) -> None:
        import torch

        from docfinder.embedding.encoder import _preferred_torch_dtype

        for dev in ("cpu", "mps", "cuda"):
            dtype = _preferred_torch_dtype(dev)
            assert dtype in (torch.bfloat16, torch.float32)
            assert dtype != torch.float16

    def test_task_instruction_prefixes(self) -> None:
        from docfinder.embedding.encoder import format_document_text, format_query_text

        q = format_query_text("invoice 2026", task="search", model_name="google/embeddinggemma-2")
        assert q == "task: search result | query: invoice 2026"
        # Idempotent
        assert format_query_text(q, model_name="google/embeddinggemma-2") == q

        d = format_document_text(
            "Total EUR 120", title="Invoice", model_name="google/embeddinggemma-2"
        )
        assert d == "title: Invoice | text: Total EUR 120"
        assert format_document_text(d, model_name="google/embeddinggemma-2") == d

    def test_ollama_tags_for_modalities(self) -> None:
        from docfinder.embedding.encoder import ollama_tag_for_modalities

        assert ollama_tag_for_modalities(["text"]) == "embeddinggemma-2:270m"
        assert ollama_tag_for_modalities(["text", "image"]) == "embeddinggemma-2:440m"
        assert ollama_tag_for_modalities(["text", "audio"]) == "embeddinggemma-2:570m"
        assert ollama_tag_for_modalities(["text", "image", "audio"]) == "embeddinggemma-2:740m"

    def test_cleanup_legacy_models(self, tmp_path) -> None:
        from docfinder.embedding.encoder import cleanup_legacy_models

        legacy_dir = tmp_path / "models--BAAI--bge-m3"
        legacy_dir.mkdir()
        (legacy_dir / "weights.bin").write_bytes(b"123")

        removed = cleanup_legacy_models(cache_dir=tmp_path)
        assert "BAAI/bge-m3" in removed
        assert not legacy_dir.exists()

    def test_detect_missing_modalities_and_new_formats(self, tmp_path) -> None:
        from docfinder.ingestion.pdf_loader import build_chunks
        from docfinder.utils.files import detect_missing_modalities, iter_document_paths

        csv_f = tmp_path / "table.csv"
        csv_f.write_text("name,amount\nAlice,42\n", encoding="utf-8")
        eml_f = tmp_path / "note.eml"
        eml_f.write_text("Subject: Budget\nFrom: a@b.com\n\nHello team\n", encoding="utf-8")
        img_f = tmp_path / "scan.png"
        img_f.write_bytes(b"fake")
        aud_f = tmp_path / "memo.mp3"
        aud_f.write_bytes(b"fake")

        # Default first launch: only text+code files are yielded
        text_only_docs = list(iter_document_paths([tmp_path], modalities=["text"]))
        assert csv_f in text_only_docs
        assert eml_f in text_only_docs
        assert img_f not in text_only_docs
        assert aud_f not in text_only_docs

        # Missing modalities detector flags both image and audio
        all_docs = list(iter_document_paths([tmp_path]))
        missing = detect_missing_modalities(all_docs, ["text"])
        assert "image" in missing and missing["image"]["count"] == 1
        assert "audio" in missing and missing["audio"]["count"] == 1

        # CSV and EML chunking
        csv_chunks = list(build_chunks(csv_f))
        assert len(csv_chunks) == 1 and "Alice" in csv_chunks[0].text
        eml_chunks = list(build_chunks(eml_f))
        assert len(eml_chunks) == 1 and "Budget" in eml_chunks[0].text
        img_chunks = list(build_chunks(img_f, modalities=["text", "image"]))
        assert len(img_chunks) == 1 and img_chunks[0].metadata["modality"] == "image"

    def test_hybrid_reranker_preserves_multimodal_score(self) -> None:
        from docfinder.index.reranker import Reranker

        reranker = Reranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.0]  # sigmoid(0.0) = 0.5

        candidates = [
            {"text": "regular text", "score": 0.4, "metadata": {"modality": "text"}},
            {"text": "[Image: scan.png]", "score": 0.92, "metadata": {"modality": "image"}},
        ]
        ranked = reranker.rerank("invoice", candidates, top_k=2)
        assert ranked[0]["metadata"]["modality"] == "image"
        assert abs(ranked[0]["score"] - 0.92) < 1e-6
