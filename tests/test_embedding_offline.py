"""Tests for offline Hugging Face loading (local_files_only on cached models)."""

from __future__ import annotations

from types import SimpleNamespace

import docfinder.embedding.encoder as encoder
import docfinder.index.reranker as reranker_module


def test_is_hf_offline_true_on_network_error(monkeypatch):
    def _fail(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(encoder.socket, "create_connection", _fail)
    assert encoder.is_hf_offline() is True


def test_is_hf_offline_false_when_reachable(monkeypatch):
    monkeypatch.setattr(
        encoder.socket,
        "create_connection",
        lambda *a, **kw: SimpleNamespace(close=lambda: None),
    )
    assert encoder.is_hf_offline() is False


def _load_kwargs(monkeypatch, offline: bool) -> dict:
    """Build an EmbeddingModel config and capture _load_model() kwargs."""
    captured: dict = {}

    class _FakeST:
        def __init__(self, model_name, **kwargs):
            captured.update(kwargs, model_name=model_name)

        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setattr(encoder, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(encoder, "is_hf_offline", lambda: offline)

    model = encoder.EmbeddingModel.__new__(encoder.EmbeddingModel)
    model.config = encoder.EmbeddingConfig(model_name="test-model", backend="torch", device="cpu")
    model._load_model()
    return captured


def test_load_model_offline_sets_local_files_only(monkeypatch):
    kwargs = _load_kwargs(monkeypatch, offline=True)
    assert kwargs["local_files_only"] is True


def test_load_model_online_allows_remote(monkeypatch):
    kwargs = _load_kwargs(monkeypatch, offline=False)
    assert kwargs["local_files_only"] is False


def test_reranker_offline_sets_local_files_only(monkeypatch):
    import sentence_transformers as st

    captured: dict = {}

    class _FakeCE:
        def __init__(self, model_name, **kwargs):
            captured.update(kwargs, model_name=model_name)

        def predict(self, pairs):
            return [0.0, 1.0][: len(pairs)]

    monkeypatch.setattr(st, "CrossEncoder", _FakeCE)
    monkeypatch.setattr(reranker_module, "is_hf_offline", lambda: True)

    reranker = reranker_module.Reranker()
    results = reranker.rerank("q", [{"text": "a", "score": 0.4}, {"text": "b", "score": 0.9}])
    assert captured["local_files_only"] is True
    assert len(results) == 2
