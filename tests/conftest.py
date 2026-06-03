"""Shared fixtures: every test runs retrieval on a deterministic offline embedder."""

import pytest

from src.recommender.rag import embedder, retriever


@pytest.fixture(autouse=True)
def offline_embedder(tmp_path):
    previous = (embedder._embedder, embedder._cache)
    embedder.set_embedder(embedder.HashingEmbedder(), cache_dir=tmp_path)
    retriever._retrievers.clear()
    yield
    embedder._embedder, embedder._cache = previous
    retriever._retrievers.clear()
