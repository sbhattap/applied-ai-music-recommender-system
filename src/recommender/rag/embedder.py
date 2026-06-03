"""Text embedders + a content-addressed vector cache.

Default is a local ONNX model via fastembed (bge-small, 384-d): no API key, ~7 ms per
query on CPU. Vectors are cached on disk by text hash, so the catalog is embedded once
and any subset of songs (e.g. after a structured filter) is a dictionary lookup.
"""

import hashlib
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .. import config


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class FastEmbedEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        self.name = model_name
        self.dim = 384
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(
                    self.name, cache_dir=str(config.CACHE_DIR / "fastembed")
                )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.array(list(self._load().embed(list(texts))), dtype=np.float32)
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self._load().query_embed([text]))), dtype=np.float32)
        return _normalize(vec)


class HashingEmbedder:
    """Deterministic bag-of-words hashing. For tests and offline smoke runs only."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text))


class VectorCache:
    """text-hash -> vector, persisted per embedder in one .npz file."""

    def __init__(self, embedder: Embedder, cache_dir: Optional[Path] = None) -> None:
        self.embedder = embedder
        self._path = (cache_dir or config.RAG_INDEX_DIR) / f"{re.sub(r'[^A-Za-z0-9._-]', '_', embedder.name)}.npz"
        self._vectors: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._loaded = False

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._path.exists():
            with np.load(self._path) as data:
                self._vectors = {k: data[k] for k in data.files}

    def get_many(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            self._load()
            keys = [self.key(t) for t in texts]
            missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._vectors]
            if missing:
                vecs = self.embedder.embed_documents([t for _, t in missing])
                for (k, _), v in zip(missing, vecs):
                    self._vectors[k] = v
                self._save()
            if not keys:
                return np.zeros((0, self.embedder.dim), dtype=np.float32)
            return np.stack([self._vectors[k] for k in keys])

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp.npz")
            np.savez(tmp, **self._vectors)
            tmp.replace(self._path)
        except OSError:
            pass  # cache is an optimization; never fail retrieval over it


_embedder: Optional[Embedder] = None
_cache: Optional[VectorCache] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = FastEmbedEmbedder()
    return _embedder


def get_vector_cache() -> VectorCache:
    global _cache
    if _cache is None or _cache.embedder is not get_embedder():
        _cache = VectorCache(get_embedder())
    return _cache


def set_embedder(embedder: Embedder, cache_dir: Optional[Path] = None) -> None:
    """Swap the process-wide embedder (tests, ablations)."""
    global _embedder, _cache
    _embedder = embedder
    _cache = VectorCache(embedder, cache_dir)
