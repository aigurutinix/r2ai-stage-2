
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from vifinqa.constants import EMBEDDING_MAX_SEQ_LENGTH
from vifinqa.embeddings.base import Embedder


def _safe_name(name: str) -> str:
    return name.replace("/", "__")


class CachedEmbedder:
    def __init__(
        self,
        inner: Embedder,
        *,
        cache_dir: Path,
        model_name: str,
        max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
    ) -> None:
        self._inner = inner
        self._dir = cache_dir / "embeddings" / _safe_name(model_name)
        if max_tokens != EMBEDDING_MAX_SEQ_LENGTH:
            self._dir = self._dir / f"tokens-{max_tokens}"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, text: str, is_query: bool) -> Path:
        key = hashlib.sha256(f"{is_query}|{text}".encode()).hexdigest()
        return self._dir / f"{key}.npy"

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))

        with self._lock:
            results: list[np.ndarray | None] = [None] * len(texts)
            miss_indices = []
            for i, text in enumerate(texts):
                path = self._path(text, is_query)
                if path.exists():
                    results[i] = np.load(path)
                else:
                    miss_indices.append(i)

            if miss_indices:
                vectors = self._inner.embed(
                    [texts[i] for i in miss_indices], is_query=is_query
                )
                for idx, vector in zip(miss_indices, vectors, strict=True):
                    results[idx] = vector
                    np.save(self._path(texts[idx], is_query), vector)

            return np.stack(results)  # type: ignore[arg-type]
