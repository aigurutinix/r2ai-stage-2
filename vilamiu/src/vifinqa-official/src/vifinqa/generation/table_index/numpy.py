"""Exact NumPy dot-product implementation of ``TableSearchIndex`` (no FAISS/Qdrant)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vifinqa.embeddings.base import Embedder
from vifinqa.generation.embedding_index import IndexedTable


@dataclass(slots=True)
class NumpyTableSearchIndex:
    _embedder: Embedder
    _entries: list[IndexedTable] = field(default_factory=list)
    _vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[IndexedTable, float]]:
        if not self._entries:
            return []
        query_vec = np.asarray(self._embedder.embed([query], is_query=True)[0], dtype=np.float32)
        scores = np.asarray(self._vectors) @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [(self._entries[i], float(scores[i])) for i in order]
