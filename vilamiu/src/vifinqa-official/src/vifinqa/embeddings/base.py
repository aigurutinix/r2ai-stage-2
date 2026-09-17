"""Minimal interface for interchangeable embedding models and backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Return normalized vectors for the supplied texts.

        ``is_query=True`` selects query-specific instructions for models that support them.
        """
        ...
