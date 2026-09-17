"""Minimal interface for interchangeable reranking models and backends."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per document in input order; higher is better."""
        ...
