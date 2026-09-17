
from __future__ import annotations


class NoOpReranker:
    def rerank(self, query: str, docs: list[str]) -> list[float]:
        n = len(docs)
        return [float(n - i) for i in range(n)]
