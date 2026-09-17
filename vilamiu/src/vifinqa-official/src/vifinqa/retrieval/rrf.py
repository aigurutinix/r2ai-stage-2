
from __future__ import annotations

from vifinqa.retrieval.base import IndexedTable, RetrievalHit, RetrievalIndex

DEFAULT_RRF_K = 60


class RRFIndex:
    def __init__(
        self,
        indexes: list[RetrievalIndex],
        *,
        fetch_top_n: int = 500,
        rrf_k: int = DEFAULT_RRF_K,
        min_scores: list[float | None] | None = None,
    ) -> None:
        if not indexes:
            raise ValueError("RRFIndex requires at least one component index")
        if min_scores is not None and len(min_scores) != len(indexes):
            raise ValueError("min_scores must have the same length as indexes")
        self._indexes = indexes
        self._fetch_top_n = fetch_top_n
        self._rrf_k = rrf_k
        self._min_scores = min_scores or [None] * len(indexes)

    @property
    def size(self) -> int:
        return max(index.size for index in self._indexes)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        fetch_n = max(top_k, self._fetch_top_n)
        scores: dict[str, float] = {}
        tables: dict[str, IndexedTable] = {}
        source_counts: dict[str, int] = {}
        best_ranks: dict[str, int] = {}

        for index, min_score in zip(self._indexes, self._min_scores, strict=True):
            for rank, hit in enumerate(index.search(query, top_k=fetch_n), start=1):
                if min_score is not None and hit.score <= min_score:
                    continue
                ref = hit.table.table_ref
                scores[ref] = scores.get(ref, 0.0) + 1.0 / (self._rrf_k + rank)
                tables[ref] = hit.table
                source_counts[ref] = source_counts.get(ref, 0) + 1
                best_ranks[ref] = min(best_ranks.get(ref, rank), rank)

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], -source_counts[item[0]], best_ranks[item[0]], item[0]),
        )[:top_k]
        return [RetrievalHit(table=tables[ref], score=score) for ref, score in ranked]
