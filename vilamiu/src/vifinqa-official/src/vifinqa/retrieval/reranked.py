
from __future__ import annotations

from dataclasses import replace
import unicodedata

from vifinqa.rerankers.base import Reranker
from vifinqa.retrieval.base import RetrievalHit, RetrievalIndex


class RerankedIndex:
    def __init__(
        self,
        first_stage: RetrievalIndex,
        reranker: Reranker,
        *,
        rerank_top_n: int,
        multi_text_top_n: int = 0,
    ) -> None:
        if multi_text_top_n < 0:
            raise ValueError("multi_text_top_n must be non-negative")
        self._first_stage = first_stage
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n
        self._multi_text_top_n = multi_text_top_n

    @property
    def size(self) -> int:
        return self._first_stage.size

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        query = unicodedata.normalize("NFC", query)
        candidate_n = max(top_k, self._rerank_top_n)
        candidates = self._first_stage.search(query, top_k=candidate_n)
        if not candidates:
            return []
        pair_docs: list[str] = []
        pair_parent_ids: list[int] = []
        for parent_id, candidate in enumerate(candidates):
            docs = candidate.table.rerank_texts or (candidate.table.text,)
            if parent_id >= self._multi_text_top_n:
                docs = docs[:1]
            pair_docs.extend(docs)
            pair_parent_ids.extend([parent_id] * len(docs))
        pair_scores = self._reranker.rerank(query, pair_docs)

        best_scores = [float("-inf")] * len(candidates)
        best_docs = [candidate.table.text for candidate in candidates]
        for parent_id, doc, score in zip(pair_parent_ids, pair_docs, pair_scores, strict=True):
            if score > best_scores[parent_id]:
                best_scores[parent_id] = score
                best_docs[parent_id] = doc

        reranked = sorted(
            (
                RetrievalHit(table=replace(candidate.table, text=best_docs[i]), score=best_scores[i])
                for i, candidate in enumerate(candidates)
            ),
            key=lambda hit: -hit.score,
        )
        return reranked[:top_k]
