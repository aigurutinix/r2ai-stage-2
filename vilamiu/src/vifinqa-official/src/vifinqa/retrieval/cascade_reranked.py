"""Fair two-stage reranking: one summary per table, then an equal row-text budget."""

from __future__ import annotations

from dataclasses import replace
import unicodedata

from vifinqa.rerankers.base import Reranker
from vifinqa.retrieval.base import RetrievalHit, RetrievalIndex


class FairCascadeRerankedIndex:
    def __init__(
        self,
        first_stage: RetrievalIndex,
        reranker: Reranker,
        *,
        rerank_top_n: int,
        cascade_top_n: int = 150,
        detail_texts_per_table: int = 3,
        rrf_k: int = 60,
    ) -> None:
        if rerank_top_n <= 0:
            raise ValueError("rerank_top_n must be greater than 0")
        if cascade_top_n <= 0:
            raise ValueError("cascade_top_n must be greater than 0")
        if detail_texts_per_table <= 0:
            raise ValueError("detail_texts_per_table must be greater than 0")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        self._first_stage = first_stage
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n
        self._cascade_top_n = cascade_top_n
        self._detail_texts_per_table = detail_texts_per_table
        self._rrf_k = rrf_k

    @property
    def size(self) -> int:
        return self._first_stage.size

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        query = unicodedata.normalize("NFC", query)
        candidates = self._first_stage.search(
            query, top_k=max(top_k, self._rerank_top_n)
        )
        if not candidates:
            return []

        summary_docs = [hit.table.summary_text or hit.table.text for hit in candidates]
        summary_scores = self._reranker.rerank(query, summary_docs)
        stage1_ids = sorted(
            range(len(candidates)), key=lambda index: (-summary_scores[index], index)
        )
        stage1_rank = {
            candidate_id: rank for rank, candidate_id in enumerate(stage1_ids, start=1)
        }

        detail_ids = stage1_ids[: min(self._cascade_top_n, len(stage1_ids))]
        detail_docs: list[str] = []
        detail_parent_ids: list[int] = []
        for candidate_id in detail_ids:
            table = candidates[candidate_id].table
            docs = list(table.detail_texts or table.rerank_texts or (table.text,))
            docs = docs[: self._detail_texts_per_table]
            docs.extend([docs[-1]] * (self._detail_texts_per_table - len(docs)))
            detail_docs.extend(docs)
            detail_parent_ids.extend([candidate_id] * self._detail_texts_per_table)
        detail_pair_scores = self._reranker.rerank(query, detail_docs)

        best_detail_scores = {
            candidate_id: float("-inf") for candidate_id in detail_ids
        }
        best_detail_docs = {
            candidate_id: candidates[candidate_id].table.text
            for candidate_id in detail_ids
        }
        for candidate_id, doc, score in zip(
            detail_parent_ids, detail_docs, detail_pair_scores, strict=True
        ):
            if score > best_detail_scores[candidate_id]:
                best_detail_scores[candidate_id] = score
                best_detail_docs[candidate_id] = doc
        detail_order = sorted(
            detail_ids,
            key=lambda index: (-best_detail_scores[index], stage1_rank[index]),
        )
        detail_rank = {
            candidate_id: rank
            for rank, candidate_id in enumerate(detail_order, start=1)
        }
        fused_ids = sorted(
            detail_ids,
            key=lambda index: (
                -(
                    1.0 / (self._rrf_k + stage1_rank[index])
                    + 1.0 / (self._rrf_k + detail_rank[index])
                ),
                stage1_rank[index],
            ),
        )
        tail_ids = stage1_ids[len(detail_ids) :]

        result: list[RetrievalHit] = []
        for final_rank, candidate_id in enumerate((*fused_ids, *tail_ids), start=1):
            candidate = candidates[candidate_id]
            if candidate_id in detail_rank:
                text = best_detail_docs[candidate_id]
            else:
                text = summary_docs[candidate_id]
            # Keep downstream balancing on one comparable scale: raw cross-encoder tail scores
            # and fused RRF head scores cannot be mixed safely.
            result.append(
                RetrievalHit(
                    table=replace(candidate.table, text=text),
                    score=1.0 / final_rank,
                )
            )
        return result[:top_k]
