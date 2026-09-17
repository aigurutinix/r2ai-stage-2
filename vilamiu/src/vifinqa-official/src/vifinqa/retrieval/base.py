"""Minimal interface for interchangeable dense, BM25, and reranked table indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vifinqa.common.schemas.table_ref import make_table_ref
from vifinqa.common.schemas.table_ref import parse_table_ref as parse_table_ref


@dataclass(frozen=True, slots=True)
class IndexedTable:
    ticker: str
    year: str
    doc_name: str
    table_id: int
    text: str  # Encoded table representation; see encoding/table_text.py.
    rerank_texts: tuple[str, ...] = ()  # Multi-view/chunk texts; empty means use ``text``.
    summary_text: str | None = (
        None  # Global representation used by stage one of the fair cascade.
    )
    detail_texts: tuple[str, ...] = ()  # Row evidence ordered by dense relevance.

    @property
    def table_ref(self) -> str:
        return make_table_ref(self.doc_name, self.table_id)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    table: IndexedTable
    score: float


class RetrievalIndex(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]: ...

    @property
    def size(self) -> int: ...
