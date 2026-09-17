"""Minimal interfaces for finding tables by ticker, report scope, and period.

Implementations are replaceable: generation code depends only on ``TableSearchIndex``
and ``TableIndexStore``. Concrete NumPy wiring belongs in the CLI or configuration layer.
"""

from __future__ import annotations

from typing import Protocol

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.generation.embedding_index import IndexedTable


class TableSearchIndex(Protocol):
    def search(self, query: str, *, top_k: int = 10) -> list[tuple[IndexedTable, float]]: ...


class TableIndexStore(Protocol):
    def get(
        self,
        *,
        ticker: str,
        report_scope: str,
        period: str,
        docs: list[DocumentRef],
    ) -> TableSearchIndex: ...
