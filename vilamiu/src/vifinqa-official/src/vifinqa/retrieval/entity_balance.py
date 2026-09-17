"""Post-rerank balancing for questions that explicitly mention multiple ticker symbols."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable

from vifinqa.retrieval.base import RetrievalHit, RetrievalIndex


def explicit_query_tickers(query: str, known_tickers: Iterable[str]) -> tuple[str, ...]:
    upper = query.upper()
    found: list[tuple[int, str]] = []
    for ticker in known_tickers:
        match = re.search(
            rf"(?<![A-Z0-9]){re.escape(ticker.upper())}(?![A-Z0-9])", upper
        )
        if match is not None:
            found.append((match.start(), ticker.upper()))
    return tuple(
        ticker for _, ticker in sorted(found, key=lambda item: (item[0], item[1]))
    )


def balance_explicit_tickers(
    hits: list[RetrievalHit],
    explicit_tickers: tuple[str, ...],
) -> list[RetrievalHit]:
    """Round-robin explicit ticker queues; order each round by current head score."""
    if len(explicit_tickers) < 2:
        return list(hits)
    order = {ticker: index for index, ticker in enumerate(explicit_tickers)}
    queues = {
        ticker: deque(hit for hit in hits if hit.table.ticker == ticker)
        for ticker in explicit_tickers
    }
    balanced: list[RetrievalHit] = []
    while any(queues.values()):
        active = [ticker for ticker, queue in queues.items() if queue]
        active.sort(key=lambda ticker: (-queues[ticker][0].score, order[ticker]))
        balanced.extend(queues[ticker].popleft() for ticker in active)
    explicit_set = set(explicit_tickers)
    balanced.extend(hit for hit in hits if hit.table.ticker not in explicit_set)
    return balanced


class MultiEntityBalancedIndex:
    def __init__(self, inner: RetrievalIndex, *, known_tickers: Iterable[str]) -> None:
        self._inner = inner
        self._known_tickers = tuple(known_tickers)

    @property
    def size(self) -> int:
        return self._inner.size

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        hits = self._inner.search(query, top_k=top_k)
        tickers = explicit_query_tickers(query, self._known_tickers)
        return balance_explicit_tickers(hits, tickers)[:top_k]
