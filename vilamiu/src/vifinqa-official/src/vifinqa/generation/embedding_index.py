"""Search index for finding relevant tables in Medium scenarios.

Two implementations behind one interface. `TableIndex` embeds every entry in its
constructor; `LexicalTableIndex` scores BM25 over the same text.

Which one runs matters more than it looks. `build_pool_index` is called **once per
seed** with up to `POOL_TABLE_CAP` (300) tables, so the dense path re-embeds a few
hundred tables for every question attempted. Measured on a rented 24 GB box with
the embedder on CPU: the medium tier produced **2 records in 3 hours 7 minutes**,
which extrapolates to roughly 12 days for the 500 records wanted, and the
`.npy` cache grew 1,402 -> 45,286 files in those three hours because the seed
changes each round so almost nothing is reused. Most of that work was thrown away
on seeds later rejected as infeasible.

The index is only used to pick `top_k` tables related to a seed, and this project's
own retrieval measurements on this corpus say lexical beats dense at that job at
every k. So lexical is the default and dense is opt-in via
`VIFINQA_POOL_INDEX=dense`, which keeps the original behaviour one variable away.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vifinqa.embeddings.base import Embedder


@dataclass(frozen=True, slots=True)
class IndexedTable:
    ticker: str
    year: str
    doc_name: str
    table_id: int
    text: str  # Text representation: table title plus column and row labels.


class TableIndex:
    def __init__(self, embedder: Embedder, entries: list[IndexedTable]) -> None:
        self._embedder = embedder
        self._entries = entries
        self._vectors = embedder.embed([e.text for e in entries], is_query=False) if entries else np.zeros((0, 0))

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_fn: Callable[[IndexedTable], bool] | None = None,
    ) -> list[tuple[IndexedTable, float]]:
        candidate_idx = [i for i, e in enumerate(self._entries) if filter_fn is None or filter_fn(e)]
        if not candidate_idx:
            return []
        query_vec = self._embedder.embed([query], is_query=True)[0]
        sims = self._vectors[candidate_idx] @ query_vec
        order = np.argsort(-sims)[:top_k]
        return [(self._entries[candidate_idx[i]], float(sims[i])) for i in order]


# Statement labels are short and repetitive, so tone-folding is not optional: the
# same line is "Lợi nhuận thuần" in one report and "Loi nhuan thuan" after OCR
# loses diacritics, and an exact-token index scores those as unrelated. Digits are
# kept as tokens of their own because the year is often the only thing separating
# two otherwise identical tables.
_WORD_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

_K1 = 1.5
_B = 0.75


def _fold(text: str) -> list[str]:
    stripped = "".join(
        ch
        for ch in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(ch) != "Mn"
    )
    return _WORD_RE.findall(stripped)


class LexicalTableIndex:
    """BM25 over the same `IndexedTable.text` the dense index embeds.

    Written without a BM25 dependency on purpose. `vifinqa.retrieval.bm25` needs
    `bm25s`, and it also builds one global index over a fixed corpus, while this is
    a throwaway index over a few hundred tables rebuilt per seed. Constructing it
    has to be cheap, and counting tokens is cheap in a way that loading a model is
    not.
    """

    def __init__(self, entries: list[IndexedTable]) -> None:
        self._entries = entries
        self._docs: list[Counter[str]] = [Counter(_fold(e.text)) for e in entries]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        frequency: Counter[str] = Counter()
        for doc in self._docs:
            frequency.update(doc.keys())
        total = len(self._docs)
        self._idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_fn: Callable[[IndexedTable], bool] | None = None,
    ) -> list[tuple[IndexedTable, float]]:
        candidate_idx = [
            i for i, e in enumerate(self._entries) if filter_fn is None or filter_fn(e)
        ]
        if not candidate_idx:
            return []
        terms = _fold(query)
        scored: list[tuple[float, int]] = []
        for i in candidate_idx:
            doc = self._docs[i]
            length = self._lengths[i]
            score = 0.0
            for term in terms:
                count = doc.get(term)
                if not count:
                    continue
                norm = count + _K1 * (
                    1.0 - _B + _B * (length / self._avg_len if self._avg_len else 1.0)
                )
                score += self._idf.get(term, 0.0) * count * (_K1 + 1.0) / norm
            scored.append((score, i))
        # Index order breaks ties, so a rebuilt index ranks identically: the caller
        # seeds its own RNG and expects the same pool to behave the same way.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(self._entries[i], float(score)) for score, i in scored[:top_k]]


def make_table_index(
    embedder: Embedder, entries: list[IndexedTable]
) -> TableIndex | LexicalTableIndex:
    """The pool index the environment asks for; lexical unless told otherwise."""

    if os.environ.get("VIFINQA_POOL_INDEX", "lexical").strip().casefold() == "dense":
        return TableIndex(embedder, entries)
    return LexicalTableIndex(entries)
