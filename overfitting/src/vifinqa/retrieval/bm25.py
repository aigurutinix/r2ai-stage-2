"""Self-contained BM25 (Okapi) — no external dependency needed."""
from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs_tokens]
        self.doc_len = [sum(c.values()) for c in self.docs]
        self.N = len(self.docs)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
                    for t, n in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for i, c in enumerate(self.docs):
            dl = self.doc_len[i] or 1
            s = 0.0
            for t in query_tokens:
                f = c.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
