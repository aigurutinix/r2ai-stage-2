"""Weight a token by how rare it is, because Vietnamese line items differ in one word.

Plain Jaccard treats every token alike, and that has now produced the same error three
times in three different planners:

  "vay ngắn hạn"        answered with 310 "Nợ ngắn hạn"      instead of 320
  "vốn cổ phần"         answered with 412 "Thặng dư vốn cổ phần"
  "chi phí nhân công"   answered with 32  "Chi phí khác"

In every case the two labels share their common words — ngắn, hạn, vốn, chi, phí —
and differ in the one that carries the meaning. Those common words appear in most of
the three hundred labels in the dictionary and therefore distinguish nothing, while
`vay` appears in a handful.

Inverse document frequency over the dictionary's own labels fixes the family rather
than the three instances. Nothing else changes: the score is still an overlap of two
token sets, still symmetric, and still refuses on a thin margin.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


class Weighting:
    """Token weights from how many labels in the dictionary contain each token."""

    def __init__(self, label_sets: Iterable[frozenset[str]]) -> None:
        labels = list(label_sets)
        self.total = max(1, len(labels))
        frequency: Counter[str] = Counter()
        for tokens in labels:
            for token in tokens:
                frequency[token] += 1
        self.frequency = frequency

    def weight(self, token: str) -> float:
        # Smoothed IDF: a token in every label scores near zero, one in a single
        # label scores highest, and a token the dictionary has never seen is treated
        # as rare rather than as free.
        return math.log((self.total + 1) / (self.frequency.get(token, 0) + 1)) + 1.0

    def mass(self, tokens: Iterable[str]) -> float:
        return sum(self.weight(t) for t in tokens)

    def overlap(self, left: frozenset[str], right: frozenset[str]) -> float:
        """Weighted Jaccard: shared mass over union mass."""

        union = self.mass(left | right)
        if not union:
            return 0.0
        return self.mass(left & right) / union

    def coverage(self, probe: frozenset[str], label: frozenset[str]) -> float:
        """Share of the PROBE's mass the label accounts for."""

        total = self.mass(probe)
        if not total:
            return 0.0
        return self.mass(probe & label) / total
