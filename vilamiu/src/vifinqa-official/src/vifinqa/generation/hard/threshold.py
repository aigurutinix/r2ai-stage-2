
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Literal

Comparison = Literal["gt", "gte", "lt", "lte"]


def _passes(value: float, threshold: float, comparison: Comparison) -> bool:
    if comparison == "gt":
        return value > threshold
    if comparison == "gte":
        return value >= threshold
    if comparison == "lt":
        return value < threshold
    return value <= threshold


def _splits_nontrivially(values: Sequence[float], threshold: float, comparison: Comparison) -> bool:
    passed = sum(1 for v in values if _passes(v, threshold, comparison))
    return 0 < passed < len(values)


def _round_to_natural(mid: float, lo: float, hi: float) -> float:
    if mid == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(mid)))
    for exponent in range(magnitude + 1, magnitude - 4, -1):
        for multiplier in (1, 5, 2):
            granularity = multiplier * (10**exponent)
            if granularity <= 0:
                continue
            candidate = round(mid / granularity) * granularity
            if lo < candidate < hi:
                return float(candidate)
    return mid


def generate_threshold(
    values: Sequence[float],
    *,
    rng: random.Random,
    comparison: Comparison = "gt",
    max_attempts: int = 20,
) -> float | None:
    unique_sorted = sorted(set(values))
    if len(unique_sorted) < 2:
        return None

    gap_indices = list(range(len(unique_sorted) - 1))
    rng.shuffle(gap_indices)

    for idx in gap_indices[:max_attempts]:
        lo, hi = unique_sorted[idx], unique_sorted[idx + 1]
        mid = (lo + hi) / 2
        threshold = _round_to_natural(mid, lo, hi)
        if _splits_nontrivially(values, threshold, comparison):
            return threshold

    idx = gap_indices[0]
    lo, hi = unique_sorted[idx], unique_sorted[idx + 1]
    mid = (lo + hi) / 2
    if _splits_nontrivially(values, mid, comparison):
        return mid
    return None
