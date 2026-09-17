"""Offline length diagnostics; the runtime row-chunk encoder never depends on a tokenizer."""

from __future__ import annotations

import math
from collections.abc import Callable

from vifinqa.encoding.row_chunks import EncodedChunk

_PERCENTILES = (50, 90, 95, 99)


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]


def _distribution(values: list[int]) -> dict[str, int]:
    return {**{f"p{p}": _percentile(values, p) for p in _PERCENTILES}, "max": max(values, default=0)}


def audit_chunk_lengths(
    chunks: list[EncodedChunk],
    *,
    count_tokens: Callable[[list[str]], list[int]],
    max_tokens: int,
    batch_size: int = 256,
) -> dict:
    token_lengths: list[int] = []
    for start in range(0, len(chunks), batch_size):
        token_lengths.extend(count_tokens([chunk.text for chunk in chunks[start : start + batch_size]]))
    char_lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "chars": _distribution(char_lengths),
        "tokens": _distribution(token_lengths),
        "chunks_over_max_tokens": sum(length > max_tokens for length in token_lengths),
        "max_tokens": max_tokens,
    }
