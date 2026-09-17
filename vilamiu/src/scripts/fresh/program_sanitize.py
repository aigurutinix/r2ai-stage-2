"""Sanitize broken generated programs (duplicate num defs, fragile parser)."""

from __future__ import annotations

import re

from num_helper import SOURCE


_NUM_BLOCK = re.compile(
    r"^def num\(frame, r, c\):.*?^return float\(text\)\s*\n",
    re.MULTILINE | re.DOTALL,
)


def strip_duplicate_num(query: str) -> str:
    """Keep one Vietnamese number parser; rewrite num(df,r,c) reads."""

    if "def num(frame, r, c)" not in query and "def num(" not in query:
        return query
    cleaned = _NUM_BLOCK.sub("", query)
    cleaned = re.sub(
        r"\bnum\((\w+),\s*(\d+),\s*(\d+)\)",
        r"_num(\1.iloc[\2, \3])",
        cleaned,
    )
    if "_num(" not in cleaned and "def _num" not in cleaned:
        cleaned = re.sub(
            r"\bnum\((\w+),\s*(\d+),\s*(\d+)\)",
            r"_num(\1.iloc[\2, \3])",
            _NUM_BLOCK.sub("", query),
        )
    if "def _num" in cleaned:
        return cleaned
    return SOURCE + cleaned


def sanitize_query(query: str) -> str:
    """Return query with standard _num helper and iloc reads."""

    if "def num(" in query or "def num (frame" in query:
        return strip_duplicate_num(query)
    return query
