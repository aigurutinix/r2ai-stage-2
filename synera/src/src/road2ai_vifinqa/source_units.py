"""Conservative unit inheritance for tables split across source pages."""

from __future__ import annotations

import re

from .corpus import Corpus, TableAsset
from .text import fold_text, source_scale


_DECLARATION = re.compile(
    r"\b(?:don vi(?: tinh)?|dvt)\s*[:\-]?\s*"
    r"((?:(?:nghin ty|tram ty|ty|trieu|nghin|ngan)\s+)?(?:dong|vnd))\b"
)


def declared_scale(text: str) -> float | None:
    """Only explicit unit declarations, not currency mentioned in narrative."""
    matches = list(_DECLARATION.finditer(fold_text(text)))
    if not matches:
        return None
    unit = matches[-1].group(1)
    if unit == "ty vnd":
        return 1e9
    return source_scale(unit)


def continuation_scale(corpus: Corpus, table: TableAsset) -> float | None:
    """Inherit only from an adjacent page with identical repeated headers.

    A new section, a local unit, or different columns prevents inheritance.
    The requested answer's unit is deliberately not an input.
    """
    if declared_scale(table.context) is not None:
        return None
    context = re.sub(r"=+\s*PAGE\s+\d+\s*=+", "", table.context, flags=re.I)
    if re.sub(r"[\s\d./-]+", "", context):
        return None
    if table.table_id <= 1 or len(table.rows) < 2:
        return None
    previous = corpus.table(table.doc_id, table.table_id - 1)
    if table.page not in (previous.page, previous.page + 1):
        return None
    headers = [[fold_text(cell) for cell in row] for row in table.rows[:2]]
    prior_headers = [[fold_text(cell) for cell in row] for row in previous.rows[:2]]
    if headers != prior_headers or len(headers[0]) < 2:
        return None
    text = " ".join(headers[0])
    if not any(marker in text for marker in ("nam", "thang", "dau ky", "cuoi ky")):
        return None
    return declared_scale(previous.context)
