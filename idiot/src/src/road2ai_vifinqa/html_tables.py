"""Expansion of OCR HTML tables, including merged cells."""

from __future__ import annotations

from lxml import html as lxml_html

from .text import clean_text


def _positive_int(value: str | None, default: int = 1) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def parse_html_table(fragment: str) -> list[list[str]]:
    """Return a rectangular grid and repeat rowspan/colspan values."""

    root = lxml_html.fragment_fromstring(fragment, create_parent=True)
    tr_nodes = root.xpath(".//tr")
    pending: dict[int, tuple[str, int]] = {}
    sparse_rows: list[dict[int, str]] = []
    width = 0

    for tr in tr_nodes:
        current: dict[int, str] = {}
        next_pending: dict[int, tuple[str, int]] = {}
        for col, (text, remaining) in pending.items():
            current[col] = text
            if remaining > 1:
                next_pending[col] = (text, remaining - 1)

        col = 0
        for cell in tr.xpath("./th|./td"):
            while col in current:
                col += 1
            text = clean_text(" ".join(cell.itertext()))
            colspan = _positive_int(cell.get("colspan"))
            rowspan = _positive_int(cell.get("rowspan"))
            while any((col + offset) in current for offset in range(colspan)):
                col += 1
            for offset in range(colspan):
                target = col + offset
                current[target] = text
                if rowspan > 1:
                    next_pending[target] = (text, rowspan - 1)
            col += colspan

        pending = next_pending
        if current:
            width = max(width, max(current) + 1)
            sparse_rows.append(current)

    while pending:
        current: dict[int, str] = {}
        next_pending = {}
        for col, (text, remaining) in pending.items():
            current[col] = text
            if remaining > 1:
                next_pending[col] = (text, remaining - 1)
        width = max(width, max(current) + 1)
        sparse_rows.append(current)
        pending = next_pending

    return [[row.get(col, "") for col in range(width)] for row in sparse_rows]

