"""Parse the inline HTML tables embedded in the OCR corpus.

The released `.txt` reports encode every table as a single-line
`<table><tr><td>...</td></tr></table>` blob. Roughly 39% carry `colspan` and
28% carry `rowspan`, so a naive tag split produces ragged rows that silently
misalign financial columns. This module expands spans into a dense grid.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", unescape(text)).strip()


class _TableParser(HTMLParser):
    """Collect `(text, colspan, rowspan)` triples per row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._row: list[tuple[str, int, int]] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        for key, value in attrs:
            if key == name and value:
                try:
                    # OCR noise occasionally yields "2." or "02"; clamp to a sane range.
                    return max(1, min(int(re.sub(r"\D", "", value) or 1), 64))
                except ValueError:
                    return 1
        return 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._close_cell()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = []
            self._cell = []
            self._colspan = self._span(attrs, "colspan")
            self._rowspan = self._span(attrs, "rowspan")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_cell()
            self._close_row()
        elif tag == "table":
            self._close_cell()
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _close_cell(self) -> None:
        if self._cell is None:
            return
        if self._row is None:
            self._row = []
        self._row.append((_clean("".join(self._cell)), self._colspan, self._rowspan))
        self._cell = None
        self._colspan = 1
        self._rowspan = 1

    def _close_row(self) -> None:
        if self._row:
            self.rows.append(self._row)
        self._row = None


def parse_html_table(html: str) -> list[list[str]]:
    """Expand one `<table>` blob into a rectangular grid of strings.

    Spanned cells repeat their text across every position they cover, which keeps
    a value addressable no matter which merged column a query lands on.
    """

    parser = _TableParser()
    parser.feed(html)
    parser.close()

    grid: list[list[str]] = []
    # Column index -> (remaining rows, text) carried down by an open rowspan.
    pending: dict[int, tuple[int, str]] = {}

    for raw_row in parser.rows:
        row: list[str] = []
        col = 0
        cursor = 0
        while cursor < len(raw_row) or col in pending:
            if col in pending:
                remaining, text = pending[col]
                row.append(text)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col] = (remaining - 1, text)
                col += 1
                continue
            text, colspan, rowspan = raw_row[cursor]
            cursor += 1
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col] = (rowspan - 1, text)
                col += 1
        grid.append(row)

    # Rows left holding only carried-down rowspans still occupy real grid rows.
    while pending:
        row: list[str] = []
        for col in range(max(pending) + 1):
            if col in pending:
                remaining, text = pending[col]
                row.append(text)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col] = (remaining - 1, text)
            else:
                row.append("")
        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]
