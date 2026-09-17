"""Find & parse inline-HTML tables inside OCR'd .txt reports.

Tables are indexed 0-based by order of appearance of the <table> tag in the
file — this order is the "table position" used in relevant_tables
(`report_id|position`, offset by config.TABLE_POS_BASE at submission time).
"""
from __future__ import annotations

import re
from bisect import bisect_right

from bs4 import BeautifulSoup

TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
PAGE_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.M)
TAG_RE = re.compile(r"<[^>]+>")

try:
    BeautifulSoup("<table><tr><td>x</td></tr></table>", "lxml")
    _PARSER = "lxml"
except Exception:  # pragma: no cover
    _PARSER = "html.parser"


def iter_tables(text: str):
    """Yield (pos, page, line_no, html, context_before) per table in order.

    pos     = 0-based order of appearance of <table> in the file (internal key)
    line_no = 1-based line number of the line where <table> starts — this IS
              the official `report_id|position` scheme used in relevant_tables,
              CONFIRMED by the organizers and verified on the leaderboard.
    """
    page_marks = [(m.start(), int(m.group(1))) for m in PAGE_RE.finditer(text)]
    starts = [p[0] for p in page_marks]

    def page_at(off: int) -> int:
        i = bisect_right(starts, off) - 1
        return page_marks[i][1] if i >= 0 else 1

    prev_end = 0
    for pos, m in enumerate(TABLE_RE.finditer(text)):
        ctx_start = max(prev_end, m.start() - 1500)
        ctx = text[ctx_start:m.start()]
        ctx = TAG_RE.sub(" ", ctx)
        ctx = PAGE_RE.sub(" ", ctx)
        ctx = re.sub(r"\s+", " ", ctx).strip()[-400:]
        line_no = text.count("\n", 0, m.start()) + 1
        yield pos, page_at(m.start()), line_no, m.group(0), ctx
        prev_end = m.end()


def parse_grid(html: str, max_rows: int = 600, max_cols: int = 40) -> list[list[str]]:
    """Expand an HTML table (incl. rowspan/colspan) into a rectangular grid of
    strings. Spanned cells replicate their text (good for row-label spans)."""
    soup = BeautifulSoup(html, _PARSER)
    occupied: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(soup.find_all("tr")):
        if r >= max_rows:
            break
        c = 0
        for cell in tr.find_all(["td", "th"]):
            while (r, c) in occupied:
                c += 1
            if c >= max_cols:
                break
            txt = " ".join(cell.get_text(" ", strip=True).split())
            rs = _span(cell.get("rowspan"))
            cs = _span(cell.get("colspan"))
            for dr in range(min(rs, max_rows - r)):
                for dc in range(min(cs, max_cols - c)):
                    occupied[(r + dr, c + dc)] = txt
            c += cs
    if not occupied:
        return []
    n_rows = max(rc[0] for rc in occupied) + 1
    n_cols = max(rc[1] for rc in occupied) + 1
    return [[occupied.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def _span(v) -> int:
    try:
        n = int(str(v))
        return max(1, min(n, 50))
    except (TypeError, ValueError):
        return 1
