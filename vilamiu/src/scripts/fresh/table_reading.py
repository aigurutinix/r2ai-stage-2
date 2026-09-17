"""Read a row's label, its money columns and its scale from what the table prints.

Three defects found by reading eight disagreements by hand, each with a concrete cause:

  the label     was taken from `row[0]`, which in a split-label table is "I." while the
                indicator sits in the next column. Question 53 matched "I." and answered
                a question about investment property with the cash line.
  the column    was the first figure left to right. In question 12 that was the share
                COUNT beside the VND amount, so "vốn cổ phần đã phát hành" came back as
                500,000,000 shares divided by a trillion — zero.
  the scale     fell through to a document-wide guess whenever the table's own header was
                not phrased as "Đơn vị tính". Headers like `2018Triệu VND` and
                `31/12/2025VND` say the unit outright, and questions 17, 32 and 54 were
                each wrong by a factor of a million because that was not read.

All three are read off the table itself rather than guessed, and the scale a table states
always beats a document-level scan — the table is the more specific statement.
"""

from __future__ import annotations

import re
import unicodedata

# "Triệu VND", "2018Triệu VND", "Đơn vị: tỷ đồng", "31/12/2025VND".
_MILLION = re.compile(r"trieu")
_BILLION = re.compile(r"\bty\b")
_THOUSAND = re.compile(r"nghin|ngan")
_MONEY = re.compile(r"vnd|dong")
# A column that is plainly not money, said so by its own header.
_NOT_MONEY = re.compile(r"co phieu|so luong|usd|eur|jpy|nguyen te|%|ty le|ty trong|"
                        r"lai suat|thoi han|ma so|thuyet minh")
# A header that names a period is naming a value column, whether or not the currency word
# happened to land in that cell.
_PERIOD = re.compile(r"\b20[0-2]\d\b|\d{1,2}/\d{1,2}/20[0-2]\d|"
                     r"nam nay|nam truoc|nam hien tai|"
                     r"so cuoi (?:nam|ky)|so dau (?:nam|ky)|"
                     r"cuoi (?:nam|ky)|dau (?:nam|ky)|ky nay|ky truoc")
_NUMERIC = re.compile(r"^[\d.,()%\-\s]+$")


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", flat).strip()


def row_label(row: list[str], skip: set[int]) -> str:
    """The longest cell that reads as text, which is where the indicator name lives."""

    best = ""
    for index, cell in enumerate(row):
        if index in skip:
            continue
        text = str(cell).strip()
        if not text or _NUMERIC.match(text):
            continue
        if len(text) > len(best):
            best = text
    return best


def scale_from_headers(grid: list[list[str]]) -> float | None:
    """The unit a table states in its own column headers, if it states one.

    Checked before any document-level scan: a header that says `Triệu VND` is a
    statement about this table, while a document scan is a statement about the report.
    """

    if not grid:
        return None
    for row in grid[:2]:
        for cell in row:
            flat = fold(cell)
            if not _MONEY.search(flat):
                continue
            if _MILLION.search(flat):
                return 1e6
            if _BILLION.search(flat):
                return 1e9
            if _THOUSAND.search(flat):
                return 1e3
            # Names the currency and no multiplier: the figures are in đồng.
            return 1.0
    return None


def money_columns(grid: list[list[str]]) -> set[int] | None:
    """Columns whose header marks them as money, when any column does.

    Returns None when no header settles it, in which case the caller should fall back to
    position. The point is question 12: a row carrying both a share count and a VND
    amount has to resolve to the VND one, and the header is what says which is which.
    """

    if not grid:
        return None
    width = max((len(row) for row in grid), default=0)
    money: set[int] = set()
    excluded: set[int] = set()
    for column in range(width):
        header = " ".join(fold(row[column]) for row in grid[:2]
                          if column < len(row))
        if _NOT_MONEY.search(header):
            excluded.add(column)
        elif _MONEY.search(header):
            money.add(column)
    if money:
        # Sibling columns headed by a period are value columns too: the currency word is
        # a statement about the table, not about the one cell OCR glued it to.
        for column in range(width):
            if column in money or column in excluded:
                continue
            header = " ".join(fold(row[column]) for row in grid[:2]
                              if column < len(row))
            if _PERIOD.search(header):
                money.add(column)
        return money
    if excluded and width > len(excluded):
        return {c for c in range(width) if c not in excluded}
    return None


def section_label(grid: list[list[str]], row_index: int, skip: set[int]) -> str:
    """The row's own label with the nearest section heading above it prefixed.

    A row label inside a note is often meaningless alone. Question 54 asked about loans
    and was answered from a row called `Tổ chức kinh tế`, which sits under the heading
    `Tiền gửi của khách hàng theo đối tượng` — customer deposits. Read alone the label
    matches almost any question about organisations; read with its heading it matches
    only questions about deposits.

    A heading is a row that carries text and no figures, which is how OCR leaves the
    section titles that a statement prints across the width of the table.
    """

    own = row_label(grid[row_index + 1], skip) if row_index + 1 < len(grid) else ""
    heading = ""
    for index in range(row_index, 0, -1):
        row = grid[index] if index < len(grid) else []
        if not row:
            continue
        texts = [str(c).strip() for c in row if str(c).strip()]
        if not texts:
            continue
        if any(_NUMERIC.match(t) for t in texts):
            continue          # carries figures, so it is a data row, not a heading
        candidate = max(texts, key=len)
        if len(candidate) >= 8 and candidate != own:
            heading = candidate
            break
    return f"{heading} {own}".strip() if heading else own
