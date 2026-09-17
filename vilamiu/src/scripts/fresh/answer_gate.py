"""Reject answers the question itself rules out.

A program's result can be checked against the question without knowing the gold, because
the question constrains the answer's type and range:

  which year was highest      the answer has to be one of the years the question lists,
                              and the question lists them — three to five candidates. A
                              program that returns a figure in the billions here has
                              misread the task, not the table.
  how many companies          an integer, and never more than the companies named.
  a percentage                a share or a margin lives in a band; a figure in the
                              millions is a money cell that escaped its conversion.
  a money figure              bounded by the unit the question asks for, generously —
                              a company can be large, but not 10^15 tỷ đồng.

These gates cost nothing and they catch the failure that has been most expensive here:
an answer of the wrong KIND, produced with complete confidence. They are deliberately
loose everywhere except the year check, which is exact because the question enumerates
the answer space.

The gate never repairs a value. A rejected answer is dropped so the incumbent stands.
"""

from __future__ import annotations

import re

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
WHICH_YEAR_RE = re.compile(r"năm nào", re.I)
COUNT_RE = re.compile(
    r"bao nhiêu (công ty|đơn vị|doanh nghiệp|khoản mục|thành viên|chi nhánh|"
    r"cổ đông|người|nhân viên|lao động)", re.I)
TIMES_RE = re.compile(r"bao nhiêu lần", re.I)
PCT_RE = re.compile(r"phần trăm|%|tỷ lệ|tỉ lệ", re.I)
TICKER_RE = re.compile(r"\b[A-Z]{3,4}\b")

# The largest a Vietnamese listed company's figure plausibly reaches, per unit. VND
# totals run to the hundreds of trillions, so the đồng bound is the loose one.
UNIT_CEILINGS = (
    ("nghìn tỷ đồng", 1e4),
    ("trăm tỷ đồng", 1e5),
    ("nghìn đồng", 1e15),
    ("triệu đồng", 1e12),
    ("tỷ đồng", 1e9),
    ("đồng", 1e18),
)


def verdict(question: str, value: float) -> str | None:
    """The reason to reject, or None to accept."""

    if WHICH_YEAR_RE.search(question):
        years = {int(y) for y in YEAR_RE.findall(question)}
        if not years:
            return None
        if value != int(value) or int(value) not in years:
            return "cau hoi 'nam nao' nhung dap an khong phai nam duoc liet ke"
        return None

    if COUNT_RE.search(question):
        if value < 0 or value != int(value):
            return "cau hoi dem nhung dap an khong phai so nguyen khong am"
        named = len(set(TICKER_RE.findall(question)))
        if named and value > named:
            return "cau hoi dem nhung dap an lon hon so cong ty duoc neu"
        return None

    if TIMES_RE.search(question):
        return None if abs(value) <= 500 else "he so 'lan' qua lon"

    if PCT_RE.search(question) and not any(
            unit in question.lower() for unit, _ in UNIT_CEILINGS[:5]):
        return None if abs(value) <= 5000 else "ty le phan tram qua lon"

    lowered = question.lower()
    for unit, ceiling in UNIT_CEILINGS:
        if unit in lowered:
            return None if abs(value) <= ceiling else f"gia tri vuot tran cua '{unit}'"
    return None
