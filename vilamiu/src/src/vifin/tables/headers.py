"""Resolve relative period headers into absolute dates before a model sees them.

The isolated probe hands the model the one table that holds the answer and asks
for a cell. It gets the row right and the column wrong in 26.4% of cases — the
single largest failure class in the whole system, larger than every wrong row put
together. The prompt already spells the conversion out ("Số đầu năm của báo cáo
{year} là cuối năm {year-1}") and the model still misses it, so telling it again
is not the fix.

The conversion needs no reasoning at all. A statement's period columns are
relative to the report it sits in, and the report year is in the file name. So we
do it in code and hand the model absolute dates it cannot misread:

    Số cuối năm   ->  Số cuối năm [31/12/2021]
    Số đầu năm    ->  Số đầu năm [31/12/2020]
    Năm nay       ->  Năm nay [năm 2021]
    Năm trước     ->  Năm trước [năm 2020]

The original text is kept and the resolution appended, for two reasons: the unit
is often glued onto the header by the extractor ("Số cuối nămTriệu đồng"), and
throwing the raw text away would throw the unit away with it; and a header that
already carries an explicit date is left untouched, so a table mixing both styles
stays internally consistent.
"""

from __future__ import annotations

import re
import unicodedata

# An explicit year or date in the header means the extractor already resolved the
# period, and rewriting it would be inventing a second opinion.
ABSOLUTE_RE = re.compile(r"(19|20)\d{2}")

# Matched against NFC-normalised, case-folded text with the diacritics intact —
# the store keeps them, and an ASCII pattern silently matches nothing. That trap
# has cost this project four separate debugging sessions.
END_OF_YEAR = ("số cuối năm", "số cuối kỳ", "cuối năm", "cuối kỳ")
START_OF_YEAR = ("số đầu năm", "số đầu kỳ", "đầu năm", "đầu kỳ")
THIS_PERIOD = ("năm nay", "kỳ này", "năm hiện tại")
PRIOR_PERIOD = ("năm trước", "kỳ trước", "năm ngoái")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).casefold()


def resolve_header(cell: str, report_year: int) -> str:
    """One header cell, with its period spelled out. Unchanged if not relative."""

    if not report_year:
        return str(cell)
    raw = str(cell)
    folded = _fold(raw)
    if not folded.strip() or ABSOLUTE_RE.search(folded):
        return raw

    # Order matters: "số đầu năm" contains "đầu năm", and both are in the same
    # bucket, but "cuối năm" must not be tested before "đầu năm" on a header that
    # happens to carry both words.
    if any(word in folded for word in START_OF_YEAR):
        return f"{raw} [31/12/{report_year - 1}]"
    if any(word in folded for word in END_OF_YEAR):
        return f"{raw} [31/12/{report_year}]"
    if any(word in folded for word in PRIOR_PERIOD):
        return f"{raw} [năm {report_year - 1}]"
    if any(word in folded for word in THIS_PERIOD):
        return f"{raw} [năm {report_year}]"
    return raw


def resolve_grid(grid, report_year: int):
    """A copy of `grid` whose header row carries absolute periods.

    Only row 0 is touched. Period words do appear in body labels ("Số dư đầu
    năm" is a legitimate line item in an equity movement table), but there they
    describe the row, not the column, and stamping a date on them would assert a
    period the row does not have.
    """

    if not grid or not report_year:
        return grid
    header = [resolve_header(cell, report_year) for cell in grid[0]]
    return [header] + [list(row) for row in grid[1:]]


DOC_YEAR_RE = re.compile(r"_((?:19|20)\d{2})_")


def year_of_document(doc_name: str) -> int:
    """The report year carried by the file name, or 0 when it has none."""

    match = DOC_YEAR_RE.search(str(doc_name))
    return int(match.group(1)) if match else 0
