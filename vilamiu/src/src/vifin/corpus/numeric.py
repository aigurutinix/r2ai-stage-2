"""Numeric-cell rules mirrored from the organisers' evaluation code.

Kept byte-compatible with `vifinqa/common/filtering/table_filters.py` and
`vifinqa/common/numeric/parsing.py` so our table-eligibility filter and answer
comparison agree with the official scorer. Note the regex deliberately rejects
separator-free runs of 4+ digits (a bare `2015` is not a numeric cell).
"""

from __future__ import annotations

import math

import re

# The organisers' repo hard-codes ANSWER_ABS_TOL = 1e-2 (absolute, rel_tol=0),
# but the deployed scorer accepts "không quá 0,02% so với đáp án" — a *relative*
# tolerance. They disagree in both directions: 0.02% of a billion-đồng figure is
# far looser than 0.01 absolute, while for an answer like 0.5 it is far tighter.
# We hold ourselves to the stricter of the two so a run that passes locally
# passes there.
ANSWER_REL_TOL = 2e-4
ANSWER_ABS_TOL = 1e-2

MIN_ROWS = 3
MIN_NUMERIC_CELLS = 6

_NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")


def is_numeric_cell(cell: str) -> bool:
    cell = cell.strip()
    if not cell or cell == "-":
        return False
    return bool(_NUMERIC_RE.match(cell))


def count_numeric_cells(rows: list[list[str]]) -> int:
    return sum(1 for row in rows for cell in row if is_numeric_cell(cell))


def coerce_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    if is_numeric_cell(text):
        cleaned = text
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        if negative:
            cleaned = cleaned[1:-1]
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        cleaned = cleaned.replace(".", "").replace(",", ".")
        try:
            num = float(cleaned)
        except ValueError:
            return None
        return -num if negative else num
    return None


def is_correct(
    expected: object,
    actual: object,
    *,
    rel_tol: float = ANSWER_REL_TOL,
    abs_tol: float = 0.0,
) -> bool:
    """Grade an answer the way the deployed scorer does: 0.02% relative."""

    exp = coerce_number(expected)
    act = coerce_number(actual)
    if exp is None or act is None:
        return False
    return math.isclose(exp, act, rel_tol=rel_tol, abs_tol=abs_tol)


def is_correct_strict(expected: object, actual: object) -> bool:
    """Pass both the published relative rule and the repo's absolute one."""

    return is_correct(expected, actual) and is_correct(
        expected, actual, rel_tol=0.0, abs_tol=ANSWER_ABS_TOL
    )
