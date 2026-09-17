"""Source-only structural text for financial-table retrieval.

The table catalog historically indexed only values from the first CSV column.
That loses three important signals in OCR financial tables:

* the note/subsection immediately preceding the table;
* merged, multi-level column headers expanded by ``pandas.read_html``;
* an unlabeled subtotal/grand-total row.

This module restores those signals without reading questions, answers, question
IDs, or leaderboard feedback.  Blank rows are called totals only when exact
``Decimal`` additivity proves the relationship in at least one physical column.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable

import pandas as pd


_NUMBER_RE = re.compile(r"^\(?-?\d+(?:[.,]\d+)*\)?%?$")
_DATE_RE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_NOTE_RE = re.compile(r"^\s*(?:\d{1,3}|[IVXLC]{1,5})\s*[.)]\s+\S", re.I)
_SUBSECTION_RE = re.compile(r"^\s*\([a-z]\)\s+\S", re.I)
_STATEMENT_RE = re.compile(
    r"BẢNG CÂN ĐỐI|BÁO CÁO KẾT QUẢ|LƯU CHUYỂN TIỀN|"
    r"KẾT QUẢ (?:HOẠT ĐỘNG|KINH DOANH)",
    re.I,
)
_PAGE_RE = re.compile(r"^\s*=+\s*PAGE", re.I)
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:đơn vị\s*:|mẫu\s+b|tại ngày|cho năm tài chính|"
    r"ngày \d{1,2}/\d{1,2}/\d{4}|\d+\s*$)",
    re.I,
)


def _text(value: object) -> str:
    text = str(value).strip()
    return "" if not text or text.casefold() == "nan" else " ".join(text.split())


def _has_alpha(value: object) -> bool:
    return any(char.isalpha() for char in _text(value))


def is_financial_number(value: object) -> bool:
    """Whether ``value`` is an accounting scalar rather than a period header."""

    text = _text(value).replace(" ", "")
    if not text or _DATE_RE.fullmatch(text) or _YEAR_RE.fullmatch(text):
        return False
    return bool(_NUMBER_RE.fullmatch(text))


def financial_decimal(value: object) -> Decimal | None:
    """Parse a locale-formatted accounting scalar exactly.

    This parser is intentionally small: it is used only to prove equality for
    inferred totals, never to calculate a submitted answer.
    """

    text = _text(value).replace(" ", "")
    if not is_financial_number(text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("%", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        pieces = text.split(",")
        text = "".join(pieces) if len(pieces[-1]) == 3 else ".".join(pieces)
    elif "." in text:
        pieces = text.split(".")
        text = "".join(pieces) if len(pieces) > 2 or len(pieces[-1]) == 3 else text
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return -abs(number) if negative else number


def header_row_indexes(frame: pd.DataFrame) -> list[int]:
    """Return the leading physical rows that form the table header band."""

    indexes: list[int] = []
    for row in range(len(frame)):
        values = [frame.iloc[row, column] for column in range(len(frame.columns))]
        if any(is_financial_number(value) for value in values):
            break
        indexes.append(row)
    return indexes


def header_paths(frame: pd.DataFrame, *, limit: int = 24) -> tuple[str, ...]:
    """Flatten merged header ancestry into stable ``parent > child`` paths."""

    header_rows = header_row_indexes(frame)
    paths: list[str] = []
    seen: set[str] = set()
    for column in range(len(frame.columns)):
        values: list[str] = []
        for row in header_rows:
            value = _text(frame.iloc[row, column])
            if value and value not in values:
                values.append(value)
        column_name = _text(frame.columns[column])
        if (
            column_name
            and not column_name.isdigit()
            and not column_name.casefold().startswith("unnamed:")
            and column_name not in values
        ):
            values.insert(0, column_name)
        if not values:
            continue
        path = " > ".join(values)
        if path not in seen:
            seen.add(path)
            paths.append(path)
        if len(paths) >= limit:
            break
    return tuple(paths)


def _is_label(value: object) -> bool:
    text = _text(value)
    return bool(text and _has_alpha(text) and not is_financial_number(text))


def label_column(frame: pd.DataFrame) -> int | None:
    """Choose the physical column most likely to contain row labels."""

    header_rows = set(header_row_indexes(frame))
    best: tuple[tuple[int, int, float], int] | None = None
    for column in range(len(frame.columns)):
        labels = [
            _text(frame.iloc[row, column])
            for row in range(len(frame))
            if row not in header_rows and _is_label(frame.iloc[row, column])
        ]
        if not labels:
            continue
        score = (
            len(labels),
            len({value.casefold() for value in labels}),
            min(sum(min(len(value), 120) for value in labels) / len(labels), 40.0),
        )
        if best is None or score > best[0]:
            best = (score, column)
    return None if best is None else best[1]


def row_labels(frame: pd.DataFrame, *, limit: int = 40) -> tuple[str, ...]:
    """Return deduplicated labels from the best physical label column."""

    column = label_column(frame)
    if column is None:
        return ()
    header_rows = set(header_row_indexes(frame))
    labels: list[str] = []
    seen: set[str] = set()
    for row in range(len(frame)):
        value = _text(frame.iloc[row, column])
        key = value.casefold()
        if row in header_rows or not _is_label(value) or key in seen:
            continue
        seen.add(key)
        labels.append(value)
        if len(labels) >= limit:
            break
    return tuple(labels)


def _header_path_for_column(frame: pd.DataFrame, column: int) -> str:
    header_rows = header_row_indexes(frame)
    values: list[str] = []
    for row in header_rows:
        value = _text(frame.iloc[row, column])
        if value and value not in values:
            values.append(value)
    return " > ".join(values)


def inferred_blank_totals(
    frame: pd.DataFrame,
    *,
    max_span: int = 12,
    limit: int = 12,
) -> tuple[str, ...]:
    """Infer labels for blank rows only when exact additivity proves a total."""

    if frame.empty:
        return ()
    label_index = label_column(frame)
    if label_index is None:
        # A fully numeric table has no semantic row axis to bind a total to.
        return ()
    first_data = len(header_row_indexes(frame))
    inferred: list[str] = []
    seen: set[str] = set()
    for candidate_row in range(first_data, len(frame)):
        if _text(frame.iloc[candidate_row, label_index]):
            continue
        for column in range(len(frame.columns)):
            if column == label_index:
                continue
            candidate = financial_decimal(frame.iloc[candidate_row, column])
            if candidate is None:
                continue
            operands: list[Decimal] = []
            row = candidate_row - 1
            while row >= first_data and len(operands) < max_span:
                # Require named physical children.  Crossing another blank
                # subtotal could make a coincidental equality look additive
                # while double-counting a nested block.
                if not _text(frame.iloc[row, label_index]):
                    break
                value = financial_decimal(frame.iloc[row, column])
                if value is None:
                    break
                operands.append(value)
                row -= 1
            if len(operands) < 2 or sum(operands, Decimal(0)) != candidate:
                continue
            header = _header_path_for_column(frame, column)
            descriptor = "Tổng cộng" + (f" > {header}" if header else "")
            if descriptor not in seen:
                seen.add(descriptor)
                inferred.append(descriptor)
            if len(inferred) >= limit:
                return tuple(inferred)
    return tuple(inferred)


def section_ancestors(
    lines: Iterable[str],
    table_line: int,
    *,
    lookback: int = 40,
    limit: int = 240,
) -> str:
    """Return the nearest structural note/statement/subsection ancestry.

    A closest ``(b)`` subsection supersedes prose belonging to ``(a)``.  Plain
    prose is used only when no structural heading exists in the window.
    ``table_line`` is one-based, matching ``relevant_tables`` identifiers.
    """

    rows = list(lines)
    start = max(0, table_line - lookback - 1)
    window = rows[start : max(0, table_line - 1)]
    note = ""
    subsection = ""
    statement = ""
    fallback: list[str] = []
    for raw in window:
        value = _text(raw)
        if (
            not value
            or "<" in value
            or ">" in value
            or _PAGE_RE.search(value)
            or _BOILERPLATE_RE.search(value)
        ):
            continue
        if _NOTE_RE.search(value):
            note = value
            subsection = ""
        elif _SUBSECTION_RE.search(value):
            subsection = value
        elif _STATEMENT_RE.search(value):
            statement = value
        if len(value) >= 6 and _has_alpha(value):
            fallback.append(value)

    parts: list[str] = []
    if note:
        parts.append(note)
    elif statement:
        parts.append(statement)
    if subsection and subsection not in parts:
        parts.append(subsection)
    if not parts:
        for value in fallback[-2:]:
            if value not in parts:
                parts.append(value)
    return " | ".join(parts)[:limit]


def table_context(frame: pd.DataFrame) -> dict[str, str]:
    """Return bounded structural fields ready to store in a catalog record."""

    return {
        "header_text": " | ".join(header_paths(frame)),
        "row_label_text": " | ".join(row_labels(frame)),
        "inferred_row_text": " | ".join(inferred_blank_totals(frame)),
    }
