
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vifinqa.common.corpus.document import Document
from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.hard.numbers import VnNumberError, parse_vn_number, scale_from_unit_text

StatementKind = Literal["kqkd", "cdkt", "lctt"]

_MA_SO_RE = re.compile(r"^\d{1,3}$")
_BARE_INT_RE = re.compile(r"^\d+$")
_HEADER_MA_SO_HINT = "mãsố"

_KQKD_REQUIRED_CODES = frozenset({"01", "11", "20", "25", "26"})
_MIN_KQKD_MATCHES = 3
_MIN_CDKT_3DIGIT_CODES = 5


@dataclass(frozen=True, slots=True)
class StatementCell:
    ma_so: str
    label: str
    value: float  # Keep unit and scale handling explicit.
    raw: str
    table_ref: str  # "{doc_name}|table_{id}"
    row_idx: int
    col_idx: int
    scale: float


@dataclass(frozen=True, slots=True)
class StatementTable:
    table_ref: str
    csv_path: Path
    kind: StatementKind
    current: dict[str, StatementCell]  # Keep period handling explicit and deterministic.
    prior: dict[str, StatementCell]  # Keep period handling explicit and deterministic.


def _normalize_header_cell(text: str) -> str:
    return "".join(text.split()).casefold()


def _fold_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _looks_like_cash_flow_label(text: str) -> bool:
    folded = _fold_diacritics(text).casefold()
    return "chuyen tien" in folded


def _header_ma_so_index(header: tuple[str, ...]) -> int | None:
    for idx, cell in enumerate(header):
        if _HEADER_MA_SO_HINT in _normalize_header_cell(cell):
            return idx
    return None


def _find_ma_so_index(row: tuple[str, ...], header_hint: int | None, *, scan_width: int = 3) -> int | None:
    candidates: list[int] = []
    if header_hint is not None:
        candidates.append(header_hint)
    for idx in range(min(scan_width, len(row))):
        if idx not in candidates:
            candidates.append(idx)
    for idx in candidates:
        if idx < len(row) and _MA_SO_RE.match(row[idx].strip()):
            return idx
    return None


def _is_bare_int(raw: str) -> bool:
    return bool(_BARE_INT_RE.match(raw.strip()))


def _row_value_cells(row: tuple[str, ...], *, exclude_idx: int) -> list[tuple[int, float, str]]:
    values: list[tuple[int, float, str]] = []
    for idx, cell in enumerate(row):
        if idx == exclude_idx:
            continue
        raw = cell.strip()
        if not raw or _is_bare_int(raw):
            continue
        try:
            value = parse_vn_number(raw)
        except VnNumberError:
            continue
        except ValueError:
            continue
        values.append((idx, value, raw))
    return values


def _row_label(row: tuple[str, ...], *, ma_so_idx: int, value_indices: set[int]) -> str:
    candidates = [
        cell.strip()
        for idx, cell in enumerate(row)
        if idx != ma_so_idx and idx not in value_indices and cell.strip()
    ]
    if not candidates:
        return ""
    return max(candidates, key=len)


def _classify_kind(codes: set[str], labels: list[str]) -> StatementKind | None:
    three_digit = {c for c in codes if len(c) == 3}
    if len(three_digit) >= _MIN_CDKT_3DIGIT_CODES:
        return "cdkt"
    if any(_looks_like_cash_flow_label(label) for label in labels):
        return "lctt"
    if len(codes & _KQKD_REQUIRED_CODES) >= _MIN_KQKD_MATCHES:
        return "kqkd"
    return None


def _table_scale(table: TableAsset, document: Document) -> tuple[float, str] | None:
    resolved = scale_from_unit_text(",".join(table.header))
    if resolved is not None:
        return resolved
    for snippet in document.table_unit_snippets(table.table_id):
        resolved = scale_from_unit_text(snippet)
        if resolved is not None:
            return resolved
    return None


def parse_statement_table(table: TableAsset, document: Document) -> StatementTable | None:
    header_hint = _header_ma_so_index(table.header)

    parsed_rows: list[tuple[str, int, str, list[tuple[int, float, str]]]] = []
    for row_idx, row in enumerate(table.rows):
        ma_so_idx = _find_ma_so_index(row, header_hint)
        if ma_so_idx is None:
            continue
        value_cells = _row_value_cells(row, exclude_idx=ma_so_idx)
        if not value_cells:
            continue
        ma_so = row[ma_so_idx].strip()
        label = _row_label(row, ma_so_idx=ma_so_idx, value_indices={idx for idx, _, _ in value_cells})
        parsed_rows.append((ma_so, row_idx, label, value_cells))

    if not parsed_rows:
        return None

    codes = {ma_so for ma_so, _, _, _ in parsed_rows}
    labels = [label for _, _, label, _ in parsed_rows]
    kind = _classify_kind(codes, labels)
    if kind is None:
        return None

    scale_result = _table_scale(table, document)
    if scale_result is None:
        return None
    scale, _normalized_unit = scale_result

    table_ref = f"{table.doc_name}|table_{table.table_id}"
    current: dict[str, StatementCell] = {}
    prior: dict[str, StatementCell] = {}
    for ma_so, row_idx, label, value_cells in parsed_rows:
        if ma_so in current:
            continue
        col_idx, raw_value, raw_text = value_cells[0]
        current[ma_so] = StatementCell(
            ma_so=ma_so,
            label=label,
            value=raw_value * scale,
            raw=raw_text,
            table_ref=table_ref,
            row_idx=row_idx,
            col_idx=col_idx,
            scale=scale,
        )
        if len(value_cells) >= 2:
            prior_col_idx, prior_raw_value, prior_raw_text = value_cells[1]
            prior[ma_so] = StatementCell(
                ma_so=ma_so,
                label=label,
                value=prior_raw_value * scale,
                raw=prior_raw_text,
                table_ref=table_ref,
                row_idx=row_idx,
                col_idx=prior_col_idx,
                scale=scale,
            )

    return StatementTable(table_ref=table_ref, csv_path=table.csv_path, kind=kind, current=current, prior=prior)
