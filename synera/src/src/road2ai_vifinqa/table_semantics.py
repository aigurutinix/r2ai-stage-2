"""Semantic interpretation of flattened financial-report tables.

The OCR corpus stores every HTML table as a rectangular grid.  Rowspan and
colspan expansion is useful for execution, but it loses the distinction
between a value inherited from a header and an ordinary data cell.  This
module rebuilds the small amount of structure needed by all deterministic
solvers: hierarchical column headers, section ancestry, period roles, units
and statement type.

The functions deliberately accept plain ``list[list[str]]`` values so they
can be used while building the SQLite index as well as at query time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .text import clean_text, fold_text, parse_vn_number, source_scale


YEAR_RE = re.compile(r"\b(20\d{2})\b")
CODE_RE = re.compile(r"^\d{1,3}$")
UNIT_MARKERS = (
    "nghin ty",
    "tram ty",
    "ty dong",
    "trieu dong",
    "trieu vnd",
    "nghin dong",
    "ngan dong",
    "nghin vnd",
    "ngan vnd",
    "don vi vnd",
    "don vi dong",
    "vnd",
)
TOTAL_MARKERS = (
    "tong cong",
    "cong",
    "tong so",
    "tong tai san",
    "tong nguon von",
    "tong doanh thu",
    "tong chi phi",
)
COMPONENT_MARKERS = (
    "trong do",
    "bao gom",
    "chi tiet",
    "phan bo",
    "nam hien hanh",
    "nam truoc",
)


class StatementKind(StrEnum):
    UNKNOWN = "unknown"
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    EQUITY_MOVEMENT = "equity_movement"
    NOTE = "note"


class PeriodRole(StrEnum):
    UNKNOWN = "unknown"
    CURRENT = "current"
    PRIOR = "prior"
    OPENING = "opening"
    CLOSING = "closing"


@dataclass(frozen=True, slots=True)
class PeriodInfo:
    role: PeriodRole
    year: int | None
    explicit: bool


@dataclass(frozen=True, slots=True)
class CellSemantics:
    row_idx: int
    col_idx: int
    row_label: str
    section: str
    column_header: str
    period: PeriodInfo
    unit_scale: float
    is_total: bool
    is_component: bool


class TableAnalyzer:
    """Precomputed semantic view for repeated cell access within one table."""

    def __init__(
        self,
        rows: list[list[str]],
        *,
        context: str = "",
        report_year: int | None = None,
    ) -> None:
        self.rows = rows
        self.context = context
        self.report_year = report_year
        self.header_indices = tuple(idx for idx, row in enumerate(rows) if is_header_row(row))
        self.labels = tuple(row_label(row) for row in rows)
        sections: list[str] = []
        active: list[str] = []
        header_set = set(self.header_indices)
        for idx, row in enumerate(rows):
            sections.append(_dedupe(active[-3:]))
            if idx in header_set:
                continue
            populated = [clean_text(cell) for cell in row if clean_text(cell)]
            if populated and not any(parse_vn_number(cell) is not None for cell in populated):
                label = self.labels[idx]
                if label:
                    active.append(label)
        self.sections = tuple(sections)
        header_text = " ".join(" ".join(row) for row in rows[: min(8, len(rows))])
        self.table_scale = _explicit_unit_scale(header_text)
        self.context_scale = _explicit_unit_scale(context)

    def column_header(self, row_idx: int, col_idx: int) -> str:
        parts: list[str] = []
        for idx in self.header_indices:
            if idx >= row_idx:
                break
            if idx >= 8 and idx < max(0, row_idx - 5):
                continue
            row = self.rows[idx]
            if col_idx < len(row):
                parts.append(row[col_idx])
        return _dedupe(parts)

    def cell(self, row_idx: int, col_idx: int, *, code_idx: int | None = None) -> CellSemantics:
        label = row_label(self.rows[row_idx], code_idx=code_idx) if code_idx is not None else self.labels[row_idx]
        section = self.sections[row_idx]
        header = self.column_header(row_idx, col_idx)
        local_scale = _explicit_unit_scale(_dedupe([header, label]))
        scale = local_scale or self.table_scale or self.context_scale or 1.0
        folded_label = fold_text(label)
        folded_scope = fold_text(f"{section} {label}")
        is_total = any(
            folded_label == marker or folded_label.startswith(f"{marker} ")
            for marker in TOTAL_MARKERS
        )
        is_component = any(marker in folded_scope for marker in COMPONENT_MARKERS)
        return CellSemantics(
            row_idx=row_idx,
            col_idx=col_idx,
            row_label=label,
            section=section,
            column_header=header,
            period=period_info(header, report_year=self.report_year),
            unit_scale=scale,
            is_total=is_total,
            is_component=is_component,
        )


def _numeric_ratio(row: list[str]) -> float:
    populated = [cell for cell in row if clean_text(cell)]
    if not populated:
        return 0.0
    numeric = 0
    for cell in populated:
        value = parse_vn_number(cell)
        if value is None:
            continue
        compact = clean_text(cell).replace(".", "").replace(",", "")
        if compact.isdigit() and 1900 <= int(compact) <= 2100:
            continue
        numeric += 1
    return numeric / len(populated)


def _dedupe(parts: list[str]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = clean_text(part)
        folded = fold_text(value)
        if not value or not folded or folded in seen:
            continue
        seen.add(folded)
        result.append(value)
    return " | ".join(result)


def row_label(row: list[str], *, code_idx: int | None = None) -> str:
    """Return the most informative non-numeric label in a data row."""

    labels: list[str] = []
    for idx, raw in enumerate(row):
        text = clean_text(raw)
        if not text or idx == code_idx or parse_vn_number(text) is not None:
            continue
        # A bare line-item code is not a semantic label.
        if CODE_RE.fullmatch(text):
            continue
        labels.append(text)
    return max(labels, key=lambda item: (len(fold_text(item).split()), len(item)), default="")


def is_header_row(row: list[str]) -> bool:
    """Conservative classifier for global or repeated local header rows."""

    populated = [clean_text(cell) for cell in row if clean_text(cell)]
    if not populated:
        return False
    folded = fold_text(" ".join(populated))
    header_cues = (
        "nam nay",
        "nam truoc",
        "ky nay",
        "ky truoc",
        "so cuoi",
        "so dau",
        "tai ngay",
        "don vi",
        "thuyet minh",
        "chi tieu",
    )
    if any(cue in folded for cue in header_cues) or YEAR_RE.search(folded):
        return _numeric_ratio(row) <= 0.5
    # Colspan expansion frequently creates repeated text across a header.
    folded_cells = [fold_text(cell) for cell in populated]
    repeated = len(folded_cells) >= 2 and len(set(folded_cells)) < len(folded_cells)
    return repeated and _numeric_ratio(row) <= 0.25


def column_header(rows: list[list[str]], row_idx: int, col_idx: int) -> str:
    """Reconstruct the header lineage above one numeric cell.

    Both the global leading header and repeated headers immediately above the
    target are considered.  Data rows between them are ignored so a value in
    another line item cannot accidentally become a year label.
    """

    parts: list[str] = []
    for idx, row in enumerate(rows[:row_idx]):
        if idx >= 8 and idx < max(0, row_idx - 5):
            continue
        if not is_header_row(row):
            continue
        if col_idx < len(row):
            parts.append(row[col_idx])
    return _dedupe(parts)


def section_text(rows: list[list[str]], row_idx: int, *, lookback: int = 10) -> str:
    """Return nearby non-numeric section labels preceding ``row_idx``."""

    parts: list[str] = []
    for idx in range(max(0, row_idx - lookback), row_idx):
        row = rows[idx]
        if is_header_row(row):
            continue
        populated = [clean_text(cell) for cell in row if clean_text(cell)]
        numeric = sum(parse_vn_number(cell) is not None for cell in populated)
        if not populated or numeric > 0:
            continue
        label = row_label(row)
        if label:
            parts.append(label)
    return _dedupe(parts[-3:])


def period_info(header: str, *, report_year: int | None = None) -> PeriodInfo:
    folded = fold_text(header)
    years = [int(value) for value in YEAR_RE.findall(folded)]
    year = years[-1] if years else None
    explicit = bool(years)
    opening_date = bool(re.search(r"\b0?1 0?1(?: 20\d{2})?\b", folded))
    closing_date = bool(re.search(r"\b31 12(?: 20\d{2})?\b", folded))
    if opening_date or any(marker in folded for marker in ("so dau", "dau ky", "dau nam")):
        role = PeriodRole.OPENING
    elif closing_date or any(marker in folded for marker in ("so cuoi", "cuoi ky", "cuoi nam")):
        role = PeriodRole.CLOSING
    elif any(marker in folded for marker in ("nam truoc", "ky truoc", "so sanh")):
        role = PeriodRole.PRIOR
    elif any(marker in folded for marker in ("nam nay", "ky nay", "hien tai")):
        role = PeriodRole.CURRENT
    elif report_year is not None and year == report_year:
        role = PeriodRole.CURRENT
    elif report_year is not None and year == report_year - 1:
        role = PeriodRole.PRIOR
    else:
        role = PeriodRole.UNKNOWN
    return PeriodInfo(role=role, year=year, explicit=explicit)


def _explicit_unit_scale(text: str) -> float | None:
    folded = fold_text(text)
    if not any(marker in folded for marker in UNIT_MARKERS):
        return None
    # A plain VND/Dong marker is explicitly base currency.
    has_base_marker = any(marker in folded for marker in ("don vi vnd", "don vi dong")) or (
        "vnd" in folded.split()
    )
    if has_base_marker and source_scale(folded) == 1.0 and not any(
        marker in folded
        for marker in ("trieu", "nghin", "ngan", "tram ty", "nghin ty")
    ):
        return 1.0
    return source_scale(folded)


def unit_scale(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    *,
    context: str = "",
) -> float:
    """Infer a cell's source unit using nearest/most-specific evidence first."""

    header = column_header(rows, row_idx, col_idx)
    local = _dedupe([header, row_label(rows[row_idx]) if row_idx < len(rows) else ""])
    if (scale := _explicit_unit_scale(local)) is not None:
        return scale
    table_header = " ".join(" ".join(row) for row in rows[: min(8, len(rows))])
    if (scale := _explicit_unit_scale(table_header)) is not None:
        return scale
    if (scale := _explicit_unit_scale(context)) is not None:
        return scale
    return 1.0


def statement_kind(context: str, rows: list[list[str]]) -> StatementKind:
    lead = " ".join(" ".join(row) for row in rows[: min(8, len(rows))])
    folded = fold_text(f"{context} {lead}")
    if any(marker in folded for marker in ("bao cao luu chuyen tien te", "luu chuyen tien te")):
        return StatementKind.CASH_FLOW
    if any(marker in folded for marker in ("bao cao ket qua hoat dong kinh doanh", "ket qua kinh doanh")):
        return StatementKind.INCOME_STATEMENT
    if any(marker in folded for marker in ("bang can doi ke toan", "bao cao tinh hinh tai chinh")):
        return StatementKind.BALANCE_SHEET
    if any(marker in folded for marker in ("bao cao bien dong von chu so huu", "thay doi von chu so huu")):
        return StatementKind.EQUITY_MOVEMENT
    if any(marker in folded for marker in ("thuyet minh", "chi tiet")):
        return StatementKind.NOTE
    return StatementKind.UNKNOWN


def cell_semantics(
    rows: list[list[str]],
    row_idx: int,
    col_idx: int,
    *,
    context: str = "",
    report_year: int | None = None,
    code_idx: int | None = None,
) -> CellSemantics:
    label = row_label(rows[row_idx], code_idx=code_idx)
    section = section_text(rows, row_idx)
    header = column_header(rows, row_idx, col_idx)
    folded_label = fold_text(label)
    folded_scope = fold_text(f"{section} {label}")
    is_total = any(
        folded_label == marker or folded_label.startswith(f"{marker} ")
        for marker in TOTAL_MARKERS
    )
    is_component = any(marker in folded_scope for marker in COMPONENT_MARKERS)
    return CellSemantics(
        row_idx=row_idx,
        col_idx=col_idx,
        row_label=label,
        section=section,
        column_header=header,
        period=period_info(header, report_year=report_year),
        unit_scale=unit_scale(rows, row_idx, col_idx, context=context),
        is_total=is_total,
        is_component=is_component,
    )


def question_statement_preference(question: str) -> tuple[StatementKind, ...]:
    folded = fold_text(question)
    if any(marker in folded for marker in ("luu chuyen", "dong tien", "tien thuan")):
        return (StatementKind.CASH_FLOW, StatementKind.NOTE)
    if any(marker in folded for marker in ("doanh thu", "loi nhuan", "chi phi", "lai lo")):
        return (StatementKind.INCOME_STATEMENT, StatementKind.NOTE)
    if any(
        marker in folded
        for marker in (
            "tai san",
            "nguon von",
            "so du",
            "cuoi nam",
            "dau nam",
            "tai ngay",
            "phai thu",
            "phai tra",
        )
    ):
        return (StatementKind.BALANCE_SHEET, StatementKind.NOTE)
    return ()


def asks_total(question: str) -> bool:
    folded = fold_text(question)
    return any(
        marker in folded
        for marker in ("tong cong", "tong so", "tong tai san", "tong nguon von", "tong doanh thu")
    )


def period_match_score(period: PeriodInfo, *, requested_year: int, report_year: int) -> float:
    """Score whether a header represents the requested reporting period."""

    if requested_year <= 0:
        return 0.0
    if period.year is not None:
        if period.year == requested_year:
            return 10.0
        return -12.0
    if requested_year == report_year:
        if period.role in (PeriodRole.CURRENT, PeriodRole.CLOSING):
            return 8.0
        if period.role in (PeriodRole.PRIOR, PeriodRole.OPENING):
            return -7.0
    if requested_year == report_year - 1:
        if period.role in (PeriodRole.PRIOR, PeriodRole.OPENING):
            return 8.0
        if period.role in (PeriodRole.CURRENT, PeriodRole.CLOSING):
            return -7.0
    return 0.0
