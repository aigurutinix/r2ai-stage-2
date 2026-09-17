"""Period-basis audit for JIT dependency checks.

The cube is only a provisional coordinate index. This module re-reads the exact
table column and the table anchor page window for one dependency closure, then
classifies whether the selected column's time basis can support the requested
period. Verdict identity is per table column, not per metric row.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.document import Document
from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.hard.recipe.audit.base import AuditVerdict, DependencyStatus, RejectionReason
from vifinqa.generation.hard.recipe.audit.dependencies import DependencyRecord

ExpectedPeriodBasis = Literal["annual_duration_flow", "point_in_time"]


class PeriodClassification(str, Enum):
    VALID_FOR_EXPECTED_PERIOD = "VALID_FOR_EXPECTED_PERIOD"
    PARTIAL_PERIOD = "PARTIAL_PERIOD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PeriodSubject:
    doc_name: str
    table_id: int
    col_idx: int
    statement_kind: str
    expected_measurement_basis: ExpectedPeriodBasis

    @property
    def identity(self) -> str:
        return "|".join(
            (
                self.doc_name,
                f"table_{self.table_id}",
                str(self.col_idx),
                self.statement_kind,
                self.expected_measurement_basis,
            )
        )


@dataclass(frozen=True, slots=True)
class PeriodEvidence:
    subject: PeriodSubject
    ticker: str
    folder_period: str
    doc_name: str
    report_scope: str
    metric_key: str
    ma_so: str
    row_idx: int
    col_idx: int
    selected_column_header: str
    full_header: tuple[str, ...]
    target_raw_row: tuple[str, ...]
    table_preview_rows: tuple[tuple[str, ...], ...]
    anchor_page: int | None
    page_window: str
    scale: float

    def stable_hash(self) -> str:
        payload = {
            "subject": self.subject.identity,
            "ticker": self.ticker,
            "folder_period": self.folder_period,
            "doc_name": self.doc_name,
            "report_scope": self.report_scope,
            "metric_key": self.metric_key,
            "ma_so": self.ma_so,
            "row_idx": self.row_idx,
            "col_idx": self.col_idx,
            "selected_column_header": self.selected_column_header,
            "full_header": self.full_header,
            "target_raw_row": self.target_raw_row,
            "table_preview_rows": self.table_preview_rows,
            "anchor_page": self.anchor_page,
            "page_window": self.page_window,
            "scale": self.scale,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_prompt_text(self) -> str:
        preview = "\n".join(str(list(row)) for row in self.table_preview_rows)
        return "\n".join(
            [
                f"period_subject: {self.subject.identity}",
                f"ticker: {self.ticker}",
                f"folder_period: {self.folder_period}",
                f"doc_name: {self.doc_name}",
                f"report_scope: {self.report_scope}",
                f"table_ref: {self.doc_name}|table_{self.subject.table_id}",
                f"anchor_page: {self.anchor_page}",
                f"statement_kind: {self.subject.statement_kind}",
                f"expected_measurement_basis: {self.subject.expected_measurement_basis}",
                f"metric_key: {self.metric_key}",
                f"ma_so: {self.ma_so}",
                f"row_idx: {self.row_idx}",
                f"col_idx: {self.col_idx}",
                f"selected_column_header: {self.selected_column_header!r}",
                f"full_header: {list(self.full_header)!r}",
                f"target_raw_row: {list(self.target_raw_row)!r}",
                f"table_preview_rows:\n{preview}",
                "page_window_around_target_anchor:",
                self.page_window,
            ]
        )


_RANGE_RE = re.compile(
    r"(?:tu|từ)\s+ngay\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s+"
    r"(?:den|đến)\s+ngay\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})"
)
_DATE_RE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_AMBIGUOUS_HEADER_RE = re.compile(r"\b(nam nay|ky nay|năm nay|kỳ này|current year|current period)\b")


def _fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").casefold()


def _table_id_from_ref(table_ref: str) -> int:
    return int(table_ref.rsplit("_", 1)[-1])


def expected_basis_for_statement_kind(statement_kind: str) -> ExpectedPeriodBasis:
    if statement_kind == "cdkt":
        return "point_in_time"
    return "annual_duration_flow"


def period_subject_for(record: DependencyRecord) -> PeriodSubject:
    statement_kind = record.binding.metric_key.split(":", 1)[0]
    return PeriodSubject(
        doc_name=record.binding.table_ref.split("|", 1)[0],
        table_id=_table_id_from_ref(record.binding.table_ref),
        col_idx=record.binding.col_idx,
        statement_kind=statement_kind,
        expected_measurement_basis=expected_basis_for_statement_kind(statement_kind),
    )


def build_period_evidence(
    record: DependencyRecord,
    *,
    doc: DocumentRef,
    document: Document,
    table: TableAsset,
    report_scope: str = "consolidated",
    preview_rows: int = 4,
) -> PeriodEvidence:
    subject = period_subject_for(record)
    target_row = table.rows[record.binding.row_idx] if record.binding.row_idx < len(table.rows) else ()
    header = table.header
    selected_header = header[record.binding.col_idx] if record.binding.col_idx < len(header) else ""
    return PeriodEvidence(
        subject=subject,
        ticker=doc.ticker,
        folder_period=doc.year,
        doc_name=doc.doc_name,
        report_scope=report_scope,
        metric_key=record.binding.metric_key,
        ma_so=record.binding.metric_key.split(":", 1)[1],
        row_idx=record.binding.row_idx,
        col_idx=record.binding.col_idx,
        selected_column_header=selected_header,
        full_header=header,
        target_raw_row=target_row,
        table_preview_rows=tuple(table.rows[:preview_rows]),
        anchor_page=document.table_page.get(subject.table_id),
        page_window=document.table_context(subject.table_id, before=1, after=1),
        scale=record.binding.scale,
    )


def _range_classification(header_text: str, expected_year: str) -> PeriodClassification | None:
    folded = _fold(header_text)
    match = _RANGE_RE.search(folded)
    if match is None:
        return None
    start_day, start_month, start_year, end_day, end_month, end_year = match.groups()
    if start_year == expected_year and end_year == expected_year and (
        int(start_day), int(start_month), int(end_day), int(end_month)
    ) == (1, 1, 31, 12):
        return PeriodClassification.VALID_FOR_EXPECTED_PERIOD
    if start_year == expected_year or end_year == expected_year:
        return PeriodClassification.PARTIAL_PERIOD
    return PeriodClassification.UNKNOWN


def classify_period_basis(evidence: PeriodEvidence) -> PeriodClassification:
    basis = evidence.subject.expected_measurement_basis
    header_text = " ".join((evidence.selected_column_header, " | ".join(evidence.full_header)))
    selected_folded = _fold(evidence.selected_column_header)
    folded_header = _fold(header_text)

    if basis == "point_in_time":
        for day, month, year in _DATE_RE.findall(header_text):
            if year == evidence.folder_period and int(day) == 31 and int(month) == 12:
                return PeriodClassification.VALID_FOR_EXPECTED_PERIOD
        if evidence.folder_period in _YEAR_RE.findall(header_text):
            return PeriodClassification.VALID_FOR_EXPECTED_PERIOD
        return PeriodClassification.UNKNOWN

    range_verdict = _range_classification(evidence.selected_column_header, evidence.folder_period)
    if range_verdict is not None:
        return range_verdict

    if _AMBIGUOUS_HEADER_RE.search(selected_folded):
        return PeriodClassification.UNKNOWN

    full_header_ranges = [m.group(0) for m in _RANGE_RE.finditer(folded_header)]
    for raw_range in full_header_ranges:
        verdict = _range_classification(raw_range, evidence.folder_period)
        if verdict is PeriodClassification.VALID_FOR_EXPECTED_PERIOD:
            return verdict

    if _AMBIGUOUS_HEADER_RE.search(folded_header):
        return PeriodClassification.UNKNOWN

    if evidence.folder_period in _YEAR_RE.findall(header_text) and "quy" not in folded_header:
        return PeriodClassification.VALID_FOR_EXPECTED_PERIOD
    return PeriodClassification.UNKNOWN


def deterministic_period_verdict(evidence: PeriodEvidence) -> AuditVerdict:
    if not evidence.selected_column_header and not evidence.full_header:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.PERIOD_BASIS_UNRESOLVED,
            detail="no column header is available to verify the selected column's period basis",
        )
    classification = classify_period_basis(evidence)
    if classification == PeriodClassification.VALID_FOR_EXPECTED_PERIOD:
        return AuditVerdict(DependencyStatus.VALID, None, detail="period basis matches the selected column")
    if classification == PeriodClassification.PARTIAL_PERIOD:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.PARTIAL_REPORTING_PERIOD,
            detail="selected column is a stub/partial duration and does not cover the requested annual period",
        )
    return AuditVerdict(
        DependencyStatus.REVIEW,
        RejectionReason.PERIOD_BASIS_UNRESOLVED,
        detail="insufficient explicit target-column evidence to verify period basis",
    )
