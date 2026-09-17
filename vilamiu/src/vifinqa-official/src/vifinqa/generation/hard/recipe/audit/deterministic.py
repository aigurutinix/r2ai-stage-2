
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.document import Document, parse_document
from vifinqa.common.corpus.statement import StatementTable, parse_statement_table
from vifinqa.common.corpus.table import load_table
from vifinqa.generation.hard.recipe.audit.base import AuditVerdict, DependencyStatus, RejectionReason
from vifinqa.generation.hard.recipe.audit.dependencies import DependencyRecord
from vifinqa.generation.panel.catalog import metric_name

_EXCLUDED_INDUSTRIES = frozenset({"Tổ chức tín dụng", "Chứng khoán", "Bảo hiểm"})

_ADJUSTMENT_KEYWORDS = ("dieu chinh", "trinh bay lai", "hoi to", "phan anh lai", "restated")

_FISCAL_YEAR_RE = re.compile(r"nam tai chinh ket thuc ngay.{0,60}?nam (\d{4})")
# Keep period handling explicit and deterministic.
_FOOTNOTE_CONTEXT_RE = re.compile(r"\([^)]{0,80}nam tai chinh ket thuc ngay.{0,60}?nam \d{4}\s{0,3}:")

_STOPWORDS = frozenset(
    {"va", "cac", "cua", "cho", "tu", "the", "trong", "hoat", "dong", "duoc", "la", "mot", "phat", "sinh"}
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_CURRENT_COLUMN_MODE_MIN_ROWS = 4
_CURRENT_COLUMN_MODE_DOMINANCE = 0.7
_LABEL_MATCH_VALID_RATIO = 0.6


def _fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").casefold()


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall(_fold(text)) if tok and tok not in _STOPWORDS}


def _table_id_from_ref(table_ref: str) -> int:
    return int(table_ref.rsplit("_", 1)[-1])


def check_chart_of_accounts(ticker: str, company_meta: dict[str, CompanyInfo]) -> AuditVerdict | None:
    info = company_meta.get(ticker)
    if info is not None and info.industry_l2 in _EXCLUDED_INDUSTRIES:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.UNSUPPORTED_CHART_OF_ACCOUNTS,
            detail=f"{ticker} belongs to industry {info.industry_l2!r}, whose chart of accounts differs from nonfinancial Circular 200 entities",
        )
    return None


def check_adjustment_only_table(document: Document, table_id: int) -> AuditVerdict | None:
    context = _fold(document.table_anchor_context(table_id, lines_before=8, lines_after=2))
    hit = next((kw for kw in _ADJUSTMENT_KEYWORDS if kw in context), None)
    if hit is not None:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.ADJUSTMENT_ONLY_TABLE,
            detail=f"anchor context for table_{table_id} contains keyword {hit!r} (adjusted/restated)",
        )
    return None


def check_fiscal_year(document: Document, *, expected_year: str) -> AuditVerdict | None:
    full_text = _fold("\n".join(content for _, content in document.pages))
    footnote_spans = [m.span() for m in _FOOTNOTE_CONTEXT_RE.finditer(full_text)]
    years = [
        m.group(1)
        for m in _FISCAL_YEAR_RE.finditer(full_text)
        if not any(start <= m.start() < end for start, end in footnote_spans)
    ]
    if not years:
        return None
    counts = Counter(years)
    mode_year, mode_count = counts.most_common(1)[0]
    if mode_year == expected_year:
        return None
    other_counts = [c for year, c in counts.items() if year != mode_year]
    if mode_count >= 2 and (not other_counts or mode_count > max(other_counts)):
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.FISCAL_YEAR_MISMATCH,
            detail=f"the document mostly reports fiscal years ending in {mode_year!r}, not period={expected_year!r} ({dict(counts)})",
        )
    return None


def check_metric_label(label: str, metric_key: str) -> AuditVerdict | None:
    expected_name = metric_name(metric_key)
    if expected_name is None:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.PROVENANCE_MISMATCH,
            detail=f"metric_key={metric_key!r} is not in the canonical catalog",
        )
    expected_tokens = _tokenize(expected_name)
    if not expected_tokens:
        return None
    label_tokens = _tokenize(label)
    overlap = expected_tokens & label_tokens
    ratio = len(overlap) / len(expected_tokens)
    if ratio >= _LABEL_MATCH_VALID_RATIO:
        return None
    if ratio == 0:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.METRIC_LABEL_MISMATCH,
            detail=f"label={label!r} does not match canonical name {expected_name!r} for {metric_key}",
        )
    return AuditVerdict(
        DependencyStatus.REVIEW,
        RejectionReason.METRIC_LABEL_MISMATCH,
        detail=f"label={label!r} only partially matches canonical name {expected_name!r} (overlap={ratio:.2f})",
    )


def check_current_column(statement: StatementTable, ma_so: str) -> AuditVerdict | None:
    col_counts = Counter(cell.col_idx for cell in statement.current.values())
    if not col_counts:
        return None
    mode_col, mode_count = col_counts.most_common(1)[0]
    target_col = statement.current[ma_so].col_idx
    if target_col == mode_col:
        return None
    total = sum(col_counts.values())
    if total >= _CURRENT_COLUMN_MODE_MIN_ROWS and mode_count / total >= _CURRENT_COLUMN_MODE_DOMINANCE:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.CURRENT_COLUMN_UNRESOLVED,
            detail=f"col_idx={target_col} differs from the dominant current column (mode={mode_col}, {dict(col_counts)})",
        )
    return AuditVerdict(
        DependencyStatus.REVIEW,
        RejectionReason.CURRENT_COLUMN_UNRESOLVED,
        detail=f"col_idx={target_col} differs from mode={mode_col}, but the mode is not dominant enough ({dict(col_counts)})",
    )


def check_duplicate_conflict(
    record: DependencyRecord, doc: DocumentRef, document: Document, *, this_value: float, tolerance: float = 1.0
) -> AuditVerdict | None:
    kind, ma_so = record.binding.metric_key.split(":", 1)
    this_table_id = _table_id_from_ref(record.binding.table_ref)
    for table_id in doc.table_ids:
        if table_id == this_table_id:
            continue
        table = load_table(
            doc.table_csv_path(table_id), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id
        )
        statement = parse_statement_table(table, document)
        if statement is None or statement.kind != kind:
            continue
        other_cell = statement.current.get(ma_so)
        if other_cell is None:
            continue
        if abs(other_cell.value - this_value) > tolerance:
            return AuditVerdict(
                DependencyStatus.REJECTED,
                RejectionReason.DUPLICATE_PRIMARY_CONFLICT,
                detail=(
                    f"code {ma_so} ({kind}) conflicts between table_{this_table_id} "
                    f"(value={this_value}) and table_{table_id} (value={other_cell.value})"
                ),
            )
    return None


def run_deterministic_checks(
    record: DependencyRecord, *, doc: DocumentRef, company_meta: dict[str, CompanyInfo]
) -> AuditVerdict:
    industry_verdict = check_chart_of_accounts(record.ticker, company_meta)
    if industry_verdict is not None:
        return industry_verdict

    kind, ma_so = record.binding.metric_key.split(":", 1)
    table_id = _table_id_from_ref(record.binding.table_ref)
    table = load_table(
        doc.table_csv_path(table_id), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id
    )
    document = parse_document(doc.text_path)  # type: ignore[arg-type]

    adjustment_verdict = check_adjustment_only_table(document, table_id)
    if adjustment_verdict is not None:
        return adjustment_verdict

    fiscal_verdict = check_fiscal_year(document, expected_year=record.period)
    if fiscal_verdict is not None:
        return fiscal_verdict

    statement = parse_statement_table(table, document)
    if statement is None or statement.kind != kind:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.PROVENANCE_MISMATCH,
            detail=f"table_{table_id} can no longer be parsed as statement kind={kind} during re-parse",
        )
    fresh_cell = statement.current.get(ma_so)
    if fresh_cell is None:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.RAGGED_VALUE_DETACHED,
            detail=f"code {ma_so} no longer resolves to exactly one number when re-parsing table_{table_id}",
        )
    if fresh_cell.row_idx != record.binding.row_idx or fresh_cell.col_idx != record.binding.col_idx:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.PROVENANCE_MISMATCH,
            detail=(
                f"re-parse points to a different cell: ({fresh_cell.row_idx},{fresh_cell.col_idx}) != "
                f"({record.binding.row_idx},{record.binding.col_idx})"
            ),
        )
    if fresh_cell.scale != record.binding.scale:
        return AuditVerdict(
            DependencyStatus.REJECTED,
            RejectionReason.UNIT_CONFLICT,
            detail=f"re-parsed scale={fresh_cell.scale} differs from binding scale={record.binding.scale}",
        )

    column_verdict = check_current_column(statement, ma_so)
    if column_verdict is not None:
        return column_verdict

    label_verdict = check_metric_label(fresh_cell.label, record.binding.metric_key)
    if label_verdict is not None:
        return label_verdict

    duplicate_verdict = check_duplicate_conflict(record, doc, document, this_value=fresh_cell.value)
    if duplicate_verdict is not None:
        return duplicate_verdict

    return AuditVerdict(DependencyStatus.VALID, None)
