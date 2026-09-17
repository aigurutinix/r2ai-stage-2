
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from vifinqa.constants import TABLE_HEADER_MAX_CHARS
from vifinqa.common.corpus.document import Document
from vifinqa.common.corpus.table import TableAsset
from vifinqa.common.filtering.table_filters import is_numeric_cell

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
# Keep report-scope handling explicit.
_HOP_NHAT_MARKERS = ("hopnhat", "consol", "bctchn")
_CONG_TY_ME_MARKERS = ("congtyme", "rieng", "separate")
METADATA_CSV_LAYOUTS = ("head", "head_tail")


def _normalize_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = stripped.replace("đ", "d").replace("Đ", "D").lower()
    return _NON_ALNUM_RE.sub("", stripped)


def detect_report_scope(doc_name: str) -> str | None:
    normalized = _normalize_ascii(doc_name)
    is_hop_nhat = any(marker in normalized for marker in _HOP_NHAT_MARKERS)
    is_cong_ty_me = any(marker in normalized for marker in _CONG_TY_ME_MARKERS)
    if is_hop_nhat and not is_cong_ty_me:
        return "hợp nhất"
    if is_cong_ty_me and not is_hop_nhat:
        return "công ty mẹ"
    return None


def _metadata_prefix(table: TableAsset, company_name: str) -> str:
    if not isinstance(company_name, str):
        raise TypeError(
            "company_name must be a string; do not serialize CompanyInfo objects into embedding text"
        )
    company_part = (
        f"Công ty: {company_name} (mã {table.ticker})"
        if company_name
        else f"Mã: {table.ticker}"
    )
    scope = detect_report_scope(table.doc_name)
    scope_part = f". Phạm vi báo cáo: {scope}" if scope else ""
    return f"{company_part}, năm {table.year}{scope_part}."


def _row_labels(table: TableAsset, *, max_rows: int) -> str:
    return " | ".join(
        " / ".join(cell for cell in row[:3] if cell and not is_numeric_cell(cell))
        for row in table.rows[:max_rows]
        if any(cell and not is_numeric_cell(cell) for cell in row[:3])
    )


def table_index_text(
    table: TableAsset, *, max_rows: int = 20, max_chars: int = TABLE_HEADER_MAX_CHARS
) -> str:
    columns = " | ".join(h for h in table.header if h)
    row_labels = _row_labels(table, max_rows=max_rows)
    return f"Cột: {columns}. Dòng: {row_labels}"[:max_chars]


def table_retrieval_text(
    table: TableAsset,
    document: Document,
    company_name: str,
    *,
    max_context_chars: int = 2500,
) -> str:
    surrounding_pages = document.table_context(table.table_id, before=0, after=0)
    context = " ".join(surrounding_pages.split())[:max_context_chars]
    return f"{_metadata_prefix(table, company_name)} {table_index_text(table)}. Ngữ cảnh: {context}"


def anchor_snippet_text(
    table: TableAsset,
    document: Document,
    company_name: str,
    *,
    max_rows: int = 80,
    max_context_chars: int = 2500,
    max_header_chars: int = TABLE_HEADER_MAX_CHARS,
) -> str:
    columns = " | ".join(h for h in table.header if h)
    row_labels = _row_labels(table, max_rows=max_rows)
    header_and_rows = f"Cột: {columns}. Dòng: {row_labels}"[:max_header_chars]
    anchor_context = document.table_anchor_context(
        table.table_id, lines_before=6, lines_after=2
    )
    context = " ".join(anchor_context.split())[:max_context_chars]
    return f"{_metadata_prefix(table, company_name)} {header_and_rows}. Ngữ cảnh: {context}"


def metadata_csv_text(
    table: TableAsset,
    _document: Document,
    company_name: str,
    *,
    max_csv_chars: int = TABLE_HEADER_MAX_CHARS,
    layout: str = "head",
) -> str:
    if layout not in METADATA_CSV_LAYOUTS:
        raise ValueError(f"Invalid metadata CSV layout: {layout!r}")
    csv_text = table.csv_path.read_text(encoding="utf-8-sig")
    condensed = " ".join(csv_text.split())
    prefix = f"{_metadata_prefix(table, company_name)} "
    if layout == "head":
        # Preserve the historical control representation exactly.
        return f"{prefix}{condensed[:max_csv_chars]}"

    if len(prefix) >= max_csv_chars:
        return prefix[:max_csv_chars]
    available = max_csv_chars - len(prefix)
    if len(condensed) <= available:
        return f"{prefix}{condensed}"
    marker = " […] "
    content_budget = max(0, available - len(marker))
    head_chars = content_budget * 2 // 3
    tail_chars = content_budget - head_chars
    tail = condensed[-tail_chars:] if tail_chars else ""
    return f"{prefix}{condensed[:head_chars]}{marker}{tail}"


def semantic_labels_text(
    table: TableAsset,
    document: Document,
    company_name: str,
    *,
    max_chars: int = TABLE_HEADER_MAX_CHARS,
    max_context_chars: int = 800,
) -> str:
    context = document.table_context_before(table.table_id)
    if len(context) > max_context_chars:
        context = context[-max_context_chars:]
    columns = " | ".join(cell for cell in table.header if cell)
    labels = [
        " / ".join(cell for cell in row[:3] if cell and not is_numeric_cell(cell))
        for row in table.rows
    ]
    labels = [label for label in labels if label]
    text = (
        f"{_metadata_prefix(table, company_name)}\n"
        f"[TITLE_CONTEXT]\n{context}\n"
        f"[COLUMNS]\n{columns}\n"
        f"[ROW_LABELS]\n" + "\n".join(labels)
    )
    if len(text) <= max_chars:
        return text
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars - len("\n…\n")
    return f"{text[:head_chars]}\n…\n{text[-tail_chars:]}"


TABLE_ENCODERS: dict[str, Callable[[TableAsset, Document, str], str]] = {
    "table_retrieval_text": table_retrieval_text,
    "anchor_snippet": anchor_snippet_text,
    "metadata_csv": metadata_csv_text,
}
