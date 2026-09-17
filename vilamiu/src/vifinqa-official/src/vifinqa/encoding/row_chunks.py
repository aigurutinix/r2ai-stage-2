"""Pure row-window encoding for the dense-only ``metadata_context_row_chunks`` representation."""

from __future__ import annotations

from dataclasses import dataclass

from vifinqa.constants import (
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP_ROWS,
    DEFAULT_CHUNK_ROWS,
    DEFAULT_CONTEXT_MAX_CHARS,
)
from vifinqa.common.corpus.document import Document
from vifinqa.common.corpus.table import TableAsset
from vifinqa.encoding.table_text import (
    METADATA_CSV_LAYOUTS,
    detect_report_scope,
    metadata_csv_text,
    semantic_labels_text,
)

CHUNKED_TABLE_ENCODING = "metadata_context_row_chunks"
MULTI_VIEW_TABLE_ENCODING = "metadata_context_multi_view"
ROW_CHUNK_VIEW = "row_chunks"
METADATA_CSV_VIEW = "metadata_csv"
SEMANTIC_LABELS_VIEW = "semantic_labels"
TABLE_VIEWS = (ROW_CHUNK_VIEW, METADATA_CSV_VIEW, SEMANTIC_LABELS_VIEW)


@dataclass(frozen=True, slots=True)
class RowChunkConfig:
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    chunk_overlap_rows: int = DEFAULT_CHUNK_OVERLAP_ROWS
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS

    def __post_init__(self) -> None:
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be greater than 0")
        if not 0 <= self.chunk_overlap_rows < self.chunk_rows:
            raise ValueError("chunk_overlap_rows must be non-negative and less than chunk_rows")
        if self.context_max_chars < 0:
            raise ValueError("context_max_chars must be non-negative")
        if self.chunk_max_chars <= 0:
            raise ValueError("chunk_max_chars must be greater than 0")

    def as_dict(self) -> dict[str, int]:
        return {
            "chunk_rows": self.chunk_rows,
            "chunk_overlap_rows": self.chunk_overlap_rows,
            "context_max_chars": self.context_max_chars,
            "chunk_max_chars": self.chunk_max_chars,
        }

    @property
    def namespace(self) -> str:
        return (
            f"r{self.chunk_rows}-o{self.chunk_overlap_rows}"
            f"-ctx{self.context_max_chars}-max{self.chunk_max_chars}"
        )


@dataclass(frozen=True, slots=True)
class ChunkTruncation:
    table_ref: str
    section: str
    row_index: int | None
    cell_index: int | None
    original_chars: int
    kept_chars: int


@dataclass(frozen=True, slots=True)
class EncodedChunk:
    ticker: str
    year: str
    doc_name: str
    table_id: int
    chunk_index: int
    row_start: int
    row_end: int  # exclusive
    text: str
    view: str = ROW_CHUNK_VIEW

    @property
    def table_ref(self) -> str:
        return f"{self.doc_name}|table_{self.table_id}"


@dataclass(frozen=True, slots=True)
class RowChunkEncoding:
    chunks: tuple[EncodedChunk, ...]
    safety_split_windows: int
    safety_split_count: int
    truncated_chunks: int
    truncations: tuple[ChunkTruncation, ...]


def _tail_cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:] if max_chars else ""


def _meta_text(table: TableAsset, company_name: str) -> str:
    if not isinstance(company_name, str):
        raise TypeError(
            "company_name must be a string; do not serialize CompanyInfo objects into embedding text"
        )
    scope = detect_report_scope(table.doc_name) or "không xác định"
    return (
        "[META]\n"
        f"Công ty: {company_name}\n"
        f"Mã: {table.ticker}\n"
        f"Năm báo cáo: {table.year}\n"
        f"Phạm vi: {scope}"
    )


def _serialize(
    meta: str, context: str, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> str:
    columns = " | ".join(header)
    row_text = "\n".join(" | ".join(row) for row in rows)
    return f"{meta}\n\n[CONTEXT_BEFORE]\n{context}\n\n[COLUMNS]\n{columns}\n\n[ROWS]\n{row_text}"


def _truncate_cell(cell: str, keep: int) -> str:
    if keep >= len(cell):
        return cell
    if keep <= 0:
        return ""
    if keep == 1:
        return "…"
    return f"{cell[: keep - 1]}…"


def _fit_single_row(
    *,
    table: TableAsset,
    meta: str,
    context: str,
    header: tuple[str, ...],
    row: tuple[str, ...],
    row_index: int,
    max_chars: int,
) -> tuple[str, tuple[ChunkTruncation, ...]]:
    """Fit one pathological row by trimming only its longest cells, then header/context if needed."""
    mutable_row = list(row)
    mutable_header = list(header)
    mutable_context = context
    events: list[ChunkTruncation] = []

    while True:
        text = _serialize(
            meta, mutable_context, tuple(mutable_header), (tuple(mutable_row),)
        )
        excess = len(text) - max_chars
        if excess <= 0:
            return text, tuple(events)

        candidates = [
            (len(cell), "row", i) for i, cell in enumerate(mutable_row) if cell
        ]
        candidates.extend(
            (len(cell), "header", i) for i, cell in enumerate(mutable_header) if cell
        )
        if candidates:
            _, section, cell_index = max(
                candidates, key=lambda item: (item[0], item[1] == "row", -item[2])
            )
            cells = mutable_row if section == "row" else mutable_header
            original = cells[cell_index]
            kept = max(0, len(original) - excess)
            cells[cell_index] = _truncate_cell(original, kept)
            events.append(
                ChunkTruncation(
                    table_ref=table.table_ref,
                    section=section,
                    row_index=row_index if section == "row" else None,
                    cell_index=cell_index,
                    original_chars=len(original),
                    kept_chars=len(cells[cell_index]),
                )
            )
            continue

        if mutable_context:
            mutable_context = _tail_cap(
                mutable_context, max(0, len(mutable_context) - excess)
            )
            continue
        raise ValueError(
            f"chunk_max_chars={max_chars} is too small for the {table.table_ref} skeleton"
        )


def encode_row_chunks(
    table: TableAsset,
    document: Document,
    company_name: str,
    *,
    config: RowChunkConfig = RowChunkConfig(),
) -> RowChunkEncoding:
    """Encode every CSV row at least once; oversized windows are recursively split by row."""
    meta = _meta_text(table, company_name)
    context = _tail_cap(
        document.table_context_before(table.table_id), config.context_max_chars
    )
    step = config.chunk_rows - config.chunk_overlap_rows
    primary_windows: list[tuple[int, int]] = []
    start = 0
    while start < len(table.rows):
        end = min(start + config.chunk_rows, len(table.rows))
        primary_windows.append((start, end))
        if end == len(table.rows):
            break
        start += step

    fitted: list[tuple[int, int, str]] = []
    truncations: list[ChunkTruncation] = []
    safety_split_count = 0
    safety_split_windows = 0
    truncated_chunks = 0

    def fit(start_row: int, end_row: int) -> None:
        nonlocal safety_split_count, truncated_chunks
        rows = table.rows[start_row:end_row]
        text = _serialize(meta, context, table.header, rows)
        if len(text) <= config.chunk_max_chars:
            fitted.append((start_row, end_row, text))
            return
        if end_row - start_row > 1:
            safety_split_count += 1
            middle = start_row + (end_row - start_row) // 2
            fit(start_row, middle)
            fit(middle, end_row)
            return
        text, events = _fit_single_row(
            table=table,
            meta=meta,
            context=context,
            header=table.header,
            row=table.rows[start_row],
            row_index=start_row,
            max_chars=config.chunk_max_chars,
        )
        truncations.extend(events)
        truncated_chunks += 1
        fitted.append((start_row, end_row, text))

    for window_start, window_end in primary_windows:
        splits_before = safety_split_count
        fit(window_start, window_end)
        if safety_split_count > splits_before:
            safety_split_windows += 1

    chunks = tuple(
        EncodedChunk(
            ticker=table.ticker,
            year=table.year,
            doc_name=table.doc_name,
            table_id=table.table_id,
            chunk_index=i,
            row_start=start_row,
            row_end=end_row,
            text=text,
        )
        for i, (start_row, end_row, text) in enumerate(fitted)
    )
    return RowChunkEncoding(
        chunks=chunks,
        safety_split_windows=safety_split_windows,
        safety_split_count=safety_split_count,
        truncated_chunks=truncated_chunks,
        truncations=tuple(truncations),
    )


def encode_table_views(
    table: TableAsset,
    document: Document,
    company_name: str,
    *,
    views: tuple[str, ...],
    row_config: RowChunkConfig,
    summary_max_chars: int,
    metadata_csv_layout: str = "head",
) -> RowChunkEncoding:
    """Encode a table into independent views that share one canonical parent table_ref."""
    unknown = set(views) - set(TABLE_VIEWS)
    if unknown:
        raise ValueError(f"Invalid table views: {sorted(unknown)}")
    if not views:
        raise ValueError("views must not be empty")
    if summary_max_chars <= 0:
        raise ValueError("summary_max_chars must be greater than 0")
    if metadata_csv_layout not in METADATA_CSV_LAYOUTS:
        raise ValueError(f"Invalid metadata_csv_layout: {metadata_csv_layout!r}")

    row_result = (
        encode_row_chunks(table, document, company_name, config=row_config)
        if ROW_CHUNK_VIEW in views
        else RowChunkEncoding(
            chunks=(),
            safety_split_windows=0,
            safety_split_count=0,
            truncated_chunks=0,
            truncations=(),
        )
    )
    chunks = list(row_result.chunks)
    next_index = len(chunks)
    if METADATA_CSV_VIEW in views:
        chunks.append(
            EncodedChunk(
                ticker=table.ticker,
                year=table.year,
                doc_name=table.doc_name,
                table_id=table.table_id,
                chunk_index=next_index,
                row_start=0,
                row_end=len(table.rows),
                text=metadata_csv_text(
                    table,
                    document,
                    company_name,
                    max_csv_chars=summary_max_chars,
                    layout=metadata_csv_layout,
                ),
                view=METADATA_CSV_VIEW,
            )
        )
        next_index += 1
    if SEMANTIC_LABELS_VIEW in views:
        chunks.append(
            EncodedChunk(
                ticker=table.ticker,
                year=table.year,
                doc_name=table.doc_name,
                table_id=table.table_id,
                chunk_index=next_index,
                row_start=0,
                row_end=len(table.rows),
                text=semantic_labels_text(
                    table,
                    document,
                    company_name,
                    max_chars=summary_max_chars,
                    max_context_chars=row_config.context_max_chars,
                ),
                view=SEMANTIC_LABELS_VIEW,
            )
        )
    return RowChunkEncoding(
        chunks=tuple(chunks),
        safety_split_windows=row_result.safety_split_windows,
        safety_split_count=row_result.safety_split_count,
        truncated_chunks=row_result.truncated_chunks,
        truncations=row_result.truncations,
    )
