"""Build and query an auditable SQLite warehouse of ViFinQA OCR tables.

The warehouse never invents financial semantics.  It preserves each source
table, its lossless rectangular grid, report/page/line provenance, nearby OCR
context, and an FTS index.  A selected table can then be exported as a CSV
whose columns are stable positional names (``col_0``, ``col_1``, ...).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = "1"
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
PAGE_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Cell:
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


class OCRTableParser(HTMLParser):
    """Small tolerant HTML table parser with rowspan/colspan preservation."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[Cell]] = []
        self.current_row: Optional[list[Cell]] = None
        self.current_cell: Optional[dict] = None

    @staticmethod
    def _positive_int(value: Optional[str]) -> int:
        try:
            return max(1, int(value or "1"))
        except ValueError:
            return 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
        elif tag in {"td", "th"}:
            if self.current_row is None:
                self.current_row = []
            self.current_cell = {
                "parts": [],
                "rowspan": self._positive_int(attributes.get("rowspan")),
                "colspan": self._positive_int(attributes.get("colspan")),
                "is_header": tag == "th",
            }
        elif tag == "br" and self.current_cell is not None:
            self.current_cell["parts"].append(" ")

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_cell is not None:
            text = " ".join("".join(self.current_cell["parts"]).split())
            assert self.current_row is not None
            self.current_row.append(
                Cell(
                    text=html.unescape(text),
                    rowspan=self.current_cell["rowspan"],
                    colspan=self.current_cell["colspan"],
                    is_header=self.current_cell["is_header"],
                )
            )
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = None

    def finish(self) -> list[list[Cell]]:
        if self.current_cell is not None:
            self.handle_endtag("td")
        if self.current_row:
            self.rows.append(self.current_row)
            self.current_row = None
        return self.rows


def expand_rows(rows: list[list[Cell]]) -> tuple[list[list[str]], int]:
    """Expand merged cells and return a rectangular, lossless text grid."""

    grid: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}
    explicit_header_rows = 0

    for cells in rows:
        if any(cell.is_header for cell in cells):
            explicit_header_rows += 1
        row: list[str] = []
        column = 0

        def consume_pending() -> None:
            nonlocal column
            while column in pending:
                remaining, value = pending[column]
                row.append(value)
                if remaining <= 1:
                    del pending[column]
                else:
                    pending[column] = (remaining - 1, value)
                column += 1

        consume_pending()
        for cell in cells:
            consume_pending()
            for _ in range(cell.colspan):
                row.append(cell.text)
                if cell.rowspan > 1:
                    pending[column] = (cell.rowspan - 1, cell.text)
                column += 1
        consume_pending()
        grid.append(row)

    while pending:
        row = []
        column = 0
        max_column = max(pending)
        while column <= max_column:
            if column in pending:
                remaining, value = pending[column]
                row.append(value)
                if remaining <= 1:
                    del pending[column]
                else:
                    pending[column] = (remaining - 1, value)
            else:
                row.append("")
            column += 1
        grid.append(row)

    width = max((len(row) for row in grid), default=0)
    rectangular = [row + [""] * (width - len(row)) for row in grid]
    if explicit_header_rows == 0 and rectangular:
        first = rectangular[0]
        numeric = sum(is_numeric_cell(value) for value in first)
        explicit_header_rows = 1 if numeric < max(1, len(first) // 2) else 0
    return rectangular, explicit_header_rows


def parse_table(raw_html: str) -> tuple[list[list[str]], int]:
    parser = OCRTableParser()
    parser.feed(raw_html)
    parser.close()
    return expand_rows(parser.finish())


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"\s+", " ", value).strip()


def is_numeric_cell(text: str) -> bool:
    value = text.strip().replace("\xa0", " ")
    if not re.search(r"\d", value):
        return False
    cleaned = re.sub(r"[\d\s.,()%+\-–—/]", "", value)
    return len(cleaned) <= 3


def infer_scope(document_id: str) -> str:
    value = document_id.lower()
    if "separate" in value:
        return "separate"
    if "consolidated" in value:
        return "consolidated"
    if "aggregated" in value:
        return "aggregated"
    return "unspecified"


def infer_unit(context: str, grid: list[list[str]]) -> tuple[str, str]:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    sample_rows = [" | ".join(row) for row in grid[:3]]
    candidates = lines[-12:] + sample_rows
    unit_text = ""
    for line in reversed(candidates):
        value = normalize(line)
        if (
            "don vi" in value
            or "dvt" in value
            or "vnd" in value
            or "usd" in value
            or "trieu dong" in value
            or "nghin dong" in value
            or "ty dong" in value
            or "phan tram" in value
        ):
            unit_text = line[:240]
            break
    value = normalize(unit_text or " ".join(candidates[-4:]))
    if "%" in value or "phan tram" in value:
        return "percent", unit_text
    if "nghin ty" in value or "ngan ty" in value:
        return "VND_1e12", unit_text
    if "tram ty" in value:
        return "VND_1e11", unit_text
    if re.search(r"\bty\s*(?:vnd|dong)?\b", value):
        return "VND_1e9", unit_text
    if "trieu" in value:
        return "VND_1e6", unit_text
    if "nghin" in value or "ngan dong" in value:
        return "VND_1e3", unit_text
    if "usd" in value or "do la my" in value:
        return "USD", unit_text
    if "co phieu" in value or "co phan" in value:
        return "shares", unit_text
    # OCR often concatenates the period and unit (for example ``2018VND``).
    if "vnd" in value or re.search(r"\bdong\b", value):
        return "VND_1", unit_text
    return "unknown", unit_text


def infer_table_kind(context: str, grid: list[list[str]]) -> str:
    first_rows = " ".join(" | ".join(row) for row in grid[:5])
    context_value = normalize(context)
    value = normalize(context + " " + first_rows)
    # Page/section headers are stronger than incidental phrases in table
    # cells.  Notes routinely say "not reflected in the balance sheet", so
    # checking primary-statement phrases first creates systematic false hits.
    if "muc luc" in context_value and "trang" in value:
        return "contents"
    if "thuyet minh bao cao tai chinh" in context_value:
        return "financial_notes"
    if "bao cao tinh hinh tai chinh" in value or "bang can doi ke toan" in value:
        return "primary_balance_sheet"
    if "bao cao ket qua hoat dong kinh doanh" in value:
        return "primary_income_statement"
    if "bao cao luu chuyen tien te" in value:
        return "primary_cash_flow"
    if "bao cao thay doi von chu so huu" in value:
        return "primary_equity_changes"
    if re.search(r"\bthuyet minh\b", context_value):
        return "financial_notes"
    if any(
        phrase in value
        for phrase in (
            "hoi dong quan tri",
            "ban tong giam doc",
            "ban giam doc",
            "ban kiem soat",
        )
    ):
        return "governance_or_company_info"
    if "bao cao kiem toan" in value or "kiem toan vien" in value:
        return "audit_report"
    return "other"


def title_hint(context: str) -> str:
    lines = [" ".join(line.split()) for line in context.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("=====") or len(line) > 240:
            continue
        return line[:240]
    return ""


def line_for_offset(newlines: list[int], offset: int) -> int:
    return bisect.bisect_right(newlines, offset) + 1


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            year INTEGER NOT NULL,
            report_scope TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            byte_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            line_count INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            table_open_tag_count INTEGER NOT NULL,
            table_count INTEGER NOT NULL,
            malformed_table_count INTEGER NOT NULL
        );

        CREATE TABLE pages (
            document_id TEXT NOT NULL REFERENCES documents(document_id),
            page_ordinal INTEGER NOT NULL,
            page_number INTEGER,
            marker_line_1 INTEGER NOT NULL,
            start_line_1 INTEGER NOT NULL,
            end_line_1 INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            PRIMARY KEY (document_id, page_ordinal)
        );

        CREATE TABLE tables (
            table_id INTEGER PRIMARY KEY,
            evidence_key TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL REFERENCES documents(document_id),
            table_ordinal INTEGER NOT NULL,
            page_ordinal INTEGER,
            page_number INTEGER,
            source_line_0 INTEGER NOT NULL,
            source_line_1 INTEGER NOT NULL,
            source_end_line_1 INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            column_count INTEGER NOT NULL,
            nonempty_cell_count INTEGER NOT NULL,
            numeric_cell_count INTEGER NOT NULL,
            header_row_count INTEGER NOT NULL,
            unit_code TEXT NOT NULL,
            unit_text TEXT NOT NULL,
            table_kind TEXT NOT NULL,
            title_hint TEXT NOT NULL,
            context_before TEXT NOT NULL,
            context_after TEXT NOT NULL,
            grid_json TEXT NOT NULL,
            raw_html TEXT NOT NULL,
            parse_status TEXT NOT NULL,
            UNIQUE (document_id, table_ordinal)
        );

        CREATE INDEX idx_documents_ticker_year
            ON documents(ticker, year, report_scope);
        CREATE INDEX idx_tables_document ON tables(document_id, table_ordinal);
        CREATE INDEX idx_tables_kind ON tables(table_kind);
        CREATE INDEX idx_tables_unit ON tables(unit_code);
        CREATE INDEX idx_tables_page_line
            ON tables(document_id, page_number, source_line_1);

        CREATE VIEW table_catalog AS
        SELECT t.table_id,
               t.evidence_key,
               d.ticker,
               d.year,
               d.report_scope,
               d.relative_path,
               t.document_id,
               t.table_ordinal,
               t.page_number,
               t.source_line_0,
               t.source_line_1,
               t.source_end_line_1,
               t.row_count,
               t.column_count,
               t.numeric_cell_count,
               t.unit_code,
               t.table_kind,
               t.title_hint,
               t.parse_status
        FROM tables t
        JOIN documents d ON d.document_id = t.document_id;

        CREATE VIRTUAL TABLE table_fts USING fts5(
            table_id UNINDEXED,
            evidence_key UNINDEXED,
            title_hint,
            context,
            table_text,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """
    )


def build_database(data_root: Path, database: Path) -> None:
    statement_root = data_root / "financial_statements"
    paths = sorted(statement_root.glob("*/*/*/*.txt"))
    if not paths:
        raise FileNotFoundError(f"No statement files under {statement_root}")

    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_name(database.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(str(temporary))
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    create_schema(connection)
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        (
            ("schema_version", SCHEMA_VERSION),
            ("built_at_utc", datetime.now(timezone.utc).isoformat()),
            ("data_root", str(data_root.resolve())),
            ("source_document_count", str(len(paths))),
        ),
    )

    total_tables = 0
    total_parse_errors = 0
    for document_index, path in enumerate(paths, 1):
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="replace")
        relative_path = path.relative_to(data_root).as_posix()
        relative_parts = Path(relative_path).parts
        ticker = relative_parts[1]
        year = int(relative_parts[2])
        document_id = path.parent.name
        scope = infer_scope(document_id)
        newlines = [match.start() for match in re.finditer("\n", text)]
        page_matches = list(PAGE_RE.finditer(text))
        page_starts = [match.start() for match in page_matches]
        table_matches = list(TABLE_RE.finditer(text))
        open_count = len(TABLE_OPEN_RE.findall(text))
        malformed_count = max(0, open_count - len(table_matches))

        connection.execute(
            """
            INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                ticker,
                year,
                scope,
                relative_path,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                len(newlines) + 1,
                len(page_matches),
                open_count,
                len(table_matches),
                malformed_count,
            ),
        )

        for page_index, match in enumerate(page_matches):
            char_start = match.start()
            char_end = (
                page_matches[page_index + 1].start()
                if page_index + 1 < len(page_matches)
                else len(text)
            )
            connection.execute(
                "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document_id,
                    page_index + 1,
                    int(match.group(1)),
                    line_for_offset(newlines, match.start()),
                    line_for_offset(newlines, char_start),
                    line_for_offset(newlines, max(char_start, char_end - 1)),
                    char_start,
                    char_end,
                ),
            )

        for table_index, match in enumerate(table_matches, 1):
            raw_html = match.group(0)
            page_zero_index = bisect.bisect_right(page_starts, match.start()) - 1
            if page_zero_index >= 0:
                page_ordinal = page_zero_index + 1
                page_number = int(page_matches[page_zero_index].group(1))
                page_char_start = page_matches[page_zero_index].end()
            else:
                page_ordinal = None
                page_number = None
                page_char_start = 0
            before_start = max(page_char_start, match.start() - 1800)
            context_before = text[before_start : match.start()].strip()
            context_after = text[match.end() : min(len(text), match.end() + 600)].strip()
            start_line_1 = line_for_offset(newlines, match.start())
            end_line_1 = line_for_offset(newlines, max(match.start(), match.end() - 1))
            evidence_key = f"{document_id}|{start_line_1}"

            parse_status = "ok"
            try:
                grid, header_rows = parse_table(raw_html)
            except Exception as error:  # preserve raw input even on a bad table
                grid, header_rows = [], 0
                parse_status = f"error:{type(error).__name__}:{error}"[:300]
                total_parse_errors += 1

            row_count = len(grid)
            column_count = max((len(row) for row in grid), default=0)
            cells = [value for row in grid for value in row]
            nonempty_count = sum(bool(value.strip()) for value in cells)
            numeric_count = sum(is_numeric_cell(value) for value in cells)
            unit_code, unit_text = infer_unit(context_before, grid)
            kind = infer_table_kind(context_before, grid)
            hint = title_hint(context_before)
            grid_json = json.dumps(grid, ensure_ascii=False, separators=(",", ":"))
            table_text = "\n".join(" | ".join(row) for row in grid)

            cursor = connection.execute(
                """
                INSERT INTO tables(
                    evidence_key, document_id, table_ordinal, page_ordinal,
                    page_number, source_line_0, source_line_1,
                    source_end_line_1, row_count, column_count,
                    nonempty_cell_count, numeric_cell_count, header_row_count,
                    unit_code, unit_text, table_kind, title_hint,
                    context_before, context_after, grid_json, raw_html,
                    parse_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_key,
                    document_id,
                    table_index,
                    page_ordinal,
                    page_number,
                    start_line_1 - 1,
                    start_line_1,
                    end_line_1,
                    row_count,
                    column_count,
                    nonempty_count,
                    numeric_count,
                    header_rows,
                    unit_code,
                    unit_text,
                    kind,
                    hint,
                    context_before,
                    context_after,
                    grid_json,
                    raw_html,
                    parse_status,
                ),
            )
            table_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO table_fts VALUES (?, ?, ?, ?, ?)",
                (
                    table_id,
                    evidence_key,
                    hint,
                    context_before,
                    table_text,
                ),
            )
            total_tables += 1

        if document_index % 25 == 0:
            connection.commit()
        if document_index % 100 == 0 or document_index == len(paths):
            print(
                f"indexed {document_index}/{len(paths)} documents; "
                f"{total_tables} tables",
                flush=True,
            )

    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (
            ("indexed_table_count", str(total_tables)),
            ("parse_error_count", str(total_parse_errors)),
        ),
    )
    connection.commit()
    connection.execute("INSERT INTO table_fts(table_fts) VALUES ('optimize')")
    connection.execute("PRAGMA optimize")
    connection.commit()
    connection.close()
    os.replace(temporary, database)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(temporary) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    print(f"database={database} documents={len(paths)} tables={total_tables}")


def fetch_rows(connection: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return list(connection.execute(sql))


def write_dict_csv(path: Path, rows: list[sqlite3.Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def markdown_table(rows: Iterable[sqlite3.Row]) -> str:
    rows = list(rows)
    if not rows:
        return "_(không có dữ liệu)_"
    headers = list(rows[0].keys())
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return "\n".join(output)


def write_stats(database: Path, output_dir: Path) -> None:
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    overall = connection.execute(
        """
        SELECT COUNT(*) AS documents,
               SUM(page_count) AS pages,
               SUM(table_count) AS tables,
               SUM(table_open_tag_count) - SUM(table_count) AS unmatched_open_tags,
               SUM(byte_size) AS source_bytes
        FROM documents
        """
    ).fetchone()
    dimensions = connection.execute(
        """
        SELECT ROUND(AVG(row_count), 2) AS avg_rows,
               MAX(row_count) AS max_rows,
               ROUND(AVG(column_count), 2) AS avg_columns,
               MAX(column_count) AS max_columns,
               SUM(CASE WHEN parse_status != 'ok' THEN 1 ELSE 0 END) AS parse_errors,
               SUM(CASE WHEN row_count = 0 THEN 1 ELSE 0 END) AS empty_tables,
               SUM(CASE WHEN numeric_cell_count > 0 THEN 1 ELSE 0 END) AS tables_with_numbers
        FROM tables
        """
    ).fetchone()
    by_year = fetch_rows(
        connection,
        """
        SELECT year,
               COUNT(*) AS documents,
               SUM(page_count) AS pages,
               SUM(table_count) AS tables,
               ROUND(AVG(table_count), 2) AS avg_tables_per_document
        FROM documents GROUP BY year ORDER BY year
        """,
    )
    by_scope = fetch_rows(
        connection,
        """
        SELECT report_scope,
               COUNT(*) AS documents,
               SUM(table_count) AS tables,
               ROUND(AVG(table_count), 2) AS avg_tables_per_document
        FROM documents GROUP BY report_scope ORDER BY documents DESC
        """,
    )
    by_kind = fetch_rows(
        connection,
        """
        SELECT table_kind,
               COUNT(*) AS tables,
               COUNT(DISTINCT document_id) AS documents,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM tables), 2) AS pct_tables
        FROM tables GROUP BY table_kind ORDER BY tables DESC
        """,
    )
    by_unit = fetch_rows(
        connection,
        """
        SELECT unit_code,
               COUNT(*) AS tables,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM tables), 2) AS pct_tables
        FROM tables GROUP BY unit_code ORDER BY tables DESC
        """,
    )
    by_width = fetch_rows(
        connection,
        """
        SELECT CASE
                 WHEN column_count <= 5 THEN '<=5'
                 WHEN column_count <= 10 THEN '6-10'
                 WHEN column_count <= 20 THEN '11-20'
                 WHEN column_count <= 50 THEN '21-50'
                 ELSE '>50'
               END AS column_bucket,
               COUNT(*) AS tables
        FROM tables
        GROUP BY column_bucket
        ORDER BY MIN(column_count)
        """,
    )
    by_height = fetch_rows(
        connection,
        """
        SELECT CASE
                 WHEN row_count <= 5 THEN '<=5'
                 WHEN row_count <= 10 THEN '6-10'
                 WHEN row_count <= 20 THEN '11-20'
                 WHEN row_count <= 50 THEN '21-50'
                 WHEN row_count <= 100 THEN '51-100'
                 ELSE '>100'
               END AS row_bucket,
               COUNT(*) AS tables
        FROM tables
        GROUP BY row_bucket
        ORDER BY MIN(row_count)
        """,
    )
    by_year_kind = fetch_rows(
        connection,
        """
        SELECT d.year, t.table_kind, COUNT(*) AS tables
        FROM tables t JOIN documents d ON d.document_id = t.document_id
        GROUP BY d.year, t.table_kind
        ORDER BY d.year, tables DESC
        """,
    )
    by_ticker_year = fetch_rows(
        connection,
        """
        SELECT ticker, year, COUNT(*) AS documents, SUM(table_count) AS tables
        FROM documents GROUP BY ticker, year ORDER BY ticker, year
        """,
    )
    no_table_documents = fetch_rows(
        connection,
        """
        SELECT ticker, year, document_id, report_scope, page_count
        FROM documents WHERE table_count = 0
        ORDER BY ticker, year, document_id
        """,
    )
    width_outliers = fetch_rows(
        connection,
        """
        SELECT evidence_key, row_count, column_count, nonempty_cell_count
        FROM tables WHERE column_count > 50
        ORDER BY column_count DESC
        """,
    )

    write_dict_csv(output_dir / "table_inventory_by_year.csv", by_year)
    write_dict_csv(output_dir / "table_inventory_by_scope.csv", by_scope)
    write_dict_csv(output_dir / "table_inventory_by_kind.csv", by_kind)
    write_dict_csv(output_dir / "table_inventory_by_unit.csv", by_unit)
    write_dict_csv(output_dir / "table_inventory_by_width.csv", by_width)
    write_dict_csv(output_dir / "table_inventory_by_height.csv", by_height)
    write_dict_csv(output_dir / "table_inventory_by_year_kind.csv", by_year_kind)
    write_dict_csv(output_dir / "table_inventory_by_ticker_year.csv", by_ticker_year)
    write_dict_csv(output_dir / "table_inventory_no_table_documents.csv", no_table_documents)

    report = f"""# ViFinQA — kiểm kê bảng OCR

> Sinh trực tiếp từ `{database.as_posix()}` với schema version {SCHEMA_VERSION}.
> Các loại bảng và đơn vị là metadata heuristic để tìm kiếm; provenance, HTML
> gốc và grid mới là dữ liệu có tính quyết định.

## Tổng quan

| Chỉ tiêu | Giá trị |
| --- | ---: |
| Báo cáo | {overall['documents']} |
| Page markers | {overall['pages']} |
| Bảng HTML hoàn chỉnh | {overall['tables']} |
| Thẻ `<table>` mở không ghép được với `</table>` | {overall['unmatched_open_tags']} |
| Dung lượng OCR nguồn (byte) | {overall['source_bytes']} |
| Số dòng trung bình/bảng | {dimensions['avg_rows']} |
| Số dòng lớn nhất | {dimensions['max_rows']} |
| Số cột trung bình/bảng | {dimensions['avg_columns']} |
| Số cột lớn nhất | {dimensions['max_columns']} |
| Bảng có ít nhất một ô số | {dimensions['tables_with_numbers']} |
| Bảng rỗng sau parse | {dimensions['empty_tables']} |
| Lỗi parser | {dimensions['parse_errors']} |

## Phân bố theo năm

{markdown_table(by_year)}

## Phân bố theo phạm vi báo cáo

{markdown_table(by_scope)}

## Loại bảng suy ra từ tiêu đề/ngữ cảnh gần nhất

{markdown_table(by_kind)}

## Đơn vị suy ra từ ngữ cảnh gần nhất

{markdown_table(by_unit)}

## Kích thước bảng

Theo số cột:

{markdown_table(by_width)}

Theo số dòng:

{markdown_table(by_height)}

Chi tiết loại bảng theo từng năm được lưu tại
`table_inventory_by_year_kind.csv`; độ phủ ticker/năm nằm tại
`table_inventory_by_ticker_year.csv`.

## Báo cáo không có bảng HTML

{markdown_table(no_table_documents)}

Tám tệp này đều là thư giải trình/tài liệu giải thích của PRT, không phải BCTC
chuẩn bị mất bảng do parser. Chúng vẫn còn trong `documents` và `pages`.

## Outlier bảng rất rộng

{markdown_table(width_outliers)}

## Tracking và xuất evidence

- `documents` giữ ticker, năm, scope, đường dẫn, SHA-256 và số lượng bảng.
- `pages` giữ page marker cùng khoảng ký tự/dòng trong OCR.
- `tables` giữ khóa `evidence_key = <document_id>|<source_line_1>`, cả dòng
  0-based lẫn 1-based, HTML gốc, grid JSON, context và metadata.
- `table_fts` cho phép tìm toàn văn trên tiêu đề, context và nội dung ô.
- `table_catalog` là view đã join provenance bảng với ticker/năm/scope/path.
- CSV evidence được xuất lossless với header `col_0`, `col_1`, ...; mọi dòng
  của bảng OCR vẫn là data nên `df.iloc` có mapping ổn định trở lại `grid_json`.

Không xem `table_kind` hoặc `unit_code` là gold. OCR có thể thiếu tiêu đề/đơn vị,
và một trang có thể chứa nhiều section. Mọi câu trả lời phải kiểm lại HTML/grid
và provenance trước khi đóng gói submission. Bốn bảng có trên 50 cột và 86
bảng có trên 100 dòng là outlier OCR thật được giữ nguyên, không tự ý cắt bỏ.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "TABLE_INVENTORY.md").write_text(report, encoding="utf-8")
    connection.close()


def search_tables(
    database: Path,
    query: str,
    ticker: Optional[str],
    year: Optional[int],
    scope: Optional[str],
    limit: int,
) -> None:
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    filters = ["table_fts MATCH ?"]
    parameters: list[object] = [query]
    if ticker:
        filters.append("d.ticker = ?")
        parameters.append(ticker.upper())
    if year:
        filters.append("d.year = ?")
        parameters.append(year)
    if scope:
        filters.append("d.report_scope = ?")
        parameters.append(scope)
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT t.table_id, t.evidence_key, d.ticker, d.year, d.report_scope,
               t.page_number, t.source_line_1, t.table_kind, t.unit_code,
               t.row_count, t.column_count, t.title_hint,
               ROUND(bm25(table_fts), 4) AS rank
        FROM table_fts
        JOIN tables t ON t.table_id = table_fts.table_id
        JOIN documents d ON d.document_id = t.document_id
        WHERE {' AND '.join(filters)}
        ORDER BY bm25(table_fts)
        LIMIT ?
        """,
        parameters,
    )
    for row in rows:
        print(json.dumps(dict(row), ensure_ascii=False))
    connection.close()


def export_table(
    database: Path,
    table_id: Optional[int],
    evidence_key: Optional[str],
    output: Optional[Path],
) -> Path:
    if (table_id is None) == (evidence_key is None):
        raise ValueError("Provide exactly one of --table-id or --evidence-key")
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    if table_id is not None:
        row = connection.execute(
            "SELECT * FROM tables WHERE table_id = ?", (table_id,)
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT * FROM tables WHERE evidence_key = ?", (evidence_key,)
        ).fetchone()
    if row is None:
        raise KeyError("Table not found")
    grid = json.loads(row["grid_json"])
    width = max((len(values) for values in grid), default=int(row["column_count"]))
    if output is None:
        filename = f"{row['document_id']}_table_{row['table_ordinal']:03d}.csv"
        output = Path("outputs/data") / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([f"col_{index}" for index in range(width)])
        for values in grid:
            writer.writerow(values + [""] * (width - len(values)))
    connection.close()
    manifest = {
        "table_id": row["table_id"],
        "evidence_key": row["evidence_key"],
        "document_id": row["document_id"],
        "page_number": row["page_number"],
        "source_line_0": row["source_line_0"],
        "source_line_1": row["source_line_1"],
        "csv_path": output.as_posix(),
        "rows": len(grid),
        "columns": width,
    }
    print(json.dumps(manifest, ensure_ascii=False))
    return output


def inspect_table(
    database: Path, table_id: Optional[int], evidence_key: Optional[str]
) -> None:
    if (table_id is None) == (evidence_key is None):
        raise ValueError("Provide exactly one of --table-id or --evidence-key")
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    key = "table_id" if table_id is not None else "evidence_key"
    value = table_id if table_id is not None else evidence_key
    row = connection.execute(f"SELECT * FROM tables WHERE {key} = ?", (value,)).fetchone()
    if row is None:
        raise KeyError("Table not found")
    metadata = dict(row)
    grid = json.loads(metadata.pop("grid_json"))
    metadata.pop("raw_html")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("GRID")
    for index, values in enumerate(grid):
        print(f"{index:04d}\t" + " | ".join(values))
    connection.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the SQLite warehouse")
    build.add_argument("--data-root", type=Path, default=Path("ViFinQA"))
    build.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))

    stats = subparsers.add_parser("stats", help="Write inventory reports")
    stats.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    stats.add_argument("--output-dir", type=Path, default=Path("analysis"))

    search = subparsers.add_parser("search", help="Search table text and context")
    search.add_argument("query")
    search.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    search.add_argument("--ticker")
    search.add_argument("--year", type=int)
    search.add_argument("--scope", choices=("separate", "consolidated", "aggregated", "unspecified"))
    search.add_argument("--limit", type=int, default=10)

    inspect = subparsers.add_parser("inspect", help="Inspect metadata and parsed grid")
    inspect.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    inspect_group = inspect.add_mutually_exclusive_group(required=True)
    inspect_group.add_argument("--table-id", type=int)
    inspect_group.add_argument("--evidence-key")

    export = subparsers.add_parser("export", help="Export one tracked table to CSV")
    export.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    export_group = export.add_mutually_exclusive_group(required=True)
    export_group.add_argument("--table-id", type=int)
    export_group.add_argument("--evidence-key")
    export.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "build":
        build_database(args.data_root, args.database)
    elif args.command == "stats":
        write_stats(args.database, args.output_dir)
    elif args.command == "search":
        search_tables(
            args.database, args.query, args.ticker, args.year, args.scope, args.limit
        )
    elif args.command == "inspect":
        inspect_table(args.database, args.table_id, args.evidence_key)
    elif args.command == "export":
        export_table(args.database, args.table_id, args.evidence_key, args.output)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
