"""OCR HTML-table parsing and CSV evidence materialization."""

import csv
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from functools import lru_cache
from pathlib import Path
import re


PAGE_RE = re.compile(r"=====\s*PAGE\s+(\d+)\s*=====", re.IGNORECASE)
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
SCOPE_RE = re.compile(r"_(consolidated|separate|aggregated)(?:_\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class ReportIdentity:
    report_id: str
    ticker: str
    year: int
    scope: str
    source_path: str


def span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.parts: list[str] | None = None
        self.column = 0
        self.rowspan = 1
        self.colspan = 1
        self.spans: dict[int, tuple[int, str]] = {}
        self.occupied: set[int] = set()

    def put(self, column: int, value: str) -> None:
        if self.row is None:
            return
        if len(self.row) <= column:
            self.row.extend([""] * (column + 1 - len(self.row)))
        self.row[column] = value

    def skip_spans(self) -> None:
        while self.column in self.occupied:
            self.column += 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row, self.column = [], 0
            self.occupied = set(self.spans)
            for column, (remaining, value) in list(self.spans.items()):
                self.put(column, value)
                if remaining == 1:
                    del self.spans[column]
                else:
                    self.spans[column] = remaining - 1, value
        elif tag in {"td", "th"} and self.row is not None:
            attributes = dict(attrs)
            self.parts = []
            self.rowspan = span(attributes.get("rowspan"))
            self.colspan = span(attributes.get("colspan"))

    def handle_data(self, data: str) -> None:
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.row is not None and self.parts is not None:
            value = " ".join("".join(self.parts).split())
            for _ in range(self.colspan):
                self.skip_spans()
                self.put(self.column, value)
                if self.rowspan > 1:
                    self.spans[self.column] = self.rowspan - 1, value
                self.column += 1
            self.parts = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = None


def parse_report_identity(report_path: Path, dataset_root: Path) -> ReportIdentity:
    relative = report_path.relative_to(dataset_root)
    try:
        marker = relative.parts.index("financial_statements")
        ticker, year, report_dir = relative.parts[marker + 1:marker + 4]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Unexpected report path: {relative}") from error
    scope = SCOPE_RE.search(report_dir)
    return ReportIdentity(
        report_path.stem.removesuffix("_extracted"), ticker, int(year),
        scope.group(1).lower() if scope else "unclassified", relative.as_posix(),
    )


def parse_table_rows(raw_html: str) -> list[list[str]]:
    parser = TableParser()
    parser.feed(unescape(raw_html))
    parser.close()
    return parser.rows


def iter_report_paths(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "financial_statements").rglob("*_extracted.txt"))


def extract_rows_at_line(report_text: str, start_line: int) -> list[list[str]]:
    line, cursor = 1, 0
    for match in TABLE_RE.finditer(report_text):
        line += report_text.count("\n", cursor, match.start())
        cursor = match.start()
        if line == start_line:
            return parse_table_rows(match.group(0))
    raise KeyError(f"No HTML table begins at source line {start_line}")


@lru_cache(maxsize=256)
def cached_report_text(path_text: str) -> str:
    """Reuse immutable OCR text across multiple evidence tables."""
    return Path(path_text).read_text(encoding="utf-8")


def materialize(report_path: Path, start_line: int, table_id: str, csv_path: Path) -> None:
    rows = extract_rows_at_line(cached_report_text(str(report_path)), start_line)
    if not rows:
        raise ValueError(f"Table {table_id} is empty")
    width = max(len(row) for row in rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([row + [""] * (width - len(row)) for row in rows])
