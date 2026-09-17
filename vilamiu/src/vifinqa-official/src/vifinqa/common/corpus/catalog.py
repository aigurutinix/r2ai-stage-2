
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TABLE_ID_RE = re.compile(r"^table_(\d+)\.csv$")


@dataclass(frozen=True, slots=True)
class DocumentRef:
    ticker: str
    year: str
    doc_name: str
    doc_dir: Path
    text_path: Path | None
    tables_dir: Path | None
    table_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def has_text(self) -> bool:
        return self.text_path is not None

    def table_csv_path(self, table_id: int) -> Path:
        if self.tables_dir is None:
            raise ValueError(f"{self.doc_name} extracted-table directory is missing")
        return self.tables_dir / f"table_{table_id}.csv"


def _find_single(doc_dir: Path, pattern: str) -> Path | None:
    matches = sorted(doc_dir.glob(pattern))
    return matches[0] if matches else None


def _scan_table_ids(tables_dir: Path | None) -> tuple[int, ...]:
    if tables_dir is None or not tables_dir.is_dir():
        return ()
    ids = []
    for p in tables_dir.iterdir():
        m = _TABLE_ID_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return tuple(sorted(ids))


def scan_document(doc_dir: Path, ticker: str, year: str) -> DocumentRef:
    text_path = _find_single(doc_dir, "*_extracted.txt")
    tables_dir = _find_single(doc_dir, "*_extracted_tables")
    if tables_dir is None:
        # Accept the earlier Vietnamese directory suffix for compatible corpora.
        tables_dir = _find_single(doc_dir, "*_extracted_bang")
    tables_dir = tables_dir if tables_dir is not None and tables_dir.is_dir() else None
    return DocumentRef(
        ticker=ticker,
        year=year,
        doc_name=doc_dir.name,
        doc_dir=doc_dir,
        text_path=text_path,
        tables_dir=tables_dir,
        table_ids=_scan_table_ids(tables_dir),
    )


def build_doc_name_lookup(data_root: Path) -> dict[str, DocumentRef]:
    return {d.doc_name: d for d in scan_catalog(data_root)}


def scan_catalog(data_root: Path) -> list[DocumentRef]:
    docs: list[DocumentRef] = []
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    for ticker_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for year_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
            for doc_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
                docs.append(scan_document(doc_dir, ticker=ticker_dir.name, year=year_dir.name))
    return docs
