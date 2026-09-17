
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TableAsset:
    ticker: str
    year: str
    doc_name: str
    table_id: int
    csv_path: Path
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.header)

    @property
    def table_ref(self) -> str:
        return f"{self.doc_name}|table_{self.table_id}"

    def raw_csv_text(self) -> str:
        return self.csv_path.read_text(encoding="utf-8")


def load_table(csv_path: Path, ticker: str, year: str, doc_name: str, table_id: int) -> TableAsset:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = [tuple(row) for row in reader]
    if not rows:
        header: tuple[str, ...] = ()
        body: tuple[tuple[str, ...], ...] = ()
    else:
        header = rows[0]
        body = tuple(rows[1:])
    return TableAsset(
        ticker=ticker,
        year=year,
        doc_name=doc_name,
        table_id=table_id,
        csv_path=csv_path,
        header=header,
        rows=body,
    )
