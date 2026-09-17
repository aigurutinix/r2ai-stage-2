"""Lookup layer over artifacts/tables.parquet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TableKey:
    doc_name: str
    table_id: int


class TableStore:
    """Random access to any of the 146,246 extracted tables."""

    def __init__(self, frame) -> None:
        self.frame = frame
        self._by_key = {
            (row.doc_name, int(row.table_id)): index
            for index, row in enumerate(frame.itertuples(index=False))
        }

    @classmethod
    def load(cls, parquet_path: Path) -> "TableStore":
        import pandas as pd

        return cls(pd.read_parquet(parquet_path))

    def __contains__(self, key: TableKey) -> bool:
        return (key.doc_name, key.table_id) in self._by_key

    def rows(self, key: TableKey) -> list[list[str]]:
        index = self._by_key.get((key.doc_name, key.table_id))
        if index is None:
            raise KeyError(f"unknown table: {key.doc_name}|{key.table_id}")
        return json.loads(self.frame.iloc[index].rows_json)

    def meta(self, key: TableKey):
        index = self._by_key.get((key.doc_name, key.table_id))
        if index is None:
            raise KeyError(f"unknown table: {key.doc_name}|{key.table_id}")
        return self.frame.iloc[index]

    @cached_property
    def doc_names(self) -> set[str]:
        return set(self.frame.doc_name.unique())
