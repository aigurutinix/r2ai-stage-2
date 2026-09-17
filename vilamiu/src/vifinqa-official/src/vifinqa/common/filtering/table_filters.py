
from __future__ import annotations

import re

from vifinqa.constants import MIN_NUMERIC_CELLS, MIN_ROWS
from vifinqa.common.corpus.table import TableAsset

_NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")


def is_numeric_cell(cell: str) -> bool:
    cell = cell.strip()
    if not cell or cell == "-":
        return False
    return bool(_NUMERIC_RE.match(cell))


def count_numeric_cells(table: TableAsset) -> int:
    return sum(1 for row in table.rows for cell in row if is_numeric_cell(cell))


def is_table_eligible(table: TableAsset) -> bool:
    if table.n_rows < MIN_ROWS:
        return False
    return count_numeric_cells(table) >= MIN_NUMERIC_CELLS
