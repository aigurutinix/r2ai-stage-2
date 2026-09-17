
from __future__ import annotations

from vifinqa.common.corpus.table import TableAsset


def _normalize(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _match_indices(candidates: list[str], label: str) -> list[int]:
    norm_label = _normalize(label)
    if not norm_label:
        return []
    exact = [i for i, c in enumerate(candidates) if _normalize(c) == norm_label]
    if exact:
        return exact
    return [i for i, c in enumerate(candidates) if norm_label in _normalize(c)]


def locate_cell(table: TableAsset, row_label: str, column_label: str) -> tuple[int, int] | None:
    norm_label = _normalize(row_label)
    exact_rows = {
        row_idx
        for row_idx, row in enumerate(table.rows)
        if any(_normalize(cell) == norm_label for cell in row if cell)
    }
    row_matches = sorted(exact_rows)
    if not row_matches:
        row_matches = sorted(
            {
                row_idx
                for row_idx, row in enumerate(table.rows)
                if any(norm_label in _normalize(cell) for cell in row if cell)
            }
        )
    if len(row_matches) != 1:
        return None
    col_matches = _match_indices(list(table.header), column_label)
    if len(col_matches) != 1:
        return None
    row_idx, col_idx = row_matches[0], col_matches[0]
    if col_idx >= len(table.rows[row_idx]):
        return None
    return row_idx, col_idx
