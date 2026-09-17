"""Turn a table record into (a) a token document for BM25 and (b) a tidy
long-format DataFrame/CSV used for codegen + submission evidence.

Tidy schema (homogeneous dtypes -> survives pd.read_csv round-trip):
    row(int), label(str), code(str), col(int), col_name(str),
    value(float), unit_scale(float)
"""
from __future__ import annotations

import io
import json

import pandas as pd

from ..utils.viet_num import parse_vn_number
from ..utils.viet_text import norm, tokens

TIDY_COLS = ["row", "label", "code", "col", "col_name", "value", "unit_scale"]


def grid_of(meta_row) -> list[list[str]]:
    return json.loads(meta_row["grid_json"])


def table_doc_tokens(meta_row, grid: list[list[str]] | None = None, max_labels: int = 60) -> list[str]:
    grid = grid if grid is not None else grid_of(meta_row)
    parts = [str(meta_row.get("context", ""))[-250:]]
    if grid and grid[0]:
        parts.append(" ".join(str(c) for c in grid[0][:12]))
    labels = []
    for row in grid[1:]:
        for cell in row[:3]:
            if cell and parse_vn_number(cell) is None:
                labels.append(cell)
                break
        if len(labels) >= max_labels:
            break
    parts.append(" ".join(labels))
    return tokens(" ".join(parts))


def tidy_rows_from_grid(grid: list[list[str]], unit_scale: float | None,
                        max_rows: int = 400) -> list[dict]:
    if not grid:
        return []
    header = grid[0]
    code_columns = {
        c for row in grid[:3] for c, cell in enumerate(row)
        if "ma so" in norm(cell) or norm(cell) in {"code", "ma"}
    }
    header_text = norm(" ".join(str(cell) for row in grid[:2] for cell in row))
    voting_rate_matrix = (
        "ty le bieu quyet" in header_text
        or "ty le so huu va bieu quyet" in header_text
    )
    annual_value_matrix = (
        not code_columns and len(header) >= 3 and not norm(header[0])
        and all(norm(cell).startswith("nam 20") for cell in header[1:] if cell)
    )
    matrix_text = norm(" ".join(str(cell) for row in grid for cell in row))
    note_value_matrix = (
        ("quyen su dung dat" in matrix_text and "gia tri con lai" in matrix_text)
        or all(value in matrix_text for value in (
            "no duoi tieu chuan", "no nghi ngo", "no co kha nang mat von"))
        or (("chuc danh" in matrix_text or "chuc vu" in matrix_text)
            and "nam nay" in matrix_text and "tong cong" in matrix_text)
        or all(value in matrix_text for value in (
            "chung chi tien gui", "duoi 12 thang",
            "tu 12 thang den duoi 5 nam"))
        or ("ngoai te" in matrix_text and "usd" in matrix_text
            and "eur" in matrix_text)
        or all(value in matrix_text for value in (
            "usd", "eur", "jpy", "cny", "cad", "aud"))
    )
    # NaN-safe (parquet stores None as NaN, and NaN is truthy)
    us = 1.0
    if unit_scale is not None and unit_scale == unit_scale and unit_scale:
        us = float(unit_scale)
    out = []
    from ..extraction.report_parser import _CODE_RE
    from ..utils.viet_num import is_year_like
    for r, row in enumerate(grid[:max_rows]):
        label_parts, row_code = [], ""
        for c, cell in enumerate(row[:6]):
            v = parse_vn_number(cell)
            if v is None and cell.strip():
                label_parts.append(cell.strip())
            elif (v is not None and not row_code and c <= 3
                  and _CODE_RE.match(cell.strip())
                  and not note_value_matrix
                  and (not voting_rate_matrix or c in code_columns)
                  and (not annual_value_matrix or c in code_columns)):
                # Preserve the legacy code heuristic everywhere except voting
                # matrices, where rates such as 100 live in the first columns.
                if not is_year_like(v):
                    row_code = cell.strip()
        label = " ".join(label_parts)[:160]
        for c, cell in enumerate(row):
            v = parse_vn_number(cell)
            if v is None:
                continue
            if cell.strip() == row_code:
                continue
            out.append({
                "row": r, "label": label, "code": row_code, "col": c,
                "col_name": (header[c] if r > 0 and c < len(header) else "")[:80],
                "value": float(v), "unit_scale": us,
            })
    return out


def tidy_df(meta_row) -> pd.DataFrame:
    rows = tidy_rows_from_grid(grid_of(meta_row), meta_row.get("unit_scale"))
    df = pd.DataFrame(rows, columns=TIDY_COLS)
    return df


def tidy_csv_text(meta_row) -> str:
    return tidy_df(meta_row).to_csv(index=False)


def df_roundtrip(csv_text: str) -> pd.DataFrame:
    """EXACTLY what the grader sees after a plain pd.read_csv(csv_path):
    empty labels -> NaN, numeric-looking codes -> numbers. Generated queries
    must therefore use na=False and .astype(str) for `code`."""
    return pd.read_csv(io.StringIO(csv_text))


def preview_for_prompt(csv_text: str, max_rows: int = 14, max_labels: int = 25) -> str:
    df = df_roundtrip(csv_text)
    if not len(df):
        return "(empty table)"
    labels = [str(l) for l in df["label"].unique()
              if isinstance(l, str) and l][:max_labels]
    head = df.head(max_rows).to_csv(index=False)
    return head + "\ndistinct labels: " + " | ".join(labels)
