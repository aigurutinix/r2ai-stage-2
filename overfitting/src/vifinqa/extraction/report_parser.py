"""Parse one report .txt -> table records with unit detection.

Unit handling ("Don vi tinh: trieu dong" ...):
- explicit unit found in the text right before the table (or its first rows)
- otherwise "sticky": inherit the last explicit unit seen in the report
- otherwise None (treated as VND downstream, flagged low-confidence)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.viet_num import parse_vn_number, is_year_like
from ..utils.viet_text import norm, strip_diacritics
from .html_tables import iter_tables, parse_grid

# regex on diacritic-stripped lowercase text
_UNIT_EXPL = re.compile(
    r"don\s*vi(?:\s*tinh)?\s*[:\-]?\s*(nghin\s*ty|ngan\s*ty|tram\s*ty|ty|trieu|nghin|ngan)?\s*(dong|vnd|d\b)?",
)
_UNIT_BARE = re.compile(r"\b(nghin\s*ty|ngan\s*ty|tram\s*ty|ty|trieu|nghin|ngan)\s+(dong|vnd)\b")
_UNIT_BAD_CTX = ("ty le", "ty suat", "ty gia", "ty trong")

_SCALE = {"nghin ty": 1e12, "ngan ty": 1e12, "tram ty": 1e11,
          "ty": 1e9, "trieu": 1e6,
          "nghin": 1e3, "ngan": 1e3, "": 1.0, None: 1.0}


_UNIT_HEADER = re.compile(r"(nghin\s*ty|ngan\s*ty|tram\s*ty|ty|trieu|nghin|ngan)?\s*(dong|vnd)")


def detect_unit(context: str, header_text: str = "") -> tuple[float | None, str | None]:
    """Return (scale, source) with source in {explicit, header, bare} or (None, None).

    Priority: explicit "Don vi (tinh): ..." declaration > unit token inside the
    table header (e.g. '31/12/2018 VND', 'Trieu VND') > bare unit phrase in the
    last chars right before the table (rejected when preceded by a digit — that
    is prose like 'trai phieu 100 ty dong', not a unit declaration).
    """
    ctx = re.sub(r"\s+", " ", strip_diacritics(context).lower())
    m = _UNIT_EXPL.search(ctx)
    if m and (m.group(1) or m.group(2)):
        word = re.sub(r"\s+", " ", (m.group(1) or "").strip())
        return _SCALE.get(word, 1.0), "explicit"
    if header_text:
        h = re.sub(r"\s+", " ", strip_diacritics(header_text).lower())
        m = _UNIT_HEADER.search(h)
        if m:
            s = m.start()
            window = h[max(0, s - 8):m.end()]
            # bare "dong" alone is too ambiguous in headers ("dong tien" = cash
            # flow); require "vnd" or an explicit multiplier word
            solid = m.group(1) or m.group(2) == "vnd"
            if solid and not any(b in window for b in _UNIT_BAD_CTX):
                word = re.sub(r"\s+", " ", (m.group(1) or "").strip())
                return _SCALE.get(word, 1.0), "header"
    tail = ctx[-120:]
    m = _UNIT_BARE.search(tail)
    if m:
        s = m.start()
        window = tail[max(0, s - 8):m.end()]
        before = tail[:s].rstrip()
        prose = bool(before) and before[-1].isdigit()
        if not prose and not any(b in window for b in _UNIT_BAD_CTX):
            word = re.sub(r"\s+", " ", m.group(1).strip())
            return _SCALE.get(word, 1.0), "bare"
    return None, None


@dataclass
class TableRec:
    report_id: str
    ticker: str
    year: int
    doc_type: str          # consolidated | separate | aggregated | other
    table_pos: int         # 0-based order of <table> in the file
    line_no: int           # 1-based line number of <table> (alt position scheme)
    page: int
    n_rows: int
    n_cols: int
    unit_scale: float | None
    unit_source: str       # explicit | sticky | none
    context: str           # cleaned text right before the table
    grid: list = field(repr=False, default_factory=list)

    def meta_row(self) -> dict:
        d = self.__dict__.copy()
        d["grid_json"] = json.dumps(self.grid, ensure_ascii=False)
        d.pop("grid")
        return d


def report_meta_from_path(txt_path: Path) -> dict:
    doc_dir = txt_path.parent
    report_id = doc_dir.name
    year_dir = doc_dir.parent
    ticker_dir = year_dir.parent
    name = report_id.lower()
    if re.search(r"(?:^|_)separate(?:_|$)", name):
        doc_type = "separate"
    elif re.search(r"(?:^|_)consolidated(?:_|$)", name):
        doc_type = "consolidated"
    elif re.search(r"(?:^|_)aggregated(?:_|$)", name):
        doc_type = "aggregated"
    else:
        doc_type = "other"
    return {
        "report_id": report_id,
        "ticker": ticker_dir.name,
        "year": int(year_dir.name),
        "doc_type": doc_type,
        "path": str(txt_path),
    }


def parse_report(txt_path: Path) -> tuple[dict, list[TableRec]]:
    meta = report_meta_from_path(Path(txt_path))
    text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
    tables: list[TableRec] = []
    sticky_scale: float | None = None
    for pos, page, line_no, html, ctx in iter_tables(text):
        grid = parse_grid(html)
        if not grid:
            grid = [[]]
        head_txt = " ".join(c for row in grid[:2] for c in row[:8])
        scale, source = detect_unit(ctx, head_txt)
        if scale is not None:
            sticky_scale = scale
        elif sticky_scale is not None:
            scale, source = sticky_scale, "sticky"
        else:
            scale, source = None, "none"
        tables.append(TableRec(
            report_id=meta["report_id"], ticker=meta["ticker"], year=meta["year"],
            doc_type=meta["doc_type"], table_pos=pos, line_no=line_no, page=page,
            n_rows=len(grid), n_cols=max((len(r) for r in grid), default=0),
            unit_scale=scale, unit_source=source, context=ctx, grid=grid,
        ))
    meta["n_tables"] = len(tables)
    return meta, tables


# ---------- cell-level long index ----------

_CODE_RE = re.compile(r"^\d{1,3}(?:\.\d+)?[a-z]?$")


def extract_cells(rec: TableRec, max_rows: int = 400) -> list[dict]:
    """One row per numeric cell: the long-format index used for structured
    lookup, rule-based codegen and synthetic validation."""
    grid = rec.grid
    if not grid:
        return []
    header = grid[0] if grid else []
    matrix_text = norm(" ".join(str(cell) for row in grid for cell in row))
    note_value_matrix = (
        ("quyen su dung dat" in matrix_text and "gia tri con lai" in matrix_text)
        or all(value in matrix_text for value in (
            "no duoi tieu chuan", "no nghi ngo", "no co kha nang mat von"))
        or (("chuc danh" in matrix_text or "chuc vu" in matrix_text)
            and "nam nay" in matrix_text and "tong cong" in matrix_text)
    )
    out = []
    for r, row in enumerate(grid[:max_rows]):
        label_parts, row_code = [], ""
        for c, cell in enumerate(row[:6]):
            v = parse_vn_number(cell)
            if v is None and cell.strip():
                label_parts.append(cell.strip())
            elif (v is not None and not row_code and c <= 3
                  and _CODE_RE.match(cell.strip())
                  and not note_value_matrix):
                if not is_year_like(v):
                    row_code = cell.strip()
        label = " ".join(label_parts)[:160]
        for c, cell in enumerate(row):
            v = parse_vn_number(cell)
            if v is None:
                continue
            if cell.strip() == row_code and _CODE_RE.match(cell.strip() or "x"):
                continue  # don't index the code column as a value
            out.append({
                "report_id": rec.report_id, "ticker": rec.ticker, "year": rec.year,
                "doc_type": rec.doc_type, "table_pos": rec.table_pos, "page": rec.page,
                "row": r, "col": c, "label": label, "row_code": row_code,
                "col_name": (header[c] if r > 0 and c < len(header) else "")[:80],
                "value": v,
                "unit_scale": rec.unit_scale if rec.unit_scale is not None else 1.0,
                "unit_known": rec.unit_source != "none",
            })
    return out
