"""Build a normalized financial-statement cube from the local ViFinQA corpus.

The competition corpus stores HTML tables inside OCR text files.  The local
catalog has already extracted those tables to CSV, but the same line-item code
can occur again in note disclosures.  This module deliberately walks tables in
document order and keeps the first well-formed primary-statement cell for each
``(ticker, year, scope, statement kind, ma_so)`` key.

Values in the cube are always expressed in VND.  The original raw value,
source table and scale are retained so every answer remains auditable.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from kingpro.answering.operand_pipeline import (
    _num,
    label_column,
    maso_column,
    resolve_columns,
    year_column,
)
from kingpro.answering.pandas_answer import detect_table_unit
from kingpro.financial.report_scope import container_scope, physical_scope
from kingpro.financial.source_units import build_local_unit_factors
from kingpro.retrieval.bm25_index import fold

StatementKind = Literal["cdkt", "kqkd", "lctt"]

_KQKD_REQUIRED = frozenset({"01", "11", "20", "25", "26"})
_KQKD_CODES = frozenset(
    "01 02 10 11 20 21 22 23 24 25 26 30 31 32 40 50 51 52 60 61 62 70 71".split()
)
_CDKT_CODES = frozenset(
    "100 110 111 112 120 121 122 123 130 131 132 133 134 135 136 137 139 "
    "140 141 149 150 151 152 153 154 155 200 210 211 212 213 214 215 216 "
    "219 220 221 222 223 224 225 226 227 228 229 230 231 232 240 241 242 "
    "250 251 252 253 254 255 260 261 262 263 268 270 300 310 311 312 313 "
    "314 315 316 317 318 319 320 321 322 330 331 332 333 334 335 336 337 "
    "338 339 340 341 342 343 400 410 411 412 413 414 415 416 417 418 419 "
    "420 421 422 429 430 431 432 440".split()
)
_LCTT_CODES = frozenset("01 02 03 04 05 06 07 08 09 10 20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 40 50 60 61 70".split())
_COST_KEYS = frozenset(f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52"))
_CODE_RE = re.compile(r"^\d{1,3}$")
_STATEMENT_HINTS = (
    "tai san",
    "nguon von",
    "no phai tra",
    "doanh thu",
    "loi nhuan",
    "luu chuyen tien",
    "hang ton kho",
)


@dataclass(frozen=True, slots=True)
class StatementCell:
    ticker: str
    year: str
    scope: str
    metric_key: str
    ma_so: str
    label: str
    value: float
    raw: str
    table_ref: str
    csv_path: str
    row_idx: int
    col_idx: int
    scale: float
    physical_scope: str | None = None


@dataclass(slots=True)
class FinancialCube:
    """Sparse cube keyed by ticker/year/scope/metric key."""

    data: dict[str, dict[str, dict[str, dict[str, StatementCell]]]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def cell(
        self,
        ticker: str,
        year: str | int,
        metric_key: str,
        scope: str = "consolidated",
    ) -> StatementCell | None:
        found = (
            self.data.get(str(ticker).upper(), {})
            .get(str(year), {})
            .get(scope, {})
            .get(metric_key)
        )
        if found is not None or not str(scope).startswith("physical:"):
            return found
        wanted = str(scope).split(":", 1)[1]
        scopes = (
            self.data.get(str(ticker).upper(), {})
            .get(str(year), {})
        )
        matches = [
            bucket[metric_key]
            for bucket in scopes.values()
            if metric_key in bucket
            and bucket[metric_key].physical_scope == wanted
        ]
        return matches[0] if len(matches) == 1 else None

    def put_first(self, cell: StatementCell) -> bool:
        bucket = (
            self.data.setdefault(cell.ticker, {})
            .setdefault(cell.year, {})
            .setdefault(cell.scope, {})
        )
        if cell.metric_key in bucket:
            return False
        bucket[cell.metric_key] = cell
        return True

    def iter_cells(self) -> Iterable[StatementCell]:
        for ticker in sorted(self.data):
            for year in sorted(self.data[ticker]):
                for scope in sorted(self.data[ticker][year]):
                    for key in sorted(self.data[ticker][year][scope]):
                        yield self.data[ticker][year][scope][key]

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for cell in self.iter_cells():
                handle.write(json.dumps(asdict(cell), ensure_ascii=False) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "FinancialCube":
        cube = cls()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cube.put_first(StatementCell(**json.loads(line)))
        cube.stats["cells"] = sum(1 for _ in cube.iter_cells())
        return cube


def _scope(report_id: str) -> str:
    """Backward-compatible alias for callers that have no physical entry."""
    return container_scope(report_id)


def _classify(codes: set[str], labels: list[str]) -> StatementKind | None:
    text = " ".join(fold(label) for label in labels)
    if len({code for code in codes if len(code) == 3}) >= 5:
        return "cdkt"
    if "luu chuyen tien" in text or "chuyen tien thuan" in text:
        return "lctt"
    if len(codes & _KQKD_REQUIRED) >= 3:
        return "kqkd"
    return None


def _allowed_codes(kind: StatementKind) -> frozenset[str]:
    if kind == "cdkt":
        return _CDKT_CODES
    if kind == "kqkd":
        return _KQKD_CODES
    return _LCTT_CODES


def _parse_table(
    entry: dict,
    tables_root: Path,
    local_unit_factors: dict[str, float] | None = None,
) -> tuple[StatementKind, list[StatementCell]] | None:
    csv_path = tables_root / entry["csv_path"]
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, UnicodeError):
        return None
    if df.empty or df.shape[1] < 3:
        return None

    cols = resolve_columns(df, n_header=2)
    label_col = label_column(df, cols)
    code_col = maso_column(df, cols)
    if code_col is None:
        return None

    parsed: list[tuple[int, str, str]] = []
    codes: set[str] = set()
    labels: list[str] = []
    for row_idx in range(len(df)):
        code = str(df.iloc[row_idx, code_col]).strip()
        if not _CODE_RE.fullmatch(code):
            continue
        label = str(df.iloc[row_idx, label_col]).strip()
        parsed.append((row_idx, code, label))
        codes.add(code)
        labels.append(label)

    kind = _classify(codes, labels)
    if kind is None:
        return None

    report_year = str(entry["year"])
    value_col = year_column(
        cols,
        report_year,
        label_col,
        code_col,
        want_dau=False,
        report_year=report_year,
    )
    if value_col is None:
        return None
    # Full-VND statements often omit the unit next to the table and the report
    # fallback can then latch onto an unrelated later phrase such as "tỷ đồng".
    # Four or more dot-separated 3-digit groups cannot plausibly be a compact
    # million/billion-unit statement value (it would imply quadrillions), but it
    # is the normal OCR representation of raw VND.  Resolve that strong local
    # signal before consulting report-level prose.
    raw_value_samples = [str(df.iloc[row_idx, value_col]).strip() for row_idx, _, _ in parsed]
    raw_vnd_pattern = re.compile(r"^\(?\d{1,3}(?:\.\d{3}){3,}\)?$")
    has_full_vnd_values = sum(bool(raw_vnd_pattern.fullmatch(raw)) for raw in raw_value_samples) >= 2
    local_scale = (local_unit_factors or {}).get(str(entry["table_ref"]))
    unit = (
        ("explicit-local-marker", local_scale)
        if local_scale is not None
        else (("đồng", 1) if has_full_vnd_values else detect_table_unit(df, entry["table_ref"]))
    )
    if unit is None:
        # Some OCR exports omit the unit line while retaining full VND values
        # (for example MWG 2021-2024).  A primary-statement column containing
        # values above 1e8 cannot plausibly be a mã số/thuyết-minh column and is
        # safe to interpret as raw VND.  Do not guess for compact-valued tables.
        samples = [_num(str(df.iloc[row_idx, value_col])) for row_idx, _, _ in parsed]
        finite = [abs(value) for value in samples if value is not None and math.isfinite(value)]
        if not finite or max(finite) < 100_000_000:
            return None
        scale = 1
    else:
        _unit_name, scale = unit

    # Preserve the catalog/container index for unqualified benchmark queries.
    # A second physical-scope index is added by ``build_statement_cube``.
    scope = _scope(entry["report_id"])
    cells: list[StatementCell] = []
    seen: set[str] = set()
    for row_idx, code, label in parsed:
        if code in seen or code not in _allowed_codes(kind):
            continue
        raw = str(df.iloc[row_idx, value_col]).strip()
        numeric = _num(raw)
        if numeric is None or not math.isfinite(numeric):
            continue
        key = f"{kind}:{code}"
        value = numeric * float(scale)
        if key in _COST_KEYS and value < 0:
            value = abs(value)
        cells.append(
            StatementCell(
                ticker=str(entry["ticker"]).upper(),
                year=report_year,
                scope=scope,
                metric_key=key,
                ma_so=code,
                label=label,
                value=value,
                raw=raw,
                table_ref=entry["table_ref"],
                csv_path=str(csv_path),
                row_idx=row_idx,
                col_idx=int(value_col),
                scale=float(scale),
            )
        )
        seen.add(code)
    return kind, cells


def build_statement_cube(
    catalog_path: str | Path = "build/catalog.jsonl",
    tables_root: str | Path = "build/tables",
    *,
    scopes: tuple[str, ...] = ("consolidated", "separate", "aggregated", "unknown"),
    max_page: int | None = 30,
    local_units_root: str | Path | None = None,
) -> FinancialCube:
    """Normalize the primary statements represented by a local table catalog."""

    catalog_path = Path(catalog_path)
    tables_root = Path(tables_root)
    corpus_root = catalog_path.resolve().parent.parent
    entries: list[dict] = []
    physical_scope_overrides = 0
    physical_scope_evidence = 0
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if _scope(entry["report_id"]) not in scopes:
                continue
            # Primary Vietnamese statements almost always have 4-6 columns.
            # Wider and late-document tables are overwhelmingly note disclosures;
            # skipping them also prevents their duplicate codes becoming candidates.
            if not 3 <= int(entry.get("n_cols", 0)) <= 6:
                continue
            page = int(entry.get("page", 0))
            if max_page is not None and page > max_page:
                continue
            search_text = fold(entry.get("search_text", ""))
            # ``search_text`` is label-rich when labels are in column 0.  Some
            # issuers put ``Mã số`` first and labels in column 1, so also retain
            # all statement-shaped tables from the opening report pages.
            has_hint = any(hint in search_text for hint in _STATEMENT_HINTS)
            early_statement_shape = page <= 18 and int(entry.get("n_cols", 0)) in (4, 5, 6)
            if not has_hint and not early_statement_shape:
                continue
            entries.append(entry)

    entries.sort(key=lambda row: (row["ticker"], int(row["year"]), row["report_id"], int(row["line"])))
    local_unit_factors: dict[str, float] = {}
    local_unit_stats: dict[str, int] = {}
    if local_units_root is not None:
        local_unit_factors, local_unit_stats = build_local_unit_factors(
            entries, local_units_root
        )
    cube = FinancialCube()
    parsed_tables = 0
    classified_tables = 0
    duplicates = 0
    physical_scope_annotated_cells = 0
    for entry in entries:
        parsed_tables += 1
        result = _parse_table(entry, tables_root, local_unit_factors)
        if result is None:
            continue
        _kind, cells = result
        classified_tables += 1
        resolved_scope, scope_evidence = physical_scope(entry, corpus_root)
        if scope_evidence is not None:
            physical_scope_evidence += 1
            if scope_evidence["overrides_container"]:
                physical_scope_overrides += 1
        for cell in cells:
            cell = replace(cell, physical_scope=resolved_scope)
            physical_scope_annotated_cells += 1
            if not cube.put_first(cell):
                duplicates += 1

    cube.stats.update(
        catalog_candidates=len(entries),
        parsed_tables=parsed_tables,
        classified_tables=classified_tables,
        duplicate_cells=duplicates,
        physical_scope_evidence=physical_scope_evidence,
        physical_scope_overrides=physical_scope_overrides,
        physical_scope_annotated_cells=physical_scope_annotated_cells,
        cells=sum(1 for _ in cube.iter_cells()),
        **local_unit_stats,
    )
    return cube
