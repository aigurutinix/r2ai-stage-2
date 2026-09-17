"""Walk the corpus and build the dual store (per-ticker parquet shards).

Output layout (STORE_DIR):
    reports.parquet          one row per report (report_id, ticker, year, doc_type, n_tables)
    tables/{TICKER}.parquet  one row per table  (meta + grid_json)
    cells/{TICKER}.parquet   one row per numeric cell (long-format index)

Per-ticker sharding keeps per-question loading cheap (a few MB) without any DB.
"""
from __future__ import annotations

import re
import traceback
from pathlib import Path

import pandas as pd

from .report_parser import parse_report, extract_cells


def _explicit_doc_type(report_id: str, fallback: str) -> str:
    """Repair stale store metadata when the report id names its type."""
    name = str(report_id).lower()
    for doc_type in ("separate", "consolidated", "aggregated"):
        if re.search(rf"(?:^|_){doc_type}(?:_|$)", name):
            return doc_type
    return fallback


def build_store(fs_dir: Path, store_dir: Path, tickers: list[str] | None = None,
                max_reports_per_ticker: int = 0, with_cells: bool = True,
                quiet: bool = False) -> pd.DataFrame:
    fs_dir, store_dir = Path(fs_dir), Path(store_dir)
    (store_dir / "tables").mkdir(parents=True, exist_ok=True)
    if with_cells:
        (store_dir / "cells").mkdir(parents=True, exist_ok=True)

    all_tickers = sorted(d.name for d in fs_dir.iterdir() if d.is_dir())
    if tickers:
        want = {t.upper() for t in tickers}
        all_tickers = [t for t in all_tickers if t.upper() in want]

    try:
        from tqdm import tqdm
        it = tqdm(all_tickers, desc="tickers", disable=quiet)
    except ImportError:
        it = all_tickers

    report_rows = []
    for ticker in it:
        t_tables, t_cells, n_done = [], [], 0
        txts = sorted((fs_dir / ticker).rglob("*_extracted.txt"))
        for txt in txts:
            if max_reports_per_ticker and n_done >= max_reports_per_ticker:
                break
            try:
                meta, tables = parse_report(txt)
            except Exception:
                print(f"[WARN] failed to parse {txt}")
                traceback.print_exc()
                continue
            report_rows.append(meta)
            for rec in tables:
                t_tables.append(rec.meta_row())
                if with_cells:
                    t_cells.extend(extract_cells(rec))
            n_done += 1
        if t_tables:
            pd.DataFrame(t_tables).to_parquet(store_dir / "tables" / f"{ticker}.parquet", index=False)
        if with_cells and t_cells:
            pd.DataFrame(t_cells).to_parquet(store_dir / "cells" / f"{ticker}.parquet", index=False)

    reports = pd.DataFrame(report_rows)
    # merge with a previous partial build (incremental --tickers runs)
    prev_path = store_dir / "reports.parquet"
    if prev_path.exists() and len(reports):
        prev = pd.read_parquet(prev_path)
        prev = prev[~prev.ticker.isin(set(reports.ticker))]
        reports = pd.concat([prev, reports], ignore_index=True)
    if len(reports):
        reports.to_parquet(prev_path, index=False)
    return reports


# ---------- store readers ----------

class Store:
    """Lazy per-ticker reader with a small cache."""

    def __init__(self, store_dir: Path, cache_size: int = 8):
        self.dir = Path(store_dir)
        self.reports = pd.read_parquet(self.dir / "reports.parquet")
        if len(self.reports) and "doc_type" in self.reports.columns:
            self.reports["doc_type"] = [
                _explicit_doc_type(report_id, doc_type)
                for report_id, doc_type in zip(
                    self.reports["report_id"], self.reports["doc_type"])
            ]
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._cache_size = cache_size
        # Preserve every report for duplicate (ticker, year, doc_type) keys
        # such as separate_1/separate_2. report_index remains the legacy
        # one-string view (last row wins) for backward-compatible callers.
        self.report_index_multi: dict[tuple[str, int, str], list[str]] = {}
        for r in self.reports.itertuples():
            key = (r.ticker, int(r.year), r.doc_type)
            self.report_index_multi.setdefault(key, []).append(r.report_id)
        self.report_index: dict[tuple[str, int, str], str] = {
            key: report_ids[-1]
            for key, report_ids in self.report_index_multi.items()
        }

    def _load(self, kind: str, ticker: str) -> pd.DataFrame:
        key = (kind, ticker)
        if key not in self._cache:
            path = self.dir / kind / f"{ticker}.parquet"
            df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            if len(df) and {"report_id", "doc_type"}.issubset(df.columns):
                df = df.copy()
                df["doc_type"] = [
                    _explicit_doc_type(report_id, doc_type)
                    for report_id, doc_type in zip(
                        df["report_id"], df["doc_type"])
                ]
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = df
        return self._cache[key]

    def tables_of(self, ticker: str, report_ids: list[str] | None = None) -> pd.DataFrame:
        df = self._load("tables", ticker)
        if report_ids is not None and len(df):
            df = df[df.report_id.isin(report_ids)]
        return df

    def cells_of(self, ticker: str, report_ids: list[str] | None = None) -> pd.DataFrame:
        df = self._load("cells", ticker)
        if report_ids is not None and len(df):
            df = df[df.report_id.isin(report_ids)]
        return df

    def find_report(self, ticker: str, year: int, doc_type: str) -> str | None:
        """Backward-compatible single-report lookup (last match wins)."""
        report_ids = self.find_reports(ticker, year, doc_type)
        return report_ids[-1] if report_ids else None

    def find_reports(self, ticker: str, year: int, doc_type: str,
                     allow_fallback: bool = True) -> list[str]:
        """Return all reports for a key, optionally falling back by type."""
        key = (ticker, int(year), doc_type)
        exact = self.report_index_multi.get(key)
        if exact:
            return list(exact)
        if not allow_fallback:
            return []

        conventional_other = (
            "separate" if doc_type == "consolidated" else "consolidated"
        )
        fallback_order = [conventional_other, "aggregated", "other"]
        for fallback_type in fallback_order:
            if fallback_type == doc_type:
                continue
            matches = self.report_index_multi.get((ticker, int(year), fallback_type))
            if matches:
                return list(matches)
        return []

    def years_of(self, ticker: str) -> list[int]:
        return sorted({y for (t, y, _d) in self.report_index if t == ticker})

    def line_no_of(self, report_id: str, table_pos: int) -> int:
        """Official submitted position = line number of <table> in the .txt."""
        ticker = report_id.split("_")[0]
        df = self._load("tables", ticker)
        if "line_no" not in df.columns:
            raise SystemExit("store has no line_no column - rebuild it first: "
                             "python scripts/01_build_store.py")
        hit = df[(df.report_id == report_id) & (df.table_pos == table_pos)]
        return int(hit.iloc[0].line_no) if len(hit) else int(table_pos)
