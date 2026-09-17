"""Synthetic validation set (no train/dev is provided by the organizers).

Template questions over VAS line codes whose gold cell is verified:
- the row_code matches a known VAS code AND the label fuzzy-matches its
  canonical Vietnamese name (>=85)
- the column header explicitly contains the report year (no positional guess)
=> gold answer is deterministic; gold relevant table = that table.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from ..extraction.build_store import Store
from ..router.entities import StockMap
from ..utils.io import write_jsonl, write_json
from ..utils.viet_text import fuzz_token_set

VAS_METRICS = {
    "10": ("Doanh thu thuần về bán hàng và cung cấp dịch vụ", "doanh thu thuần"),
    "11": ("Giá vốn hàng bán", "giá vốn hàng bán"),
    "60": ("Lợi nhuận sau thuế thu nhập doanh nghiệp", "lợi nhuận sau thuế"),
    "270": ("TỔNG CỘNG TÀI SẢN", "tổng tài sản"),
    "300": ("NỢ PHẢI TRẢ", "nợ phải trả"),
    "400": ("VỐN CHỦ SỞ HỮU", "vốn chủ sở hữu"),
}
UNITS = [("đồng", 1.0), ("triệu đồng", 1e6), ("tỷ đồng", 1e9)]


def generate(store_dir: Path, code_stock_csv: Path, out_dir: Path,
             n_questions: int = 300, seed: int = 13) -> None:
    rng = random.Random(seed)
    store = Store(store_dir, cache_size=101)
    stock = StockMap(code_stock_csv)
    out_dir = Path(out_dir)

    pool = []
    tickers = sorted({t for (t, _y, _d) in store.report_index})
    for ticker in tickers:
        cells = store.cells_of(ticker)
        if not len(cells):
            continue
        cells = cells[cells.row_code.isin(VAS_METRICS.keys()) & cells.unit_known]
        for r in cells.itertuples():
            canon, _short = VAS_METRICS[str(r.row_code)]
            if fuzz_token_set(str(r.label), canon) < 85:
                continue
            if not re.search(rf"\b{r.year}\b|31\s*/\s*12\s*/\s*{r.year}",
                             str(r.col_name)):
                continue
            if abs(r.value) < 1:
                continue
            pool.append(r)

    rng.shuffle(pool)
    seen, questions, gold = set(), [], {}
    qid = 0
    for r in pool:
        key = (r.report_id, str(r.row_code))
        if key in seen:
            continue
        seen.add(key)
        qid += 1
        if qid > n_questions:
            break
        unit_name, unit_scale = rng.choice(UNITS)
        canon, short = VAS_METRICS[str(r.row_code)]
        name = stock.ticker2name.get(r.ticker, r.ticker)
        me = "công ty mẹ " if r.doc_type == "separate" else ""
        q = (f"{short.capitalize()} của {me}{name} ({r.ticker}) năm {r.year} "
             f"là bao nhiêu {unit_name}?")
        ans = round(float(r.value) * float(r.unit_scale) / unit_scale, 2)
        questions.append({"id": qid, "question": q})
        # gold positions use the OFFICIAL scheme: line number of <table>
        line = store.line_no_of(r.report_id, int(r.table_pos))
        gold[str(qid)] = {
            "answer": ans, "unit": unit_name,
            "relevant_docs": [r.report_id],
            "relevant_tables": [f"{r.report_id}|{line}"],
            "ticker": r.ticker, "year": int(r.year), "row_code": str(r.row_code),
        }
    write_jsonl(out_dir / "val_questions.jsonl", questions)
    write_json(out_dir / "val_gold.json", gold)
    print(f"validation set: {len(questions)} questions -> {out_dir}")
