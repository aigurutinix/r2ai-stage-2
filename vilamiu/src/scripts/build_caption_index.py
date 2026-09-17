"""Precompute the caption groups once, so the generator fleet stays light.

`_caption_grouped_pool` needs to know which tables are the same statement at
different companies. It was reading `artifacts/tables.parquet` to work that out —
in every process. Eight generator processes therefore each loaded pandas and a
multi-hundred-megabyte parquet to answer a question whose whole answer fits in a
few tens of kilobytes, and the laptop stalled. The generation itself is API-bound:
it calls OpenRouter and waits, so every byte of that was wasted.

This writes the answer once. Each process then reads a small JSON.

Group key is caption + scope + year, so a group is the same note, on the same
basis, for the same period, across companies — which is what stage 2 needs when it
looks for one concept present in two or three tables.

Usage:  PYTHONPATH=src python scripts/build_caption_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts" / "caption_groups.json"
MIN_COMPANIES = 3


def main() -> None:
    import pandas as pd

    frame = pd.read_parquet(
        ROOT / "artifacts" / "tables.parquet",
        columns=["doc_name", "ticker", "year", "scope", "table_id", "caption",
                 "eligible"])
    frame = frame[frame["eligible"].astype(bool)]
    caption = frame["caption"].astype(str).str.strip()
    frame = frame[caption.str.len() >= 8]
    caption = caption[caption.str.len() >= 8]

    key = (caption.str.lower() + "||" + frame["scope"].astype(str)
           + "||" + frame["year"].astype(str))
    frame = frame.assign(_key=key)

    groups = []
    for _, rows in frame.groupby("_key"):
        # One table per company: stage 1 compares entities, not pages.
        rows = rows.drop_duplicates(subset=["ticker"])
        if rows["ticker"].nunique() < MIN_COMPANIES:
            continue
        groups.append([
            [str(r.doc_name), int(r.table_id), str(r.ticker), str(r.year)]
            for r in rows.head(12).itertuples()
        ])

    OUT.write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"{len(groups)} groups of >= {MIN_COMPANIES} companies -> {OUT} "
          f"({size:.0f} KB)")
    if groups:
        biggest = max(groups, key=len)
        print(f"  largest group: {len(biggest)} companies, e.g. {biggest[0][0]}")
    print("  the fleet reads this instead of the parquet, so each process costs")
    print("  a JSON parse rather than a pandas load")


if __name__ == "__main__":
    main()
