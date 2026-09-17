"""Paired per-question comparisons, on the unit-corrected gold.

The marginal totals moved 8 points when the gold was rebuilt in the unit each
question asks for, so the paired verdicts drawn from the broken gold have to be
re-run before any of today's conclusions stand.

Usage:  PYTHONPATH=src python scripts/_paired_unit_fixed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _rescore_unit_fixed import (  # noqa: E402
    asked_unit, close, parse_number, table_scale,
)

from vifin.store import TableKey, TableStore  # noqa: E402

PAIRS = [
    ("3 tables vs 8 tables", "_sc_t3.jsonl", "_sc_t8.jsonl"),
    ("captions on vs off", "_sc_caption.jsonl", "_sc_nocaption.jsonl"),
    ("row indices on vs off", "_sc_idx_on.jsonl", "_sc_idx_off.jsonl"),
]


def build_gold(store):
    gold = {}
    for line in (ROOT / "artifacts" / "easy_full.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        answer = parse_number(record.get("answer"))
        refs = record.get("relevant_tables") or []
        rid = record.get("id")
        if answer is None or not refs or rid is None:
            continue
        wanted = asked_unit(record["question"])
        if wanted is None:
            gold[rid] = answer
            continue
        doc, tid = refs[0].rsplit("|table_", 1)
        grid = store.rows(TableKey(doc, int(tid)))
        if not grid:
            gold[rid] = answer
            continue
        is_cell = any(parse_number(c) == answer for r in grid[1:] for c in r
                      if parse_number(c) is not None)
        if not is_cell:
            gold[rid] = answer
            continue
        meta = store.meta(TableKey(doc, int(tid)))
        gold[rid] = answer * table_scale(grid, getattr(meta, "caption", "")) / wanted
    return gold


def load(name):
    path = ROOT / "artifacts" / name
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    gold = build_gold(store)

    for label, a_name, b_name in PAIRS:
        a, b = load(a_name), load(b_name)
        ids = [i for i in (set(a) & set(b)) if i in gold]
        if not ids:
            print(f"\n{label}: no shared questions")
            continue
        fixed = broken = both = neither = 0
        for i in ids:
            ok_a = any(close(v, gold[i]) for v in a[i].get("values") or [])
            ok_b = any(close(v, gold[i]) for v in b[i].get("values") or [])
            if ok_a and ok_b:
                both += 1
            elif ok_a and not ok_b:
                fixed += 1
            elif ok_b and not ok_a:
                broken += 1
            else:
                neither += 1
        print(f"\n{label}  ({len(ids)} questions)")
        print(f"   both right {both:3d}   both wrong {neither:3d}")
        print(f"   first fixes {fixed:3d}   first breaks {broken:3d}"
              f"   net {fixed - broken:+d}")


if __name__ == "__main__":
    main()
