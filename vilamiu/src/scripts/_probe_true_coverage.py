"""How often is the number actually in the prompt, counting every table that holds it?

"Gold table in shortlist: 69%" measures whether the annotated table was retrieved.
But the answer sits in 2.8 tables on average, so the model only needs one of them,
and the figure that bounds what it can possibly do is the share of questions where
*some* holder made it into the eight tables we show.

That number is the real ceiling, and the gap between it and our accuracy is the
real size of the reading problem. If coverage is near 90% while we answer 38%, no
amount of retrieval work can matter and every remaining hour belongs to reading.

Usage:
  PYTHONPATH=src python scripts/_probe_true_coverage.py --limit 150
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def parse_number(text):
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def close(a, b, tol: float = 5e-4) -> bool:
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= tol)


def holds(store, key, answer) -> bool:
    grid = store.rows(key)
    if not grid:
        return False
    meta = store.meta(key)
    context = f"{getattr(meta, 'unit_page', '')} {getattr(meta, 'unit_doc', '')} " \
              f"{getattr(meta, 'caption', '')}"
    for line in grid[1:]:
        for column, cell in enumerate(line):
            value = parse_number(cell)
            if value is None or value == 0:
                continue
            if close(value, answer):
                return True
            if close(value * lookup_mod.column_scale(grid, column, context), answer):
                return True
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full.jsonl")
    parser.add_argument("--rank", default="artifacts/_gold_anchor_rank.jsonl")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--tables", type=int, default=8)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    ranking = {}
    for line in (ROOT / args.rank).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ranking[row["id"]] = [
                TableKey(r["doc_name"], int(r["table_id"])) for r in row.get("refs", [])
            ]

    records = [
        json.loads(l) for l in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    stat = collections.Counter()
    for index, record in enumerate(records):
        if stat["checked"] >= args.limit:
            break
        answer = parse_number(record.get("answer"))
        if answer is None or answer == 0:
            continue
        shortlist = ranking.get(index, [])[: args.tables]
        if not shortlist:
            continue
        gold_keys = set()
        for ref in record.get("relevant_tables") or []:
            doc, tid = ref.rsplit("|table_", 1)
            gold_keys.add(TableKey(doc, int(tid)))

        stat["checked"] += 1
        if gold_keys & set(shortlist):
            stat["annotated_in_prompt"] += 1
        if any(holds(store, key, answer) for key in shortlist):
            stat["some_holder_in_prompt"] += 1

    n = stat["checked"] or 1
    print(f"{stat['checked']} gold questions, {args.tables} tables shown per prompt\n")
    print(f"  annotated gold table in the prompt : {stat['annotated_in_prompt']:4d}  "
          f"{100 * stat['annotated_in_prompt'] / n:5.1f}%")
    print(f"  SOME table holding the answer      : {stat['some_holder_in_prompt']:4d}  "
          f"{100 * stat['some_holder_in_prompt'] / n:5.1f}%   <-- the real ceiling")


if __name__ == "__main__":
    main()
