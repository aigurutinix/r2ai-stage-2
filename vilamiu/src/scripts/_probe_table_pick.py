"""How often does our own machinery pick the table that holds the answer?

Two numbers measured on the same gold records point in opposite directions, and the
difference between them is table selection:

  given the gold table   matcher 30.2%   model 69.1%
  through the real path  matcher 42.8%   model 21.7%

The matcher does better end to end while being worse at reading, so it must be much
better at choosing which table to read. If that is so, the two should be combined the
other way round from anything tried so far: let the matcher choose the table, and let
the model read inside it.

That is only worth building if the matcher's table choice is actually strong, which
is what this measures. Retrieval gives the shortlist; `find()` scores each table; the
table it scores highest is compared with the gold one.

Usage:
  PYTHONPATH=src python scripts/_probe_table_pick.py --limit 800 --shortlist 8
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--shortlist", type=int, default=8)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)

    tally: collections.Counter[str] = collections.Counter()
    seen = 0
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        gold_key = TableKey(doc, int(table_id))
        parsed = parse_question(record.get("id", 0), record["question"], roster)
        groups = max(1, len(parsed.tickers)) * max(1, len(parsed.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            parsed, per_group=per_group, cap=args.shortlist)]
        if not keys:
            continue
        seen += 1
        if gold_key not in keys:
            tally["bảng gold không trong shortlist"] += 1
            continue
        tally["gold có trong shortlist"] += 1

        best_key, best_score = None, -1.0
        for key in keys:
            grid = store.rows(key)
            if not grid:
                continue
            caption = str(getattr(store.meta(key), "caption", ""))[:60]
            found = lookup_mod.find(grid, parsed, caption)
            if found is not None and found.score > best_score:
                best_score, best_key = found.score, key
        if best_key is None:
            tally["không chọn được bảng nào"] += 1
        elif best_key == gold_key:
            tally["CHỌN ĐÚNG BẢNG"] += 1
        else:
            tally["chọn sai bảng"] += 1

    have = max(tally["gold có trong shortlist"], 1)
    decided = tally["CHỌN ĐÚNG BẢNG"] + tally["chọn sai bảng"]
    print(f"{seen} bản ghi gold, shortlist {args.shortlist}\n")
    for name, count in tally.most_common():
        print(f"  {name:32s} {count:5d}  {count / max(seen, 1):6.1%}")
    print(f"\n  chọn đúng | gold có mặt     {tally['CHỌN ĐÚNG BẢNG']}/{have} = "
          f"{tally['CHỌN ĐÚNG BẢNG'] / have:.1%}")
    if decided:
        print(f"  chọn đúng | có quyết định   {tally['CHỌN ĐÚNG BẢNG']}/{decided} = "
              f"{tally['CHỌN ĐÚNG BẢNG'] / decided:.1%}")


if __name__ == "__main__":
    main()
