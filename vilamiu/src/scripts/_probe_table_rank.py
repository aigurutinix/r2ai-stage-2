"""Where in our own ordering does the table holding the answer sit?

The isolated cell probe hands the model exactly one table — the one containing the
answer — and it names the right cell 69.1% of the time. The pipeline hands it eight
and does far worse. The difference is the burden of choosing among them, and how
large that burden is depends on where the right table actually ranks.

If the gold table is first most of the time, showing three instead of eight moves the
model close to the probe's condition. If it is spread through the list, cutting the
budget throws away the answer instead.

Measured on `easy_full_units.jsonl`, whose `relevant_tables` are real gold.

Usage:
  PYTHONPATH=src python scripts/_probe_table_rank.py --limit 600
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

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


TICKER_RE = re.compile(r"\(\s*[A-Z][A-Z0-9]{2,3}\s*\)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--cap", type=int, default=8)
    # 90% of the generated gold questions carry the ticker in parentheses against
    # 22.6% of the real exam, and the ticker is the strongest retrieval signal
    # there is. Every retrieval number measured on the full set is therefore
    # optimistic — `totalrow.zip` was built on one such number and cost 24
    # questions. `--no-ticker` keeps only the questions that withhold it.
    parser.add_argument("--no-ticker", action="store_true")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)

    ranks: collections.Counter[str] = collections.Counter()
    seen = 0
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        if args.no_ticker and TICKER_RE.search(record["question"]):
            continue
        refs = record.get("relevant_tables") or []
        if not refs:
            continue
        wanted = set()
        for ref in refs:
            doc, _, table_id = ref.rpartition("|table_")
            if doc and table_id.isdigit():
                wanted.add(TableKey(doc, int(table_id)))
        if not wanted:
            continue
        seen += 1

        question = parse_question(record.get("id", 0), record["question"], roster)
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.cap // groups))
        order = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.cap)]

        position = None
        for index, key in enumerate(order, start=1):
            if key in wanted:
                position = index
                break
        if position is None:
            ranks["không có trong top %d" % args.cap] += 1
        elif position == 1:
            ranks["hạng 1"] += 1
        elif position <= 3:
            ranks["hạng 2-3"] += 1
        elif position <= 5:
            ranks["hạng 4-5"] += 1
        else:
            ranks["hạng 6-%d" % args.cap] += 1

    print(f"{seen} câu gold, cap={args.cap}\n")
    running = 0
    for name in ("hạng 1", "hạng 2-3", "hạng 4-5", f"hạng 6-{args.cap}",
                 f"không có trong top {args.cap}"):
        count = ranks.get(name, 0)
        if not name.startswith("không"):
            running += count
        print(f"  {name:22s} {count:5d}  {count / max(seen, 1):6.1%}"
              f"   luỹ kế {running / max(seen, 1):6.1%}")


if __name__ == "__main__":
    main()
