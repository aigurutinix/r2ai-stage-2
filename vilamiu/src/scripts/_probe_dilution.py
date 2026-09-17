"""Does showing more tables bury the right row under better-looking wrong ones?

Presence rises with the table budget: the gold table is in the shown set 73.9% of
the time at 8 tables and 90.3% at 24. That only converts into answers if the reader
can still find it. The multiplicative ceiling estimate assumes it can, and that
assumption is the whole risk in regenerating with a larger budget.

A model is not needed to bound it. The lexical label matcher is a stand-in reader:
for each question, score every row of every shown table against the question's metric
and see whether the winner lies in the gold table. If the winner moves out of the gold
table as the budget grows, distractors are genuinely competitive and a model would
face the same pull. If it does not, the extra tables are inert and the only effect of
raising the budget is the presence gain.

This is a lower bound on the reader, not a prediction of it — the matcher is weaker
than the model. But its *change* with the budget is the quantity at issue.

Usage:
  PYTHONPATH=src python scripts/_probe_dilution.py --limit 400 --no-ticker
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

TICKER_RE = re.compile(r"\(\s*[A-Z][A-Z0-9]{2,3}\s*\)")


def shown_tables(retriever, question, cap: int):
    groups = max(1, len(question.tickers)) * max(1, len(question.years))
    per_group = max(2, -(-cap // groups))
    return [hit.key for hit in retriever.search_balanced(
        question, per_group=per_group, cap=cap)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--no-ticker", action="store_true")
    parser.add_argument("--caps", default="8,16,24")
    args = parser.parse_args()

    caps = [int(part) for part in args.caps.split(",")]
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)

    stats = {cap: {"present": 0, "won": 0} for cap in caps}
    seen = 0

    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        if args.no_ticker and TICKER_RE.search(record["question"]):
            continue
        refs = record.get("relevant_tables") or []
        wanted = set()
        for ref in refs:
            doc, _, table_id = ref.rpartition("|table_")
            if doc and table_id.isdigit():
                wanted.add(TableKey(doc, int(table_id)))
        if not wanted:
            continue
        seen += 1

        question = parse_question(record.get("id", 0), record["question"], roster)
        metric = lookup_mod.extract_metric(record["question"])
        biggest = max(caps)
        order = shown_tables(retriever, question, biggest)

        for cap in caps:
            subset = order[:cap]
            if not (wanted & set(subset)):
                continue
            stats[cap]["present"] += 1
            best_score, best_key = -1.0, None
            for key in subset:
                grid = store.rows(key)
                if not grid:
                    continue
                found = lookup_mod.match_row(grid, metric)
                if found is not None and found[1] > best_score:
                    best_score, best_key = found[1], key
            if best_key in wanted:
                stats[cap]["won"] += 1

    print(f"{seen} câu gold" + (" (bỏ câu có mã CK)" if args.no_ticker else "") + "\n")
    print(f"{'cap':>5}  {'có bảng gold':>13}  {'khớp nhãn thắng':>16}  "
          f"{'thắng|có mặt':>13}")
    for cap in caps:
        present = stats[cap]["present"]
        won = stats[cap]["won"]
        rate = won / present if present else 0.0
        print(f"{cap:>5}  {present:>13}  {won:>16}  {rate:>12.1%}")


if __name__ == "__main__":
    main()
