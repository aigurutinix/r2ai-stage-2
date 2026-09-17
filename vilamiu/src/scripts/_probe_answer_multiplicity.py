"""How many tables hold the answer, not just the one the annotation names?

A rival declares the annotated gold table for 49.3% of questions and answers
67.0%. We declare it for 72.1% and answer 38.1%. If the figure a question asks
for sits in only one table, those two numbers cannot both be true. If it sits in
several -- the balance sheet, a note, a summary page -- then "find any table
holding the number" is a much easier target than "find the annotated one", and a
pipeline aimed at the first would score well while measuring badly on TABLES_F2.

This counts, for each gold question, how many distinct tables of the same company
and year contain a cell equal to the gold answer once the column scale is applied.
Nothing is generated and no API is called.

The outcome decides where the remaining effort goes:

* mostly one table  -> retrieval precision is genuinely the binding constraint
* often several     -> our retrieval is already sufficient and the reading step is
                       the whole story, so stop tuning k

Usage:
  PYTHONPATH=src python scripts/_probe_answer_multiplicity.py --limit 150
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full.jsonl")
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame

    # Index eligible tables by (ticker, year) so the search stays inside the
    # company-year the question is about; a matching number in another company's
    # report is a coincidence, not an alternative source.
    by_group = collections.defaultdict(list)
    for row in frame.itertuples():
        if not bool(getattr(row, "eligible", True)):
            continue
        by_group[(str(row.ticker), str(row.year))].append(row)

    records = [
        json.loads(l) for l in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    counts: collections.Counter[int] = collections.Counter()
    checked = 0
    gold_among = 0
    for record in records:
        if checked >= args.limit:
            break
        answer = parse_number(record.get("answer"))
        if answer is None or answer == 0:
            continue
        refs = record.get("relevant_tables") or []
        if not refs:
            continue
        doc_name = refs[0].rsplit("|table_", 1)[0]
        match = frame[frame.doc_name == doc_name]
        if match.empty:
            continue
        ticker = str(match.iloc[0].ticker)
        year = str(match.iloc[0].year)
        gold_keys = set()
        for ref in refs:
            doc, tid = ref.rsplit("|table_", 1)
            gold_keys.add(TableKey(doc, int(tid)))

        holders = set()
        for row in by_group.get((ticker, year), []):
            key = TableKey(str(row.doc_name), int(row.table_id))
            grid = store.rows(key)
            found = False
            for line in grid[1:]:
                for column, cell in enumerate(line):
                    value = parse_number(cell)
                    if value is None or value == 0:
                        continue
                    scaled = value * lookup_mod.column_scale(
                        grid, column,
                        f"{row.unit_page} {row.unit_doc} {row.caption}")
                    if close(value, answer) or close(scaled, answer):
                        found = True
                        break
                if found:
                    break
            if found:
                holders.add(key)

        checked += 1
        counts[min(len(holders), 6)] += 1
        if holders & gold_keys:
            gold_among += 1

    print(f"{checked} gold questions searched within their own company-year\n")
    total = sum(counts.values()) or 1
    for n in sorted(counts):
        label = f"{n}+" if n == 6 else str(n)
        print(f"  {label:>3} table(s) hold the answer: {counts[n]:4d}  "
              f"{100 * counts[n] / total:5.1f}%")
    many = sum(v for k, v in counts.items() if k >= 2)
    print(f"\n  answer present in two or more tables: {many} ({100 * many / total:.1f}%)")
    print(f"  the annotated gold table is one of the holders: {gold_among} "
          f"({100 * gold_among / total:.1f}%)")


if __name__ == "__main__":
    main()
