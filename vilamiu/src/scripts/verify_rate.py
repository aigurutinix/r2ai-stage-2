"""How many shipped answers match a line the report actually carries.

Hand-reading verified one answer exactly (MBB's loan provision, 4.354.219 against
the cell "(4.354.219)" in the 31/12/2020 column) and refuted another (VSC's
operating cash flow, 145,73 against 427,74 in the report). Doing that one question
at a time does not scale, so this counts it.

For every question naming one company and one year, the report's own tables are
searched for a row whose label carries most of the question's metric phrase. If
the shipped answer equals one of that row's cells, converted to the unit the
question asks for, the answer is *corroborated by the report*. That is weaker than
gold — the row could be the wrong one — but it is far stronger than agreement
between two of our own mechanisms, and it needs no labels.

Usage:  PYTHONPATH=src python scripts/verify_rate.py --base guarded3_clean.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_against_reports import (  # noqa: E402
    asked_scale, cell_number, key_words, metric_phrase,
)

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def close(a, b, tol: float = 2e-3) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= tol)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--min-words", type=int, default=3,
                        help="metric words that must appear in the row label")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    rows = payload if isinstance(payload, list) else (
        payload.get("predictions") or list(payload.values())[0])

    by_group = collections.defaultdict(list)
    for record in frame.itertuples():
        if bool(getattr(record, "eligible", True)):
            by_group[(str(record.ticker), str(record.year))].append(record)

    stat = collections.Counter()
    for row in rows:
        parsed = parse_question(0, row["question"], roster)
        if len(parsed.tickers) != 1 or len(parsed.years) != 1:
            continue
        answer = cell_number(row.get("answer"))
        if answer is None:
            continue
        stat["checkable"] += 1

        wanted = key_words(metric_phrase(row["question"]))
        if len(wanted) < args.min_words:
            stat["metric phrase too short"] += 1
            continue
        _, scale = asked_scale(row["question"])

        found_row = False
        corroborated = False
        for record in by_group.get((parsed.tickers[0], str(parsed.years[0])), []):
            grid = store.rows(TableKey(str(record.doc_name), int(record.table_id)))
            if not grid:
                continue
            context = f"{record.unit_page} {record.unit_doc} {record.caption}"
            for line in grid[1:]:
                label = str(line[0]).strip()
                if not label:
                    continue
                if len(wanted & key_words(label)) < args.min_words:
                    continue
                found_row = True
                for column, cell in enumerate(line[1:], start=1):
                    value = cell_number(cell)
                    if value is None:
                        continue
                    converted = value * lookup_mod.column_scale(grid, column, context) / scale
                    if close(abs(converted), abs(answer)):
                        corroborated = True
                        break
                if corroborated:
                    break
            if corroborated:
                break

        if corroborated:
            stat["CORROBORATED by the report"] += 1
        elif found_row:
            stat["row found, answer not among its cells"] += 1
        else:
            stat["no row matches the metric phrase"] += 1

    total = stat.pop("checkable", 0) or 1
    print(f"{args.base}: {total} câu một công ty một năm, có đáp án số\n")
    for name, count in stat.most_common():
        print(f"  {count:4d}  {100 * count / total:5.1f}%  {name}")


if __name__ == "__main__":
    main()
