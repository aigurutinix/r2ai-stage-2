"""How often does the gold row's label mean what the question asks for?

Hand-reading four random small-table gold records found two whose gold cell sits on a
row that names a different line item than the question does:

  "Chi phí nhân sự" (personnel cost)      -> gold row "Chi phí khấu hao" (depreciation)
  "lợi nhuận chưa phân phối" (undistributed) -> gold row "Lợi nhuận chưa thực hiện" (unrealised)

Both are generated records: the generator picked a cell, then wrote a question, and the
execution gate cannot catch a question that names the wrong item because the program
still reproduces the cell's value. A reader that answers such a question correctly is
scored wrong.

If that class is large, every reader accuracy this project measured offline — 30.2% for
the label matcher, 42.8% for the lookup branch, 27.3% for the model — is depressed by
the test set rather than by the reader, and the ceiling implied by those numbers is not
real.

Token overlap between the question's metric and the gold row's label is a crude proxy
for "means the same thing", but it is the same proxy the matcher itself uses, so a row
the matcher could never score is exactly a row this flags.

Usage:
  PYTHONPATH=src python scripts/_probe_gold_wording.py --limit 1500
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

ILOC_NAME = re.compile(r"\.iloc\[\s*(-?\d+)\s*\]\s*\[\s*(['\"])(.*?)\2\s*\]", re.S)
ILOC_RC = re.compile(r"\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def overlap(metric: str, label: str) -> float:
    wanted = {t for t in fold(metric).split() if len(t) > 2}
    have = {t for t in fold(label).split() if len(t) > 2}
    if not wanted or not have:
        return 0.0
    shared = len(wanted & have)
    if not shared:
        return 0.0
    coverage = shared / len(wanted)
    focus = shared / len(have)
    return 2 * coverage * focus / (coverage + focus)


def gold_row_of(query: str, grid) -> int | None:
    match = ILOC_NAME.search(query) or ILOC_RC.search(query)
    if match is None:
        return None
    index = int(match.group(1))
    row = len(grid) + index if index < 0 else index + 1
    return row if 0 <= row < len(grid) else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    tally: collections.Counter[str] = collections.Counter()
    examples = []
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
        grid = store.rows(TableKey(doc, int(table_id)))
        if not grid or len(grid) < 2:
            continue
        row = gold_row_of(record.get("pandas_query") or "", grid)
        if row is None:
            continue
        seen += 1
        metric = lookup_mod.extract_metric(record["question"])
        label = str(grid[row][0]).strip()
        if not label:
            tally["dòng gold không có nhãn (dòng cộng)"] += 1
            continue
        score = overlap(metric, label)
        if score >= 0.6:
            tally["nhãn khớp tốt (>=0,6)"] += 1
        elif score >= 0.3:
            tally["khớp một phần (0,3-0,6)"] += 1
        elif score > 0:
            tally["khớp yếu (<0,3)"] += 1
        else:
            tally["KHÔNG chung một từ nào"] += 1
            if len(examples) < args.show:
                # The best-matching row in the same table, for comparison: if some
                # other row matches the question well, the gold is the odd one out.
                best, best_score = "", 0.0
                for other in grid[1:]:
                    text = str(other[0]).strip()
                    value = overlap(metric, text)
                    if value > best_score:
                        best, best_score = text, value
                examples.append((record.get("id"), metric[:40], label[:40],
                                 best[:40], best_score))

    print(f"{seen} bản ghi gold\n")
    for name, count in tally.most_common():
        print(f"  {name:34s} {count:5d}  {count / max(seen, 1):6.1%}")
    bad = tally["KHÔNG chung một từ nào"] + tally["khớp yếu (<0,3)"]
    print(f"\n  gold mà nhãn gần như không liên quan: {bad}/{seen} = "
          f"{bad / max(seen, 1):.1%}")
    for qid, metric, label, best, best_score in examples:
        print(f"\n  id={qid} hỏi {metric!r}")
        print(f"     gold ở dòng {label!r}")
        print(f"     dòng khớp nhất trong bảng: {best!r} ({best_score:.2f})")


if __name__ == "__main__":
    main()
