"""Count answers built from a report year the question never mentions.

A question names its periods explicitly — "trong các năm 2017, 2019 và 2022", "cuối
năm 2024". Every table belongs to one report whose year is in its file name. So a
program that read a cell out of a report for some other year is wrong on the face of
it, and no gold answer is needed to say so.

Hand-tracing found two instances immediately: id=473 read a "2021 Nghìn VND" column
for a question about 2022, and id=832 answered "2021" to a question listing 2017,
2018, 2019, 2022 and 2025. 535 of 1012 answers come from multi-cell programs and
none of them had been checked at all.

A report may legitimately be read for the previous year end — "Số đầu năm" of the
2021 report is the close of 2020 — so a read from report year Y is accepted when the
question names Y or Y+1.

Usage:
  PYTHONPATH=src python scripts/audit_year_reads.py --base aimed.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trace_answer import logged_reads, read_grid  # noqa: E402

YEAR_IN_DOC = re.compile(r"_((?:19|20)\d{2})_")
YEAR_IN_TEXT = re.compile(r"\b((?:19|20)\d{2})\b")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    import pandas as pd

    tally: collections.Counter[str] = collections.Counter()
    offenders = []

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        for row in rows:
            named = {int(y) for y in YEAR_IN_TEXT.findall(row["question"])}
            if not named:
                tally["câu không nêu năm"] += 1
                continue

            frames, owner = {}, {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if not grid:
                    continue
                frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
                match = YEAR_IN_DOC.search(item["csv_path"])
                owner[item["variable"]] = int(match.group(1)) if match else 0
            if not frames:
                tally["không có khung"] += 1
                continue

            reads = logged_reads(row.get("pandas_query") or "", frames)
            if not reads:
                tally["không đọc qua num()"] += 1
                continue

            # A report for year Y carries the close of Y-1 in its opening column,
            # so Y is in scope when the question names Y or Y+1.
            stray = sorted({
                owner.get(variable, 0) for variable, _, _, _ in reads
                if owner.get(variable, 0)
                and owner[variable] not in named
                and owner[variable] + 1 not in named
            })
            if stray:
                tally["ĐỌC NĂM KHÔNG ĐƯỢC NÊU"] += 1
                offenders.append((row["id"], sorted(named), stray,
                                  row.get("answer"), row["question"][:70]))
            else:
                tally["năm hợp lệ"] += 1

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:28s} {count:5d}  {count / max(total, 1):5.1%}")
    scope = tally["năm hợp lệ"] + tally["ĐỌC NĂM KHÔNG ĐƯỢC NÊU"]
    if scope:
        print(f"\ntrong {scope} câu truy được, {tally['ĐỌC NĂM KHÔNG ĐƯỢC NÊU']} câu "
              f"đọc năm lạ ({tally['ĐỌC NĂM KHÔNG ĐƯỢC NÊU'] / scope:.1%})")
    for qid, named, stray, answer, question in offenders[: args.show]:
        print(f"  id={qid:<5d} hỏi {named} nhưng đọc {stray}  nộp={answer}")
        print(f"        {question}")


if __name__ == "__main__":
    main()
