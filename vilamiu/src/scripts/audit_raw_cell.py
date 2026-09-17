"""Compare each single-cell answer with the raw parsed cell, and nothing else.

The organisers' generator prompt (`generation/prompts/easy.py:16-21`) fixes the
contract: `pandas_query` must return the value **as parsed from the CSV**, keeping
full precision and adding no correction constant, and `unit` then records the unit
*that value is already in*. So the question's stated unit is the table's own unit,
and the gold answer for a single-cell question is the raw parsed cell.

`audit_unit_math.py` could not see a violation of this, because it computed its
expectation with the same `column_scale` the pipeline uses — circular. This compares
against the cell text itself, which is independent of every assumption we make about
units.

Usage:
  PYTHONPATH=src python scripts/audit_raw_cell.py --base aimed.zip
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

from trace_answer import logged_reads, read_grid  # noqa: E402

POWERS = ((1e-12, "×1e-12"), (1e-9, "×1e-9"), (1e-6, "×1e-6"), (1e-3, "×1e-3"),
          (1e3, "×1e3"), (1e6, "×1e6"), (1e9, "×1e9"), (1e12, "×1e12"))


def parse_vn(text: str):
    """The organisers' own reading of a cell: parentheses negative, "." grouping."""

    raw = str(text).strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1]
    raw = raw.rstrip("%").strip()
    cleaned = raw.replace(".", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    import pandas as pd

    tally: collections.Counter[str] = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        for row in rows:
            try:
                answer = float(row.get("answer"))
            except (TypeError, ValueError):
                tally["đáp án không phải số"] += 1
                continue
            frames = {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if grid:
                    frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
            if not frames:
                tally["không có khung"] += 1
                continue
            reads = logged_reads(row.get("pandas_query") or "", frames)
            unique = {(v, r, c): cell for v, r, c, cell in reads}
            if len(unique) != 1:
                tally["nhiều ô hoặc không đọc"] += 1
                continue
            cell = next(iter(unique.values()))
            raw = parse_vn(cell)
            if raw is None or raw == 0:
                tally["ô không parse được"] += 1
                continue

            if abs(abs(answer) - abs(raw)) <= 0.01:
                tally["KHỚP ô thô"] += 1
                continue
            label = "lệch khác"
            for power, name in POWERS:
                if abs(abs(answer) - abs(raw) * power) <= max(0.01, abs(raw) * power * 1e-6):
                    label = f"lệch {name}"
                    break
            tally[label] += 1
            if len(examples[label]) < 3:
                examples[label].append((row["id"], row.get("answer"), cell,
                                        row["question"][:72]))

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:28s} {count:5d}  {count / max(total, 1):5.1%}")
    print()
    for name, _ in tally.most_common():
        for qid, answer, cell, question in examples.get(name, []):
            print(f"  [{name}] id={qid} nộp={answer} ô={cell!r}")
            print(f"      {question}")


if __name__ == "__main__":
    main()
