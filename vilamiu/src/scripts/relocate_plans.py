"""Re-locate every plan cell with the label matcher instead of the model.

The `plan` branch answers 238 questions — the second largest — and its cells were
placed by the model/embedding localiser, measured at about 21%. The lexical label
matcher in `lookup.match_row` is measured at 42.8% on its own pool. Nothing about
the plan requires the weaker locator; it was simply what produced the file.

A plan does not store its cell coordinates, but `compile_plan` writes them into the
code as literals: `v0 = num(df1, 12, 3) * 1.0`. They can be read back out, and every
cell of one plan shares a single metric — a plan is the same line item across years
or reports, never several different ones — so the question's own metric phrase is
what each cell should be matched against.

Only the row moves. The column carries the period and the unit scale, both already
decided by machinery that has been measured; re-deciding them here would confound
two changes in one file.

Usage:
  PYTHONPATH=src python scripts/relocate_plans.py \
      --in planned_v3.jsonl --out planned_v4.jsonl
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
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# `num(df1, 12, 3)` — the frame variable, the DataFrame row, the column.
LITERAL_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<name>v\d+)\s*=\s*(?P<value>-?[\d.eE+-]+)\s*\*\s*"
    r"(?P<scale>-?[\d.eE+-]+)\s*#\s*(?P<frame>df\d*)\.iloc\[(?P<row>-?\d+),\s*"
    r"(?P<col>\d+)\]\s*$",
    re.M,
)
READ_RE = re.compile(
    r"\bnum\(\s*(?P<frame>df\d*)\s*,\s*(?P<row>-?\d+)\s*,\s*(?P<col>\d+)\s*\)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source", default="planned_v3.jsonl")
    parser.add_argument("--out", dest="target", default="planned_v4.jsonl")
    parser.add_argument("--min-score", type=float, default=lookup_mod.MIN_LABEL_SCORE)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    questions = {
        question.id: question
        for question in parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                                  ROOT / "data" / "code_stock.csv")
    }

    tally: collections.Counter[str] = collections.Counter()
    examples = []
    out_lines = []

    for line in (ROOT / "artifacts" / args.source).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        code = record.get("code") or ""
        question = questions.get(record.get("id"))
        if not code or question is None:
            out_lines.append(line)
            tally["giữ nguyên"] += 1
            continue

        metric = lookup_mod.extract_metric(question.question)
        frames = {
            name: TableKey(doc, int(table_id))
            for name, (doc, table_id) in zip(record.get("variables") or [],
                                             record.get("keys") or [])
        }

        moved = [0]

        def located(frame_name: str, current_row: int):
            """The row the label matcher picks, or None to leave the cell alone."""

            key = frames.get(frame_name)
            if key is None:
                return None
            grid = store.rows(key)
            if not grid:
                return None
            found = lookup_mod.match_row(grid, metric)
            if found is None or found[1] < args.min_score:
                return None
            # `match_row` indexes the grid, whose row 0 is the header;
            # `frame_from_rows` consumes that header, so DataFrame index = grid - 1.
            new_row = found[0] - 1
            if new_row < 0 or new_row == current_row:
                return None
            if len(examples) < args.show:
                old_label = ""
                if 0 <= current_row + 1 < len(grid):
                    old_label = str(grid[current_row + 1][0])[:44]
                examples.append((record["id"], metric[:40], old_label, found[2][:44]))
            return new_row

        def on_literal(match: re.Match) -> str:
            frame_name = match.group("frame")
            row = int(match.group("row"))
            new_row = located(frame_name, row)
            if new_row is None:
                return match.group(0)
            moved[0] += 1
            return (f"{match.group('indent')}{match.group('name')} = "
                    f"num({frame_name}, {new_row}, {match.group('col')}) * "
                    f"{match.group('scale')}")

        def on_read(match: re.Match) -> str:
            frame_name = match.group("frame")
            row = int(match.group("row"))
            new_row = located(frame_name, row)
            if new_row is None:
                return match.group(0)
            moved[0] += 1
            return f"num({frame_name}, {new_row}, {match.group('col')})"

        new_code = READ_RE.sub(on_read, LITERAL_RE.sub(on_literal, code))
        if moved[0]:
            record["code"] = new_code
            # The cached value belongs to the old cells. Clearing it forces
            # `run_submit` to re-execute and report what the new program produces —
            # shipping a value the code does not compute scores zero on execution.
            record["value"] = None
            record["ok"] = False
            tally["đã dời ô"] += 1
        else:
            tally["giữ nguyên"] += 1
        out_lines.append(json.dumps(record, ensure_ascii=False))

    (ROOT / "artifacts" / args.target).write_text(
        "\n".join(out_lines) + "\n", encoding="utf-8")

    total = sum(tally.values())
    print(f"{total} kế hoạch -> artifacts/{args.target}")
    for name, count in tally.most_common():
        print(f"  {name:14s} {count:5d}  {count / max(total, 1):5.1%}")
    print()
    for qid, metric, old_label, new_label in examples:
        print(f"  id={qid:<5d} {metric!r}")
        print(f"      cũ : {old_label!r}")
        print(f"      mới: {new_label!r}")


if __name__ == "__main__":
    main()
