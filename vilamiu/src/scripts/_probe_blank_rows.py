"""Can the label matcher be taught to see the total row, and does it help?

25.6% of gold cells sit on a row whose label cell is empty — the total line, which
Vietnamese statements leave unlabelled under the detail rows it sums. `match_row`
tokenises the label and skips the row when there are no tokens, so it can never
select one of them. That is a structural ceiling on the deterministic path, not a
tuning problem.

A total-row rule was tried once and lost 24 questions. Its post-mortem named three
causes: "Tổng Công ty" inside a company name firing the rule on questions that ask
for no total, "Cộng:" being an addition rather than a total, and "Tổng <other item>"
returning the total of a different line. All three are failures of a bolted-on rule
that guesses when to fire.

This measures the alternative — give the blank row a label it can be matched on, and
let the ordinary scoring decide — against the gold coordinates directly. Exact, free,
and it compares candidate labelling schemes rather than arguing about them.

Usage:
  PYTHONPATH=src python scripts/_probe_blank_rows.py --limit 1500
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

ILOC_NAME = re.compile(r"\.iloc\[\s*(-?\d+)\s*\]\s*\[\s*(['\"])(.*?)\2\s*\]", re.S)
ILOC_RC = re.compile(r"\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

# A label that opens a section rather than naming one line of it.
SECTION_RE = re.compile(r"^[IVXLC]+[.)]|^[A-ZĐÀ-Ỹ][A-ZĐÀ-Ỹ\s,()/-]{6,}$")


def gold_row_of(query: str, grid) -> int | None:
    match = ILOC_NAME.search(query) or ILOC_RC.search(query)
    if match is None:
        return None
    index = int(match.group(1))
    row = len(grid) + index if index < 0 else index + 1
    return row if 0 <= row < len(grid) else None


def relabel(grid, scheme: str, caption: str):
    """A copy of the grid where blank labels carry an inherited name."""

    if scheme == "none":
        return grid
    out = [list(row) for row in grid]
    section = ""
    previous = ""
    for index in range(1, len(out)):
        label = str(out[index][0]).strip() if out[index] else ""
        if label:
            if SECTION_RE.match(label):
                section = label
            previous = label
            continue
        if not any(str(cell).strip() for cell in out[index][1:]):
            continue
        if scheme == "previous":
            out[index][0] = f"Tổng {previous}" if previous else ""
        elif scheme == "section":
            base = section or previous
            out[index][0] = f"Tổng {base}" if base else ""
        elif scheme == "caption":
            out[index][0] = f"Tổng {caption}" if caption else ""
        elif scheme == "both":
            # The caption names the note the table belongs to and the preceding
            # label names the last detail it sums; a question about the total may
            # echo either, so the synthesised label carries both.
            parts = [p for p in (previous, caption) if p]
            out[index][0] = "Tổng " + " ".join(parts) if parts else ""
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--schemes", default="none,previous,section,caption")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    records = []
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(records) >= args.limit:
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
        meta = store.meta(TableKey(doc, int(table_id)))
        records.append((record, grid, row, str(getattr(meta, "caption", ""))[:60]))

    print(f"{len(records)} bản ghi gold\n")
    print(f"{'cách gán nhãn':12s} {'cam kết':>8s} {'đúng dòng':>10s} "
          f"{'đúng|cam kết':>13s} {'đúng ở dòng rỗng':>18s}")
    for scheme in args.schemes.split(","):
        tally: collections.Counter[str] = collections.Counter()
        for record, grid, gold_row, caption in records:
            blank = not str(grid[gold_row][0]).strip()
            table = relabel(grid, scheme, caption)
            metric = lookup_mod.extract_metric(record["question"])
            found = lookup_mod.match_row(table, metric)
            if found is None:
                continue
            tally["commit"] += 1
            if found[0] == gold_row:
                tally["hit"] += 1
                if blank:
                    tally["hit_blank"] += 1
        commit = max(tally["commit"], 1)
        print(f"{scheme:12s} {tally['commit']:8d} {tally['hit']:10d} "
              f"{tally['hit'] / commit:12.1%} {tally['hit_blank']:18d}")


if __name__ == "__main__":
    main()
