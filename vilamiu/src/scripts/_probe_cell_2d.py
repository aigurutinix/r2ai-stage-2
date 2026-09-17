"""Score cells on both axes at once, instead of scoring rows and then a period.

`find()` matches the question's metric against row labels and then chooses a column by
period alone. Reading the target tables shows why five separate improvements to that
row matcher all came back flat: for a large share of questions the metric is not a row
label at all.

  "cho vay khách hàng"  -> row is the blank total line, column is "Cho vay khách hàng"
  "Tỷ lệ lợi ích kinh tế" -> column header, row is a subsidiary's name
  "Tổng vốn chủ sở hữu" -> row "Số dư cuối năm nay", column "Tổng cộng"

The target tables are mostly small notes — median 9 rows, 54% of them three columns
wide — and many are matrices: rows are companies, segments, currencies or movements,
and the metric is written across the top.

So this scores every cell by how much of the question both of its axes explain, and
takes the best. The row label and the column header are matched against the same
metric phrase, and the period words in the question are matched against the header
separately, so a matrix table and an ordinary statement are handled by one rule.

Measured against the generator's own cell coordinates: the (row, column) either is the
pair the gold program read or it is not.

Usage:
  PYTHONPATH=src python scripts/_probe_cell_2d.py --limit 1500
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
NON_VALUE_RE = re.compile(r"mã\s*số|thuyết\s*minh|^tm$|^stt$|^mã$", re.I)
NUMBER_RE = re.compile(r"^\(?-?[\d.,]+\)?%?$")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def tokens_of(text: str) -> set:
    return {t for t in fold(text).split() if len(t) > 2}


def explained(wanted: set, label: str) -> tuple[float, set]:
    """How much of the question this axis explains, and which words it took.

    Coverage of the *question* is what matters here, not the F1 the row matcher
    uses: two axes together should explain the metric, and each is allowed to
    explain only part of it.
    """

    have = tokens_of(label)
    if not wanted or not have:
        return 0.0, set()
    shared = wanted & have
    if not shared:
        return 0.0, set()
    # Penalise an axis that drags in many words of its own, so a long unrelated
    # label cannot win by accident.
    focus = len(shared) / len(have)
    return len(shared) / len(wanted) * (0.5 + 0.5 * focus), shared


def is_value(cell: str) -> bool:
    text = str(cell).strip()
    return bool(text) and text not in ("-", "--") and bool(
        NUMBER_RE.match(text.replace(" ", "")))


def best_cell(grid, metric: str) -> tuple[int, int, float] | None:
    """The cell whose row and column together explain the metric best."""

    wanted = tokens_of(metric)
    if not wanted or len(grid) < 2:
        return None
    header = [str(cell) for cell in grid[0]]
    column_scores = []
    for index, text in enumerate(header):
        if NON_VALUE_RE.search(text):
            column_scores.append((0.0, set()))
            continue
        column_scores.append(explained(wanted, text))

    best = None
    for row_index in range(1, len(grid)):
        row = grid[row_index]
        if not row:
            continue
        row_score, row_words = explained(wanted, str(row[0]))
        for column in range(1, len(row)):
            if not is_value(row[column]):
                continue
            column_score, column_words = column_scores[column] \
                if column < len(column_scores) else (0.0, set())
            union = row_words | column_words
            # The pair is credited for the words it jointly explains, plus a
            # little for how confidently each axis explained its share.
            score = len(union) / len(wanted) * (
                0.7 + 0.3 * max(row_score, column_score))
            if best is None or score > best[2]:
                best = (row_index, column, score)
    return best


def gold_cell(query: str, grid):
    match = ILOC_NAME.search(query)
    if match:
        index = int(match.group(1))
        row = len(grid) + index if index < 0 else index + 1
        name = match.group(3)
        header = [str(cell) for cell in grid[0]]
        squashed = [re.sub(r"\s+", "", cell) for cell in header]
        target = re.sub(r"\s+", "", name)
        column = (header.index(name) if name in header
                  else (squashed.index(target) if target in squashed else None))
        return (row, column) if 0 <= row < len(grid) else None
    match = ILOC_RC.search(query)
    if match:
        index = int(match.group(1))
        row = len(grid) + index if index < 0 else index + 1
        column = int(match.group(2))
        if column < 0:
            column = len(grid[0]) + column
        return (row, column) if 0 <= row < len(grid) else None
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
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
        grid = store.rows(TableKey(doc, int(table_id)))
        if not grid or len(grid) < 2:
            continue
        located = gold_cell(record.get("pandas_query") or "", grid)
        if located is None:
            continue
        gold_row, gold_col = located
        seen += 1

        metric = lookup_mod.extract_metric(record["question"])
        found = best_cell(grid, metric)
        if found is None or found[2] < args.min_score:
            tally["im lặng"] += 1
            continue
        row, column, _ = found
        tally["cam kết"] += 1
        if row == gold_row and (gold_col is None or column == gold_col):
            tally["ĐÚNG CẢ Ô"] += 1
        elif row == gold_row:
            tally["đúng dòng, sai cột"] += 1
        else:
            tally["sai dòng"] += 1

    commit = max(tally["cam kết"], 1)
    print(f"{seen} bản ghi gold\n")
    for name, count in tally.most_common():
        print(f"  {name:24s} {count:5d}  {count / max(seen, 1):6.1%}")
    right_row = tally["ĐÚNG CẢ Ô"] + tally["đúng dòng, sai cột"]
    print(f"\n  ĐÚNG dòng (mọi cột)  {right_row}/{seen} = {right_row / max(seen, 1):.1%}")
    print(f"  ĐÚNG cả ô            {tally['ĐÚNG CẢ Ô']}/{seen} = "
          f"{tally['ĐÚNG CẢ Ô'] / max(seen, 1):.1%}")
    print("  (so: find() chấm dòng 30,2% toàn bộ / 51,7% trên số cam kết)")


if __name__ == "__main__":
    main()
