"""Would a total-row rule close the matcher's largest silence?

`lookup.find` returns nothing on 63% of gold questions, and a third of those
silences are the same shape: the question asks for a total, and the answer sits in
a row the matcher cannot name — either labelled `Cộng`/`Tổng`, or with no label at
all, which is how these statements print a total.

The rule under test: when no metric variant matches a row, and the question is
asking for a total, take the total row.

Prototyped here rather than edited into `lookup.py` first, because `find` is shared
by every branch of the submission — the project's own lesson is to fix the shared
component rather than add a parallel one, but that only holds once the fix is known
to be a fix. This measures the gain and, more importantly, the damage: a rule that
fires on questions that are *not* asking for a total would replace correct silence
with confident error, which scores worse than nothing.

Usage:  PYTHONPATH=src python scripts/_probe_total_row.py [limit]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SOURCE = ROOT / "artifacts" / "easy_full.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# "Tổng cộng", "Tổng số", "Tổng giá trị", "Cộng" — and the plain "Tổng X" that
# opens a third of the generated questions.
ASKS_TOTAL = re.compile(r"\b(tổng|cộng)\b", re.I)
IS_TOTAL_ROW = re.compile(r"^\s*(cộng|tổng|total)\b", re.I)


def parse_number(text: str) -> float | None:
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


def close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= 5e-4)


def total_row(grid: list[list[str]], label_col: int) -> int | None:
    """The row a statement uses for its total, or None.

    Two conventions, checked in order of certainty: a row named `Cộng`/`Tổng`,
    then a trailing row whose label cell is empty. The blank-label form is only
    accepted at the end of the table, because a blank label mid-table is usually
    a section break or a continuation, not a total.
    """

    body = grid[1:]
    for index in range(len(body) - 1, -1, -1):
        line = body[index]
        label = (line[label_col] or "").strip() if label_col < len(line) else ""
        if IS_TOTAL_ROW.match(label):
            return index + 1
    for index in range(len(body) - 1, max(len(body) - 3, -1), -1):
        line = body[index]
        label = (line[label_col] or "").strip() if label_col < len(line) else ""
        if not label and any(parse_number(cell) for cell in line):
            return index + 1
    return None


def main() -> None:
    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if LIMIT:
        records = records[:LIMIT]

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    stats: Counter[str] = Counter()

    for index, record in enumerate(records):
        gold_answer = parse_number(record["answer"])
        if gold_answer is None:
            continue
        doc_name, table_id = record["relevant_tables"][0].rsplit("|table_", 1)
        grid = store.rows(TableKey(doc_name, int(table_id)))
        if not grid:
            continue
        question = parse_question(index, record["question"], roster)
        label_col = lookup_mod.label_column(grid)

        matched = None
        for variant in lookup_mod.metric_variants(question.question):
            candidate = lookup_mod.match_row(grid, variant, label_col)
            if candidate is not None and (matched is None or candidate[1] > matched[1]):
                matched = candidate
        if matched is not None:
            stats["matcher_already_answers"] += 1
            continue

        stats["silent_today"] += 1
        if not ASKS_TOTAL.search(question.question):
            stats["  rule_stays_silent (not a total question)"] += 1
            continue

        row = total_row(grid, label_col)
        if row is None:
            stats["  rule_finds_no_total_row"] += 1
            continue

        column = lookup_mod.pick_column(grid, question, label_col)
        if column is None:
            stats["  rule_fires_but_no_column"] += 1
            continue

        value = parse_number(grid[row][column]) if column < len(grid[row]) else None
        if value is None:
            stats["  rule_fires_but_cell_empty"] += 1
        elif close(value, gold_answer):
            stats["  RULE_CORRECT"] += 1
        else:
            stats["  RULE_WRONG"] += 1

    for key, count in stats.most_common():
        print(f"  {key:44s} {count:5d}")

    fired = stats["  RULE_CORRECT"] + stats["  RULE_WRONG"]
    silent = stats["silent_today"] or 1
    print(f"\n  fires on {fired}/{silent} = {fired / silent:.1%} of today's silences")
    if fired:
        print(f"  and is right {stats['  RULE_CORRECT'] / fired:.1%} of the times it fires")
    print("\n  Precision is what decides this, not coverage. Silence costs one")
    print("  question; a confident wrong answer costs the same question and")
    print("  displaces whatever branch would otherwise have handled it.")


if __name__ == "__main__":
    main()
