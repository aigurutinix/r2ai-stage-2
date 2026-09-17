"""Are the generated questions like the real ones, or like the easy half?

This is the check that decides whether a 7B is good enough to build the training
set with, and it is the same test `probe_synth.py` was written for — reused here
against the organisers' generator instead of a hand-rolled one.

The pipeline's judge exists to reject a question that copies a row or column
label. That rejection is exactly why our lexical label matcher tops out at 42.8%
on the real set: the real questions are paraphrased away from the table's own
wording. A weak judge lets label-copies through, and a training set full of them
teaches the model the half we already answer while telling it nothing about the
half we lose.

The measurement needs no gold. Run our own label matcher over the generated
questions and compare its hit rate against the 42.8% it scores on the real ones:

    ~42%  the distribution matches — the generator is usable
    ~90%  the questions echo their labels — the judge is too weak, step up a size

Usage:
  PYTHONPATH=src python scripts/_probe_synth_quality.py runs/gen7b/per_question.jsonl
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

RECORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "runs" / "gen7b" / "per_question.jsonl")


def parse_ref(ref: str) -> tuple[str, int] | None:
    doc, _, tail = ref.partition("|")
    if tail.startswith("table_") and tail[6:].isdigit():
        return doc, int(tail[6:])
    return None


def main() -> None:
    rows = [
        json.loads(line)
        for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        print(f"{RECORDS}: no records yet")
        return

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    solved = copied = scored = 0
    for record in rows:
        refs = [parse_ref(r) for r in record.get("relevant_tables", [])]
        refs = [r for r in refs if r]
        if not refs:
            continue
        doc, table_id = refs[0]
        try:
            grid = store.rows(TableKey(doc, table_id))
        except Exception:
            continue
        question = parse_question(0, record["question"], roster)
        scored += 1

        found = lookup_mod.find(grid, question)
        if found is not None and found.score >= lookup_mod.MIN_LABEL_SCORE:
            solved += 1

        # A question that reuses its row's label verbatim is the failure mode the
        # judge is supposed to catch; measure it directly as well as through the
        # matcher, because the matcher can also succeed for honest reasons.
        labels = [str(row[0]) for row in grid[1:] if row and str(row[0]).strip()]
        best = 0.0
        for label in labels:
            overlap = lookup_mod.match_row([[""], [label]], record["question"])
            if overlap and overlap[1] > best:
                best = overlap[1]
        copied += best >= 0.75

    n = scored or 1
    print(f"{RECORDS}: {len(rows)} records, {scored} with a resolvable gold table\n")
    print(f"  label matcher solves the cell : {solved}/{n} = {solved / n:.1%}")
    print(f"  question echoes its row label : {copied}/{n} = {copied / n:.1%}")
    print()
    print("  reference: the same matcher scores 42.8% on the real question set.")
    print("    near 42%  -> distribution matches, the generator is usable")
    print("    near 90%  -> label copies, the judge is too weak for this size")


if __name__ == "__main__":
    main()
