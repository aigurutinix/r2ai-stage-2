"""Can a note's unit be inferred from the report's statements? Measure it where both exist.

84% of note tables sit on a page that never mentions a unit. That is not a parsing
failure: a Vietnamese report declares its unit once, at the head of the notes, and every
table after that inherits it silently. There is nothing to parse, only something to
infer.

The inference on offer is the document's own statements, which do declare their unit.
And it is checkable, because a minority of note tables declare one too: for those, does
the modal scale of the document's statement tables equal the scale the note itself
states?

If it agrees, the inference unlocks the 84%. If it disagrees, notes are denominated
independently of the statements and the scale has to come from somewhere else — from a
total that ties to a statement line, which recovers it exactly but only where a tie
exists.

Carrying the unit line across pages was already measured and rejected: it added 786
statement tables and pushed cross-document agreement from 90.9% to 89.3% while doubling
scale errors. Inferring from the document's own modal scale is a different claim, and
this is the test of it.

Usage:  python scripts/fresh/check_scale_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    args = parser.parse_args()

    # Modal scale of each document's statement tables.
    statement_scales: dict[str, Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            statement_scales[record["doc"]][record["scale"]] += 1
    modal = {doc: counter.most_common(1)[0][0]
             for doc, counter in statement_scales.items()}
    spread = sum(1 for c in statement_scales.values() if len(c) > 1)
    print(f"{len(modal)} tai lieu co bang bao cao chinh; "
          f"{spread} tai lieu khai NHIEU HON MOT he so trong bao cao chinh")

    # Notes that declare their own scale — the test set.
    counters: Counter[str] = Counter()
    mismatch: Counter[str] = Counter()
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["scale"] is None:
            counters["thuyet minh khong khai don vi"] += 1
            continue
        guess = modal.get(record["doc"])
        if guess is None:
            counters["tai lieu khong co bao cao chinh de suy"] += 1
            continue
        counters["doi chieu duoc"] += 1
        if abs(record["scale"] - guess) < 1e-9:
            counters["  KHOP he so pho bien cua bao cao chinh"] += 1
        else:
            counters["  LECH"] += 1
            mismatch[f"thuyet minh {record['scale']:g} vs bao cao {guess:g}"] += 1

    checked = counters["doi chieu duoc"]
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    if checked:
        agree = counters["  KHOP he so pho bien cua bao cao chinh"]
        print(f"\ntren {checked} thuyet minh co khai don vi: "
              f"KHOP {100 * agree / checked:.1f}%")
    if mismatch:
        print("\ncac kieu lech:")
        for name, count in mismatch.most_common(6):
            print(f"  {name}: {count}")

    # Second, independent check: where a tie recovered a note's true scale, does the
    # modal statement scale match that instead?
    tie_agree = tie_total = 0
    notes_path = ROOT / args.notes
    if notes_path.exists():
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            note = json.loads(line)
            if not note.get("paired"):
                continue
            guess = modal.get(note["doc"])
            if guess is None:
                continue
            true_scale = Counter(t["scale"] for t in note["ties"]).most_common(1)[0][0]
            tie_total += 1
            tie_agree += abs(true_scale - guess) < 1e-9
        if tie_total:
            print(f"\nkiem doc lap — he so ma MOI NOI suy ra, so voi he so pho bien:")
            print(f"  {tie_agree}/{tie_total} khop "
                  f"({100 * tie_agree / tie_total:.1f}%)")


if __name__ == "__main__":
    main()
