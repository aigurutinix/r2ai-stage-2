"""How much of a later batch repeats an earlier one, at four levels of strictness.

The dedup that runs on append keys on `(question, relevant_tables)`, which only
catches a byte-identical repeat. That is the weakest useful definition. A batch
can dodge it and still be worthless for training by asking a differently-worded
question about the same cell, the same table, or the same document — each of
which over-weights one part of the corpus in the training mix.

So the answer to "is batch 2 a repeat" needs all four numbers, not one.

Usage:  PYTHONPATH=src python scripts/_check_overlap.py pool.jsonl 60
        (the second argument is where the first batch ends)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "pool_120.jsonl"
SPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def load(path: Path) -> list[dict]:
    return [
        json.loads(line.replace(": NaN", ": null"))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    rows = load(POOL)
    first, second = rows[:SPLIT], rows[SPLIT:]
    print(f"{POOL.name}: {len(rows)} records — batch 1 = {len(first)}, "
          f"batch 2 = {len(second)}\n")

    def question(r): return r.get("question", "").strip()
    def tables(r): return tuple(r.get("relevant_tables") or [])
    def docs(r): return tuple(r.get("relevant_docs") or [])
    def cell(r): return (tables(r), str(r.get("answer")))

    levels = [
        ("identical question + table", lambda r: (question(r), tables(r))),
        ("same table AND same answer", cell),
        ("same table (any question)", tables),
        ("same document", docs),
    ]

    for label, key in levels:
        seen = {key(r) for r in first}
        repeats = [r for r in second if key(r) in seen]
        share = len(repeats) / max(len(second), 1)
        print(f"  {label:30s} {len(repeats):3d}/{len(second)} = {share:5.1%}")

    # Within batch 2 itself: a batch can also repeat itself.
    print()
    for label, key in levels[:3]:
        counts = Counter(key(r) for r in second)
        dupes = sum(c - 1 for c in counts.values() if c > 1)
        print(f"  self-repeats in batch 2, {label:30s} {dupes}")

    # Coverage is the positive side of the same question.
    print()
    all_docs = {d for r in rows for d in (r.get("relevant_docs") or [])}
    all_tables = {t for r in rows for t in (r.get("relevant_tables") or [])}
    print(f"  distinct documents across the pool: {len(all_docs)}")
    print(f"  distinct tables across the pool   : {len(all_tables)}")
    print(f"  corpus has 1,965 documents and 146,246 tables, so the pool is")
    print(f"  sampling {len(all_docs) / 1965:.1%} of documents so far.")


if __name__ == "__main__":
    main()
