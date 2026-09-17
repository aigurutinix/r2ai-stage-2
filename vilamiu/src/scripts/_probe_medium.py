"""What a derived (medium) record actually contains, before deciding how to keep it.

`build_sft.py` discards almost every medium record, and the proposed fix searches
every cell of both frames for a pair that reproduces the answer. Whether that is
sound depends on something only the data can say: if the generator's own
`pandas_query` already names both cells, parsing it is exact, while a search over
~400x400 cell pairs with three operations invites a coincidental match — and a
coincidental match is a training pair with a fabricated label, which this file's
own docstring calls worse than no pair.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line.replace(": NaN", ": null")))
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = load(ROOT / "artifacts" / "sft_records.jsonl")
    medium = [r for r in rows if r.get("difficulty") == "medium"]
    print(f"{len(rows)} records, {len(medium)} medium\n")

    # How many gold tables, and what shape is the gold program?
    print("gold tables per medium record:",
          dict(collections.Counter(len(r.get("relevant_tables", [])) for r in medium)))

    iloc = re.compile(r"(df\d*)\s*\.\s*iloc\s*\[")
    shapes: collections.Counter[str] = collections.Counter()
    for record in medium:
        program = str(record.get("pandas_query", ""))
        frames = sorted(set(iloc.findall(program)))
        lines = len([l for l in program.splitlines() if l.strip()])
        shapes[f"{len(frames)} frame(s) via iloc, {lines} line(s)"] += 1
    print("program shape:", dict(shapes))

    print("\n" + "=" * 70)
    for record in medium[:3]:
        print(f"Q: {record['question'][:150]}")
        print(f"answer: {record['answer']!r}")
        print(f"tables: {record.get('relevant_tables')}")
        print("program:")
        for line in str(record.get("pandas_query", "")).splitlines():
            print(f"    {line}")
        print("-" * 70)


if __name__ == "__main__":
    main()
