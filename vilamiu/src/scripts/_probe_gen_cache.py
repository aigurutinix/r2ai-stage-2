"""How much of the corpus does the LLM branch actually get to see?

Two teams now score above 0.50 EXECUTION where we score 0.3281, and our own
numbers say where the difference could live: DOCS_F2 is 0.9628 and TABLES_F2 is
0.5641, so we hand the model the right report and the wrong table inside it about
two thirds of the time. A model cannot write pandas over a table it was not given.

This reports the shape of the cached generation: coverage over the 1012 questions,
how many tables each prompt carried, and how far the branch gets — because the
cheapest possible fix is to widen the context rather than to sharpen the pick.

Usage:  PYTHONPATH=src python scripts/_probe_gen_cache.py [cache.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "gen_helpers.jsonl"


def main() -> None:
    rows = [
        json.loads(line)
        for line in CACHE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = {row["id"] for row in rows}
    print(f"{CACHE.name}: {len(rows)} rows, {len(ids)} distinct question ids of 1012")

    ok = [r for r in rows if r.get("ok")]
    valued = [r for r in rows if r.get("value") is not None]
    print(f"  ok=True                : {len(ok)}")
    print(f"  value is not None      : {len(valued)}")
    print(f"  usable (ok and valued) : {len({r['id'] for r in rows if r.get('ok') and r.get('value') is not None})}")

    print(f"\n  tables handed to the model, per question:")
    for count, n in sorted(Counter(len(r.get("keys", [])) for r in rows).items()):
        print(f"    {count} tables  {n:5d} questions")

    print(f"\n  keys of the questions with no usable program (first 5):")
    bad = [r for r in rows if not (r.get("ok") and r.get("value") is not None)]
    for r in bad[:5]:
        print(f"    id={r['id']:4d} keys={len(r.get('keys', []))} "
              f"error={str(r.get('error'))[:70]}")

    missing = sorted(set(range(1, 1013)) - ids)
    print(f"\n  question ids with NO cached generation at all: {len(missing)}")
    print(f"    {missing[:25]}")

    sample = next(r for r in rows if r.get("ok"))
    print(f"\n  sample row keys: {list(sample)}")
    print(f"  sample variables: {sample.get('variables')}")
    print(f"  sample code (first 300 chars):\n{str(sample.get('code'))[:300]}")


if __name__ == "__main__":
    main()
