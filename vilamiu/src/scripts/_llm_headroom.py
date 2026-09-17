"""How much of the shipped submission still declines an available 14B program.

`patch_submission` only substitutes where the current answer is provably wrong or
came from a branch measured weaker than the generator. Everything else keeps the
rule answer. This counts that population and how far the two disagree, which is
the size of the decision we cannot settle without a labelled set.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else "retry14b.zip"
    cache = sys.argv[2] if len(sys.argv) > 2 else "artifacts/gen14b_merged.jsonl"

    with zipfile.ZipFile(ROOT / "submissions" / base) as z:
        shipped = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    generated: dict[int, dict] = {}
    for line in (ROOT / cache).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("value") is not None:
                generated[row["id"]] = row

    used = agree = disagree = no_program = 0
    for qid, record in shipped.items():
        row = generated.get(qid)
        if row is None:
            no_program += 1
            continue
        if (record.get("pandas_query") or "").strip() == (row.get("code") or "").strip():
            used += 1
            continue
        shipped_value = float(record.get("answer") or 0.0)
        llm_value = float(row["value"])
        if abs(shipped_value - llm_value) <= 0.01:
            agree += 1
        else:
            disagree += 1

    print(f"base={base}  cache={cache}")
    print(f"  shipping the 14B program        : {used}")
    print(f"  program exists, answers agree   : {agree}")
    print(f"  program exists, answers DIFFER  : {disagree}   <- the undecided pool")
    print(f"  no usable program               : {no_program}")


if __name__ == "__main__":
    main()
