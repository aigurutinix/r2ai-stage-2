"""The bare-prompt model hard-codes the answer. Is the answer in the tables?

Stripping our six clauses and the `num`/`find_row` prelude produced 108 programs
that execute — and 103 of them never touch a DataFrame. They look like:

    result = 10.009282192e9 / 1e6
    result = round(result, 2)

The model read the figure out of the rendered table in the prompt and wrote it
down. Such a program is inadmissible: it fails the organisers' manual review, and
`reads_no_frame` already rejects it.

But it says something the code-writing path cannot. **The model located the
number.** If our bottleneck were finding the right cell, this would not happen; it
happens because expressing the lookup as table-navigating code is a separate
skill, and that is where the programs die.

If a hard-coded constant matches a cell in one of the tables the model was given,
the program can be rebuilt mechanically — emit `num(df_k, r, c)` for that cell —
and the result is legitimate, verifiable, and reads a real frame. This measures
how often that match exists.

Usage:  PYTHONPATH=src python scripts/_probe_constant_to_cell.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

CACHE = ROOT / "artifacts" / "gen_bare.jsonl"
# The scorer's tolerance is 0.02% relative; matching a cell needs the same.
TOL = 2e-4
# Unit conversions a program may legitimately apply between cell and answer.
SCALES = (1.0, 1e3, 1e6, 1e9, 1e12, 1e-3, 1e-6, 1e-9, 1e-12, 100.0, 0.01)


def parse_cell(text: str) -> float | None:
    raw = str(text).strip()
    if not raw or raw in ("-", "--", "nan", "None"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?\d{1,3}(\.\d{3})*(,\d+)?|-?\d+([.,]\d+)?", raw):
        return None
    raw = raw.replace(".", "").replace(",", ".") if raw.count(".") > 1 or "," in raw else raw
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    rows = [
        json.loads(line)
        for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    constants = []
    for row in rows:
        if not row.get("ok") or row.get("value") is None:
            continue
        if not reads_no_frame(row["code"]):
            continue
        constants.append(row)
    print(f"{len(rows)} bare rows, {len(constants)} of them hard-coded constants")

    matched = unmatched = 0
    scale_used: dict[float, int] = {}
    for row in constants:
        target = float(row["value"])
        if target == 0:
            unmatched += 1
            continue
        keys = [TableKey(d, int(t)) for d, t in row["keys"]]
        found = False
        for key in keys:
            grid = store.rows(key)
            for line in grid:
                for cell in line:
                    value = parse_cell(cell)
                    if value is None or value == 0:
                        continue
                    for scale in SCALES:
                        got = value * scale
                        if abs(got - target) <= TOL * max(abs(got), abs(target)):
                            found = True
                            scale_used[scale] = scale_used.get(scale, 0) + 1
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        matched += found
        unmatched += not found

    total = matched + unmatched or 1
    print(f"\n  constant matches a cell in a table the model was given: "
          f"{matched}/{total} = {matched / total:.1%}")
    print(f"  no matching cell                                       : {unmatched}")
    print(f"\n  scale between cell and answer: "
          f"{dict(sorted(scale_used.items(), key=lambda kv: -kv[1]))}")
    print("\n  A high match rate means these answers can be rebuilt as real")
    print("  cell reads — legitimate programs, and evidence that locating the")
    print("  figure is not what our pipeline is failing at.")


if __name__ == "__main__":
    main()
