"""Use the stored cell plans as a third voter on the undecided pool.

`_llm_headroom` found 344 questions where the shipped rule answer and the 14B
program answer disagree and nothing on disk can say which is right. A tie needs
an independent third opinion, and `artifacts/planned.jsonl` already holds one:
cell coordinates chosen by a model that was asked for coordinates, not for code.

Coordinates are the reason this is worth counting. The rule branch reaches its
figure by matching a label, the program branch by writing pandas; the plan branch
by naming a row and a column. Three mechanisms that fail differently, so where
two of them land on the same number the number is unlikely to be a coincidence
of a shared bug.
"""

from __future__ import annotations

import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The organisers compare with math.isclose(rel_tol=0.0, abs_tol=1e-2), so two
# answers that differ by more than a cent are different answers, full stop.
TOL = 1e-2


def load_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("value") is not None:
                rows[row["id"]] = row
    return rows


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else "retry14b.zip"

    with zipfile.ZipFile(ROOT / "submissions" / base) as z:
        shipped = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    program = load_jsonl(ROOT / "artifacts" / "gen14b_merged.jsonl")
    plan = load_jsonl(ROOT / "artifacts" / "planned.jsonl")

    print(f"shipped={len(shipped)}  program={len(program)}  plan={len(plan)}")
    print(f"plan ops: {collections.Counter(r.get('op') for r in plan.values()).most_common()}")

    buckets: collections.Counter = collections.Counter()
    flip: list[tuple[int, float, float]] = []

    for qid, record in shipped.items():
        ship = float(record.get("answer") or 0.0)
        prog = program.get(qid)
        cell = plan.get(qid)
        ships_program = (
            prog is not None
            and (record.get("pandas_query") or "").strip() == (prog.get("code") or "").strip()
        )
        if ships_program:
            buckets["ships the program already"] += 1
            continue
        if prog is None:
            buckets["no program" + (" / has plan" if cell else " / no plan")] += 1
            continue
        if close(ship, float(prog["value"])):
            buckets["program agrees with shipped"] += 1
            continue

        # Undecided: rule answer vs program answer, and they differ.
        if cell is None:
            buckets["UNDECIDED, no plan to break tie"] += 1
        elif close(float(cell["value"]), float(prog["value"])):
            buckets["UNDECIDED -> plan backs the PROGRAM"] += 1
            flip.append((qid, ship, float(prog["value"])))
        elif close(float(cell["value"]), ship):
            buckets["UNDECIDED -> plan backs the SHIPPED rule"] += 1
        else:
            buckets["UNDECIDED -> plan says a third thing"] += 1

    for name, count in buckets.most_common():
        print(f"  {count:4d}  {name}")

    out = ROOT / "artifacts" / "_plan_backed_flips.json"
    out.write_text(json.dumps([q for q, _, _ in flip]), encoding="utf-8")
    print(f"\n{len(flip)} ids where an independent plan backs the program -> {out.name}")
    for qid, ship, prog_value in flip[:12]:
        print(f"    id={qid:4d}  shipped={ship!r}  program+plan={prog_value!r}")


if __name__ == "__main__":
    main()
