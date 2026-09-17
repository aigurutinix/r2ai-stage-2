"""A/B twenty tables against eight, on the control set, at temperature 0.

The one intervention that has measured positive on generation quality was widening
the table budget: 2 -> 6/8 moved agreement with the shipped answer from 41.4% to
48.1%. Self-consistency measured negative (-7.1%) and currency-unit handling came
back clean, so the table budget is the live hypothesis — and it fits the shape of
the leaderboard, where picking the document is solved for everyone (DOCS_F2 0.94-0.96)
and picking the table inside it is not (TABLES precision 0.28-0.34).

If more tables keeps helping, the read is that our weakest step — choosing the
table — is one the model can be allowed to do itself, given enough candidates.
If it stops helping or reverses, the budget has a peak and the 8 we use is near it.

Both arms are temperature 0 and share model, prompt and question set. The 8-table
arm is `gen14b_merged.jsonl`, generated earlier under exactly those settings.

Usage:  PYTHONPATH=src python scripts/_ab_t20.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

BASE = ROOT / "submissions" / "sub15.zip"


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 2e-4


def load(name: str) -> dict[int, dict]:
    path = ROOT / "artifacts" / name
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def main() -> None:
    arm_a = sys.argv[1] if len(sys.argv) > 1 else "gen14b_merged.jsonl"
    arm_b = sys.argv[2] if len(sys.argv) > 2 else "gen_t20.jsonl"
    eight, twenty = load(arm_a), load(arm_b)
    print(f"A = {arm_a}   B = {arm_b}")
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ids = [
        qid for qid in json.loads(
            (ROOT / "artifacts" / "_consensus_ids.json").read_text())
        if qid in twenty
    ]

    def value_of(row) -> float | None:
        if row is None or not row.get("ok"):
            return None
        code = row["code"]
        if reads_no_frame(code):
            return None
        keys = [TableKey(d, int(t)) for d, t in row["keys"]]
        names = list(row["variables"])
        try:
            outcome = run_query(code, {n: store.rows(k) for n, k in zip(names, keys)})
        except Exception:
            return None
        if not outcome.ok or outcome.value is None:
            return None
        value = float(outcome.value)
        return None if value != value or abs(value) == float("inf") else value

    tally = {"n": 0, "a_runs": 0, "b_runs": 0, "a_hit": 0, "b_hit": 0,
             "both": 0, "same": 0, "b_only_hit": 0, "a_only_hit": 0}
    widened = 0
    for qid in ids:
        record = base.get(qid)
        if record is None:
            continue
        try:
            shipped = float(record.get("answer") or 0)
        except (TypeError, ValueError):
            continue
        tally["n"] += 1
        if len(twenty[qid].get("keys", [])) > len(eight.get(qid, {}).get("keys", [])):
            widened += 1
        a, b = value_of(eight.get(qid)), value_of(twenty.get(qid))
        tally["a_runs"] += a is not None
        tally["b_runs"] += b is not None
        hit_a = a is not None and close(a, shipped)
        hit_b = b is not None and close(b, shipped)
        tally["a_hit"] += hit_a
        tally["b_hit"] += hit_b
        if a is not None and b is not None:
            tally["both"] += 1
            tally["same"] += close(a, b)
            tally["b_only_hit"] += hit_b and not hit_a
            tally["a_only_hit"] += hit_a and not hit_b

    n = max(tally["n"], 1)
    print(f"control questions with a 20-table run: {tally['n']}"
          f"   (budget actually grew on {widened})")
    print(f"\n  {'':22s} {'8 tables':>10} {'20 tables':>10}")
    print(f"  {'program executes':22s} {tally['a_runs']:10d} {tally['b_runs']:10d}")
    print(f"  {'  as a rate':22s} {tally['a_runs']/n:9.1%} {tally['b_runs']/n:9.1%}")
    print(f"  {'agrees with shipped':22s} {tally['a_hit']:10d} {tally['b_hit']:10d}")
    print(f"  {'  as a rate':22s} {tally['a_hit']/n:9.1%} {tally['b_hit']/n:9.1%}")
    print(f"\n  both arms ran        {tally['both']}")
    print(f"  the two agree        {tally['same']}")
    print(f"  20 right, 8 wrong    {tally['b_only_hit']}")
    print(f"  8 right, 20 wrong    {tally['a_only_hit']}")
    delta = (tally["b_hit"] - tally["a_hit"]) / n
    print(f"\n  lift from 20 tables: {delta:+.1%}")


if __name__ == "__main__":
    main()
