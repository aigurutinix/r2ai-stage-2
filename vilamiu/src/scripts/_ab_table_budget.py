"""A/B the table budget on the questions both caches cover.

`gen14b_all.jsonl` was generated while `search_balanced` was left at its default
`per_group=2`, so a one-company question saw two tables however large `--tables`
was. `gen14b_v2.jsonl` spreads the budget over however many (ticker, year) groups
the question actually has, so the same questions now see six or eight.

Restricting to the ids present in both makes this a controlled comparison: same
model, same prompt, same questions, one variable. What it measures is agreement
with the shipped answer, which is a proxy — but the *difference* between two
proxies measured the same way is a fair read on whether the wider context helped.

Usage:  PYTHONPATH=src python scripts/_ab_table_budget.py
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

BASE = ROOT / "submissions" / "screen_ratio_gated.zip"


def load(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 2e-4


def main() -> None:
    old = load(ROOT / "artifacts" / "gen14b_all.jsonl")
    new = load(ROOT / "artifacts" / "gen14b_v2.jsonl")
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    shared = sorted(set(old) & set(new))
    # Only ids whose table budget actually changed carry information.
    widened = [i for i in shared if len(new[i].get("keys", [])) > len(old[i].get("keys", []))]
    print(f"ids in both caches: {len(shared)}   budget actually widened: {len(widened)}")
    if not widened:
        print("nothing to compare yet — wait for more rows")
        return

    def evaluate(row) -> float | None:
        if not row.get("ok"):
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
        return float(outcome.value)

    tally = {"old_runs": 0, "new_runs": 0, "old_agree": 0, "new_agree": 0,
             "both_run": 0, "both_agree": 0, "flipped_to_agree": 0,
             "flipped_to_disagree": 0}
    for qid in widened:
        try:
            shipped = float(base[qid].get("answer") or 0)
        except (TypeError, ValueError):
            continue
        a, b = evaluate(old[qid]), evaluate(new[qid])
        tally["old_runs"] += a is not None
        tally["new_runs"] += b is not None
        agree_old = a is not None and close(a, shipped)
        agree_new = b is not None and close(b, shipped)
        tally["old_agree"] += agree_old
        tally["new_agree"] += agree_new
        if a is not None and b is not None:
            tally["both_run"] += 1
            tally["both_agree"] += close(a, b)
            if agree_new and not agree_old:
                tally["flipped_to_agree"] += 1
            if agree_old and not agree_new:
                tally["flipped_to_disagree"] += 1

    n = len(widened)
    print(f"\non the {n} questions whose table budget grew:")
    print(f"  program executes      old {tally['old_runs']:4d} ({tally['old_runs']/n:5.1%})"
          f"   new {tally['new_runs']:4d} ({tally['new_runs']/n:5.1%})")
    print(f"  agrees with shipped   old {tally['old_agree']:4d} ({tally['old_agree']/n:5.1%})"
          f"   new {tally['new_agree']:4d} ({tally['new_agree']/n:5.1%})")
    print(f"\n  both programs run     {tally['both_run']}")
    print(f"  the two agree         {tally['both_agree']}")
    print(f"  new agrees, old did not   {tally['flipped_to_agree']}")
    print(f"  old agreed, new does not  {tally['flipped_to_disagree']}")
    print("\n  Agreement with the shipped answer is a proxy, not accuracy — but both")
    print("  columns are the same proxy, so the difference between them is readable.")


if __name__ == "__main__":
    main()
