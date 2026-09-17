"""Estimate the new model's accuracy without gold, by agreement with a measured branch.

We have no labelled answers, so no branch's accuracy can be measured here directly.
One number is known from the leaderboard though: the lexical label matcher scores
**42.8%** on the questions it serves. That makes it a yardstick. If a freshly
generated program agrees with the matcher on a large share of those questions,
the generator is at least in the matcher's league; if it disagrees almost
everywhere, it is producing noise that happens to execute.

The reasoning is bounded, and worth stating so it is not over-read. Agreement is
a lower bound on nothing and an upper bound on nothing — two mechanisms can be
wrong together. What it does do is separate the two hypotheses that matter here:

    high agreement  -> the generator reads tables the way the matcher does, so
                       routing more questions to it is a small extrapolation
    low agreement   -> the generator is doing something else entirely, and its
                       27% historical accuracy is the only estimate we have

`helpers` is the standing warning: +147 executable programs bought +2 correct
answers, because executability was measured and accuracy was assumed.

Usage:  PYTHONPATH=src python scripts/_probe_agreement.py artifacts/gen14b_all.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "gen14b_all.jsonl"
BASE = ROOT / "submissions" / "screen_ratio_gated.zip"


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 2e-4  # the scorer's tolerance


def main() -> None:
    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    rows = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    print(f"{CACHE.name}: {len(rows)} rows")

    def branch(record) -> str:
        """Which mechanism wrote this program, read off its shape.

        Counting occurrences of the substring "df" mislabels every single-table
        program, because a lookup names the same frame three or four times. The
        number of *distinct* frame variables is what separates a one-table read
        from a composition.
        """

        code = (record.get("pandas_query") or "").strip()
        if code in ("", "result = 0.0"):
            return "zero"
        frames = set(re.findall(r"\bdf\d*\b", code))
        if "def num(" in code or "find_row(" in code:
            return "llm"
        if len(frames) >= 2:
            return "multi"
        if "iloc" in code:
            return "single"
        return "other"

    stats: Counter[str] = Counter()
    agree: Counter[str] = Counter()
    total: Counter[str] = Counter()

    for qid, row in rows.items():
        record = base.get(qid)
        if record is None:
            continue
        stats["cached"] += 1
        if not row.get("ok") or row.get("value") is None:
            stats["not_executable"] += 1
            continue
        code = row["code"]
        if reads_no_frame(code):
            stats["hardcoded"] += 1
            continue
        keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
        names = list(row["variables"])
        try:
            outcome = run_query(code, {n: store.rows(k) for n, k in zip(names, keys)})
        except Exception:
            stats["raised"] += 1
            continue
        if not outcome.ok or outcome.value is None:
            stats["failed_here"] += 1
            continue
        stats["runs"] += 1
        try:
            theirs = float(record.get("answer") or 0)
        except (TypeError, ValueError):
            continue
        kind = branch(record)
        total[kind] += 1
        if close(float(outcome.value), theirs):
            agree[kind] += 1

    print(f"\n{dict(stats)}")
    print(f"\nagreement with the shipped answer, by the branch that produced it:")
    print(f"  {'branch':10s} {'n':>6} {'agree':>7} {'rate':>7}   known accuracy")
    known = {"single": "42.8% (label matcher)", "llm": "27%", "multi": "7-30%",
             "zero": "0% by construction", "other": ""}
    for kind in sorted(total, key=lambda k: -total[k]):
        rate = agree[kind] / total[kind] if total[kind] else 0.0
        print(f"  {kind:10s} {total[kind]:6d} {agree[kind]:7d} {rate:6.1%}   {known.get(kind, '')}")

    n = sum(total.values()) or 1
    print(f"\n  overall {sum(agree.values())}/{n} = {sum(agree.values()) / n:.1%}")
    print("\n  Read this as a discriminator, not an accuracy. High agreement on the")
    print("  `single` row means the generator reads tables like the 42.8% branch.")


if __name__ == "__main__":
    main()
