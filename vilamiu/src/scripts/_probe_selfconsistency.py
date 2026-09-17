"""Does majority-of-k beat a single sample, and by how much?

Their EXECUTION went 0.3538 -> 0.5119 in a day with every retrieval metric
unchanged, on a model they say is 9B. So the gap is technique. Self-consistency
is the technique whose signature fits: same model, same retrieval, large jump.
We have never sampled — `ChatClient.temperature` is 0.0, so a second run
reproduces the first exactly.

Three independent samples at temperature 0.8 were generated for 200 questions
from the label matcher's own pool. That pool is the only one with a pinned
reference: the matcher scores **42.8%** on the leaderboard. Agreement with it is
a proxy, not accuracy — but every column below is the same proxy, so the
comparison between them is fair.

What decides whether to roll this out to all 1,012 questions:

  * **coverage** — on what share of questions do at least two of three samples
    agree? A consensus that almost never forms cannot carry the pipeline.
  * **separation** — is agreement with the matcher much higher inside the
    consensus subset than outside it? That is what makes the majority worth
    keeping and the disagreements worth routing elsewhere.
  * **lift** — does the majority answer agree with the matcher more often than a
    single sample does? That is the number that would move the score.

Usage:  PYTHONPATH=src python scripts/_probe_selfconsistency.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

BASE = ROOT / "submissions" / "sub15.zip"
SAMPLES = ("gen_s1.jsonl", "gen_s2.jsonl", "gen_s3.jsonl")


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
    caches = [load(name) for name in SAMPLES]
    single = load("gen14b_merged.jsonl")
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ids = json.loads((ROOT / "artifacts" / "_consensus_ids.json").read_text())

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

    def majority(values: list[float]) -> tuple[float, int] | None:
        """The most-agreed value and how many samples backed it."""

        best: tuple[float, int] | None = None
        for candidate in values:
            votes = sum(1 for other in values if close(candidate, other))
            if best is None or votes > best[1]:
                best = (candidate, votes)
        return best

    counts: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    ran = 0
    for qid in ids:
        record = base.get(qid)
        if record is None:
            continue
        try:
            shipped = float(record.get("answer") or 0)
        except (TypeError, ValueError):
            continue
        values = [v for v in (value_of(c.get(qid)) for c in caches) if v is not None]
        if not values:
            counts["no_sample_ran"] += 1
            continue
        ran += 1
        top = majority(values)
        assert top is not None
        value, votes = top
        bucket = f"{votes}_of_{len(values)}"
        counts[bucket] += 1
        hits[bucket] += close(value, shipped)

        counts["majority_total"] += 1
        hits["majority_total"] += close(value, shipped)
        one = value_of(single.get(qid))
        if one is not None:
            counts["single_total"] += 1
            hits["single_total"] += close(one, shipped)

    print(f"control set: {len(ids)} questions from the matcher pool "
          f"(matcher scores 42.8% on the board)")
    print(f"  at least one sample ran: {ran}   none ran: {counts['no_sample_ran']}\n")

    print("  agreement with the shipped answer, split by how many samples agreed:")
    for bucket in sorted(k for k in counts if "_of_" in k):
        n = counts[bucket]
        print(f"    {bucket:10s} {n:4d}   match {hits[bucket]:4d}  "
              f"({hits[bucket] / max(n, 1):5.1%})")

    for label in ("single_total", "majority_total"):
        n = counts[label]
        if n:
            print(f"\n  {label:16s} {n:4d}   match {hits[label]:4d}  "
                  f"({hits[label] / n:5.1%})")

    if counts["single_total"] and counts["majority_total"]:
        lift = (hits["majority_total"] / counts["majority_total"]
                - hits["single_total"] / counts["single_total"])
        print(f"\n  lift from majority-of-3 over one sample: {lift:+.1%}")
        print("  A clear positive lift, plus a wide split between the unanimous and")
        print("  the split buckets, is what justifies sampling all 1,012 questions.")


if __name__ == "__main__":
    main()
