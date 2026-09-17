"""Is agreement between two independent models a signal worth acting on?

Their EXECUTION jumped 0.3538 -> 0.5119 in a day with every retrieval metric
byte-identical, and they say the model is 9B — so the gap is method, not model or
retrieval. The technique whose signature matches that shape is self-consistency:
sample the same question several times and keep the majority answer instead of
the first program that runs. We have never done it — `ChatClient.temperature` is
0.0, so a second run reproduces the first exactly.

Before spending hours sampling, the hypothesis can be tested for free on data
already on disk. Two caches hold independent opinions on the same questions:
`gen_helpers.jsonl` (Qwen3-8B, 6 tables) and `gen14b_v2.jsonl` (Qwen3-14B, 8).
If consensus carries information, then on the subset where the two models agree,
agreement with the 42.8% label matcher should be markedly higher than on the
subset where they disagree.

The label matcher is the yardstick because its accuracy is the one number the
leaderboard has pinned. It is not gold, so what follows is a comparison of two
conditional rates, not an accuracy — but both are measured the same way against
the same reference, so their difference is readable.

Usage:  PYTHONPATH=src python scripts/_probe_consensus_signal.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

BASE = ROOT / "submissions" / "screen_ratio_gated.zip"


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 2e-4


def load(name: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in (ROOT / "artifacts" / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def main() -> None:
    eight, fourteen = load("gen_helpers.jsonl"), load("gen14b_v2.jsonl")
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

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
        return float(outcome.value)

    # Restrict to the matcher's own pool — the only questions with a pinned
    # reference accuracy. A shipped program that is a single-frame lookup with no
    # generated prelude came from that branch or from the fallback that shares
    # its generator; the pool is mixed, but identically mixed in both columns.
    pool = []
    for qid, record in base.items():
        code = (record.get("pandas_query") or "").strip()
        if not code or code == "result = 0.0":
            continue
        if "def num(" in code or "def find_row(" in code:
            continue
        if len(set(re.findall(r"\bdf\d*\b", code))) != 1:
            continue
        pool.append(qid)

    agree_both = disagree_both = 0
    hit_when_agree = hit_when_disagree = 0
    only_one = 0
    for qid in pool:
        a, b = value_of(eight.get(qid)), value_of(fourteen.get(qid))
        if a is None or b is None:
            only_one += 1
            continue
        try:
            shipped = float(base[qid].get("answer") or 0)
        except (TypeError, ValueError):
            continue
        models_agree = close(a, b)
        # Where the two models agree, does their shared answer match the branch
        # whose accuracy we know?
        matches = close(b, shipped)
        if models_agree:
            agree_both += 1
            hit_when_agree += matches
        else:
            disagree_both += 1
            hit_when_disagree += matches

    print(f"matcher-pool questions: {len(pool)}")
    print(f"  only one model produced a value: {only_one}")
    print(f"\n  the two models AGREE     : {agree_both:4d}"
          f"   of those, match the matcher: {hit_when_agree:4d}"
          f"  ({hit_when_agree / max(agree_both, 1):.1%})")
    print(f"  the two models DISAGREE  : {disagree_both:4d}"
          f"   of those, 14B matches      : {hit_when_disagree:4d}"
          f"  ({hit_when_disagree / max(disagree_both, 1):.1%})")

    total = agree_both + disagree_both
    if total:
        print(f"\n  consensus covers {agree_both}/{total} = {agree_both / total:.1%} "
              f"of the pool")
    print("\n  A large gap between the two rates means consensus separates the")
    print("  questions the models get right from the ones they do not — which is")
    print("  what makes majority-of-k worth the sampling cost. A small gap means")
    print("  the two models fail together, and sampling will not rescue them.")


if __name__ == "__main__":
    main()
