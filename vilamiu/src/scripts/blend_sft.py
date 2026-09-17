"""Blend SFT pair files so the training mix matches the exam's, not ours.

The point of the exercise. A set that is 92.7% one-cell lookups scored 69.4% on
its own held-out slice and went backwards on an exam that is 36.6% one-cell
lookups, because it teaches "the answer is a cell" and overwrites the base
model's ability to write anything longer (IMPLEMENTATION.md §18).

So the mix is chosen from the exam, not from what happens to be available.
Classes the prompt cannot physically serve — cohort screens over four or more
companies need one statement each, and eight truncated tables inside a 12k-token
budget cannot hold them — are excluded, and the remaining shares are
renormalised over what is left.

Sampling is by class, without replacement, from a seeded shuffle. Where a class
is short of its quota the shortfall is reported rather than back-filled from
another class, because silently over-representing whatever is abundant is the
failure this script exists to prevent.

Usage:
  PYTHONPATH=src python scripts/blend_sft.py \
      --single artifacts/sft_prog2.jsonl \
      --generated artifacts/sft_shapes.jsonl \
      --out artifacts/sft_mixed.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _probe_exam_classes import classify  # noqa: E402
from _mix import UNTRAINABLE  # noqa: E402

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exam_target(roster) -> dict[str, float]:
    counts: collections.Counter[str] = collections.Counter()
    for line in (ROOT / "data" / "questions" / "questions.jsonl") \
            .read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["question"]
        parsed = parse_question(0, text, roster)
        counts[classify(text, len(parsed.tickers), len(parsed.years))] += 1
    trainable = {k: n for k, n in counts.items() if k not in UNTRAINABLE}
    total = sum(trainable.values())
    return {k: n / total for k, n in trainable.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default="artifacts/sft_prog2.jsonl",
                        help="pairs whose target is a one-cell read")
    parser.add_argument("--generated", default="artifacts/sft_shapes.jsonl",
                        help="pairs from gen_shapes.py, already built into prompts")
    parser.add_argument("--out", default="artifacts/sft_mixed.jsonl")
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    target = exam_target(roster)

    pool: dict[str, list[dict]] = collections.defaultdict(list)
    for path in (ROOT / args.single, ROOT / args.generated):
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for pair in load(path):
            text = pair["meta"]["question"]
            parsed = parse_question(0, text, roster)
            name = classify(text, len(parsed.tickers), len(parsed.years))
            if name in UNTRAINABLE:
                continue
            pool[name].append(pair)

    rng = random.Random(args.seed)
    for pairs in pool.values():
        rng.shuffle(pairs)

    # The blend is as large as the binding class allows: for each class, how many
    # pairs in total its supply could support at the target share, and the
    # smallest such number is the ceiling.
    #
    # Only the classes that carry real weight get a vote. Two-company arithmetic
    # is 1.9% of the target and the pool holds a couple of dozen, so letting it
    # vote would cap the whole blend at ~1,200 pairs to protect a class worth
    # twenty of them. A class below the floor contributes whatever it has and the
    # shortfall is printed; a class above it is honoured exactly.
    MAJOR = 0.05
    ceiling = min(
        int(len(pool.get(name, [])) / share)
        for name, share in target.items()
        if share >= MAJOR and pool.get(name)
    )

    quota = {name: int(round(share * ceiling)) for name, share in target.items()}
    chosen: list[dict] = []
    print(f"{'class':<26}{'target':>8}{'quota':>8}{'supply':>8}{'taken':>8}")
    for name, share in sorted(target.items(), key=lambda kv: -kv[1]):
        supply = pool.get(name, [])
        take = min(quota[name], len(supply))
        chosen.extend(supply[:take])
        flag = "" if take == quota[name] else "  SHORT"
        print(f"{name:<26}{share:>8.1%}{quota[name]:>8}{len(supply):>8}{take:>8}{flag}")

    rng.shuffle(chosen)
    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as handle:
        for pair in chosen:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(chosen)} pairs -> {out}")
    achieved: collections.Counter[str] = collections.Counter()
    for pair in chosen:
        text = pair["meta"]["question"]
        parsed = parse_question(0, text, roster)
        achieved[classify(text, len(parsed.tickers), len(parsed.years))] += 1
    print("achieved mix:")
    for name, count in achieved.most_common():
        print(f"  {name:<26}{count:>6}{count / len(chosen):>8.1%}"
              f"   (exam {target.get(name, 0):.1%})")


if __name__ == "__main__":
    main()
