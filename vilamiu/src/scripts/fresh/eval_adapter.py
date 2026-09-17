"""Score an adapter run against the base, on references no adapter helped produce.

Four adapters have been trained in this project and every one was declared better by the
metric it was tuned on, then lost on the board. The gate this time is written down before
the training starts, and both arms are scored by the same code:

  arm A (base)     few-shot prompts (`prompts_full.jsonl`), the strongest measured base
                   configuration — p_pick 0.68, 65% tie-breaking. Beating zero-shot does
                   not count; that bar fooled us twice.
  arm B (adapter)  train-matched prompts (`prompts_adapter.jsonl`), no examples.

References, both from before any adapter existed:

  eval_trio.json       65 questions where three independent mechanisms computed the same
                       figure from the same cell — the purest labels available
  eval_consensus.json  417 questions where two reasoning passes agreed — noisier, but the
                       noise is identical for both arms, so the COMPARISON stands

The adapter ships only if it beats the base on BOTH references. A split verdict means the
adapter learned the generator's dialect but not the exam's — the recorded failure mode.

Usage, after executing each arm's replies through run_programs.py:
  python scripts/fresh/eval_adapter.py --base artifacts/fresh/base_results.jsonl \
      --adapter artifacts/fresh/adapter_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(relative: str) -> dict[int, float]:
    out: dict[int, float] = {}
    path = ROOT / relative
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("answer") in (None, ""):
            continue
        try:
            out[int(row["id"])] = float(row["answer"])
        except (TypeError, ValueError):
            continue
    return out


def score(name: str, answers: dict[int, float],
          reference: dict[int, float]) -> tuple[int, int]:
    hit = miss = 0
    for qid, truth in reference.items():
        got = answers.get(qid)
        if got is None:
            continue
        if abs(got - truth) <= max(0.011, abs(truth) * 1e-6):
            hit += 1
        else:
            miss += 1
    total = hit + miss
    print(f"    {name:10s} tra loi {total:4d}/{len(reference)}   "
          f"dung {hit:4d} ({hit / max(total, 1) * 100:.0f}% cua tra loi, "
          f"{hit / max(len(reference), 1) * 100:.0f}% cua bo cham)")
    return hit, total


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()

    base = load(args.base)
    adapter = load(args.adapter)

    verdicts = []
    for label, path in (("TRIO (65 cau, nhan sach)", "artifacts/fresh/eval_trio.json"),
                        ("DONG THUAN (417 cau)",
                         "artifacts/fresh/eval_consensus.json")):
        reference = {int(k): float(v) for k, v in json.loads(
            (ROOT / path).read_text(encoding="utf-8")).items()}
        print(f"\n  {label}")
        base_hit, _ = score("base", base, reference)
        adapter_hit, _ = score("adapter", adapter, reference)
        verdicts.append(adapter_hit > base_hit)

    print()
    if all(verdicts):
        print("  => DAT: adapter thang base tren CA HAI bo cham. Duoc phep dung.")
    else:
        print("  => TRUOT: adapter khong thang base o it nhat mot bo cham."
              " Giu base, khong dung adapter — day la che do loi da ghi so.")


if __name__ == "__main__":
    main()
