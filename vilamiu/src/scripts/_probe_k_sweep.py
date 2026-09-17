"""Does declaring fewer tables raise F2, at a fixed set of answers?

F2 = 5h / (4g + k) for g gold tables, k declared, h of them right. So k can be
swept offline with no API cost, and the sweep has been run before: the peak sits
near 6.45 and the curve is flat to within 0.001 either side.

What the earlier sweep could not answer is the question actually being asked --
fewer tables *without* losing recall. Trimming the ranked tail lowers k and drops
recall together. A rival at precision 0.6164 and recall 0.6496 is not trimming; it
is declaring the right tables in the first place.

So this sweep separates the two. Declarations are built the way the submission
builds them: the tables the program actually read come first (evidence), then the
ranked list pads to k. Evidence quality is fixed by the answering step, so the
sweep shows how much of the gap k can close on its own -- and how much it cannot.

Usage:
  PYTHONPATH=src python scripts/_probe_k_sweep.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = [
    ("easy_416", "artifacts/ab5_easy_416_base.jsonl"),
    ("gold_multistep", "artifacts/ab5_gold_multistep_base.jsonl"),
]
FRAME_RE = re.compile(r"num\(\s*(df\d+)")


def f2(hits: int, gold: int, declared: int) -> float:
    if gold == 0 or declared == 0:
        return 0.0
    return 5.0 * hits / (4.0 * gold + declared)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for label, path in ARMS:
        rows = [
            json.loads(l) for l in (ROOT / path).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        # The prompt bound df1..dfn to the shortlist in order, so a frame named in
        # the program identifies the table it read.
        usable = []
        for row in rows:
            gold = set(row.get("gold_tables") or [])
            if not gold:
                continue
            shortlist = row.get("shortlist") or []
            evidence_idx = sorted(
                {int(m[2:]) - 1 for m in FRAME_RE.findall(row.get("code") or "")})
            usable.append((gold, shortlist, evidence_idx, row))

        print(f"\n== {label}: {len(usable)} questions with gold tables")
        if not usable:
            print("   no shortlist recorded in this file -- cannot rebuild "
                  "declarations, so the sweep is not runnable here")
            continue

        print(f"   {'k':>4} {'F2':>8} {'precision':>10} {'recall':>8}")
        for k in range(1, 13):
            total_f2 = hits_sum = decl_sum = gold_sum = 0.0
            for gold, shortlist, evidence_idx, _ in usable:
                declared = []
                for i in evidence_idx:
                    if i < len(shortlist) and shortlist[i] not in declared:
                        declared.append(shortlist[i])
                for ref in shortlist:
                    if len(declared) >= k:
                        break
                    if ref not in declared:
                        declared.append(ref)
                declared = declared[:k]
                hits = len(set(declared) & gold)
                total_f2 += f2(hits, len(gold), len(declared))
                hits_sum += hits
                decl_sum += len(declared)
                gold_sum += len(gold)
            n = len(usable)
            print(f"   {k:>4} {total_f2 / n:8.4f} {hits_sum / max(decl_sum, 1):10.4f} "
                  f"{hits_sum / max(gold_sum, 1):8.4f}")


if __name__ == "__main__":
    main()
