"""Is there a cliff in the anchor scores that a relative cutoff could sit in?

Declaring a fixed top-k treats rank 6 the same on a question where it scores 95%
of the leader as on one where it scores 40%. If the score falls off sharply after
the first table or two, a per-question cutoff at a fraction of the leader's score
trims the tail without touching questions where the field is genuinely flat.

The trade is legible through F2 = 5r / (4 + k/g): the surviving share of refs sets
k/g, and the only unknown is how much recall goes with the trimmed refs. That
cannot be measured without gold, so this reports both sides — the shrink in k and
the rank positions being dropped — and leaves the recall assumption explicit.

Usage:  PYTHONPATH=src python scripts/_probe_rel_cut.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

RANK = ROOT / "artifacts" / "anchor_rank_scored.jsonl"
BOARD_R, BOARD_F2, BOARD_KG = 0.7213, 0.5641, 2.39
CUTS = (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def declare_k(question) -> int:
    """The span policy the best submission was built with (k = 2*span, 3..11)."""

    span = max(1, len(question.tickers)) * max(1, len(question.years))
    return min(max(round(2.0 * min(span, 8)), 3), 11)


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    ranks: dict[int, list[float]] = {}
    for line in RANK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ranks[row["id"]] = [float(r.get("score", 0.0)) for r in row["refs"]]

    print(f"{len(ranks)} ranked questions from {RANK.name}")

    # The shape of the fall-off, measured only over the refs that would be
    # declared under the current policy.
    ratios_by_rank: dict[int, list[float]] = {}
    declared = 0
    for qid, question in parsed.items():
        scores = ranks.get(qid, [])[: declare_k(question)]
        if not scores or scores[0] <= 0:
            continue
        declared += len(scores)
        for position, score in enumerate(scores):
            ratios_by_rank.setdefault(position, []).append(score / scores[0])

    print(f"\nscore as a share of the leader, by rank position:")
    print(f"  rank    n   median    p25    p75")
    for position in sorted(ratios_by_rank)[:11]:
        values = sorted(ratios_by_rank[position])
        n = len(values)
        print(f"  {position + 1:4d} {n:5d}   {statistics.median(values):.3f}  "
              f"{values[n // 4]:.3f}  {values[3 * n // 4]:.3f}")

    print(f"\ncut   refs kept        k/g   k mean   TABLES_F2 if recall holds at")
    print(f"                                       {'1.00':>6} {'0.95':>6} {'0.90':>6} {'0.85':>6}")
    for cut in CUTS:
        kept_total = 0
        per_question: list[int] = []
        for qid, question in parsed.items():
            scores = ranks.get(qid, [])[: declare_k(question)]
            if not scores:
                per_question.append(0)
                continue
            top = scores[0]
            # The leader is always kept: a question with no declared table scores
            # zero on every table metric, which is strictly worse than a guess.
            kept = 1 + sum(1 for s in scores[1:] if top > 0 and s >= cut * top)
            kept_total += kept
            per_question.append(kept)
        share = kept_total / declared
        kg = BOARD_KG * share
        row = (f" {cut:.2f}  {kept_total:6d} ({share:4.0%})  {kg:4.2f}  "
               f"{statistics.mean(per_question):5.2f}  ")
        for hold in (1.00, 0.95, 0.90, 0.85):
            row += f" {5 * BOARD_R * hold / (4 + kg):6.4f}"
        print(row)
    print(f"\n baseline: k/g {BOARD_KG}, recall {BOARD_R}, TABLES_F2 {BOARD_F2}, "
          f"k mean {declared / len(parsed):.2f}")
    print(" macro moves by (new F2 - 0.5641)/3; every 0.03 F2 is 0.01 macro.")

    # How often does the cut bite at all? A cut that fires on few questions is
    # safe but worthless; one that fires everywhere is a fixed k in disguise.
    for cut in (0.35, 0.55, 0.75):
        bites = Counter()
        for qid, question in parsed.items():
            scores = ranks.get(qid, [])[: declare_k(question)]
            if not scores:
                continue
            top = scores[0]
            kept = 1 + sum(1 for s in scores[1:] if top > 0 and s >= cut * top)
            bites[len(scores) - kept] += 1
        print(f"\n cut {cut:.2f} — refs dropped per question: {dict(sorted(bites.items()))}")


if __name__ == "__main__":
    main()
