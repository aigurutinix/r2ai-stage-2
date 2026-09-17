"""Are our currency answers in the unit the question asked for?

Self-consistency at temperature 0.8 measured *worse* than a single greedy sample,
so the 9B team's lead is not that. The next candidate is duller and bigger: unit
handling. Tonight's patches already recovered 18 graded questions that were
nothing but unit errors — a đồng amount shipped for a "%" question, a share count
shipped for a "triệu cổ phiếu" question — and those were only the ones whose
magnitude made them *provably* wrong.

The same defect one notch quieter would be invisible to that test. A question
asking "bao nhiêu nghìn tỷ đồng" answered with 1.7e12 is a raw đồng figure, but
1.7e12 does not trip a 1e16 ceiling. What catches it is the distribution: within
one unit family the answers should cluster in a band a human would recognise, and
a second cluster three orders of magnitude away is a conversion that never ran.

Usage:  PYTHONPATH=src python scripts/_probe_unit_magnitudes.py [submission.zip]
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "sub15.zip"


def main() -> None:
    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(ROOT / "submissions" / SUB) as z:
        preds = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    def answer(qid: int) -> float:
        try:
            return float(preds[qid].get("answer") or 0)
        except (TypeError, ValueError):
            return 0.0

    by_unit: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for qid, question in parsed.items():
        if question.target_unit:
            by_unit[question.target_unit].append((qid, answer(qid)))

    print(f"{SUB}: order-of-magnitude spread within each unit family\n")
    print(f"  {'unit':16s} {'n':>4}  {'scale':>7}   distribution of log10|answer|")
    for unit in sorted(by_unit, key=lambda u: -len(by_unit[u])):
        rows = by_unit[unit]
        scale = UNIT_SCALE.get(unit)
        buckets: Counter[int] = Counter()
        for _, value in rows:
            if value == 0:
                buckets[-99] += 1
            else:
                buckets[int(math.floor(math.log10(abs(value))))] += 1
        spread = " ".join(
            f"{'0' if e == -99 else e}:{n}"
            for e, n in sorted(buckets.items())
        )
        label = f"{scale:.0e}" if scale else "-"
        print(f"  {unit:16s} {len(rows):4d}  {label:>7}   {spread}")

    # For a currency question the answer is the figure divided by its scale, so
    # `answer * scale` reconstructs the đồng amount. Real line items in this
    # corpus run from about 1e6 (a small note) to 2e15 (VCB total assets).
    print("\n  reconstructed đồng amount (answer x scale), by family:")
    for unit, rows in sorted(by_unit.items(), key=lambda kv: -len(kv[1])):
        scale = UNIT_SCALE.get(unit)
        if scale is None:
            continue
        suspicious = [
            (qid, value) for qid, value in rows
            if value != 0 and abs(value) * scale > 5e15
        ]
        tiny = [
            (qid, value) for qid, value in rows
            if value != 0 and abs(value) * scale < 1e6
        ]
        print(f"  {unit:16s} {len(rows):4d}   above 5e15 đồng: {len(suspicious):3d}"
              f"   below 1e6 đồng: {len(tiny):3d}")
        for qid, value in suspicious[:4]:
            print(f"      id={qid:4d} ships {value:.4g} -> {abs(value) * scale:.3g} đồng")
            print(f"            {parsed[qid].question[:120]}")


if __name__ == "__main__":
    main()
