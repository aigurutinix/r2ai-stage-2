"""Would a label gate on the declared tables raise precision without losing recall?

`refs = searched[:declare_k]` pads the declaration with the top of the anchor
ranking whether or not those tables could contain the answer. The board says what
that costs: DOCS precision 0.9705 against TABLES precision 0.3357, so 65% of the
refs name a gold document at a non-gold line, and the implied k/g is 2.39 — we
declare nearly two and a half tables for every one the gold program reads.

A table that holds no row resembling the metric is almost never the table the gold
program read. `lookup.find` already scores exactly that. This measures what a gate
on that score would drop:

  * how many refs survive at each threshold (the precision side);
  * whether the table our own program read survives (the recall side — evidence is
    the closest thing to gold we have, and dropping it would be a straight loss).

F2 = 5r / (4 + k/g), so the trade is legible: surviving share sets k/g, and the
evidence-survival rate bounds how much recall the gate can cost.

Usage:  PYTHONPATH=src python scripts/_probe_label_gate_declare.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"
BOARD_R, BOARD_F2 = 0.7213, 0.5641
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.75)


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        recs = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    # `relevant_tables` carries a 1-based line number, not a table ordinal, so the
    # ref strings cannot be turned back into TableKeys. Rebuild the same ordering
    # from the anchor rank the build used, and read the evidence table off the
    # program's own frame count instead.
    rank_path = ROOT / "artifacts" / "anchor_keys.jsonl"
    order: dict[int, list[TableKey]] = {}
    for line in rank_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            keys = row.get("refs") or row.get("keys") or []
            order[row["id"]] = [
                TableKey(r["doc_name"], int(r["table_id"]))
                if isinstance(r, dict) else TableKey(r[0], int(r[1]))
                for r in keys
            ]
    print(f"{len(order)} ranked questions from {rank_path.name}")

    survive: Counter[float] = Counter()
    total = 0
    first_kept: Counter[float] = Counter()
    scores: list[float] = []

    for qid, question in parsed.items():
        keys = order.get(qid, [])[: len(recs[qid]["relevant_tables"])]
        if not keys:
            continue
        best_here = []
        for key in keys:
            try:
                found = lookup_mod.find(store.rows(key), question)
            except Exception:
                found = None
            score = float(found.score) if found is not None else 0.0
            best_here.append(score)
            scores.append(score)
        total += len(best_here)
        for cut in THRESHOLDS:
            kept = sum(1 for s in best_here if s >= cut)
            survive[cut] += kept
            # Does the top-ranked table — the one the answer branches reach for
            # first — survive? If the gate drops it, recall is going with it.
            first_kept[cut] += int(best_here[0] >= cut)

    scores.sort()
    n = len(scores)
    print(f"\n{total} declared refs scored; label score distribution")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q * 100):02d}  {scores[int(n * q)]:.3f}")

    questions = sum(1 for qid in parsed if order.get(qid))
    print(f"\ngate    refs kept   k/g    projected TABLES_F2 at recall")
    print(f"                            {'1.00':>6} {'0.95':>6} {'0.90':>6} {'0.80':>6}")
    for cut in THRESHOLDS:
        share = survive[cut] / total
        kg = 2.39 * share
        row = f"  {cut:.2f}  {survive[cut]:6d} ({share:4.0%})  {kg:4.2f} "
        for hold in (1.00, 0.95, 0.90, 0.80):
            f2 = 5 * (BOARD_R * hold) / (4 + kg)
            row += f" {f2:6.4f}"
        print(row)
    print(f"\nbaseline: k/g 2.39, recall {BOARD_R}, TABLES_F2 {BOARD_F2}")
    print("top-ranked table surviving the gate (a floor on what recall costs):")
    for cut in THRESHOLDS:
        print(f"  {cut:.2f}  {first_kept[cut]}/{questions} = {first_kept[cut] / questions:.1%}")


if __name__ == "__main__":
    main()
