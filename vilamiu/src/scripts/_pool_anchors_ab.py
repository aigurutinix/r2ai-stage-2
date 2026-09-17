"""Does pooling anchors across a document's fragments help, and stay honest?

Three numbers decide it, and coverage alone is the one that misleads:
`allow_label_only` raised coverage handsomely while sinking the assets identity
from 97.8% to 73.2%, i.e. it bought groups by making them wrong.

So this reports coverage, the agreement rate of the accounting identities on
groups that can be checked, and the count of exam questions the panel path can
actually serve — with pooling off and on, in separate processes so the module
flag is read fresh.

Usage:  PYTHONPATH=src python scripts/_pool_anchors_ab.py
"""

from __future__ import annotations

import collections
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import collections, sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from vifin.answering import lookup as lookup_mod
from vifin.answering.panel_context import build_context, expand_operands, is_panel_question
from vifin.corpus.metrics import METRICS, _fold, build_panel
from vifin.query.parse import parse_all
from vifin.store import TableStore
from pathlib import Path

ALIAS_TO_METRIC = {_fold(a): m.name for m in METRICS for a in m.aliases}
IDENTITIES = (("total_assets", "liabilities", "equity"),
              ("total_assets", "current_assets", "long_assets"),
              ("liabilities", "liabilities_short", "liabilities_long"))

def referenced(question):
    folded = _fold(question)
    out = []
    for alias, name in ALIAS_TO_METRIC.items():
        if alias in folded and name not in out:
            out.append(name)
    return out

store = TableStore.load(Path("artifacts/tables.parquet"))
panel, provenance = build_panel(store.frame, with_provenance=True)
parsed = parse_all(Path("data/questions/questions.jsonl"), Path("data/code_stock.csv"))

agree = disagree = 0
for row in panel.values():
    for whole, left, right in IDENTITIES:
        w, a, b = row.get(whole), row.get(left), row.get(right)
        if None in (w, a, b):
            continue
        if abs(w - (a + b)) / max(abs(w), 1.0) <= 0.005:
            agree += 1
        else:
            disagree += 1

served = cohort = 0
for question in parsed:
    if question.unit_scale is not None and lookup_mod.is_single_lookup(question.question):
        continue
    metrics = expand_operands(question.question, referenced(question.question))
    if not metrics or not question.tickers or not question.years:
        continue
    if not is_panel_question(question.question, metrics, ALIAS_TO_METRIC):
        continue
    if build_context(question, panel, provenance, metrics).complete:
        served += 1
        if len(question.tickers) >= 2:
            cohort += 1

cells = sum(len(v) for v in panel.values())
rate = 100.0 * agree / max(agree + disagree, 1)
print(f"{len(panel)}|{cells}|{rate:.1f}|{agree + disagree}|{served}|{cohort}")
'''


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{'pooling':10s} {'groups':>7s} {'cells':>7s} {'identity':>9s} "
          f"{'checked':>8s} {'served':>7s} {'cohort':>7s}")
    for flag in ("0", "1"):
        env = dict(os.environ)
        env["VIFIN_POOL_ANCHORS"] = flag
        env["PYTHONPATH"] = "src"
        result = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                                text=True, cwd=ROOT, env=env)
        line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        if "|" not in line:
            print(f"  {flag:8s} failed: {result.stderr[-400:]}")
            continue
        groups, cells, rate, checked, served, cohort = line.split("|")
        print(f"  {'on' if flag == '1' else 'off':8s} {groups:>7s} {cells:>7s} "
              f"{rate + '%':>9s} {checked:>8s} {served:>7s} {cohort:>7s}")


if __name__ == "__main__":
    main()
