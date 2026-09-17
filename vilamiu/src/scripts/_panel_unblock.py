"""Do the accounting identities unblock any cohort question, or only fill cells?

Filling 410 panel cells is not the goal; answering questions is. A cell gained in
a group no question asks about is worth nothing, and `liabilities_short` — the
second-largest hole — gains exactly one. So this replays the same gate the panel
path applies, before and after the identities, and reports the only number that
decides whether to ship them.

Usage:  PYTHONPATH=src python scripts/_panel_unblock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))

from _panel_identities import solve  # noqa: E402

import collections  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.panel_context import (  # noqa: E402
    build_context, expand_operands, is_panel_question,
)
from vifin.corpus.metrics import METRICS, _fold, build_panel  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableStore  # noqa: E402

ALIAS_TO_METRIC = {_fold(a): m.name for m in METRICS for a in m.aliases}


def referenced_metrics(question: str) -> list[str]:
    folded = _fold(question)
    out = []
    for alias, name in ALIAS_TO_METRIC.items():
        if alias in folded and name not in out:
            out.append(name)
    return out


def covered(parsed, panel, provenance, cohort_only: bool) -> set[int]:
    out = set()
    for question in parsed:
        if cohort_only and len(question.tickers) < 2:
            continue
        if question.unit_scale is not None and lookup_mod.is_single_lookup(question.question):
            continue
        metrics = expand_operands(question.question, referenced_metrics(question.question))
        if not metrics or not question.tickers or not question.years:
            continue
        if not is_panel_question(question.question, metrics, ALIAS_TO_METRIC):
            continue
        if build_context(question, panel, provenance, metrics).complete:
            out.add(question.id)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parsed = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                       ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, provenance = build_panel(store.frame, with_provenance=True)

    for label, cohort_only in (("cohort (2+ companies)", True), ("all 1012", False)):
        base = covered(parsed, panel, provenance, cohort_only)
        print(f"{label}: {len(base)} covered before")

    filled: collections.Counter[str] = collections.Counter()
    for row in panel.values():
        solve(row, filled)
    print(f"\nidentities filled {sum(filled.values())} cells\n")

    for label, cohort_only in (("cohort (2+ companies)", True), ("all 1012", False)):
        after = covered(parsed, panel, provenance, cohort_only)
        print(f"{label}: {len(after)} covered after")


if __name__ == "__main__":
    main()
