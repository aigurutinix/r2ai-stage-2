"""Why the panel is incomplete for the cohort questions.

`_panel_funnel.py` puts 54.9% of them on "panel incomplete", which is one word
for two very different faults. Either the company-year is absent from the panel
altogether — the statement was never parsed into metrics — or the group is there
and the one metric the question asks for is missing from it. The first is a
corpus-coverage problem, the second a metric-extraction problem, and they are
fixed in different files.

Usage:  PYTHONPATH=src python scripts/_panel_gap.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parsed = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                       ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, provenance = build_panel(store.frame, with_provenance=True)

    have_group = {(t, y, s) for t, y, s in panel}
    scopes = collections.Counter(s for _, _, s in panel)
    print(f"panel: {len(panel)} groups, scopes {dict(scopes)}")

    cause: collections.Counter[str] = collections.Counter()
    missing_metric: collections.Counter[str] = collections.Counter()
    missing_group_ticker: collections.Counter[str] = collections.Counter()
    partial = 0

    for question in parsed:
        if len(question.tickers) < 2:
            continue
        if question.unit_scale is not None and lookup_mod.is_single_lookup(question.question):
            continue
        metrics = expand_operands(question.question, referenced_metrics(question.question))
        if not metrics or not question.years:
            continue
        if not is_panel_question(question.question, metrics, ALIAS_TO_METRIC):
            continue
        context = build_context(question, panel, provenance, metrics)
        if context.complete:
            continue

        absent_groups = 0
        absent_metrics = 0
        for ticker in question.tickers:
            for year in question.years:
                key = (ticker, str(year), question.scope)
                if key not in have_group:
                    absent_groups += 1
                    missing_group_ticker[ticker] += 1
                    continue
                row = panel[key]
                for metric in metrics:
                    if row.get(metric) is None:
                        absent_metrics += 1
                        missing_metric[metric] += 1

        if absent_groups and absent_metrics:
            cause["both"] += 1
            partial += 1
        elif absent_groups:
            cause["company-year absent from the panel"] += 1
        elif absent_metrics:
            cause["group present, metric missing"] += 1
        else:
            cause["complete by this count — build_context disagrees"] += 1

    total = sum(cause.values())
    print(f"\n{total} incomplete cohort questions\n")
    for name, count in cause.most_common():
        print(f"  {count:4d}  {100 * count / max(total, 1):5.1f}%  {name}")
    print("\nmetrics most often missing from a present group:")
    for name, count in missing_metric.most_common(8):
        print(f"  {count:5d}  {name}")
    print("\ntickers most often absent as a company-year:")
    for name, count in missing_group_ticker.most_common(8):
        print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    main()
