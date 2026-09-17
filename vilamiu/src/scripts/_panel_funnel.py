"""Where the multi-company questions leave the panel path.

`run_panel_answer.py` drops a question at four different `continue`s and counts
only one of them, so "45 covered" says nothing about which gate to open. This
replays the same loop with a counter on every branch, over the cohort questions
specifically: 150 of the 1,012 name more than one company, and none of them can
be served by the eight-table prompt.

Usage:  PYTHONPATH=src python scripts/_panel_funnel.py
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

    stage: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)

    def note(bucket: str, question) -> None:
        stage[bucket] += 1
        if len(examples[bucket]) < 3:
            examples[bucket].append(question.question[:150])

    for question in parsed:
        if len(question.tickers) < 2:
            continue
        stage["cohort questions"] += 1
        if question.unit_scale is not None and lookup_mod.is_single_lookup(question.question):
            note("taken by the single lookup", question)
            continue
        metrics = expand_operands(question.question, referenced_metrics(question.question))
        if not metrics:
            note("no metric recognised", question)
            continue
        if not question.years:
            note("no year parsed", question)
            continue
        if not is_panel_question(question.question, metrics, ALIAS_TO_METRIC):
            note("is_panel_question said no", question)
            continue
        context = build_context(question, panel, provenance, metrics)
        if not context.complete:
            note("panel incomplete for some company-year", question)
            continue
        note("REACHES THE MODEL", question)

    total = stage.pop("cohort questions", 0)
    print(f"{total} questions naming two or more companies\n")
    for bucket, count in stage.most_common():
        print(f"  {count:4d}  {100 * count / max(total, 1):5.1f}%  {bucket}")
    print()
    for bucket, _ in stage.most_common(3):
        print(f"-- {bucket}")
        for text in examples[bucket]:
            print(f"     {text}")


if __name__ == "__main__":
    main()
