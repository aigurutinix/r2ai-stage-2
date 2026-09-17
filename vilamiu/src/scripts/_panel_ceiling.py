"""The most the panel path could ever serve, if extraction were perfect.

The identities filled 410 cells and unblocked one question, so the next candidate
is better metric extraction — a much larger job inside `corpus/metrics.py`. Worth
knowing its ceiling first. This grants every company-year in the corpus every
metric it asks for and counts what still fails, which separates "the panel is
thin" from "these questions were never panel questions".

Usage:  PYTHONPATH=src python scripts/_panel_ceiling.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.panel_context import (  # noqa: E402
    expand_operands, is_panel_question,
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
    panel, _ = build_panel(store.frame, with_provenance=True)

    # Which company-year-scope groups exist at all, ignoring which lines were
    # found inside them. That is the true corpus limit: no extraction effort can
    # invent a statement that was never in the documents.
    have_group = {key for key in panel}

    stage: collections.Counter[str] = collections.Counter()
    for question in parsed:
        cohort = len(question.tickers) >= 2
        bucket = "cohort" if cohort else "single"
        stage[f"{bucket}: total"] += 1
        if question.unit_scale is not None and lookup_mod.is_single_lookup(question.question):
            stage[f"{bucket}: taken by single lookup"] += 1
            continue
        metrics = expand_operands(question.question, referenced_metrics(question.question))
        if not metrics:
            stage[f"{bucket}: no metric recognised"] += 1
            continue
        if not question.tickers or not question.years:
            stage[f"{bucket}: no ticker or year"] += 1
            continue
        if not is_panel_question(question.question, metrics, ALIAS_TO_METRIC):
            stage[f"{bucket}: not a panel question"] += 1
            continue
        missing = [
            (t, str(y)) for t in question.tickers for y in question.years
            if (t, str(y), question.scope) not in have_group
        ]
        if missing:
            stage[f"{bucket}: company-year not in corpus"] += 1
            continue
        stage[f"{bucket}: REACHABLE with perfect extraction"] += 1

    for bucket in ("cohort", "single"):
        total = stage[f"{bucket}: total"]
        print(f"\n{bucket}: {total} questions")
        for name, count in sorted(stage.items(), key=lambda kv: -kv[1]):
            if not name.startswith(f"{bucket}: ") or name.endswith("total"):
                continue
            print(f"  {count:4d}  {100 * count / max(total, 1):5.1f}%  "
                  f"{name.split(': ', 1)[1]}")


if __name__ == "__main__":
    main()
