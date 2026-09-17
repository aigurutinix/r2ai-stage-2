"""Is the missing company-year absent from the corpus, or only from the panel?

`_panel_ceiling.py` put 102 of the 288 cohort questions on "company-year not in
corpus", but it tested membership in the panel, and a group only enters the panel
once at least one metric is extracted from it. So that bucket conflates a real
corpus limit with a total extraction failure, and the two point opposite ways:
one caps the panel path for good, the other is the same fixable problem as every
other hole.

This tests against `tables.parquet`, which lists every table the corpus holds
regardless of what was understood inside it.

Usage:  PYTHONPATH=src python scripts/_panel_corpus_limit.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import METRICS, _fold, build_panel  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableStore  # noqa: E402

ALIAS_TO_METRIC = {_fold(a): m.name for m in METRICS for a in m.aliases}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parsed = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                       ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, _ = build_panel(store.frame, with_provenance=True)

    frame = store.frame
    in_corpus = {
        (str(t), str(y), str(s))
        for t, y, s in zip(frame["ticker"], frame["year"], frame["scope"])
    }
    in_panel = set(panel)
    print(f"corpus: {len(in_corpus)} company-year-scope groups")
    print(f"panel:  {len(in_panel)} of them yielded at least one metric "
          f"({100 * len(in_panel) / max(len(in_corpus), 1):.1f}%)\n")

    cause: collections.Counter[str] = collections.Counter()
    for question in parsed:
        if len(question.tickers) < 2:
            continue
        wanted = [(t, str(y), question.scope)
                  for t in question.tickers for y in question.years]
        if not wanted:
            continue
        absent_panel = [k for k in wanted if k not in in_panel]
        if not absent_panel:
            cause["every group is in the panel"] += 1
            continue
        absent_corpus = [k for k in absent_panel if k not in in_corpus]
        if absent_corpus:
            cause["some group is genuinely absent from the corpus"] += 1
        else:
            cause["in the corpus, but extraction yielded nothing"] += 1

    total = sum(cause.values())
    print(f"{total} cohort questions\n")
    for name, count in cause.most_common():
        print(f"  {count:4d}  {100 * count / max(total, 1):5.1f}%  {name}")


if __name__ == "__main__":
    main()
