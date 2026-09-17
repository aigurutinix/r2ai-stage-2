"""How often does a ranking put a resolvable table within reach?

The offline test that decides whether a new ranking is worth a submission. It
asks the question the answer path actually asks: for each metric phrase a
question needs, is there a table in the top-k where that phrase resolves to a row
above `MIN_LABEL_SCORE`?

Measuring with the *whole question* instead was the wrong test and nearly killed
a correct plan: a `plan` question is k cells plus an operation, so the full
sentence never matches one row label, and every ranking scored 5-10%. Scoring the
component phrases separates rankings that differ by 20 points.

Usage:
  PYTHONPATH=src python scripts/probe_fill.py plan:5 locate:8 \
      artifacts/reranked_metric.jsonl artifacts/rr_anchor_both.jsonl
"""

from __future__ import annotations

import collections
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod, panel2  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

MAX_PHRASES = 4


def load(path: Path) -> dict[int, list[TableKey]]:
    order: dict[int, list[TableKey]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        entries = row.get("keys") or row.get("refs") or []
        order[row["id"]] = [
            TableKey(e[0], int(e[1])) if isinstance(e, list)
            else TableKey(e["doc_name"], int(e["table_id"]))
            for e in entries
        ]
    return order


def main() -> None:
    branches, paths = [], []
    for argument in sys.argv[1:]:
        if ":" in argument and not argument.endswith(".jsonl"):
            name, _, cut = argument.partition(":")
            branches.append((name, int(cut)))
        else:
            paths.append(Path(argument))
    if not branches:
        branches = [("plan", 5), ("locate", 8)]

    questions = {q.id: q for q in parse_all(
        ROOT / "data" / "questions" / "questions.jsonl", ROOT / "data" / "code_stock.csv")}
    sources = json.loads((ROOT / "artifacts" / "sources.json").read_text(encoding="utf-8"))
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    rankings = {path.stem: load(path if path.is_absolute() else ROOT / path) for path in paths}

    for branch, cut in branches:
        ids = [int(i) for i, s in sources.items() if s == branch]
        filled: collections.Counter = collections.Counter()
        total = 0
        for qid in ids:
            question = questions[qid]
            phrases = panel2.expand_formulas(panel2.metric_phrases(question))[:MAX_PHRASES]
            if not phrases:
                continue
            total += len(phrases)
            for name, order in rankings.items():
                for phrase in phrases:
                    probe = dataclasses.replace(question, question=phrase)
                    for key in order.get(qid, [])[:cut]:
                        got = lookup_mod.find(store.rows(key), probe)
                        if got is not None and got.score >= lookup_mod.MIN_LABEL_SCORE:
                            filled[name] += 1
                            break
        denominator = total or 1
        print(f"\n{branch}: {len(ids)} câu, {total} cụm chỉ tiêu, top-{cut}")
        for name in rankings:
            print(f"    {name:26s} {filled[name]:4d}/{total} = {filled[name] / denominator:6.1%}")


if __name__ == "__main__":
    main()
