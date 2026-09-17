"""Where does context-BM25 place the tables our answer path actually read?

There is no offline gold, so this uses the closest available stand-in: on the
questions we answer from a resolved cell, the table that cell came from is almost
certainly one of the gold tables.

The stand-in is biased and the bias runs one way. Those tables were found *inside*
the current BM25 shortlist, so a table the current ranking never surfaced could
never have become evidence. The comparison can therefore not show context-BM25
winning on its merits — but it can show it losing, which is the thing worth
knowing before a submission is spent.

Usage:  PYTHONPATH=src python scripts/compare_rank.py [submission.zip]
"""

from __future__ import annotations

import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey  # noqa: E402

CONFIDENT = {"lookup", "compose", "ratio_divide", "ratio_lookup", "screen", "lookup_rescan"}


def evidence_keys(record: dict) -> list[TableKey]:
    keys = []
    for item in record.get("evidence", []):
        stem = Path(item.get("csv_path", "")).stem
        doc_name, separator, raw = stem.rpartition("_table_")
        if separator and raw.isdigit():
            keys.append(TableKey(doc_name, int(raw)))
    return keys


def rank_of(order: list[TableKey], key: TableKey) -> int | None:
    for position, candidate in enumerate(order):
        if candidate == key:
            return position + 1
    return None


def bucket(rank: int | None) -> str:
    if rank is None:
        return "not in top 30"
    for edge in (1, 3, 5, 10, 20):
        if rank <= edge:
            return f"top {edge}"
    return "top 30"


def main() -> None:
    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submissions" / "label_doc.zip"
    records = json.loads(zipfile.ZipFile(archive).read("submission.json").decode("utf-8"))
    sources = json.loads((ROOT / "artifacts" / "sources.json").read_text(encoding="utf-8"))

    context: dict[int, list[TableKey]] = {}
    for line in (ROOT / "artifacts" / "context_rank.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            context[row["id"]] = [TableKey(r["doc_name"], int(r["table_id"])) for r in row["refs"]]

    frame = pd.read_parquet(
        ROOT / "artifacts" / "tables.parquet",
        columns=["doc_name", "ticker", "year", "scope", "table_id", "eligible"])
    retriever = LexicalRetriever(frame)
    parsed = {q.id: q for q in parse_all(
        ROOT / "data" / "questions" / "questions.jsonl", ROOT / "data" / "code_stock.csv")}

    reranked: dict[int, list[TableKey]] = {}
    path = ROOT / "artifacts" / "reranked_metric.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                reranked[row["id"]] = [TableKey(d, int(t)) for d, t in row["keys"]]

    current_buckets: collections.Counter = collections.Counter()
    context_buckets: collections.Counter = collections.Counter()
    checked = better = worse = same = 0

    for record in records:
        qid = record["id"]
        if sources.get(str(qid)) not in CONFIDENT:
            continue
        keys = evidence_keys(record)
        if not keys:
            continue
        question = parsed[qid]
        current = reranked.get(qid) or [h.key for h in retriever.search(question, top_k=30)]
        current = current[:30]
        for key in keys:
            checked += 1
            here = rank_of(current, key)
            there = rank_of(context.get(qid, []), key)
            current_buckets[bucket(here)] += 1
            context_buckets[bucket(there)] += 1
            if (there or 99) < (here or 99):
                better += 1
            elif (there or 99) > (here or 99):
                worse += 1
            else:
                same += 1

    print(f"{checked} evidence tables from confidently answered questions\n")
    order = ["top 1", "top 3", "top 5", "top 10", "top 20", "top 30", "not in top 30"]
    print(f"{'bucket':<16}{'current':>10}{'context':>10}")
    running_current = running_context = 0
    for name in order:
        if name == "not in top 30":
            print(f"{name:<16}{current_buckets[name]:>10}{context_buckets[name]:>10}")
            continue
        running_current += current_buckets[name]
        running_context += context_buckets[name]
        print(f"{name:<16}{running_current:>10}{running_context:>10}"
              f"   ({running_current / checked:5.1%} vs {running_context / checked:5.1%})")

    print(f"\ncontext ranks it higher: {better}   lower: {worse}   same: {same}")


if __name__ == "__main__":
    main()
