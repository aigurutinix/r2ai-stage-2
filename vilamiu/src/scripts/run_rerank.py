"""Rerank each question's BM25 shortlist and cache the new ordering.

Writes artifacts/reranked.jsonl: one row per question with the table keys in
cross-encoder order. Downstream scripts read the cache, so the GPU pass runs
once and every later experiment is free.

Usage:  python scripts/run_rerank.py [--candidates 30] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering.lookup import extract_metric  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.retrieval.reranker import Reranker, table_text  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="artifacts/reranked.jsonl")
    parser.add_argument("--query-mode", choices=("full", "metric"), default="full")
    # The candidate pool. BM25 over caption plus row labels recovers 86.2% of the
    # tables the answer path actually reads inside its top 30; the anchor ranking
    # recovers 94.4%. A cross-encoder can only reorder what it is given, so a
    # better pool raises its ceiling before it scores anything.
    parser.add_argument("--pool", default="", help="jsonl of candidates; empty = BM25")
    # What the cross-encoder reads about each table. `meta` is caption plus row
    # labels — the exact text the organisers' question generator forbids a
    # question from copying, which is what capped label matching at 42%. `anchor`
    # is the prose introducing the table instead.
    parser.add_argument("--text", choices=("meta", "anchor"), default="meta")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [q for q in parsed if q.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} questions to rerank ({len(done)} cached), top-{args.candidates} candidates")

    pool: dict[int, list] = {}
    if args.pool:
        for line in (root / args.pool).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entries = row.get("keys") or row.get("refs") or []
            pool[row["id"]] = [
                TableKey(e[0], int(e[1])) if isinstance(e, list)
                else TableKey(e["doc_name"], int(e["table_id"]))
                for e in entries
            ]
        print(f"{len(pool)} questions with a supplied candidate pool from {args.pool}")

    anchor_text: dict[tuple[str, int], str] = {}
    if args.text == "anchor":
        for line in (root / "artifacts" / "anchor_index.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            anchor_text[(row["doc_name"], int(row["table_id"]))] = row["text"]
        print(f"{len(anchor_text)} anchor descriptions loaded")

    def describe(key) -> str:
        if args.text == "anchor":
            found = anchor_text.get((key.doc_name, int(key.table_id)))
            if found:
                return found
        return table_text(store.meta(key))

    reranker = Reranker.load(batch_size=args.batch_size)
    print(f"loaded {reranker.model.model.config._name_or_path}")

    handle = out.open("a", encoding="utf-8")
    started = time.time()
    for position, question in enumerate(todo, start=1):
        candidates = (pool.get(question.id)
                      or [hit.key for hit in retriever.search(question, top_k=args.candidates)])
        candidates = candidates[: args.candidates]
        if not candidates:
            continue
        texts = [describe(key) for key in candidates]
        # Reranking against the whole question was measurably worse than BM25
        # (top-1 label match 33.8% -> 17.0%): the sentence is mostly company,
        # year and unit words that the metadata filter has already used, and the
        # cross-encoder spends its attention on them instead of the line item.
        query = question.question if args.query_mode == "full" else extract_metric(question.question)
        scores = reranker.score(query or question.question, texts)
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        handle.write(
            json.dumps(
                {
                    "id": question.id,
                    "keys": [[candidates[i].doc_name, candidates[i].table_id] for i in order],
                    "scores": [round(scores[i], 4) for i in order],
                    "bm25_rank_of_top": order[0],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        if position % 25 == 0 or position == len(todo):
            handle.flush()
            elapsed = time.time() - started
            print(f"  {position}/{len(todo)}  {elapsed:.0f}s  ({elapsed / position:.2f}s/câu)")
    handle.close()


if __name__ == "__main__":
    main()
