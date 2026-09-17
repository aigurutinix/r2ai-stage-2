"""Locate the answer row by embedding similarity, offline.

Writes artifacts/embed_located.jsonl in the same shape as run_locate.py, so the
packager treats the two interchangeably. Also reports how often it agrees with
the regex matcher and with the model localiser — the agreement pattern is what
tells us whether it is worth trusting anywhere, and on which questions.

Usage:  python scripts/run_embed_locate.py [--min-score 0.6] [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.embed_match import LabelEmbedder, best_label  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.corpus.numeric import is_correct  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("value") is not None:
                rows[row["id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--out", default="artifacts/embed_located.jsonl")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    reranked: dict[int, list[TableKey]] = {}
    path = root / "artifacts" / "reranked_metric.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                reranked[row["id"]] = [TableKey(d, int(t)) for d, t in row["keys"]]

    model_located = load_jsonl(root / "artifacts" / "located.jsonl")

    todo = [q for q in parsed if q.unit_scale is not None]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} currency-unit questions, min_score={args.min_score}")

    embedder = LabelEmbedder.load()
    out = Path(args.out).open("w", encoding="utf-8")
    stats = collections.Counter()
    started = time.time()

    for position, question in enumerate(todo, start=1):
        keys = (reranked.get(question.id) or
                [h.key for h in retriever.search(question, top_k=args.candidates)])[:args.candidates]
        if not keys:
            continue
        grids = [store.rows(key) for key in keys]
        metric = lookup_mod.extract_metric(question.question)
        match = best_label(embedder, metric, grids, args.min_score)
        if match is None:
            stats["no match"] += 1
            continue

        grid, key = grids[match.table], keys[match.table]
        column = lookup_mod.pick_column(grid, question)
        if column is None or column >= len(grid[match.row]):
            stats["no value column"] += 1
            continue
        value = lookup_mod._parse_cell(grid[match.row][column])
        if value is None:
            stats["cell not numeric"] += 1
            continue

        meta = store.meta(key)
        scale = lookup_mod.column_scale(
            grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        )
        picked = lookup_mod.Lookup(match.row, column, str(grid[match.row][0]), value, match.score)
        code = lookup_mod.synthesize(picked, scale, question.unit_scale, magnitude=True)
        outcome = run_query(code, {"df": grid})
        if not outcome.ok:
            stats["crash"] += 1
            continue

        stats["located"] += 1
        out.write(json.dumps({
            "id": question.id, "ok": True, "value": outcome.value, "error": "",
            "code": code, "variables": ["df"], "keys": [[key.doc_name, key.table_id]],
            "label": picked.label[:80], "score": round(match.score, 4),
        }, ensure_ascii=False) + "\n")
        if position % 100 == 0:
            print(f"  {position}/{len(todo)}  located={stats['located']}  {time.time() - started:.0f}s")
    out.close()

    print(f"\nembedding: " + "  ".join(f"{k}={v}" for k, v in stats.most_common()))
    print(f"elapsed {time.time() - started:.0f}s")

    # Agreement is the only signal available without gold: where independent
    # methods converge, confidence is higher than any single method's.
    embed_located = load_jsonl(Path(args.out))
    both = set(embed_located) & set(model_located)
    agree = sum(
        1 for i in both
        if is_correct(embed_located[i]["value"], model_located[i]["value"])
    )
    print(f"\nembedding vs model localiser: {len(both)} chung, {agree} đồng thuận"
          + (f" ({agree / len(both):.1%})" if both else ""))


if __name__ == "__main__":
    main()
