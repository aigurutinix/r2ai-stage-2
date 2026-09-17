"""Localise the answer cell with the model, then emit our deterministic pandas.

Targets the questions the regex matcher cannot serve: it covers 411 single-lookup
questions at 42.8% and leaves the rest to a fallback that answers 5.9%. Only the
matcher is replaced here — unit resolution, code emission and sandbox checking
are the existing ones, so a score change means localisation and nothing else.

Usage:
  python scripts/run_locate.py --limit 80 --local-url http://localhost:18000/v1 \
      --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.locate import locate  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

CANDIDATES = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
    parser.add_argument("--local-url", default="http://localhost:18000/v1")
    parser.add_argument("--cache", default="artifacts/located.jsonl")
    # Reasoning models spend the budget on prose before the answer: Qwen3.5-9B
    # writes ~3.5k characters of "Thinking Process:" and never reaches the JSON
    # at 120. It emits no <think> tag, so no reasoning parser can strip it and
    # neither /no_think, a system directive, nor an assistant prefill suppresses
    # it — all three were measured. The extractor already regex-matches the object
    # out of surrounding text, so the fix is budget, not parsing.
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--reranked", default="artifacts/reranked_metric.jsonl")
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    reranked: dict[int, list[TableKey]] = {}
    path = root / args.reranked
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                reranked[row["id"]] = [TableKey(d, int(t)) for d, t in row["keys"]]

    # An empty --local-url routes to OpenRouter, the same fallback run_generate
    # already uses. qwen/qwen3-14b is on the allow-list and hosted there, so the
    # 14B tier is reachable without renting a GPU at all.
    client = (ChatClient.local(args.model, args.local_url, max_tokens=args.max_tokens)
              if args.local_url
              else ChatClient.from_env(root, model=args.model,
                                       max_tokens=args.max_tokens))

    cache = Path(args.cache)
    done = set()
    if cache.exists():
        done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}

    # Only currency-unit questions: the deterministic emitter converts scales,
    # and a ratio question needs arithmetic this path does not do.
    todo = [q for q in parsed if q.unit_scale is not None and q.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} questions to localise ({len(done)} cached), {args.candidates} candidates each")

    handle = cache.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"found": 0, "none": 0, "bad_cell": 0, "crash": 0}
    started = time.time()

    def work(question) -> None:
        keys = (reranked.get(question.id) or
                [hit.key for hit in retriever.search(question, top_k=args.candidates)])[:args.candidates]
        if not keys:
            return
        grids = [store.rows(key) for key in keys]
        captions = [str(store.meta(key).caption) for key in keys]
        found = locate(client, question.question, grids, captions)

        row = {"id": question.id, "keys": [[k.doc_name, k.table_id] for k in keys]}
        if found is None or not found.found or found.table >= len(keys):
            row["ok"] = False
            row["error"] = "no cell returned"
            with lock:
                counters["none"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return

        grid = grids[found.table]
        key = keys[found.table]
        if found.row >= len(grid) or found.column >= len(grid[found.row]):
            row.update(ok=False, error="cell out of range")
            with lock:
                counters["bad_cell"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return

        value = lookup_mod._parse_cell(grid[found.row][found.column])
        if value is None:
            row.update(ok=False, error=f"cell not numeric: {grid[found.row][found.column]!r}")
            with lock:
                counters["bad_cell"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return

        meta = store.meta(key)
        scale = lookup_mod.column_scale(
            grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        )
        picked = lookup_mod.Lookup(
            row=found.row, column=found.column,
            label=str(grid[found.row][0]), value=value, score=1.0,
        )
        code = lookup_mod.synthesize(picked, scale, question.unit_scale, magnitude=True)
        outcome = run_query(code, {"df": grid})
        row.update(
            ok=outcome.ok,
            value=outcome.value,
            error=outcome.error[:200],
            code=code,
            variables=["df"],
            table=found.table,
            row_index=found.row,
            column=found.column,
            label=picked.label[:80],
        )
        row["keys"] = [[key.doc_name, key.table_id]]
        with lock:
            counters["found" if outcome.ok else "crash"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            handle.flush()
            total = sum(counters.values())
            if total % 20 == 0 or total == len(todo):
                print(f"  {total}/{len(todo)}  located={counters['found']}"
                      f"  none={counters['none']}  bad={counters['bad_cell']}"
                      f"  crash={counters['crash']}  {time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()

    total = sum(counters.values())
    if total:
        print(f"\nlocalised {counters['found']}/{total} ({counters['found'] / total:.1%})")
        print(f"  no cell returned {counters['none']}  bad cell {counters['bad_cell']}"
              f"  crash {counters['crash']}")
        print(f"elapsed {time.time() - started:.0f}s -> {(time.time() - started) / total:.2f}s/câu")


if __name__ == "__main__":
    main()
