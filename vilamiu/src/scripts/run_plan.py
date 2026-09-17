"""Answer by locating cells and applying one operation from a closed set.

Runs over every question the deterministic single-cell lookup cannot serve —
ratio questions, differences, growth, group comparisons — which is where the
fallback currently answers 5.9% of 186 graded questions.

Usage:
  python scripts/run_plan.py --limit 80 --local-url http://localhost:18000/v1
"""

from __future__ import annotations

import os

import argparse
import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.locate import render_candidates  # noqa: E402
from vifin.answering.plan_cells import SYSTEM, compile_plan, parse_plan, suggest_op  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


# `VIFIN_PLAN_LITERALS=1` restores the old literal-emitting behaviour.
PLAIN_READS = os.environ.get("VIFIN_PLAN_LITERALS") != "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
    parser.add_argument("--local-url", default="http://localhost:18000/v1")
    # Reasoning models spend the budget on prose before the answer: Qwen3.5-9B
    # writes ~3.5k characters of "Thinking Process:" and never reaches the JSON
    # at 200. It emits no <think> tag, so no reasoning parser can strip it and
    # neither /no_think, a system directive, nor an assistant prefill suppresses
    # it — all three were measured. The extractor already regex-matches the object
    # out of surrounding text, so the fix is budget, not parsing.
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--reranked", default="artifacts/reranked_metric.jsonl")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--cache", default="artifacts/planned.jsonl")
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

    # An empty `--local-url` routes through OpenRouter instead of a served model,
    # so planning no longer requires renting a GPU. The reply is a short JSON
    # object, so the hosted call is cheap.
    client = (
        ChatClient.local(args.model, args.local_url, max_tokens=args.max_tokens)
        if args.local_url else
        ChatClient.from_env(root, model=args.model, max_tokens=args.max_tokens)
    )

    cache = Path(args.cache)
    done = set()
    if cache.exists():
        done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}

    # Everything the single-cell deterministic path cannot serve.
    todo = [
        q for q in parsed
        if q.id not in done
        and (q.unit_scale is None or not lookup_mod.is_single_lookup(q.question))
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} questions to plan ({len(done)} cached), {args.candidates} tables each")

    handle = cache.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters: collections.Counter = collections.Counter()
    started = time.time()

    def work(question) -> None:
        keys = (reranked.get(question.id) or
                [h.key for h in retriever.search(question, top_k=args.candidates)])[:args.candidates]
        if not keys:
            return
        grids = [store.rows(key) for key in keys]
        captions = [str(store.meta(key).caption) for key in keys]
        hint = suggest_op(question.target_unit, question.question)
        user = (f"<question>\n{question.question}\n</question>\n"
                f"<likely_operation>{hint}</likely_operation>\n\n"
                f"{render_candidates(grids, captions)}")

        row: dict = {"id": question.id, "ok": False}
        try:
            reply = client.complete(SYSTEM, user)
        except RuntimeError as exc:
            with lock:
                counters["transport"] += 1
            return
        plan = parse_plan(reply)
        if plan is None:
            row["error"] = "no valid plan"
            with lock:
                counters["no plan"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        if any(c.table >= len(keys) for c in plan.cells):
            row["error"] = "table index out of range"
            with lock:
                counters["bad index"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return

        used = sorted({c.table for c in plan.cells})
        scales = []
        cell_values = []
        for cell in plan.cells:
            grid = grids[cell.table]
            meta = store.meta(keys[cell.table])
            scales.append(lookup_mod.column_scale(
                grid, cell.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            ))
            parsed = lookup_mod._parse_cell(grid[cell.row][cell.column])
            cell_values.append(parsed if parsed is not None else 0.0)
        # Ratios cancel units, so an output scale would double-apply them.
        out_scale = 1.0 if plan.op in ("ratio", "ratio_pct", "growth_pct") else (question.unit_scale or 1.0)

        # Renumber tables so the emitted frame names match the evidence order.
        remap = {t: i for i, t in enumerate(used)}
        from vifin.answering.plan_cells import Cell, Plan
        plan = Plan(plan.op, tuple(Cell(remap[c.table], c.row, c.column) for c in plan.cells))
        names = ["df"] if len(used) == 1 else [f"df{i + 1}" for i in range(len(used))]
        # `values=None` on purpose. Passing the parsed figures makes
        # `compile_plan` emit them as literals, which is what the organisers
        # reject on the private round's manual review — its own docstring says to
        # prefer None. The cache on disk was built with them, which is why
        # `fix_constants.py` had 242 queries to swap back into real reads.
        code = compile_plan(plan, scales, out_scale, names,
                            None if PLAIN_READS else cell_values)
        tables = {n: grids[t] for n, t in zip(names, used)}
        outcome = run_query(code, tables)
        row.update(
            ok=outcome.ok, value=outcome.value, error=outcome.error[:200], code=code,
            variables=names, keys=[[keys[t].doc_name, keys[t].table_id] for t in used],
            op=plan.op, hint=hint,
        )
        with lock:
            counters[plan.op if outcome.ok else f"crash:{outcome.error.split(':')[0]}"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            total = sum(counters.values())
            if total % 25 == 0 or total == len(todo):
                ok = sum(v for k, v in counters.items() if not k.startswith(("crash", "no ", "bad", "transport")))
                print(f"  {total}/{len(todo)}  ok={ok}  {time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()

    total = sum(counters.values())
    ok = sum(v for k, v in counters.items() if not k.startswith(("crash", "no ", "bad", "transport")))
    print(f"\nusable plans {ok}/{total} ({ok / max(1, total):.1%})")
    for key, count in counters.most_common(12):
        print(f"  {count:4d}  {key}")
    print(f"elapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
