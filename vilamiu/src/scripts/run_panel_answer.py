"""Answer derived questions from the verified metric panel.

Only questions whose every company-year and every referenced metric is present
in the panel are attempted. Anything partial falls back to the existing OCR-table
branch: a panel with holes would let the model quietly answer from the rows that
happen to be there.

Usage:  python scripts/run_panel_answer.py [--limit N] [--workers N]
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
from vifin.answering.generate import build_prompts, generate_query  # noqa: E402
from vifin.answering.panel_context import (  # noqa: E402
    PANEL_CLAUSE, build_context, expand_operands, is_panel_question,
)
from vifin.corpus.metrics import METRICS, _fold, build_panel  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableStore  # noqa: E402

ALIAS_TO_METRIC = {_fold(a): m.name for m in METRICS for a in m.aliases}


def referenced_metrics(question: str) -> list[str]:
    folded = _fold(question)
    found = []
    for alias, name in ALIAS_TO_METRIC.items():
        if alias in folded and name not in found:
            found.append(name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default="qwen/qwen3-8b")
    parser.add_argument("--cache", default="artifacts/panel_answers.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    panel, provenance = build_panel(store.frame, with_provenance=True)
    print(f"panel groups: {len(panel)}")

    client = ChatClient.from_env(root, model=args.model)
    system, user_template = build_prompts(root)
    system += PANEL_CLAUSE

    cache = Path(args.cache)
    done = set()
    if cache.exists():
        done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = []
    rejected = 0
    for question in parsed:
        if question.id in done:
            continue
        if question.unit_scale is not None and lookup_mod.is_single_lookup(question.question):
            continue  # the deterministic lookup already owns these
        metrics = expand_operands(question.question, referenced_metrics(question.question))
        if not metrics or not question.tickers or not question.years:
            continue
        if not is_panel_question(question.question, metrics, ALIAS_TO_METRIC):
            rejected += 1
            continue
        context = build_context(question, panel, provenance, metrics)
        if not context.complete:
            continue
        todo.append((question, context))

    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} questions fully covered by the panel  ({rejected} rejected as out of scope)")

    if args.dry_run:
        return

    handle = cache.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "transport": 0}
    started = time.time()

    def work(item) -> None:
        question, context = item
        tables = {"df": context.rows}
        refs = {"df": "metric_panel"}
        generated = generate_query(
            client, system, user_template, question.question, tables, refs, max_attempts=2
        )
        if generated.result.error.startswith(("llm error", "network error", "malformed")):
            with lock:
                counters["transport"] += 1
            return
        row = {
            "id": question.id,
            "ok": generated.result.ok,
            "value": generated.result.value,
            "error": generated.result.error[:300],
            "code": generated.code,
            "panel_rows": context.rows,
            "keys": [[k.doc_name, k.table_id] for k in context.tables],
        }
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            counters["ok" if generated.result.ok else "fail"] += 1
            total = counters["ok"] + counters["fail"]
            if total % 10 == 0 or total == len(todo):
                print(f"  {total}/{len(todo)}  executable={counters['ok'] / total:.1%}"
                      f"  {time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()

    total = counters["ok"] + counters["fail"]
    if total:
        print(f"\nexecutable: {counters['ok']}/{total} ({counters['ok'] / total:.1%})")


if __name__ == "__main__":
    main()
