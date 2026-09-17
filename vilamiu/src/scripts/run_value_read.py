"""Read one figure per exam question with the model, for the questions nothing else serves.

Measured on gold, through the real retrieval path: asking the model to copy the cell
value gets 27.3%, against 21.7% for asking it for coordinates and 19.5% for asking it
to write a program. All three are below the deterministic label matcher's 42.8%, so
this does not replace that branch — it fills the questions no branch can serve, which
today fall to a column scan measured at 5.9%.

The shortlist is eight tables, which the sweep found to be the peak: 21.3% at one
table, 26.0% at three, 27.3% at eight, 25.5% at sixteen.

Output is a cache keyed by question id; `run_submit` consumes it as a last-resort
branch and turns the value into a real `num(df, r, c)` read by locating the cell.

Usage:
  PYTHONPATH=src python scripts/run_value_read.py --model Qwen/Qwen3-14B \
      --local-url http://127.0.0.1:18000/v1 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _probe_value_read import JSON_RE, SYSTEM, THINK_RE, USER, render  # noqa: E402

from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--cache", default="artifacts/value_read.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    client = (ChatClient.local(args.model, args.local_url) if args.local_url
              else ChatClient.from_env(ROOT, model=args.model))

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    cache_path = ROOT / args.cache
    done = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in questions if q.id not in done]
    print(f"{len(questions)} câu, {len(done)} đã có, {len(todo)} cần chạy", flush=True)

    handle = cache_path.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"n": 0, "ok": 0}
    started = time.time()

    def work(question) -> None:
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        blocks, refs = [], []
        for key in keys:
            grid = store.rows(key)
            if not grid:
                continue
            blocks.append(render(len(blocks), grid,
                                 str(getattr(store.meta(key), "caption", ""))))
            refs.append([key.doc_name, key.table_id])
        if not blocks:
            return
        user = USER.format(question=question.question, tables="\n\n".join(blocks))
        try:
            reply = client.complete(SYSTEM, user)
        except RuntimeError:
            return
        match = JSON_RE.search(THINK_RE.sub("", reply or ""))
        raw = unit = ""
        if match:
            try:
                data = json.loads(match.group(0))
                raw = str(data.get("raw", "")).strip()
                unit = str(data.get("unit", "")).strip().lower()
            except (ValueError, TypeError):
                raw = ""
        with lock:
            handle.write(json.dumps({"id": question.id, "raw": raw, "unit": unit,
                                     "keys": refs}, ensure_ascii=False) + "\n")
            handle.flush()
            counters["n"] += 1
            counters["ok"] += 1 if raw else 0
            if counters["n"] % 50 == 0:
                print(f"  {counters['n']}/{len(todo)}  có số="
                      f"{counters['ok'] / counters['n']:.1%}  "
                      f"{time.time() - started:.0f}s", flush=True)

    def guarded(question) -> None:
        try:
            work(question)
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            print(f"  id={question.id} bỏ qua: {type(exc).__name__}: {exc}"[:150],
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(guarded, todo))
    handle.close()
    print(f"xong: {counters['n']} câu, có số {counters['ok']}", flush=True)


if __name__ == "__main__":
    main()
