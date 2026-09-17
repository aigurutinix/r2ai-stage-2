"""Run the ratio branch alone over all 1012 questions and cache what it resolves.

A full `run_submit` cannot be compared against the shipped build any more — the
code on disk differs from it on 518 answers — so an improvement to one branch has
to be delivered as a splice. This runs only `ratio.resolve`, executes each program
it returns, and writes a cache in the same shape the other caches use, ready for
`splice_reader.py`.

Prints the funnel so a change to the wording patterns is visible as a count.

Usage:
  PYTHONPATH=src python scripts/run_ratio_only.py --out artifacts/ratio_only.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import ratio as ratio_mod  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/ratio_only.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    counts = {"parse": 0, "resolve": 0, "run": 0, "fail": 0}
    records = []
    started = time.time()
    for index, question in enumerate(questions, start=1):
        if ratio_mod.shape(question) is not None:
            counts["parse"] += 1
        try:
            resolved = ratio_mod.resolve(question, store, retriever, args.top_k)
        except Exception:  # noqa: BLE001 - a broken operand must not stop the sweep
            resolved = None
        if resolved is None:
            continue
        counts["resolve"] += 1
        frames = {name: store.rows(key)
                  for name, key in zip(resolved.variables, resolved.keys)}
        outcome = run_query(resolved.code, frames)
        if not outcome.ok:
            counts["fail"] += 1
            continue
        counts["run"] += 1
        records.append({
            "id": question.id,
            "ok": True,
            "value": outcome.value,
            "error": "",
            "attempts": 1,
            "code": resolved.code,
            "variables": list(resolved.variables),
            "refs": {name: f"{key.doc_name}|{store.meta(key).start_line}"
                     for name, key in zip(resolved.variables, resolved.keys)},
            "keys": [[key.doc_name, key.table_id] for key in resolved.keys],
            "labels": list(resolved.labels),
            "score": resolved.score,
        })
        if index % 200 == 0:
            print(f"  {index}/{len(questions)}  resolve={counts['resolve']}  "
                  f"{time.time() - started:.0f}s", flush=True)

    out = ROOT / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    print(f"parse duoc tu/mau: {counts['parse']}   resolve ra chuong trinh: "
          f"{counts['resolve']}   chay duoc: {counts['run']}   loi: {counts['fail']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
