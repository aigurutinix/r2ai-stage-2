"""Run the search-based ratio resolver over all 1012 questions and cache it.

Reports the funnel next to the regex branch it replaces, so "how many more
questions does search reach than parsing" is one number rather than an argument.
Writes a cache in the shape `splice_reader.py` consumes, because a full
`run_submit` no longer reproduces the shipped build and cannot be compared to it.

Usage:
  PYTHONPATH=src python scripts/run_pair.py --out artifacts/pair.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import pair as pair_mod  # noqa: E402
from vifin.answering import ratio as ratio_mod  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/pair.jsonl")
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="override pair.MIN_COVERAGE")
    args = parser.parse_args()

    if args.min_coverage:
        pair_mod.MIN_COVERAGE = args.min_coverage

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    scope = [q for q in questions
             if q.target_unit in ("phan_tram", "lan", "vong")
             and len(q.tickers) == 1 and q.years]
    parsed = sum(1 for q in scope if ratio_mod.shape(q) is not None)
    print(f"{len(scope)} cau trong pham vi (1 ma, don vi ty le); "
          f"regex parse duoc {parsed}", flush=True)

    records, failed = [], 0
    started = time.time()
    for index, question in enumerate(scope, start=1):
        try:
            answer = pair_mod.resolve(question, store, retriever, args.shortlist)
        except Exception:  # noqa: BLE001 - one bad table must not stop the sweep
            answer = None
        if answer is None:
            continue
        frames = {name: store.rows(key)
                  for name, key in zip(answer.variables, answer.keys)}
        outcome = run_query(answer.code, frames)
        if not outcome.ok:
            failed += 1
            continue
        records.append({
            "id": question.id,
            "ok": True,
            "value": outcome.value,
            "error": "",
            "attempts": 1,
            "code": answer.code,
            "variables": list(answer.variables),
            "refs": {name: f"{key.doc_name}|{store.meta(key).start_line}"
                     for name, key in zip(answer.variables, answer.keys)},
            "keys": [[key.doc_name, key.table_id] for key in answer.keys],
            "labels": list(answer.labels),
            "score": round(answer.score, 3),
        })
        if index % 100 == 0:
            print(f"  {index}/{len(scope)}  giai duoc={len(records)}  "
                  f"{time.time() - started:.0f}s", flush=True)

    out = ROOT / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    print(f"tim kiem giai duoc {len(records)} cau (regex parse {parsed}), "
          f"chuong trinh loi {failed}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
