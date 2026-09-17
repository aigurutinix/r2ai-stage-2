"""Run the cohort branch over the 288 multi-company questions and cache it.

Prints the funnel, because the interesting number is where the branch declines: a
question whose read fails for one of the named companies is refused outright, and
knowing whether the losses are the ranking questions, the metric extraction or one
company's missing table says what to build next.

Usage:
  PYTHONPATH=src python scripts/run_cohort.py --out artifacts/cohort.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import cohort as cohort_mod  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/cohort.jsonl")
    parser.add_argument("--shortlist", type=int, default=8)
    args = parser.parse_args()

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    scope = [q for q in questions if len(q.tickers) > 1 and q.years]
    counters: Counter[str] = Counter()
    for question in scope:
        op = cohort_mod.operation(question.question)
        if op is None:
            counters["xep hang hoac khong ro phep"] += 1
        else:
            counters[f"phep {op}"] += 1
    print(f"{len(scope)} cau nhieu ma")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")

    records = []
    fail: Counter[str] = Counter()
    started = time.time()
    for index, question in enumerate(scope, start=1):
        try:
            answer = cohort_mod.resolve(question, store, retriever, args.shortlist)
        except Exception:  # noqa: BLE001
            answer = None
            fail["ngoai le"] += 1
        if answer is None:
            fail["tu choi"] += 1
            continue
        frames = {name: store.rows(key)
                  for name, key in zip(answer.variables, answer.keys)}
        outcome = run_query(answer.code, frames)
        if not outcome.ok:
            fail["chuong trinh loi"] += 1
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
            "op": answer.op,
            "n_tickers": len(question.tickers),
        })
        if index % 50 == 0:
            print(f"  {index}/{len(scope)}  tra loi={len(records)}  "
                  f"{time.time() - started:.0f}s", flush=True)

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    print(f"\ntra loi {len(records)}/{len(scope)} cau nhieu ma; " +
          ", ".join(f"{k}={v}" for k, v in fail.most_common()))
    by_op = Counter(r["op"] for r in records)
    by_n = Counter(r["n_tickers"] for r in records)
    print("  theo phep:", dict(by_op))
    print("  theo so ma:", dict(sorted(by_n.items())))
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
