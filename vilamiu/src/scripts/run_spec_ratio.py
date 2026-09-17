"""Answer ratio questions from the model's reading of the question.

Stage 1 is `run_spec.py`: the question alone goes to the model, and the operation
plus the two metric phrases come back. This is stage 2 and it contains no model at
all — the phrases go into `ratio.resolve_pair`, which is the code that already
ships 42 answers: `lookup.find` matches each phrase against row labels,
`pick_column` picks the period, `column_scale` fixes the unit, and
`SANITY_LIMIT` withholds a rate that cannot be one.

The label matcher's score doubles as a free arbiter. When two models parse the
same question differently there is no gold to say which is right, but there is
this: the parse whose phrases the matcher can actually find in the company's own
tables, at a higher score, is the parse that describes a real line item. Pass
several spec files with `--spec` and the best-scoring resolution wins.

Usage:
  PYTHONPATH=src python scripts/run_spec_ratio.py --spec artifacts/spec_q.jsonl \
      --out artifacts/spec_ratio.jsonl
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

# Only the operations a quotient can express. `hieu`, `tong` and `trung_binh` are
# real answers to real questions but they belong to `compose`, not here, and
# claiming them with a division would ship a confident wrong number.
DIVIDING = {"chia", "tang_truong"}


def load_specs(paths: list[str]) -> dict[int, list[dict]]:
    specs: dict[int, list[dict]] = {}
    for path in paths:
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["src"] = path
                specs.setdefault(record["id"], []).append(record)
    return specs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", default=[],
                        help="one or more spec files; the best resolution wins")
    parser.add_argument("--out", default="artifacts/spec_ratio.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if not args.spec:
        args.spec = ["artifacts/spec_q.jsonl"]

    specs = load_specs(args.spec)
    questions = {q.id: q for q in parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv")}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    counts = {"khong phai phep chia": 0, "khong tim duoc dong": 0, "chuong trinh loi": 0}
    records = []
    started = time.time()
    scope = sorted(specs)
    for index, qid in enumerate(scope, start=1):
        question = questions.get(qid)
        if question is None:
            continue
        # The whole downstream path assumes one company and a known period, the
        # same scope `ratio.eligible` enforces. Cohort questions need the operand
        # loop in `compose`.
        if len(question.tickers) != 1 or not question.years:
            continue

        best = None
        for spec in specs[qid]:
            if spec["op"] not in DIVIDING:
                counts["khong phai phep chia"] += 1
                continue
            numerator = spec.get("tu", "")
            denominator = spec.get("mau", "") or numerator
            try:
                if ratio_mod.compound_shape(question) is not None:
                    # ROA and ROE divide by the AVERAGE of the opening and closing
                    # balance, not the closing one. At a 0.02% tolerance that is a
                    # wrong answer, not an approximation, so these keep the
                    # compound path whatever the model called the operands.
                    answer = ratio_mod.resolve(question, store, retriever, args.top_k)
                else:
                    answer = ratio_mod.resolve_pair(
                        question, numerator, denominator, store, retriever, args.top_k)
            except Exception:  # noqa: BLE001 - one bad operand must not stop the sweep
                answer = None
            if answer is None:
                continue
            if best is None or answer.score > best[0].score:
                best = (answer, spec)
        if best is None:
            counts["khong tim duoc dong"] += 1
            continue

        answer, spec = best
        frames = {name: store.rows(key)
                  for name, key in zip(answer.variables, answer.keys)}
        outcome = run_query(answer.code, frames)
        if not outcome.ok:
            counts["chuong trinh loi"] += 1
            continue
        records.append({
            "id": qid,
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
            "op": spec["op"],
            "tu": spec.get("tu", ""),
            "mau": spec.get("mau", ""),
            "spec_src": spec.get("src", ""),
        })
        if index % 100 == 0:
            print(f"  {index}/{len(scope)}  giai duoc={len(records)}  "
                  f"{time.time() - started:.0f}s", flush=True)

    out = ROOT / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    print(f"{len(specs)} spec -> giai duoc {len(records)} cau; " +
          ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
