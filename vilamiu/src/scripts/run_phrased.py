"""Answer single-figure questions using the model's phrasings of the metric.

Stage 1 (`run_spec.py`) returns `bien_the`: three to five ways a real statement
would print the metric the question asks about. This is stage 2 and it has no model
in it. Every phrasing goes through `lookup.match_row` across the same eight
shortlisted tables, the same 0.55 threshold applies, and the best-scoring row wins.
The rule's own phrasing is always included, so the result can only be the old
answer or a better-scoring one.

The program is emitted by `lookup.synthesize`, which matches the row by its label
text rather than its index — so the query stays meaningful on the private round's
manual review, and it survives a row shifting.

Writes a cache in the shape `splice_reader.py` consumes.

Usage:
  PYTHONPATH=src python scripts/run_phrased.py --spec artifacts/spec_q.jsonl \
      --out artifacts/phrased.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering import phrased  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--out", default="artifacts/phrased.jsonl")
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--rules-only", action="store_true",
                        help="ignore the model's phrasings; the control run")
    args = parser.parse_args()
    if not args.spec:
        args.spec = ["artifacts/spec_q.jsonl"]

    extra: dict[int, list[str]] = {}
    for path in args.spec:
        file = ROOT / path
        if not file.exists():
            print(f"khong co {path}, bo qua")
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                words = [record.get("tu", "")] + list(record.get("bien_the") or [])
                extra.setdefault(record["id"], []).extend(w for w in words if w)

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    scope = [q for q in questions
             if len(q.tickers) == 1 and q.years
             and lookup_mod.is_single_lookup(q.question)]
    print(f"{len(scope)} cau mot-chi-tieu, {len(extra)} cau co bien the tu model",
          flush=True)

    records, from_model = [], 0
    started = time.time()
    for index, question in enumerate(scope, start=1):
        rule_phrase = lookup_mod.extract_metric(question.question)
        phrases = [rule_phrase]
        if not args.rules_only:
            phrases += extra.get(question.id, [])

        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        try:
            found = phrased.best_row(question, phrases, store, keys)
        except Exception:  # noqa: BLE001 - one bad table must not stop the sweep
            found = None
        if found is None:
            continue
        # `best_row` expands every phrase through `metric_variants`, so a winner
        # that differs from `rule_phrase` may still be rule-derived. Comparing
        # against the raw phrase alone reported 61 model wins on a run that had no
        # model phrasings at all.
        rule_set = {rule_phrase.casefold()} | {
            v.casefold() for v in lookup_mod.metric_variants(rule_phrase)}
        if found.phrase.casefold() not in rule_set:
            from_model += 1

        grid = store.rows(found.key)
        meta = store.meta(found.key)
        scale_in = lookup_mod.column_scale(
            grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
        cell = lookup_mod._parse_cell(grid[found.row][found.column]) \
            if found.column < len(grid[found.row]) else None
        if cell is None:
            continue
        lookup = lookup_mod.Lookup(
            row=found.row, column=found.column, label=found.label,
            value=cell, score=found.score,
            label_col=lookup_mod.label_column(grid))
        code = lookup_mod.synthesize(
            lookup, scale_in, question.unit_scale or 1.0, magnitude=True)
        outcome = run_query(code, {"df": grid})
        if not outcome.ok:
            continue
        records.append({
            "id": question.id,
            "ok": True,
            "value": outcome.value,
            "error": "",
            "attempts": 1,
            "code": code,
            "variables": ["df"],
            "refs": {"df": f"{found.key.doc_name}|{meta.start_line}"},
            "keys": [[found.key.doc_name, found.key.table_id]],
            "labels": [found.label],
            "score": round(found.score, 3),
            "phrase": found.phrase,
            "rule_phrase": rule_phrase,
        })
        if index % 200 == 0:
            print(f"  {index}/{len(scope)}  tra loi={len(records)}  "
                  f"nho model={from_model}  {time.time() - started:.0f}s", flush=True)

    out = ROOT / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    print(f"tra loi {len(records)}/{len(scope)} cau; "
          f"{from_model} cau thang nho cach noi cua model")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
