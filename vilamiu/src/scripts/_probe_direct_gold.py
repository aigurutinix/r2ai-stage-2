"""What is the direct-answer path actually worth?

`run_direct.py` asks a 14B model for the number rather than for a program — the
Chain-of-Thought shape the organisers measured as the one small models can do,
against a 99% syntax-error rate for Program-of-Thought under 10B. It has been
used once, to patch a single branch, and gained 5 questions. Its standalone
accuracy has never been measured. The only figure anyone quoted for it was
"agrees with the shipped answer 71.4% of the time", which says nothing about
whether either is right.

That gap matters now: the fine-tuned adapter lost 16 questions on the real exam,
and our program-writing branches run at 5.9-13.3% while the organisers measure
this model class far higher on easy questions. The difference between 13% and 60%
on the same size of model is worth more than another training run.

Scored against the generated gold set, which is free and unlimited, using the
same prompt, the same retriever and the same table budget as `run_direct.py` — it
imports them rather than restating them, so the two cannot drift.

The comparison in the last column is the one that decides anything: our current
rule pipeline scores 32.4% on this same set, through the same anchor ranking.

Usage:
  PYTHONPATH=src python scripts/_probe_direct_gold.py --limit 400 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_direct import SYSTEM, USER, parse_reply  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.generate import render_tables, variable_names  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SOURCE = ROOT / "artifacts" / "easy_full.jsonl"
RANK = ROOT / "artifacts" / "_gold_anchor_rank.jsonl"


def parse_number(text: str) -> float | None:
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        return -float(cleaned) if negative else float(cleaned)
    except ValueError:
        return None


def close(a: float, b: float, tol: float = 2e-3) -> bool:
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= tol) or abs(a - b) <= 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--out", default="artifacts/_direct_gold.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    ranking: dict[int, list[TableKey]] = {}
    for line in RANK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ranking[row["id"]] = [
                TableKey(r["doc_name"], int(r["table_id"])) for r in row.get("refs", [])
            ]

    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    # The gold answer is a raw cell, so its own column decides its unit. Both the
    # model's reply and the gold have to be compared in the same unit or the
    # measurement is of unit handling, not of reading — a scorer written this
    # morning got that wrong and reported a bug that did not exist.
    work = []
    for index, record in enumerate(records):
        gold_raw = parse_number(record["answer"])
        if gold_raw is None or index not in ranking:
            continue
        doc_name, table_id = record["relevant_tables"][0].rsplit("|table_", 1)
        gold_key = TableKey(doc_name, int(table_id))
        gold_grid = store.rows(gold_key)
        if not gold_grid:
            continue
        gold_column = None
        for line in gold_grid[1:]:
            for c_index, cell in enumerate(line):
                value = parse_number(cell)
                if value is not None and value != 0 and close(value, gold_raw, 5e-4):
                    gold_column = c_index
                    break
            if gold_column is not None:
                break
        if gold_column is None:
            continue
        question = parse_question(index, record["question"], roster)
        asked = UNIT_SCALE.get(question.target_unit) or 1.0
        meta = store.meta(gold_key)
        scale_in = lookup_mod.column_scale(
            gold_grid, gold_column,
            f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
        work.append((index, question, round(abs(gold_raw) * scale_in / asked, 2)))
        if len(work) >= args.limit:
            break

    print(f"{len(work)} gold questions, {args.workers} workers")
    client = ChatClient.from_env(ROOT, model="qwen/qwen3-14b")
    stats: Counter[str] = Counter()
    lock = threading.Lock()
    out = Path(ROOT / args.out).open("w", encoding="utf-8")

    def ask(item) -> None:
        index, question, want = item
        keys = ranking[index][: args.tables]
        names = variable_names(len(keys))
        grids = {n: store.rows(k) for n, k in zip(names, keys)}
        refs = {
            n: f"{k.doc_name}|{int(store.meta(k).start_line)}"
            for n, k in zip(names, keys)
        }
        user = USER.format(question=question.question,
                           tables=render_tables(grids, refs))
        try:
            reply = client.complete(SYSTEM, user)
        except Exception as exc:  # noqa: BLE001 — one bad call must not end the run
            with lock:
                stats["transport_error"] += 1
            return
        got = parse_reply(reply)
        with lock:
            stats["answered"] += 1
            if got is None:
                stats["unparseable"] += 1
            elif close(abs(got), want):
                stats["CORRECT"] += 1
            out.write(json.dumps({
                "id": index, "question": question.question,
                "want": want, "got": got, "reply": reply[:200],
            }, ensure_ascii=False) + "\n")
            out.flush()
            if stats["answered"] % 50 == 0:
                print(f"  {stats['answered']}/{len(work)}  "
                      f"correct {stats['CORRECT'] / stats['answered']:.1%}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(ask, work))
    out.close()

    n = stats["answered"] or 1
    print(f"\n  answered      {stats['answered']}")
    print(f"  unparseable   {stats['unparseable']}")
    print(f"  transport     {stats['transport_error']}")
    print(f"\n  DIRECT (CoT)  {stats['CORRECT']:4d} = {stats['CORRECT'] / n:.1%}")
    print(f"  rule pipeline           32.4%   (same set, same anchor ranking)")
    print(f"  fine-tuned adapter      69.4%   (held-out, same generator — did not"
          f" transfer)")
    print("\n  If direct clears the rule pipeline by a wide margin, the next move")
    print("  costs API credit and no GPU. If it lands near 13%, the whole")
    print("  LLM-answering direction is closed and the rules are what we have.")


if __name__ == "__main__":
    main()
