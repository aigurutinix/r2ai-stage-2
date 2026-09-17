"""Score the label branch through the path the submission actually takes.

`score_local.py` hands the matcher the gold table so it can isolate the reader.
That isolation is exactly what made a patch look safe this morning and cost 24
questions: the rule measured 72.8% on gold tables and 38.2% once retrieval chose
the table. Any change to the reader has to be re-measured here before it ships.

The path: anchor ranking (the same index the submission ranks with, run over the
generated questions by `rank_context.py` with `QUESTIONS_FILE` set) →
`Corroborator.choose` → `lookup.find`. Correctness is compared in VND so a table
printing in triệu and one printing in đồng agree when they hold the same amount.

Flags are read at import, so each configuration needs its own process:

  PYTHONPATH=src python scripts/_probe_path_ab.py
  PYTHONPATH=src VIFIN_FUZZY_LABEL=1 VIFIN_FUZZY_MIN=0.30 python scripts/_probe_path_ab.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.corroborate import Corroborator  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SOURCE = ROOT / "artifacts" / "easy_full.jsonl"
RANK = ROOT / "artifacts" / "_gold_anchor_rank.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 600
POOL = 20


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
    return a == b or (scale > 0 and abs(a - b) / scale <= tol)


def main() -> None:
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    corroborator = Corroborator(store)
    per_question = [] if os.environ.get("VIFIN_PATH_AB_OUT") else None

    ranking: dict[int, list[TableKey]] = {}
    for line in RANK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ranking[row["id"]] = [
                TableKey(ref["doc_name"], int(ref["table_id"]))
                for ref in row.get("refs", [])
            ]

    def to_vnd(key: TableKey, grid: list[list[str]], column: int, value: float) -> float:
        meta = store.meta(key)
        return value * lookup_mod.column_scale(
            grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")

    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ][:LIMIT]

    stats: Counter[str] = Counter()
    for index, record in enumerate(records):
        gold_answer = parse_number(record["answer"])
        if gold_answer is None:
            continue
        doc_name, table_id = record["relevant_tables"][0].rsplit("|table_", 1)
        gold_key = TableKey(doc_name, int(table_id))
        gold_grid = store.rows(gold_key)
        if not gold_grid:
            continue
        candidates = ranking.get(index, [])[:POOL]
        if not candidates:
            continue

        # The gold answer is a raw cell, so its column decides its unit.
        gold_column = None
        for line in gold_grid[1:]:
            for c_index, cell in enumerate(line):
                value = parse_number(cell)
                if value is not None and value != 0 and close(value, gold_answer, 5e-4):
                    gold_column = c_index
                    break
            if gold_column is not None:
                break
        if gold_column is None:
            continue
        gold_vnd = to_vnd(gold_key, gold_grid, gold_column, gold_answer)

        stats["considered"] += 1
        question = parse_question(index, record["question"], roster)

        def record_verdict(right: bool, answered: bool, gold_table: bool) -> None:
            # Recorded for every *considered* question, including the ones this
            # branch declines. Writing only the answered ones conditions the file
            # on the branch having succeeded far enough to speak, which lifted its
            # apparent accuracy from 32.4% to 42.3% and would overstate any
            # comparison drawn against it.
            if per_question is not None:
                per_question.append({
                    "id": record.get("id", index),
                    "question": record["question"],
                    "rule_right": bool(right),
                    "rule_answered": bool(answered),
                    "picked_gold_table": bool(gold_table),
                })

        picked = corroborator.choose(question, candidates)
        if picked is None:
            record_verdict(False, False, False)
            continue
        grid = store.rows(picked.key)
        found = lookup_mod.find(grid, question)
        if found is None or found.score < lookup_mod.MIN_LABEL_SCORE:
            record_verdict(False, False, picked.key == gold_key)
            continue
        stats["answered"] += 1
        if picked.key == gold_key:
            stats["picked_gold_table"] += 1
        right = close(to_vnd(picked.key, grid, found.column, found.value), gold_vnd)
        if right:
            stats["CORRECT"] += 1
        # `VIFIN_PATH_AB_OUT` writes the per-question verdict. The aggregate says
        # this branch scores 32.4% and the direct branch 46.2%, which cannot say
        # whether they fail on the same questions — and that is the only thing
        # that decides whether a selector between them is worth building.
        record_verdict(right, True, picked.key == gold_key)

    total = stats["considered"] or 1
    flag = os.environ.get("VIFIN_FUZZY_LABEL", "0")
    threshold = os.environ.get("VIFIN_FUZZY_MIN", "-")
    print(f"fuzzy={flag} min={threshold}  |  {total} questions")
    print(f"  answered {stats['answered']:4d}   correct {stats['CORRECT']:4d} = "
          f"{stats['CORRECT'] / total:5.1%}   precision "
          f"{stats['CORRECT'] / max(stats['answered'], 1):5.1%}   "
          f"gold table {stats['picked_gold_table']:4d}")
    if per_question is not None:
        out = ROOT / os.environ["VIFIN_PATH_AB_OUT"]
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                 for r in per_question), encoding="utf-8")
        print(f"  per-question verdicts -> {out}")


if __name__ == "__main__":
    main()
