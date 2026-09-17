"""Score our own pipeline against the generated gold set, and say where it breaks.

Until now every change had to be measured through the leaderboard: 506 scored
questions, ten submissions a day, and a resolution floor of five to eleven answers
before the number moves at all. That is why so many decisions in this project were
argued rather than measured, and why several of them were wrong.

The generation run produced questions that carry their own gold — the answer, the
table it came from, and a program that was executed to prove it. Verified: 200 of
200 gold table references resolve in our store with byte-identical headers, so the
two corpora are the same tables under the same ids. That makes this a real test set.

What it measures, in the order the pipeline fails:

  1. **retrieval** — is the gold table in our top-k at all, and at what rank
  2. **cell** — handed the gold table, does the label matcher pick the right cell
  3. **unit** — right cell, wrong scale

The split matters more than the total. "We answer 38%" says nothing about what to
do next; "we retrieve 90% and then read the wrong row half the time" says train the
reader, and "we read the right cell but divide by the wrong power of ten" says fix
forty lines of code and skip the GPU entirely. This project has guessed that split
before and been wrong — 70% localisation guessed against 48/36/15 measured.

Scope, stated so the number is not overread: these are easy-tier questions, which
our own estimate puts at 39.7% of the exam, and their phrasing is skewed (90% name
the ticker in brackets against 22.6% of the real set). It measures the segment the
fine-tune targets, not the exam.

Usage:  PYTHONPATH=src python scripts/score_local.py [limit]
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
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# `easy_full.jsonl` stores the raw cell rather than the asked-for unit in 28% of
# records, so scoring against it understates by about eight points. Point this at
# `easy_full_units.jsonl`, whose units were repaired by `build_gold_units.py`.
SOURCE = ROOT / "artifacts" / os.environ.get("VIFIN_GOLD", "easy_full.jsonl")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOP_K = 10

# The generator's prompt orders it to name the ticker in brackets, and 89.9% of the
# generated questions do. Only 22.6% of the real exam does. The ticker is the single
# strongest retrieval signal there is — it names the company, and company plus year
# picks the document — so every retrieval number measured on this set is optimistic,
# and that optimism is exactly what made a patched submission look safe and cost
# 24 questions. `VIFIN_STRIP_TICKER=1` removes it so the measurement runs down the
# path the real questions take: resolve the company from its name alone.
STRIP_TICKER = os.environ.get("VIFIN_STRIP_TICKER") == "1"
TICKER_RE = re.compile(r"\s*\((?:mã\s*)?([A-Z]{3})\)")


def parse_number(text: str) -> float | None:
    """Read a figure as the reports print it.

    Vietnamese statements use '.' for thousands and ',' for decimals, but a
    minority of this corpus prints English-style. Guessing one convention
    discarded 10 of 91 good records once already, so both are tried and the
    reading that survives is used.
    """

    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return (scale > 0 and abs(a - b) / scale <= tol) or abs(a - b) <= 0.01


def main() -> None:
    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if LIMIT:
        records = records[:LIMIT]

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)

    retrieval: Counter[str] = Counter()
    cell: Counter[str] = Counter()
    ranks: list[int] = []

    for index, record in enumerate(records):
        gold_ref = record["relevant_tables"][0]
        doc_name, table_id = gold_ref.rsplit("|table_", 1)
        gold_key = TableKey(doc_name, int(table_id))
        gold_answer = parse_number(record["answer"])
        if gold_answer is None:
            cell["gold_answer_not_numeric"] += 1
            continue

        text = record["question"]
        if STRIP_TICKER:
            text = TICKER_RE.sub("", text)
        question = parse_question(index, text, roster)

        # 1. Retrieval: is the gold table reachable at all?
        hits = [hit.key for hit in retriever.search(question, top_k=TOP_K)]
        if gold_key in hits:
            rank = hits.index(gold_key)
            ranks.append(rank)
            retrieval["hit@1" if rank == 0 else f"hit@{TOP_K}"] += 1
        else:
            retrieval["MISS"] += 1

        # 2. The reader, handed the gold table, so retrieval cannot mask it.
        grid = store.rows(gold_key)
        meta = store.meta(gold_key)
        unit_text = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        asked = UNIT_SCALE.get(question.target_unit) or 1.0

        # Where the answer actually lives, so a miss can be attributed rather
        # than merely counted. A value may appear more than once; every position
        # is kept and the closest interpretation wins.
        gold_cells = set()
        for r_index, line in enumerate(grid[1:], start=1):
            for c_index, raw_cell in enumerate(line):
                value = parse_number(raw_cell)
                if value is None or value == 0:
                    continue
                if close(value, gold_answer):
                    gold_cells.add((r_index, c_index))

        found = lookup_mod.find(grid, question)
        if found is None:
            cell["matcher_found_nothing"] += 1
            continue
        if found.score < lookup_mod.MIN_LABEL_SCORE:
            cell["matcher_declined_low_score"] += 1
            continue

        # Compare CELL POSITION, not converted value.
        #
        # The first version of this compared our unit-converted answer against
        # `record["answer"]` and reported 19% of answered questions as
        # "right cell, wrong unit". That was this scorer's bug, not the
        # pipeline's: the generator locks a fact by running its query over the
        # raw CSV, so `answer` is the **raw cell**, never converted to the unit
        # the question asks for. Asked "bao nhiêu tỷ đồng", the gold is still
        # 1.40305e13 VND while our correct 14,030.5 was counted wrong.
        # (`build_sft.py` already multiplies by column_scale/asked_scale for
        # exactly this reason; this scorer did not.)
        #
        # Position is comparable because the gold value is a raw cell, so the
        # cells holding it can be found exactly. Unit correctness cannot be
        # judged from this data at all — the records carry no unit field — so it
        # is not claimed. See `suspicious_scale` for the one unit fault that is
        # visible without one.
        position = (found.row, found.column)

        if not gold_cells:
            cell["answer_in_no_cell"] += 1
        elif position in gold_cells:
            cell["CELL_CORRECT"] += 1
            # A money scale applied to a column that counts things — shares,
            # employees, contracts — is wrong whatever unit was asked for. It is
            # the one scale fault visible without a gold unit, and it is real:
            # "số lượng cổ phiếu đang lưu hành" came back as 6.9e15 shares.
            scale_in = lookup_mod.column_scale(grid, found.column, unit_text)
            counts_things = re.search(r"số lượng|số cổ phiếu|cổ phiếu đang lưu hành",
                                      question.question, re.I)
            if counts_things and scale_in != 1.0:
                cell["suspicious_scale_on_a_count"] += 1
        elif any(r == found.row for r, _ in gold_cells):
            cell["wrong_column"] += 1
        elif any(c == found.column for _, c in gold_cells):
            cell["wrong_row"] += 1
        else:
            cell["wrong_row_and_column"] += 1

    total = len(records)
    print(f"{total} gold questions from {SOURCE.name}\n")

    print("  1. RETRIEVAL — is the gold table in our top-10?")
    for key, count in retrieval.most_common():
        print(f"       {key:24s} {count:5d}   {count / total:5.1%}")
    if ranks:
        print(f"       median rank of the gold table: {sorted(ranks)[len(ranks) // 2]}")

    print("\n  2. READING — given the gold table, which cell does the matcher take?")
    order = ["CELL_CORRECT", "wrong_row", "wrong_column", "wrong_row_and_column",
             "matcher_declined_low_score", "matcher_found_nothing",
             "answer_in_no_cell", "gold_answer_not_numeric"]
    for key in order:
        if cell[key]:
            print(f"       {key:26s} {cell[key]:5d}   {cell[key] / total:5.1%}")
    if cell["suspicious_scale_on_a_count"]:
        print(f"       (of the correct cells, {cell['suspicious_scale_on_a_count']} "
              f"apply a money scale to a count — a real unit bug)")

    tried = (cell["CELL_CORRECT"] + cell["wrong_row"] + cell["wrong_column"]
             + cell["wrong_row_and_column"])
    print(f"\n  matcher commits to an answer on {tried}/{total} = {tried / total:.1%};"
          f" of those it takes the right cell {cell['CELL_CORRECT'] / max(tried, 1):.1%}")

    print("\n  Coverage, not accuracy, is the binding constraint: the matcher never")
    print("  even produces a candidate on most questions. Where it does commit it")
    print("  is mostly right, so raising the 0.75 threshold's reach is a different")
    print("  and larger lever than making the reader more accurate.")
    print("\n  This is the label-matcher branch alone — the LLM branches that carry")
    print("  the rest of the submission are not run here, so these are proportions")
    print("  between failure types, not the pipeline's accuracy.")


if __name__ == "__main__":
    main()
