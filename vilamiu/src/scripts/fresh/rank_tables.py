"""Pick each question's candidate tables from the embeddings, at home, for free.

Only the encoding needs a GPU. Ranking is 1012 dot products against a few dozen candidate
tables each, which is a second of CPU — so it stays outside the paid window.

The metadata filter comes first and is not negotiable: a table only competes for a
question if its company and year match what the question names, and its scope matches
whether the question says "công ty mẹ". That filter is the part of this pipeline that has
always worked, and it removes almost all of the corpus before similarity is consulted.
Similarity then decides among what is left, which is the job word overlap was doing badly.

Usage:
  python scripts/fresh/rank_tables.py --vectors artifacts/fresh/vectors --top 9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default="artifacts/fresh/vectors")
    parser.add_argument("--top", type=int, default=9)
    parser.add_argument("--out", default="artifacts/fresh/dense_topk.json")
    args = parser.parse_args()

    base = ROOT / args.vectors
    table_vectors = np.load(base / "tables.npy").astype(np.float32)
    question_vectors = np.load(base / "questions.npy").astype(np.float32)
    table_meta = json.loads((base / "table_refs.json").read_text(encoding="utf-8"))
    question_ids = json.loads((base / "question_ids.json").read_text(encoding="utf-8"))
    row_of_question = {qid: i for i, qid in enumerate(question_ids)}

    # Index the corpus by the metadata filter, so similarity is only ever computed
    # against tables the question could possibly want.
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, meta in enumerate(table_meta):
        by_key[(meta["ticker"], meta["year"])].append(index)

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    out: dict[str, list[dict]] = {}
    empty = 0
    for question in questions:
        text = question["question"]
        row = row_of_question.get(question["id"])
        if row is None:
            continue
        tickers = sorted(resolver.resolve(text))
        years = sorted({y for y in YEAR_RE.findall(text)})
        if not tickers or not years:
            empty += 1
            continue
        candidates: list[int] = []
        for ticker in tickers:
            for year in years:
                candidates += by_key.get((ticker, year), [])
        if not candidates:
            empty += 1
            continue

        want_separate = bool(PARENT_RE.search(text))
        scores = table_vectors[candidates] @ question_vectors[row]
        # Prefer the scope the question asks for without excluding the other: a note may
        # only exist in one of the two reports.
        for position, index in enumerate(candidates):
            if ("separate" in table_meta[index]["doc"]) != want_separate:
                scores[position] -= 0.05

        # Share the slots across the company-year pairs the question names, so a
        # question about four companies cannot be answered from one company's tables.
        per_pair = max(1, args.top // max(1, len(tickers) * len(years)))
        picked: list[int] = []
        used: dict[tuple[str, str], int] = defaultdict(int)
        for position in np.argsort(-scores):
            index = candidates[int(position)]
            key = (table_meta[index]["ticker"], table_meta[index]["year"])
            if used[key] >= per_pair and len(picked) >= args.top:
                break
            if used[key] >= per_pair:
                continue
            picked.append(index)
            used[key] += 1
            if len(picked) >= args.top:
                break
        # Fill any leftover slots with the next best overall.
        if len(picked) < args.top:
            for position in np.argsort(-scores):
                index = candidates[int(position)]
                if index not in picked:
                    picked.append(index)
                if len(picked) >= args.top:
                    break

        out[str(question["id"])] = [
            {"ref": table_meta[i]["ref"], "doc": table_meta[i]["doc"],
             "ticker": table_meta[i]["ticker"], "year": table_meta[i]["year"],
             "table_id": table_meta[i]["table_id"]} for i in picked]

    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"{len(out)} cau co ung vien, {empty} cau khong xac dinh duoc ma/nam")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
