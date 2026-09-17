"""Does the dense retriever put the answer's table in the candidate set more often?

The organisers measured 47.41% recall@10 for BM25 and 80.80% for a dense embedder with a
reranker, and word overlap — what this pipeline used — sits in the BM25 band. This checks
whether the swap actually moved recall HERE, on these questions, rather than trusting a
number measured on their pipeline.

The test is the same one that exposed the earlier mistake: for a question whose answer is
known, does any cell of the candidate tables hold it, converted to the unit the question
asks for. Two guards, both learned the hard way:

  the comparison happens in the QUESTION'S unit, because the known answer was rounded to
  two decimals there — comparing in đồng made a table that was offered first look absent
  and produced a "16% recall" that was pure arithmetic error.

  a permuted control runs alongside, testing each question against ANOTHER question's
  answer. Nine tables hold thousands of cells and a hit is allowed at five scales, so a
  candidate set can match almost any figure by chance; only the gap between the real rate
  and the permuted one means anything.

Usage:
  python scripts/fresh/recall_dense.py --topk artifacts/fresh/dense_topk.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from recall_shared import cells_of, known_answers  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", default="artifacts/fresh/dense_topk.json")
    parser.add_argument("--old-prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--neutral", action="store_true")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    known = known_answers(ROOT, neutral=args.neutral)
    dense = {int(k): v for k, v in
             json.loads((ROOT / args.topk).read_text(encoding="utf-8")).items()}
    old = {}
    for line in (ROOT / args.old_prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            old[record["id"]] = record["meta"]["refs"]

    cache: dict[Path, list[float]] = {}
    order = sorted(known)

    def covered(refs: list[dict], target: float, unit: float) -> bool:
        for ref in refs:
            path = (ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"] /
                    ref["doc"] / f"{ref['doc']}_extracted_tables"
                    / f"table_{ref['table_id']}.csv")
            if path not in cache:
                if len(cache) > 500:
                    cache.clear()
                cache[path] = cells_of(path)
            for value in cache[path]:
                if any(abs(value * scale / unit - target) <= 0.01 for scale in SCALES):
                    return True
        return False

    counters: Counter[str] = Counter()
    for position, qid in enumerate(order):
        _name, unit = unit_of(questions[qid])
        if not unit:
            continue
        target = abs(known[qid])
        if args.permute:
            others = [v for other, v in known.items() if other != qid]
            target = abs(others[position % len(others)])
        if qid in old:
            counters["cu — tu khoa"] += 1
            counters["cu — CO"] += covered(old[qid], target, unit)
        if qid in dense:
            counters["moi — dense"] += 1
            counters["moi — CO"] += covered(dense[qid], target, unit)

    for label, total, hit in (("chong tu khoa (cu)", "cu — tu khoa", "cu — CO"),
                              ("dense embedding (moi)", "moi — dense", "moi — CO")):
        n, k = counters[total], counters[hit]
        if n:
            print(f"  {label}: {k}/{n} = {100 * k / n:.0f}% recall")


if __name__ == "__main__":
    main()
