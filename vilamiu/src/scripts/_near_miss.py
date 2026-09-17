"""How often is the label matcher's choice a coin flip between near-equal rows?

The matcher scores every row label against the question's metric words and takes
the argmax. That mechanism cannot tell "Chứng chỉ tiền gửi" from "Tiền gửi của
khách hàng" — they share most of their tokens and mean different things — and hand
verification caught exactly that error. A model would not make it.

But a model is only worth adding where the decision is actually close. If the
winner beats the runner-up by a wide margin almost everywhere, then the matcher is
not guessing and a chooser has nothing to fix. If the top two are near-equal often,
every one of those is a coin flip the matcher is losing half the time.

Measured with no gold and no model: for each question, collect the labels from
every shortlisted table with their scores, and report the margin between first and
second.

Usage:  PYTHONPATH=src python scripts/_near_miss.py --n 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    # The pool the matcher actually serves: one company, a single figure asked for.
    scope = [q for q in questions
             if len(q.tickers) == 1 and q.years
             and lookup_mod.is_single_lookup(q.question)][:args.n]
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    metric_of = lookup_mod.extract_metric

    margins, close_cases = [], []
    silent = 0
    started = time.time()
    for index, question in enumerate(scope, start=1):
        metric = metric_of(question.question)
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]

        scored = []
        for key in keys:
            grid = store.rows(key)
            if len(grid) < 2:
                continue
            label_col = lookup_mod.label_column(grid)
            meta = store.meta(key)
            match = lookup_mod.match_row(grid, metric, label_col, meta.caption)
            if match is None:
                continue
            row, score, label = match
            scored.append((score, label, key, row))
        if not scored:
            silent += 1
            continue
        scored.sort(key=lambda item: -item[0])
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        margin = best[0] - second[0] if second else 1.0
        margins.append(margin)
        if second is not None and margin <= 0.05 and best[1] != second[1]:
            close_cases.append((question.question, best, second))
        if index % 100 == 0:
            print(f"  {index}/{len(scope)}  {time.time() - started:.0f}s", flush=True)

    if not margins:
        print("khong do duoc")
        return
    margins.sort()
    def share(limit: float) -> str:
        n = sum(1 for m in margins if m <= limit)
        return f"{n} ({100 * n / len(margins):.0f}%)"

    print(f"\n{len(scope)} cau trong pham vi khop nhan, {silent} cau khong ung vien nao")
    print(f"do cach giua nhat va nhi (tren {len(margins)} cau co ung vien):")
    print(f"  bang nhau (<=0.00): {share(0.0)}")
    print(f"  sat (<=0.05):       {share(0.05)}")
    print(f"  gan (<=0.15):       {share(0.15)}")
    print(f"  trung vi do cach:   {margins[len(margins) // 2]:.3f}")
    print(f"\nnhung ca sat diem nhung KHAC NHAN — day la cho tung xu "
          f"({len(close_cases)} ca), {min(args.show, len(close_cases))} vi du:")
    for text, best, second in close_cases[:args.show]:
        print(f"  hoi : {text[:104]}")
        print(f"    chon  {best[0]:.3f}  {str(best[1])[:64]}")
        print(f"    a quan {second[0]:.3f}  {str(second[1])[:64]}")


if __name__ == "__main__":
    main()
