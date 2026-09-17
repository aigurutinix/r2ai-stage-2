"""Can the regex lookup be composed across years to answer derived questions?

550 of 1012 questions are derived, and every mechanism currently serving them is
model-based: LLM code generation (19.5%), cell plans (7.2%), best-effort fallback
(5.9%). But a derived question is compositional — "chênh lệch X giữa năm A và
năm B" is two label matches and a subtraction — and label matching is the
strongest component in the pipeline (~55%).

This probe measures the premise before any mechanism is written: for a
multi-year, single-metric, single-ticker question, how many of the asked years
resolve to a figure, and does the *same* label win in each year's document?
Label agreement across independently OCR'd reports is real evidence, not the
circular kind: nothing here selected the documents by their values.

Usage:
  PYTHONPATH=src python scripts/probe_derived.py [--limit N] [--show N]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

# Ordered: the first pattern that matches wins, so the specific comparatives are
# tested before the plain aggregations they can contain.
OPS = (
    ("diff", re.compile(r"chênh lệch|hiệu số|biến động|thay đổi|so với", re.I)),
    ("growth", re.compile(r"tăng trưởng", re.I)),
    ("max", re.compile(r"cao nhất|lớn nhất", re.I)),
    ("min", re.compile(r"thấp nhất|nhỏ nhất", re.I)),
    ("avg", re.compile(r"trung bình|bình quân", re.I)),
    ("sum", re.compile(r"\btổng\b|tích lũy|cộng lại", re.I)),
)

# Derived vocabulary that belongs to a metric's *name*, not to an operation.
# "Lỗ chênh lệch tỷ giá" is a line item; "chênh lệch giữa 2016 và 2017" is a
# subtraction. Matching the keyword alone conflates them.
NAME_IDIOMS = re.compile(
    r"chênh lệch tỷ giá|chênh lệch đánh giá lại|tỷ lệ quyền biểu quyết|"
    r"tỷ lệ lợi ích|tỷ lệ sở hữu|thu nhập bình quân|bình quân tháng",
    re.I,
)


def classify(question: str) -> str | None:
    stripped = NAME_IDIOMS.sub(" ", question)
    for name, pattern in OPS:
        if pattern.search(stripped):
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=14)
    parser.add_argument("--per-group", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl",
                       root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    todo = [
        q for q in parsed
        if len(q.tickers) == 1 and len(q.years) >= 2 and q.unit_scale is not None
        and classify(q.question) is not None
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} câu đa năm / một mã / đơn vị tiền / có phép tính\n")

    ops: collections.Counter = collections.Counter()
    coverage: collections.Counter = collections.Counter()
    agree: collections.Counter = collections.Counter()
    samples: list[str] = []

    for question in todo:
        op = classify(question.question)
        ops[op] += 1
        # Retrieve per year with a metric-only query. Ranking by the full
        # question wastes the signal on company and year tokens the document
        # filter has already consumed — the same defect that made the
        # cross-encoder rerank worse than BM25 until it was fed `--query-mode
        # metric` (top-1 17.0% -> 41.4%).
        metric = lookup_mod.extract_metric(question.question)

        found_years: dict[int, tuple[str, float]] = {}
        for year in question.years:
            probe = dataclasses.replace(question, years=[year], question=metric)
            keys = [h.key for h in retriever.search(probe, top_k=args.per_group)]
            best = None
            for key in keys:
                grid = store.rows(key)
                found = lookup_mod.find(grid, probe)
                if found is None:
                    continue
                if best is None or found.score > best[0]:
                    meta = store.meta(key)
                    scale = lookup_mod.column_scale(
                        grid, found.column,
                        f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
                    )
                    best = (found.score, found.label, found.value * scale)
            if best is not None:
                found_years[year] = (best[1], best[2])

        ratio = len(found_years) / len(question.years)
        coverage["đủ mọi năm" if ratio == 1 else
                 ("≥ nửa số năm" if ratio >= 0.5 else
                  ("có ít nhất 1 năm" if found_years else "không năm nào"))] += 1

        if len(found_years) >= 2:
            labels = {lookup_mod._norm(l) for l, _ in found_years.values()}
            agree["nhãn giống nhau mọi năm" if len(labels) == 1 else "nhãn khác nhau"] += 1
            if len(samples) < args.show and len(found_years) == len(question.years):
                mark = "=" if len(labels) == 1 else "≠"
                values = ", ".join(f"{y}:{v:,.0f}" for y, (_, v) in sorted(found_years.items()))
                label = next(iter(found_years.values()))[0]
                samples.append(
                    f"  id={question.id:4d} [{op}] {mark} {label[:34]!r}\n"
                    f"        {question.question[:88]}\n        {values}"
                )

    def dump(title: str, counter: collections.Counter) -> None:
        total = sum(counter.values()) or 1
        print(title)
        for key, count in counter.most_common():
            print(f"  {count:4d}  ({count / total:4.0%})  {key}")
        print()

    dump("Phép tính:", ops)
    dump("Số năm giải được:", coverage)
    dump("Nhãn có nhất quán giữa các tài liệu độc lập:", agree)
    print("Ví dụ (mọi năm đều giải được):")
    for line in samples:
        print(line)


if __name__ == "__main__":
    main()
