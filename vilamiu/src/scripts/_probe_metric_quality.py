"""Does `extract_metric` actually return a line-item name?

Everything downstream matches the question's metric against row labels, so a metric
that comes out as a fragment gives the matcher nothing to work with. Hand-reading a
batch of shipped answers turned up several:

  "Năm nào trong các năm 2022 và 2023 …"          -> '2022 và 2023'
  "… doanh thu hoạt động tài chính cao nhất …"    -> 'tài chính cao nhất'
  "… hệ số thanh toán nhanh cuối năm đó …"        -> 'hệ số thanh toán nhanh cuối năm đó'

These are not near-misses. A phrase of years cannot match any label, and no amount of
lowering the matcher's threshold or relabelling its blank rows can help a question
whose metric was destroyed before the matcher saw it.

This counts the shapes that cannot be a Vietnamese line item: only digits and
conjunctions, a leftover comparative, a leftover period phrase, or too short to carry
a noun phrase. It needs no gold answers — a metric is judged against the question it
came from.

Usage:
  PYTHONPATH=src python scripts/_probe_metric_quality.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402

# A metric made only of years, conjunctions and punctuation.
YEARS_ONLY_RE = re.compile(
    r"^[\s\d/.\-–,]*(?:(?:19|20)\d{2}[\s,/.\-–]*|và|hoặc|đến|tới)+$", re.I)
# Comparative and superlative words belong to the operation, not the line item.
COMPARATIVE_RE = re.compile(
    r"\b(cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất|ít nhất|"
    r"cao hơn|thấp hơn|lớn hơn|nhỏ hơn|trung vị|trung bình|"
    r"tăng trưởng|chênh lệch|biến động|so với)\b", re.I)
# A period phrase left inside the metric.
PERIOD_LEFT_RE = re.compile(
    r"\b(cuối năm( đó)?|đầu năm|năm đó|năm nay|năm trước|kỳ này|kỳ trước|"
    r"ngày \d|quý [IVX1-4])\b", re.I)
# Words that only ever appear in a question frame, never in a statement line.
FRAME_RE = re.compile(r"\b(doanh nghiệp nào|công ty nào|năm nào|bao nhiêu|"
                      r"trong nhóm|trong số|xét các|đồng thời)\b", re.I)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/questions/questions.jsonl")
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    tally: collections.Counter[str] = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)

    for line in (ROOT / args.questions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question = record["question"]
        metric = lookup_mod.extract_metric(question)
        text = metric.strip()
        tally["tổng"] += 1

        if not text or len(text) < 6:
            label = "quá ngắn / rỗng"
        elif YEARS_ONLY_RE.match(text):
            label = "CHỈ CÓ NĂM"
        elif FRAME_RE.search(text):
            label = "còn khung câu hỏi"
        elif COMPARATIVE_RE.search(text):
            label = "còn từ so sánh"
        elif PERIOD_LEFT_RE.search(text):
            label = "còn cụm thời kỳ"
        else:
            label = "trông như một chỉ tiêu"
        tally[label] += 1
        if label != "trông như một chỉ tiêu" and len(examples[label]) < args.show:
            examples[label].append((record["id"], text[:56], question[:74]))

    total = tally.pop("tổng", 0)
    print(f"{total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:26s} {count:5d}  {count / max(total, 1):6.1%}")
    broken = total - tally.get("trông như một chỉ tiêu", 0)
    print(f"\n  chỉ tiêu KHÔNG dùng được: {broken}/{total} = {broken / max(total, 1):.1%}")
    for name, _ in tally.most_common():
        if name == "trông như một chỉ tiêu":
            continue
        for qid, metric, question in examples.get(name, [])[:4]:
            print(f"\n  [{name}] id={qid}")
            print(f"     chỉ tiêu: {metric!r}")
            print(f"     câu     : {question}")


if __name__ == "__main__":
    main()
