"""Size the question classes that EXEC is currently losing, so effort follows volume."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COHORT = re.compile(r"trong nhóm|xét các|trong các (?:doanh nghiệp|công ty)|nhóm gồm|trong bốn|trong năm mã|các mã", re.I)
MEDIAN = re.compile(r"trung vị|trung bình của nhóm|bình quân nhóm", re.I)
SUPER = re.compile(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất|ít nhất", re.I)
RATIO_UNIT = re.compile(r"bao nhiêu\s*%|phần trăm|tỷ lệ|tỉ lệ|\blần\b|\bvòng\b|biên lợi nhuận|hệ số", re.I)
MULTIYEAR = re.compile(r"giai đoạn|từ năm \d{4} đến|so với năm|qua các năm", re.I)


def load_questions() -> dict[int, str]:
    path = ROOT / "data" / "questions" / "questions.jsonl"
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["id"])] = rec["question"]
    return out


def load_zip(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as zf:
        target = next(n for n in zf.namelist() if n.endswith("submission.json"))
        return {int(r["id"]): r for r in json.loads(zf.read(target).decode("utf-8"))}


def num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    questions = load_questions()
    rows = load_zip(sys.argv[1] if len(sys.argv) > 1 else "retry14b.zip")

    buckets: Counter[str] = Counter()
    suspect: Counter[str] = Counter()
    for qid, q in questions.items():
        cohort = bool(COHORT.search(q))
        if cohort and MEDIAN.search(q):
            name = "cohort + median filter"
        elif cohort and SUPER.search(q):
            name = "cohort + superlative"
        elif cohort:
            name = "cohort, plain"
        elif MULTIYEAR.search(q):
            name = "multi-year series"
        elif SUPER.search(q):
            name = "superlative, single entity"
        else:
            name = "plain lookup / derived"
        buckets[name] += 1

        value = num(rows.get(qid, {}).get("answer"))
        if value is None:
            continue
        wrong_unit = RATIO_UNIT.search(q) and abs(value) > 1000
        if wrong_unit or value == 0.0:
            suspect[name] += 1

    print(f"{'class':<28} {'questions':>9} {'certainly wrong':>16}")
    for name, total in buckets.most_common():
        print(f"{name:<28} {total:>9} {suspect[name]:>16}")
    print(f"{'TOTAL':<28} {sum(buckets.values()):>9} {sum(suspect.values()):>16}")


if __name__ == "__main__":
    main()
