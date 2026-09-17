"""Audit canonical metric coverage on the official question set.

This is a schema-coverage report, not a leaderboard evaluator.  It groups
questions that do not map to any canonical key so dictionary work can be done
by financial concept family instead of by question ID.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.finance.metrics import METRICS, find_metrics
from vifinqa.utils.io import read_jsonl
from vifinqa.utils.viet_text import norm


FAMILIES = (
    ("bank_lending_funding", ("cho vay", "du no", "tien gui", "tctd", "tin dung")),
    ("provisions", ("du phong", "trich lap")),
    ("receivables_payables", ("phai thu", "phai tra", "tra truoc", "tam ung")),
    ("bonds_securities", ("trai phieu", "chung khoan", "ky phieu", "giay to co gia")),
    ("tax_expenses", ("chi phi", "thue", "gia von")),
    ("investments", ("dau tu", "gop von", "cong ty con", "lien ket")),
    ("shares_ownership", ("co phieu", "co phan", "so huu", "bieu quyet")),
    ("leases_commitments", ("thue hoat dong", "cam ket", "tien thue")),
    ("compensation", ("luong", "thu lao", "thu nhap ban", "nhan vien")),
    ("cash_revenue_profit", ("tien", "doanh thu", "loi nhuan", "lai")),
)


def family(question: str) -> str:
    text = norm(question)
    for boilerplate in ("cong ty co phan", "cong ty cp", "ngan hang thuong mai co phan"):
        text = text.replace(boilerplate, " ")
    scores = [(sum(marker in text for marker in markers), name)
              for name, markers in FAMILIES]
    score, name = max(scores)
    return name if score else "other_notes"


def audit(questions: list[dict]) -> dict:
    unmatched, matched = [], []
    key_counts: Counter[str] = Counter()
    for question in questions:
        matches = find_metrics(question["question"])
        if matches:
            matched.append(question)
            key_counts.update(match.metric.key for match in matches)
        else:
            unmatched.append(question)
    family_counts = Counter(family(q["question"]) for q in unmatched)
    return {
        "questions": len(questions),
        "canonical_metrics": len(METRICS),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "coverage": len(matched) / max(1, len(questions)),
        "family_counts": family_counts,
        "key_counts": key_counts,
        "unmatched_questions": unmatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=config.QUESTIONS_JSONL)
    parser.add_argument("--sample", type=int, default=8,
                        help="unmatched examples printed per family")
    args = parser.parse_args()

    report = audit(read_jsonl(args.questions))
    print(f"metrics={report['canonical_metrics']} "
          f"questions={report['questions']} matched={report['matched']} "
          f"unmatched={report['unmatched']} "
          f"coverage={report['coverage']:.2%}")
    by_family: dict[str, list[dict]] = {}
    for question in report["unmatched_questions"]:
        by_family.setdefault(family(question["question"]), []).append(question)
    for name, count in report["family_counts"].most_common():
        print(f"\n[{name}] {count}")
        for question in by_family[name][:max(0, args.sample)]:
            print(f"  {question['id']}: {question['question']}")


if __name__ == "__main__":
    main()
