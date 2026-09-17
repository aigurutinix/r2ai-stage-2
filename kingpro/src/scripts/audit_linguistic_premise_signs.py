"""Audit answer signs against directional Vietnamese comparison wording.

This is a read-only triage tool.  A wording/sign mismatch is not automatically
an error: benchmark questions sometimes contain a false premise while the
expected answer still follows ordered subtraction or signed growth.  Findings
must therefore be checked against source cells and the durable review ledger.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "knowledge" / "vothuong" / "experiments.jsonl"


RULES = (
    (
        "positive_excess",
        re.compile(r"\b(nhiều hơn|cao hơn|lớn hơn|vượt(?: quá)?)\b", re.IGNORECASE),
        "nonnegative",
    ),
    (
        "positive_shortfall",
        re.compile(r"\b(ít hơn|thấp hơn|nhỏ hơn|kém hơn)\b", re.IGNORECASE),
        "nonnegative",
    ),
    (
        "positive_decrease_amount",
        re.compile(r"\bgiảm (?:đi )?bao nhiêu\b", re.IGNORECASE),
        "nonnegative",
    ),
    (
        "signed_increase_wording",
        re.compile(r"\btăng bao nhiêu(?: phần trăm| %|%)?\b", re.IGNORECASE),
        "nonnegative",
    ),
)


def reviewed_questions(path: Path) -> dict[int, list[dict]]:
    reviewed: dict[int, list[dict]] = {}
    if not path.is_file():
        return reviewed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "review":
            continue
        for raw_id in event.get("question_ids", []):
            reviewed.setdefault(int(raw_id), []).append(
                {
                    "review_id": event.get("id"),
                    "verdict": event.get("verdict"),
                    "summary": event.get("summary"),
                }
            )
    return reviewed


def audit(submission_dir: Path, ledger: Path) -> dict:
    records = json.loads(
        (submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    reviews = reviewed_questions(ledger)
    findings: list[dict] = []
    matched = 0
    for record in records:
        question = str(record.get("question", ""))
        answer = record.get("answer")
        if not isinstance(answer, (int, float)) or isinstance(answer, bool):
            continue
        for rule_name, pattern, expected in RULES:
            match = pattern.search(question)
            if not match:
                continue
            matched += 1
            mismatch = expected == "nonnegative" and float(answer) < 0
            if mismatch:
                qid = int(record["id"])
                code = str(record.get("pandas_query", ""))
                findings.append(
                    {
                        "id": qid,
                        "question": question,
                        "answer": answer,
                        "rule": rule_name,
                        "matched_phrase": match.group(0),
                        "expected_sign_from_wording": expected,
                        "uses_abs": bool(re.search(r"\babs\s*\(", code)),
                        "uses_growth_formula": bool(
                            re.search(r"(?:/|\bdiv\b).*(?:100|1e2)", code, re.DOTALL)
                        ),
                        "durable_reviews": reviews.get(qid, []),
                        "status": "already_reviewed" if qid in reviews else "needs_review",
                    }
                )
            break
    findings.sort(key=lambda item: (item["status"] != "needs_review", item["id"]))
    return {
        "submission": str(submission_dir.resolve()),
        "ledger": str(ledger.resolve()),
        "question_count": len(records),
        "directional_wording_count": matched,
        "finding_count": len(findings),
        "unreviewed_count": sum(item["status"] == "needs_review" for item in findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "A linguistic sign mismatch is triage only. Source values, benchmark "
            "sign conventions, and prior isolated leaderboard evidence remain authoritative."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.submission, args.ledger)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
