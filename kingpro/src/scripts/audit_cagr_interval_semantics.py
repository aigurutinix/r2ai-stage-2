"""Detect off-by-one CAGR exponents without changing submission answers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


POW_RE = re.compile(r"\*\*\s*\(\s*1(?:\.0)?\s*/\s*(\d+(?:\.0)?)\s*\)")
YEAR_RE = re.compile(r"\b(20\d{2})\b")


def audit(rows: list[dict]) -> dict:
    findings = []
    checked = 0
    for row in rows:
        question = str(row.get("question", ""))
        if "cagr" not in question.casefold():
            continue
        checked += 1
        years = sorted({int(value) for value in YEAR_RE.findall(question)})
        exponents = [float(value) for value in POW_RE.findall(str(row.get("pandas_query", "")))]
        if len(years) < 2:
            findings.append({"id": int(row["id"]), "kind": "cagr_years_unresolved", "years": years})
            continue
        intervals = years[-1] - years[0]
        if intervals <= 0 or not exponents:
            findings.append({"id": int(row["id"]), "kind": "cagr_exponent_unresolved", "years": years, "exponents": exponents})
            continue
        wrong = [value for value in exponents if abs(value - intervals) > 1e-12]
        if wrong:
            findings.append({
                "id": int(row["id"]),
                "kind": "cagr_interval_mismatch",
                "years": years,
                "expected_interval_count": intervals,
                "observed_exponent_denominators": exponents,
                "question": question,
                "claim_limit": "Operator mismatch is proven; source recomputation is still required before changing an answer.",
            })
    return {
        "kind": "cagr_interval_semantics",
        "questions_with_cagr": checked,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    rows = json.loads((args.submission / "submission.json").read_text(encoding="utf-8"))
    report = audit(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_findings and report["finding_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
