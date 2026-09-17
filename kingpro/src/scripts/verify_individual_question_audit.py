"""Fail-closed coverage check for the durable per-question audit ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROW = re.compile(r"^\|\s*(R\d+)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--expected-count", type=int, default=1012)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    submission_path = args.submission_dir.resolve() / "submission.json"
    ledger_path = args.ledger.resolve()
    rows = json.loads(submission_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("submission.json must contain a list")

    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    int_ids = [value for value in ids if isinstance(value, int) and not isinstance(value, bool)]
    counts = Counter(int_ids)
    expected = set(range(1, args.expected_count + 1))
    candidate_ids = set(int_ids)

    text = ledger_path.read_text(encoding="utf-8")
    references: dict[int, list[str]] = {}
    ledger_rows = 0
    for ref, question_field in ROW.findall(text):
        ledger_rows += 1
        for token in re.findall(r"\d+", question_field):
            qid = int(token)
            if 1 <= qid <= args.expected_count:
                references.setdefault(qid, []).append(ref)

    digest = sha256(submission_path)
    problems = {
        "candidate_non_integer_ids": len(ids) - len(int_ids),
        "candidate_duplicate_ids": sorted(qid for qid, count in counts.items() if count > 1),
        "candidate_missing_ids": sorted(expected - candidate_ids),
        "candidate_out_of_range_ids": sorted(candidate_ids - expected),
        "ledger_missing_ids": sorted(expected - set(references)),
        "ledger_out_of_range_ids": sorted(set(references) - expected),
        "sha256_mismatch": bool(args.expected_sha256 and digest != args.expected_sha256.upper()),
    }
    status = "PASS" if not any(problems.values()) else "FAIL"
    report = {
        "status": status,
        "submission": str(args.submission_dir.resolve()),
        "submission_json_sha256": digest,
        "expected_sha256": args.expected_sha256.upper() if args.expected_sha256 else None,
        "candidate_records": len(rows),
        "candidate_unique_in_range_ids": len(candidate_ids & expected),
        "ledger": str(ledger_path),
        "ledger_table_rows": ledger_rows,
        "ledger_unique_in_range_ids": len(set(references) & expected),
        "ledger_repeat_verifications": sum(max(0, len(refs) - 1) for refs in references.values()),
        "problems": problems,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
