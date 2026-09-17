"""Build the isolated q375 interest-coverage difference repair over v188.

The question asks for the difference between the mean interest-coverage ratio
of the above-median D/E group and the remaining group.  The reviewed v188
program divided those two group means.  This builder preserves every source
cell and grouping rule, replacing only that final quotient with the absolute
difference requested by the question.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v188_data_derived_year_labels"
OUTPUT = ROOT / "sub_top123_candidate_v189_interest_coverage_difference"
QUESTION_ID = 375
OLD_FORMULA = "result = avg_interest_coverage_high_de / avg_interest_coverage_low_de"
NEW_FORMULA = "result = abs(avg_interest_coverage_high_de - avg_interest_coverage_low_de)"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    submission = read_json(SOURCE / "submission.json")
    rows = {int(row["id"]): row for row in submission}
    row = rows[QUESTION_ID]
    before_rows = {int(item["id"]): digest(item) for item in submission}
    before_submission = digest(submission)

    if float(row["answer"]) != 1.74:
        raise ValueError(f"unexpected q375 baseline answer: {row['answer']!r}")
    if row["pandas_query"].count(OLD_FORMULA) != 1:
        raise ValueError("q375 reviewed quotient formula not found exactly once")
    expected_evidence = [{"variable": "df1", "csv_path": "data/q375_source_cells.csv"}]
    if row.get("evidence") != expected_evidence:
        raise ValueError(f"q375 evidence changed from reviewed baseline: {row.get('evidence')!r}")

    row["answer"] = 24.94
    row["pandas_query"] = row["pandas_query"].replace(OLD_FORMULA, NEW_FORMULA)

    changed = [
        question_id
        for question_id, old_digest in before_rows.items()
        if digest(rows[question_id]) != old_digest
    ]
    if changed != [QUESTION_ID]:
        raise AssertionError(f"unexpected submission deltas: {changed[:20]}")

    shutil.copytree(SOURCE, OUTPUT)
    write_json(OUTPUT / "submission.json", submission)

    panel_audit = read_json(SOURCE / "panel_source_audit.json")
    panel_row = next(item for item in panel_audit if int(item["id"]) == QUESTION_ID)
    if float(panel_row["answer"]) != 1.74:
        raise ValueError("q375 panel audit baseline does not match v188")
    panel_row["answer"] = 24.94
    panel_row["formula_repair"] = {
        "old": "mean(high-D/E interest coverage) / mean(other interest coverage)",
        "new": "abs(mean(high-D/E interest coverage) - mean(other interest coverage))",
        "source_cells_unchanged": True,
    }
    write_json(OUTPUT / "panel_source_audit.json", panel_audit)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "isolated source-preserving q375 arithmetic-intent repair",
        "changed_questions": [QUESTION_ID],
        "source_submission_sha256": before_submission,
        "candidate_submission_sha256": digest(submission),
        "answer_change": {"old": 1.74, "new": 24.94},
        "calculation": {
            "median_de": 0.47805178962725403,
            "high_de_tickers": ["DCM", "GVR"],
            "other_tickers": ["DPM", "PRT"],
            "mean_interest_coverage_high_de": 58.75312905097468,
            "mean_interest_coverage_other": 33.81102209051275,
            "absolute_difference": 24.942106960461928,
            "rounded_answer": 24.94,
        },
        "invariants": {
            "question_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_tables_unchanged": True,
            "evidence_unchanged": True,
            "source_cells_unchanged": True,
            "all_other_submission_rows_unchanged": True,
            "v184_locked_artifact_untouched": True,
            "v188_source_untouched": True,
        },
        "claim_limit": (
            "Source-proven local correction; BTC score is unknown until the exact "
            "artifact is evaluated. This report is not leaderboard gold."
        ),
    }
    write_json(OUTPUT / "q375_interest_coverage_difference_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
