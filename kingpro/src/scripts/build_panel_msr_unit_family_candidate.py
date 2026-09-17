"""Build source-confirmed MSR thousand-VND fixes for q473/q490/q568.

The three questions are paraphrases of the same panel calculation.  Their
compact manifests store MSR's printed revenue/profit tokens with scale 1 even
though all three exact statements explicitly say ``Nghìn VND``.  Restoring
scale 1,000 preserves the margin selector and corrects the 2022 revenue sum.

By default the builder layers the family fix on v200 (q782 + q1003).  Use
``--isolated`` to create a clean A/B candidate from the measured v196 base.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "sub_top123_candidate_v196_effective_tax_sign"
CUMULATIVE_SOURCE = ROOT / "sub_top123_candidate_v200_cumulative_unit_fixes"
QIDS = (473, 490, 568)
MSR_SOURCES = {
    2020: (
        "MSR_financial_statements_2020_consolidated|260",
        "table_7_line260.csv",
    ),
    2021: (
        "MSR_financial_statements_2021_consolidated|266",
        "table_7_line266.csv",
    ),
    2022: (
        "MSR_financial_statements_2022_consolidated|357",
        "table_3_line357.csv",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalize_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])

    changed = []
    for row in rows:
        if row.get("ticker") != "MSR":
            continue
        year = int(row["year"])
        if year not in MSR_SOURCES:
            raise AssertionError(f"unexpected MSR year in {path.name}: {year}")
        expected_table, expected_csv = MSR_SOURCES[year]
        if row.get("source_table") != expected_table:
            raise AssertionError(f"unexpected MSR table in {path.name}: {row}")
        if row.get("source_csv") != expected_csv:
            raise AssertionError(f"unexpected MSR source CSV in {path.name}: {row}")
        if row.get("metric_key") not in {"kqkd:10", "kqkd:60"}:
            raise AssertionError(f"unexpected MSR metric in {path.name}: {row}")
        if float(row.get("scale", 0)) != 1.0:
            raise AssertionError(f"unexpected old MSR scale in {path.name}: {row}")
        row["scale"] = "1000.0"
        changed.append(dict(row))

    if len(changed) != 6:
        raise AssertionError(f"expected six MSR rows in {path.name}, got {len(changed)}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def build(isolated: bool = False) -> dict:
    source = BASELINE if isolated else CUMULATIVE_SOURCE
    output = ROOT / (
        "sub_top123_candidate_v201_panel_msr_unit_isolated"
        if isolated
        else "sub_top123_candidate_v201_cumulative_three_unit_fixes"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shutil.copytree(source, output)

    source_submission = source / "submission.json"
    output_submission = output / "submission.json"
    source_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in source_rows}
    changed_evidence: dict[int, list[dict[str, str]]] = {}

    for qid in QIDS:
        row = next(item for item in rows if int(item["id"]) == qid)
        if float(row["answer"]) != 191.14:
            raise AssertionError(f"unexpected q{qid} old answer: {row['answer']}")
        changed_evidence[qid] = normalize_manifest(
            output / "data" / f"q{qid}_source_cells.csv"
        )
        row["answer"] = 206.67

    write_json(output_submission, rows)

    panel_path = output / "panel_source_audit.json"
    panel_rows = json.loads(panel_path.read_text(encoding="utf-8"))
    for qid in QIDS:
        matches = [item for item in panel_rows if int(item["id"]) == qid]
        if len(matches) != 1:
            raise AssertionError(f"expected one q{qid} panel audit row")
        audit = matches[0]
        if float(audit["answer"]) != 191.14:
            raise AssertionError(f"unexpected q{qid} panel-audit answer")
        audit["answer"] = 206.67
        audit["unit_correction"] = (
            "MSR 2020-2022 statements explicitly use thousand VND; "
            "six MSR revenue/profit source cells scaled by 1,000"
        )
    write_json(panel_path, panel_rows)

    changed_ids = []
    for row in rows:
        qid = int(row["id"])
        if row != old_by_id[qid]:
            changed_ids.append(qid)
            for key in set(row) | set(old_by_id[qid]):
                if key != "answer" and row.get(key) != old_by_id[qid].get(key):
                    raise AssertionError(f"q{qid} non-answer field changed: {key}")
    if changed_ids != list(QIDS):
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")

    report = {
        "candidate": output.name,
        "source_candidate": source.name,
        "isolated": isolated,
        "changed_question_ids_relative_to_source": list(QIDS),
        "old_answer": 191.14,
        "new_answer": 206.67,
        "normalized_rows_per_question": 6,
        "source_unit_evidence": [
            f"build/tables/{table.split('|', 1)[0]}/{csv_name}"
            for table, csv_name in MSR_SOURCES.values()
        ],
        "invariants": {
            "only_family_answers_changed_relative_to_source": True,
            "pandas_queries_unchanged": True,
            "retrieval_fields_unchanged": True,
            "only_family_msr_evidence_scales_changed": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": (
            "Source-confirmed family unit correction; leaderboard result remains unknown."
        ),
    }
    write_json(output / "q473_q490_q568_msr_unit_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    build(isolated=args.isolated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
