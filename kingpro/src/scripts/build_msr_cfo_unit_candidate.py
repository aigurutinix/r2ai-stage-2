"""Build source-confirmed q1003 MSR thousand-VND normalization candidates.

Default output is cumulative v200 on top of v199.  ``--isolated`` creates an
independent q1003-only A/B from v196, useful if the q782 leaderboard result is
negative.  Both variants change one answer and one evidence-row scale only.
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
CUMULATIVE_SOURCE = ROOT / "sub_top123_candidate_v199_hng_share_capital_unit"
QID = 1003
EVIDENCE_REL = Path("data/q1003_source_cells.csv")
MSR_TABLE = "MSR_financial_statements_2015_separate|271"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build(isolated: bool = False) -> dict:
    source = BASELINE if isolated else CUMULATIVE_SOURCE
    output = ROOT / (
        "sub_top123_candidate_v200_q1003_msr_cfo_unit_isolated"
        if isolated
        else "sub_top123_candidate_v200_cumulative_unit_fixes"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shutil.copytree(source, output)

    source_submission = source / "submission.json"
    output_submission = output / "submission.json"
    old_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    row = next(item for item in rows if int(item["id"]) == QID)

    evidence_path = output / EVIDENCE_REL
    # utf-8-sig accepts both BOM and non-BOM compact evidence files and keeps
    # the logical first field name as ``ticker``.
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
        fieldnames = list(evidence_rows[0])
    msr_rows = [item for item in evidence_rows if item["source_table"] == MSR_TABLE]
    if len(msr_rows) != 1:
        raise AssertionError(f"expected exactly one q1003 MSR row, got {len(msr_rows)}")
    msr = msr_rows[0]
    expected = {
        "ticker": "MSR",
        "year": "2015",
        "metric_key": "lctt:20",
        "raw": "41.706.084",
        "typed_factor": "1.0",
        "scale": "1.0",
        "source_table": MSR_TABLE,
        "source_csv": "table_5_line271.csv",
        "row_idx": "19",
        "col_idx": "2",
    }
    if msr != expected:
        raise AssertionError(f"unexpected q1003 MSR evidence row: {msr}")
    msr["scale"] = "1000.0"
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence_rows)

    msr_thousand_vnd = 41_706_084.0
    hpg_vnd = -45_061_179_652.0
    aaa_vnd = 29_663_853_542.0
    dcm_vnd = 326_395_298_043.0
    corrected_answer = round(
        (msr_thousand_vnd * 1000.0 + hpg_vnd + aaa_vnd + dcm_vnd) / 1e9,
        2,
    )
    if corrected_answer != 352.70:
        raise AssertionError(f"unexpected q1003 answer: {corrected_answer}")
    old_answer = row["answer"]
    row["answer"] = corrected_answer
    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_audit_path = output / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    audits = [item for item in source_audit if int(item["id"]) == QID]
    if len(audits) != 1:
        raise AssertionError(f"expected one q1003 source-audit row, got {len(audits)}")
    audit = audits[0]
    audit["answer"] = corrected_answer
    audit["note"] = (
        "total parent MSR/HPG/AAA/DCM net operating cash flow, VND billion; "
        "MSR 2015 statement is explicitly in thousand VND"
    )
    audit_msr = [item for item in audit["sources"] if item["table_ref"] == MSR_TABLE]
    if len(audit_msr) != 1 or float(audit_msr[0]["scale"]) != 1.0:
        raise AssertionError("unexpected q1003 MSR source-audit scale")
    audit_msr[0]["scale"] = 1000.0
    source_audit_path.write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for candidate_row in rows:
        qid = int(candidate_row["id"])
        old = old_by_id[qid]
        if qid != QID and candidate_row != old:
            raise AssertionError(f"q{qid} changed unexpectedly")
    for key in set(old_by_id[QID]) | set(row):
        if key != "answer" and old_by_id[QID].get(key) != row.get(key):
            raise AssertionError(f"q1003 field changed unexpectedly: {key}")

    report = {
        "candidate": output.name,
        "source_candidate": source.name,
        "isolated": isolated,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": old_answer,
        "new_answer": corrected_answer,
        "msr_printed_thousand_vnd": msr_thousand_vnd,
        "msr_scale": 1000.0,
        "other_operands_vnd": {
            "HPG": hpg_vnd,
            "AAA": aaa_vnd,
            "DCM": dcm_vnd,
        },
        "source_unit_evidence": (
            "build/tables/MSR_financial_statements_2015_separate/table_5_line271.csv"
        ),
        "invariants": {
            "only_q1003_answer_changed_relative_to_source": True,
            "pandas_query_unchanged": True,
            "retrieval_fields_unchanged": True,
            "only_q1003_msr_evidence_scale_changed": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Source-confirmed unit correction; leaderboard result is unknown until evaluated.",
    }
    (output / "q1003_msr_cfo_unit_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
