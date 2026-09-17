"""Build v199: correct q782's HNG report unit from VND to thousand VND.

The HNG consolidated 2016 financial statements present the balance sheet and
equity note in thousand VND.  The compact evidence row retained the printed
token but lost that report-level unit, so v196 compared VNM's VND amount with
HNG's thousand-VND amount directly.  This candidate restores the scale of
1,000 for that one source cell and changes no retrieval fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v196_effective_tax_sign"
OUTPUT = ROOT / "sub_top123_candidate_v199_hng_share_capital_unit"
QID = 782
EVIDENCE_REL = Path("data/q782_source_cells.csv")
HNG_TABLE = "HNG_financial_statements_2016_consolidated|259"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    source_submission = SOURCE / "submission.json"
    output_submission = OUTPUT / "submission.json"
    old_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    row = next(item for item in rows if int(item["id"]) == QID)

    evidence_path = OUTPUT / EVIDENCE_REL
    with evidence_path.open("r", encoding="utf-8", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
        fieldnames = list(evidence_rows[0])
    hng_rows = [item for item in evidence_rows if item["source_table"] == HNG_TABLE]
    if len(hng_rows) != 1:
        raise AssertionError(f"expected exactly one q782 HNG row, got {len(hng_rows)}")
    hng = hng_rows[0]
    expected = {
        "ticker": "HNG",
        "year": "2016",
        "metric_key": "cdkt:411",
        "raw": "7.671.438.950",
        "typed_factor": "1.0",
        "scale": "1.0",
        "source_table": HNG_TABLE,
        "source_csv": "table_3_line259.csv",
        "row_idx": "18",
        "col_idx": "3",
    }
    if hng != expected:
        raise AssertionError(f"unexpected q782 HNG evidence row: {hng}")
    # Use scale, not typed_factor: the parser intentionally applies
    # typed_factor only when pandas has already coerced a token to a number.
    # This dotted HNG token remains a string, while scale is always applied.
    hng["scale"] = "1000.0"
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence_rows)

    vnm_vnd = 14_514_534_290_000.0
    hng_thousand_vnd = 7_671_438_950.0
    corrected_answer = round(abs(vnm_vnd - hng_thousand_vnd * 1000.0) / 1e9, 2)
    if corrected_answer != 6843.10:
        raise AssertionError(f"unexpected q782 answer: {corrected_answer}")
    old_answer = row["answer"]
    row["answer"] = corrected_answer
    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_audit_path = OUTPUT / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    audit_rows = [item for item in source_audit if int(item["id"]) == QID]
    if len(audit_rows) != 1:
        raise AssertionError(f"expected one q782 source-audit row, got {len(audit_rows)}")
    audit = audit_rows[0]
    audit["answer"] = corrected_answer
    audit["note"] = (
        "absolute VNM/HNG ending share-capital difference, VND billion; "
        "HNG report values are stated in thousand VND"
    )
    audit_hng = [item for item in audit["sources"] if item["table_ref"] == HNG_TABLE]
    if len(audit_hng) != 1 or float(audit_hng[0]["scale"]) != 1.0:
        raise AssertionError("unexpected q782 HNG source-audit scale")
    audit_hng[0]["scale"] = 1000.0
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
            raise AssertionError(f"q782 field changed unexpectedly: {key}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids": [QID],
        "old_answer": old_answer,
        "new_answer": corrected_answer,
        "vnm_share_capital_vnd": vnm_vnd,
        "hng_share_capital_printed_thousand_vnd": hng_thousand_vnd,
        "hng_scale": 1000.0,
        "source_unit_evidence": [
            "build/tables/HNG_financial_statements_2016_consolidated/table_56_line1694.csv",
            "build/tables/HNG_financial_statements_2016_consolidated/table_12_line893.csv",
        ],
        "invariants": {
            "only_q782_answer_changed_in_submission": True,
            "pandas_query_unchanged": True,
            "retrieval_fields_unchanged": True,
            "only_q782_hng_evidence_scale_changed": True,
            "base_is_best_measured_v196": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Source-confirmed unit correction; leaderboard result is unknown until evaluated.",
    }
    (OUTPUT / "q782_hng_share_capital_unit_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
