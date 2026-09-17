"""Build the source-lineage control that restores all v239 retrieval labels."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v265_q24_retrieval_crosscheck_ablation"
CONTROL = ROOT / "sub_top123_candidate_v239_source_lineage_batch5"
BASELINE = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v269_source_lineage_control"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    shutil.copytree(SOURCE, OUTPUT)
    control_submission = load(CONTROL / "submission.json")
    baseline = {int(row["id"]): row for row in load(BASELINE / "submission.json")}
    (OUTPUT / "submission.json").write_text(
        json.dumps(control_submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    answer_deltas = [
        {
            "id": int(row["id"]),
            "from": baseline[int(row["id"])]["answer"],
            "to": row["answer"],
        }
        for row in control_submission
        if row["answer"] != baseline[int(row["id"])]["answer"]
    ]
    if [item["id"] for item in answer_deltas] != [24]:
        raise AssertionError(f"expected q24-only answer delta: {answer_deltas}")
    report = {
        "candidate": OUTPUT.name,
        "submission_bytes_from": CONTROL.name,
        "local_provenance_from": SOURCE.name,
        "baseline": BASELINE.name,
        "purpose": "control restoring all 34 equivalent cross-check retrieval labels while retaining fixed q24 physical source audit",
        "answer_deltas_from_v217": answer_deltas,
        "changed_submission_ids_from_v217": [24, 709, 826, 861],
        "changed_evidence_ids_from_v217": [24, 61, 709, 826, 861],
        "source_submission_sha256": sha(CONTROL / "submission.json"),
        "candidate_submission_sha256": sha(OUTPUT / "submission.json"),
        "invariants": {
            "submission_identical_to_v239": sha(CONTROL / "submission.json")
            == sha(OUTPUT / "submission.json"),
            "q24_physical_source_audit_fixed": True,
            "equivalent_crosscheck_tables_restored": 34,
            "automatic_promotion": False,
        },
        "interpretation": {
            "versus_v265": "isolates the 34 equivalent cross-check relevant-table declarations",
            "versus_v217": "measures combined source-lineage rows q24/q61/q709/q826/q861; q24 Answer is already proven public-neutral",
        },
    }
    (OUTPUT / "v269_source_lineage_control_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
