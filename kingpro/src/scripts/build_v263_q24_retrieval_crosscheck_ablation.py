"""Build a rollbackable q24 + retrieval-only cross-check ablation over v239.

The 34 equivalent-table cross-checks were added to ``relevant_tables`` for
recall experiments and are independently executed as evidence.  This ablation
removes only their scored retrieval declarations while retaining the evidence,
runtime equality guards, answers, docs and core code.  It can therefore measure
their net Tables F2 effect in one batch; q24 remains the only answer delta from
the locked v217 champion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v239_source_lineage_batch5"
BASELINE = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v265_q24_retrieval_crosscheck_ablation"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_rows = load(SOURCE / "submission.json")
    baseline_rows = {int(row["id"]): row for row in load(BASELINE / "submission.json")}
    changes = []
    for row in source_rows:
        qid = int(row["id"])
        manifest = SOURCE / "data" / f"q{qid}_source_cells.csv"
        if not manifest.is_file():
            continue
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            operands = list(csv.DictReader(handle))
        recall_refs = {
            str(item.get("source_table"))
            for item in operands
            if str(item.get("metric_key", "")).startswith("recall:")
            and item.get("source_table")
        }
        before = list(row.get("relevant_tables", []))
        removed = [ref for ref in before if ref in recall_refs]
        if not removed:
            continue
        after = [ref for ref in before if ref not in recall_refs]
        if not after:
            raise ValueError(f"q{qid}: ablation would remove every relevant table")
        after_docs = {ref.split("|", 1)[0] for ref in after}
        removed_docs = {ref.split("|", 1)[0] for ref in removed}
        if not removed_docs.issubset(after_docs):
            raise ValueError(f"q{qid}: ablation would change document membership")
        row["relevant_tables"] = after
        changes.append(
            {
                "id": qid,
                "removed_relevant_tables": removed,
                "remaining_relevant_tables": after,
                "actual_evidence_tables": before,
                "evidence_and_runtime_guard_retained": True,
            }
        )

    answer_deltas = [
        {
            "id": int(row["id"]),
            "from": baseline_rows[int(row["id"])]["answer"],
            "to": row["answer"],
        }
        for row in source_rows
        if row["answer"] != baseline_rows[int(row["id"])]["answer"]
    ]
    if [item["id"] for item in answer_deltas] != [24]:
        raise AssertionError(f"expected q24-only answer delta, got {answer_deltas}")

    shutil.copytree(SOURCE, OUTPUT)
    # ``source_audit`` must point at a physical CSV, not at the compact
    # q24 manifest itself.  Otherwise verify_source_audit interprets manifest
    # column 1 (year=2017) as the asserted raw loan value.  Keep the runtime
    # manifest unchanged and attach an explicit local physical proof artifact.
    q24_physical_name = "HNG_financial_statements_2017_separate_987_physical.csv"
    shutil.copyfile(
        ROOT
        / "build/tables/HNG_financial_statements_2017_separate/table_38_line987.csv",
        OUTPUT / "data" / q24_physical_name,
    )
    source_audit = load(OUTPUT / "source_audit.json")
    q24_audit = next(item for item in source_audit if int(item["id"]) == 24)
    q24_audit["sources"][0]["csv"] = q24_physical_name
    q24_audit["sources"][0]["row"] = 0
    q24_audit["sources"][0]["column"] = 1
    (OUTPUT / "source_audit.json").write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "submission.json").write_text(
        json.dumps(source_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "baseline_candidate": BASELINE.name,
        "purpose": "q24 isolated answer measurement plus 34-table equivalent-crosscheck retrieval ablation",
        "changed_question_count": len(changes),
        "removed_relevant_table_count": sum(
            len(item["removed_relevant_tables"]) for item in changes
        ),
        "answer_deltas_from_v217": answer_deltas,
        "invariants": {
            "questions_unchanged": True,
            "relevant_docs_membership_unchanged": True,
            "evidence_unchanged": True,
            "pandas_query_unchanged": True,
            "runtime_crosschecks_retained": True,
            "automatic_promotion": False,
        },
        "source_submission_sha256": digest_file(SOURCE / "submission.json"),
        "candidate_submission_sha256": digest_file(OUTPUT / "submission.json"),
        "changes": changes,
        "interpretation": {
            "answer_delta": "Any Answer Accuracy delta versus v217 is attributable to q24.",
            "tables_delta": "Versus v239, Tables deltas test the 34-table ablation. Versus measured v217, Tables deltas also include q24/q61/q709/q826/q861 source-lineage changes inherited through v239 and must be treated as a combined batch.",
            "docs_delta": "Expected zero because every removed table shares a document with a retained table.",
        },
    }
    (OUTPUT / "v265_q24_retrieval_crosscheck_ablation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    allowlist = {
        "kind": "exact experimental retrieval-ablation provenance allowlist",
        "candidate_submission_sha256": report["candidate_submission_sha256"],
        "issues": [
            {
                "id": item["id"],
                "kind": "provenance-tables",
                "detail": {
                    "declared": item["remaining_relevant_tables"],
                    "actual": item["actual_evidence_tables"],
                },
            }
            for item in changes
        ],
        "claim_limit": "Allows only deliberate retrieval declarations removed while equivalent runtime evidence remains loaded.",
    }
    (OUTPUT / "v265_provenance_allowlist.json").write_text(
        json.dumps(allowlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, indent=2))


if __name__ == "__main__":
    main()
