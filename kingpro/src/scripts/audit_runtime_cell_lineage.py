"""Reprove every declared V297 cell coordinate against physical CSV tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from kingpro.product.cell_lineage import RuntimeCellLineageIndex


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def declared_ids(candidate: Path) -> set[int]:
    ids: set[int] = set()
    audit_path = candidate / "source_audit.json"
    if audit_path.is_file():
        for row in json.loads(audit_path.read_text(encoding="utf-8-sig")):
            if row.get("id") is not None and row.get("sources"):
                ids.add(int(row["id"]))
    for path in (candidate / "data").glob("q*_source_cells.csv"):
        match = re.fullmatch(r"q(\d+)_source_cells", path.stem)
        if match:
            ids.add(int(match.group(1)))
    sidecar = (
        candidate.parent
        / "build"
        / "runtime_lineage"
        / f"{candidate.name}_counterfactual.json"
    )
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        ids.update(
            int(row["question_id"])
            for row in payload.get("records", [])
            if row.get("status") == "verified" and row.get("cells")
        )
    return ids


def audit(root: Path, candidate: Path) -> dict:
    started = time.perf_counter()
    submission = json.loads((candidate / "submission.json").read_text(encoding="utf-8-sig"))
    by_id = {int(row["id"]): row for row in submission}
    declared = declared_ids(candidate)
    index = RuntimeCellLineageIndex(root, candidate)
    verified: dict[int, list[dict]] = {}
    missing: list[int] = []
    citation_unbound: list[dict] = []
    for question_id in sorted(declared):
        cells = index.for_question(question_id)
        if not cells:
            missing.append(question_id)
            continue
        verified[question_id] = cells
        relevant = set(map(str, by_id.get(question_id, {}).get("relevant_tables", [])))
        for cell in cells:
            if str(cell["table_ref"]) not in relevant:
                citation_unbound.append(
                    {
                        "id": question_id,
                        "table_ref": cell["table_ref"],
                        "relevant_tables": sorted(relevant),
                    }
                )
    cell_count = sum(map(len, verified.values()))
    checks = {
        "submission_has_unique_ids": len(by_id) == len(submission),
        "every_declared_question_reproved": not missing and set(verified) == declared,
        "every_cell_bound_to_relevant_table": not citation_unbound,
        "every_cell_coordinate_and_raw_verified": all(
            cell.get("verified") is True
            and cell.get("verification") in {
                "coordinate_and_raw_match",
                "counterfactual_result_dependency_and_coordinate_raw_match",
            }
            and cell.get("raw_manifest") == cell.get("raw_physical")
            for cells in verified.values()
            for cell in cells
        ),
    }
    sidecar = (
        root / "build" / "runtime_lineage" / f"{candidate.name}_counterfactual.json"
    )
    return {
        "schema_version": "runtime-cell-lineage-coverage/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "artifact": candidate.name,
        "input_hashes": {
            "submission_sha256": _sha256(candidate / "submission.json"),
            "catalog_sha256": _sha256(root / "build" / "catalog.jsonl"),
            "counterfactual_sidecar_sha256": _sha256(sidecar),
            "counterfactual_analyzer_sha256": _sha256(
                root / "src" / "kingpro" / "product" / "counterfactual_lineage.py"
            ),
            "runtime_verifier_sha256": _sha256(
                root / "src" / "kingpro" / "product" / "cell_lineage.py"
            ),
        },
        "questions": len(submission),
        "declared_questions": len(declared),
        "verified_questions": len(verified),
        "verified_cells": cell_count,
        "coverage": round(len(verified) / max(1, len(submission)), 6),
        "checks": checks,
        "missing_question_ids": missing,
        "citation_unbound_cells": citation_unbound,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "claim_limit": (
            "Coverage counts only artifact-declared cells whose physical table coordinate "
            "and raw token were reread and matched; it is not private-test accuracy."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate", type=Path, default=ROOT / "sub_v297_scope2")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "demo_compliance" / "runtime_cell_lineage_coverage.json",
    )
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
