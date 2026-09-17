"""Build a two-question public-regression ablation from audited v225.

Submissions 3722 (v224) and 3723 (v225) tie all ten public metrics although
their only payload delta is q98.  Therefore q98 is public-neutral at published
precision.  Relative to champion v217, the only remaining answer deltas are
q224, q530, q714, q764, q931 and q1007.  This diagnostic restores the complete
v217 rows for the two HUT scope questions q224/q714 while leaving every other
v225 row byte-semantically unchanged.

This is a bounded A/B probe, not private-ground-truth evidence and not an
automatic replacement for the v217 champion or the semantically audited v225.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9"
TRUSTED_PUBLIC = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v226_scope_pair_public_ablation"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
TARGET_IDS = (224, 714)


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def by_id(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["id"]): row for row in rows}


def main() -> None:
    if not SOURCE.is_dir() or not TRUSTED_PUBLIC.is_dir():
        raise FileNotFoundError("v225 source or v217 public reference is missing")
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    source_rows = load(SOURCE / "submission.json")
    trusted_rows = load(TRUSTED_PUBLIC / "submission.json")
    if not isinstance(source_rows, list) or not isinstance(trusted_rows, list):
        raise AssertionError("submission payload must be a list")
    source_map = by_id(source_rows)
    trusted_map = by_id(trusted_rows)
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    rows_map = by_id(rows)

    expected = {
        224: {"source": 1200.5, "trusted": 181.54},
        714: {"source": 168.74, "trusted": 238.89},
    }
    for qid in TARGET_IDS:
        if source_map[qid].get("answer") != expected[qid]["source"]:
            raise AssertionError(f"q{qid}: unexpected v225 answer")
        if trusted_map[qid].get("answer") != expected[qid]["trusted"]:
            raise AssertionError(f"q{qid}: unexpected v217 answer")
        rows_map[qid].clear()
        rows_map[qid].update(
            json.loads(json.dumps(trusted_map[qid], ensure_ascii=False))
        )

    changed = [
        int(row["id"])
        for row in rows
        if row != source_map[int(row["id"])]
    ]
    if changed != list(TARGET_IDS):
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)

    # Restore every evidence payload referenced by the two v217 rows.
    for qid in TARGET_IDS:
        for evidence in trusted_map[qid].get("evidence", []):
            relative = Path(str(evidence["csv_path"]))
            trusted_path = TRUSTED_PUBLIC / relative
            if not trusted_path.is_file():
                raise FileNotFoundError(trusted_path)
            destination = BUILDING / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trusted_path, destination)

    source_audit = load(SOURCE / "source_audit.json")
    trusted_audit = load(TRUSTED_PUBLIC / "source_audit.json")
    if not isinstance(source_audit, list) or not isinstance(trusted_audit, list):
        raise AssertionError("source_audit.json must be a list")
    trusted_audit_map = by_id(trusted_audit)
    audit_rows = [
        row for row in source_audit if int(row.get("id", -1)) not in TARGET_IDS
    ]
    for qid in TARGET_IDS:
        if qid in trusted_audit_map:
            audit_rows.append(
                json.loads(json.dumps(trusted_audit_map[qid], ensure_ascii=False))
            )
    audit_rows.sort(key=lambda row: int(row.get("id", -1)))
    write(BUILDING / "source_audit.json", audit_rows)

    report = {
        "candidate": OUTPUT.name,
        "role": "public_regression_ablation_not_private_candidate",
        "source_candidate": SOURCE.name,
        "trusted_public_reference": TRUSTED_PUBLIC.name,
        "changed_question_ids_relative_to_source": list(TARGET_IDS),
        "answer_changes": {
            str(qid): {
                "from": source_map[qid]["answer"],
                "to": trusted_map[qid]["answer"],
            }
            for qid in TARGET_IDS
        },
        "held_constant_question_ids": [530, 764, 931, 1007],
        "hypothesis": (
            "Exactly one of q224/q714 appears in public retrieval from the "
            "0.0010 docs-precision delta; restoring both tests their aggregate "
            "answer/retrieval contribution without single-question probing."
        ),
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "trusted_submission_sha256": sha256(TRUSTED_PUBLIC / "submission.json"),
        "claim_limit": (
            "Public A/B diagnostic only. A score change must not be treated as "
            "private gold or override physical-report evidence by itself."
        ),
    }
    write(BUILDING / "v226_scope_pair_public_ablation.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = sha256(OUTPUT / "submission.json")
    write(OUTPUT / "v226_scope_pair_public_ablation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
