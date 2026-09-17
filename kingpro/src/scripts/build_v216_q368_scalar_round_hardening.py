"""Build a value-preserving q368 runtime portability repair from v215.

The public scorer reported exactly one ``AttributeError``.  q368 is the only
program left that calls ``.round`` on a Pandas reduction scalar.  NumPy scalars
expose that method, while Python ``float`` does not.  Normalize the reduction
to ``float`` and let the already-present builtin ``round`` line perform the
same final rounding.  No answer, evidence, table, or calculation changes.
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
SOURCE = ROOT / "sub_top123_candidate_v215_expense_scope_batch2"
OUTPUT = ROOT / "sub_top123_candidate_v216_q368_scalar_round_hardened"
QUESTION_ID = 368
OLD = "result = low_quick_ratio['net_margin_pct'].mean().round(2)"
NEW = "result = float(low_quick_ratio['net_margin_pct'].mean())"


def _digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}
    row = by_id[QUESTION_ID]
    code = str(row.get("pandas_query") or "")
    if code.count(OLD) != 1:
        raise AssertionError("q368 scalar-round expression changed unexpectedly")
    if "result = round(float(result), 2)" not in code:
        raise AssertionError("q368 builtin final rounding guard is missing")
    row["pandas_query"] = code.replace(OLD, NEW, 1)

    changed = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])]]
    if changed != [QUESTION_ID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")
    changed_fields = {
        key for key in row if row.get(key) != source_by_id[QUESTION_ID].get(key)
    }
    if changed_fields != {"pandas_query"}:
        raise AssertionError(f"unexpected q368 changed fields: {changed_fields}")

    shutil.copytree(SOURCE, OUTPUT)
    _write_json(OUTPUT / "submission.json", rows)
    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": changed,
        "changed_fields": ["pandas_query"],
        "answers_changed": 0,
        "evidence_changed": 0,
        "tables_changed": 0,
        "old_expression": OLD,
        "new_expression": NEW,
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "claim_limit": (
            "Value-preserving runtime hardening. The hidden failing ID remains "
            "unknown until this candidate is scored."
        ),
    }
    _write_json(OUTPUT / "v216_q368_scalar_round_hardening_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
