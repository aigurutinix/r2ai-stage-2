"""Validate compact agent result files and render a ledger-ready inbox."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ALLOWED_VERDICTS = {"answer_change", "keep", "cleanup", "tool_gap", "oracle_conflict"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def validate_result(value: dict, expected_id: int) -> list[str]:
    errors: list[str] = []
    if int(value.get("id", -1)) != expected_id:
        errors.append("id_mismatch")
    if value.get("verdict") not in ALLOWED_VERDICTS:
        errors.append("invalid_verdict")
    if value.get("confidence") not in ALLOWED_CONFIDENCE:
        errors.append("invalid_confidence")
    reason = value.get("one_line_reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        errors.append("invalid_reason")
    refs = value.get("source_refs")
    if not isinstance(refs, list) or len(refs) > 6 or not all(isinstance(item, str) for item in refs):
        errors.append("invalid_source_refs")
    if not isinstance(value.get("cluster_ids"), list):
        errors.append("invalid_cluster_ids")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path)
    args = parser.parse_args()
    root = args.task_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    valid: list[dict] = []
    invalid: list[dict] = []
    pending: list[int] = []
    for item in manifest:
        qid = int(item["id"])
        path = Path(item["result_path"])
        if not path.is_file():
            pending.append(qid)
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            errors = validate_result(value, qid)
        except Exception as exc:
            invalid.append({"id": qid, "errors": [f"{type(exc).__name__}: {exc}"]})
            continue
        if errors:
            invalid.append({"id": qid, "errors": errors})
        else:
            valid.append(value)
    (root / "validated_results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in valid),
        encoding="utf-8",
    )
    summary = {
        "valid": len(valid),
        "invalid": invalid,
        "pending": pending,
        "verdict_counts": dict(Counter(item["verdict"] for item in valid)),
    }
    (root / "results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
