"""Run the read-only typed-plan/dual-execution compiler audit.

Example::

    python scripts/audit_typed_plan_dual_execution.py \
      sub_top123_candidate_v269_source_lineage_control/submission.json \
      --output build/typed_plan_dual_execution_v269.json --strict

The input is never modified.  ``--strict`` returns a non-zero exit status when
an accepted compiler query lacks typed coverage or any of the three execution
paths disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.product.typed_plan_audit import TypedPlanAuditor, write_audit  # noqa: E402


def _rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("input must be a JSON array or JSONL stream of question objects")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="submission.json or questions.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "typed_plan_dual_execution_audit.json",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="sandbox timeout per accepted query")
    parser.add_argument("--strict", action="store_true", help="fail on uncovered accepted queries or disagreements")
    args = parser.parse_args()

    auditor = TypedPlanAuditor(ROOT, sandbox_timeout=args.timeout)
    input_path = args.input.resolve()
    report = auditor.audit_rows(_rows(input_path))
    report["inputs"]["questions"] = {
        "path": str(input_path),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    write_audit(report, args.output)
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output.resolve()}")
    if args.strict and (
        int(summary["accepted_uncovered"]) != 0
        or int(summary["disagreements_or_errors"]) != 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
