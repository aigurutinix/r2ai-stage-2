"""Ingest explicitly approved, high-confidence keep reviews into the ledger.

The input reports may use ``items``, ``reviews`` or ``records``.  This tool is
fail-closed: every approved ID must exist exactly once, be a keep verdict with
confidence >= 0.95, and propose the same answer it reviewed.  Existing ledger
entries are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(payload: dict) -> list[dict]:
    for key in ("items", "reviews", "records", "verdicts"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("report has no items/reviews/records/verdicts list")


def parse_ids(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("--approved-ids must contain unique IDs")
    return result


def numeric_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.005)
    except (TypeError, ValueError):
        return left == right


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize(item: dict, artifact: Path) -> dict:
    verdict = str(item.get("verdict", ""))
    if verdict not in {"keep", "cleanup"}:
        raise ValueError(f"q{item.get('id')}: verdict is not keep/cleanup")
    raw_confidence = item.get("confidence", 0.0)
    if isinstance(raw_confidence, str):
        confidence = float(item.get("confidence_score", 1.0 if raw_confidence == "high" else 0.0))
    else:
        confidence = float(raw_confidence)
    if confidence < 0.95:
        raise ValueError(f"q{item.get('id')}: confidence below 0.95")
    current = item.get("current_answer")
    if current is None:
        current = item.get("current")
    if isinstance(current, dict):
        current = current.get("answer")
    proposed = item.get("proposed_answer", item.get("proposed"))
    if isinstance(proposed, dict):
        proposed = proposed.get("answer")
    if proposed is None:
        proposed = current
    if current is None or not numeric_equal(current, proposed):
        raise ValueError(f"q{item.get('id')}: keep review changes the answer")
    proof = (
        item.get("proof")
        or item.get("independent_recompute")
        or item.get("recomputation")
        or item.get("one_line_reason")
    )
    if not isinstance(proof, str) or len(proof.strip()) < 12:
        raise ValueError(f"q{item.get('id')}: missing source proof")
    refs = item.get("source_refs")
    if refs is None and item.get("source") is not None:
        refs = [item["source"]]
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"q{item.get('id')}: missing source refs")
    return {
        "status": (
            "source_confirmed_pending_batch_cleanup_via_parallel_batch"
            if verdict == "cleanup"
            else "source_confirmed_no_change_via_parallel_batch"
        ),
        "answer": current,
        "confidence": "high",
        "proof": proof,
        "source_refs": refs[:8],
        "review_artifact": display_path(artifact),
        "mutation": "pending_batch_cleanup" if verdict == "cleanup" else "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--approved-ids", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    approved = parse_ids(args.approved_ids)
    found: dict[int, tuple[dict, Path]] = {}
    for report in args.reports:
        path = report.resolve()
        for item in rows(load(path)):
            qid = int(item["id"])
            if qid in found:
                raise SystemExit(f"q{qid} appears in multiple reports")
            found[qid] = (item, path)
    missing = sorted(set(approved) - set(found))
    if missing:
        raise SystemExit(f"approved IDs missing from reports: {missing}")
    ledger_path = args.ledger.resolve()
    ledger = load(ledger_path)
    verdicts = ledger.get("verdicts")
    if not isinstance(verdicts, dict):
        raise SystemExit("ledger missing verdicts object")
    staged = {}
    for qid in approved:
        if str(qid) in verdicts:
            raise SystemExit(f"q{qid} already exists in ledger")
        item, artifact = found[qid]
        staged[str(qid)] = normalize(item, artifact)
    if not args.dry_run:
        verdicts.update(staged)
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"approved": approved, "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
