"""Fail-closed structural and arithmetic gate for V275 source-review shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "build" / "v275_residual_shard_manifest.json"
DEFAULT_OUTPUT = ROOT / "build" / "v275_residual_source_adjudication.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate(
    manifest_path: Path,
    *,
    root: Path = ROOT,
    require_terminal: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load(manifest_path)
    candidate = root / str(manifest["candidate"])
    submission_path = candidate / "submission.json"
    expected_hash = str(manifest["candidate_submission_sha256"]).upper()
    actual_hash = sha256(submission_path)
    issues: list[dict[str, Any]] = []
    if actual_hash != expected_hash:
        issues.append(
            {
                "kind": "candidate-hash-mismatch",
                "expected": expected_hash,
                "actual": actual_hash,
            }
        )
    submission = {int(row["id"]): row for row in load(submission_path)}
    allowed = set(manifest["allowed_statuses"])
    required = set(manifest["required_record_fields"])
    all_expected: list[int] = []
    all_records: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    seen: set[int] = set()

    for shard_name, spec in manifest["shards"].items():
        expected_ids = [int(value) for value in spec["ids"]]
        all_expected.extend(expected_ids)
        path = root / str(spec["output"])
        if not path.is_file():
            issues.append({"kind": "shard-missing", "shard": shard_name, "path": str(path)})
            continue
        shard_hashes[shard_name] = sha256(path)
        payload = load(path)
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            issues.append({"kind": "records-missing", "shard": shard_name})
            continue
        actual_ids = [int(row.get("id", -1)) for row in records if isinstance(row, dict)]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
            issues.append(
                {
                    "kind": "shard-id-coverage-mismatch",
                    "shard": shard_name,
                    "expected": expected_ids,
                    "actual": actual_ids,
                }
            )
        for record in records:
            if not isinstance(record, dict):
                issues.append({"kind": "record-not-object", "shard": shard_name})
                continue
            record = dict(record)
            if "physical_refs" not in record:
                for alias in ("source_refs", "refs"):
                    if isinstance(record.get(alias), list):
                        record["physical_refs"] = list(record[alias])
                        break
            if "hard_negatives" not in record:
                negative = record.get("hard_negative")
                if isinstance(negative, list):
                    record["hard_negatives"] = negative
                elif isinstance(negative, dict) and negative:
                    record["hard_negatives"] = [negative]
                elif isinstance(negative, str) and negative.strip():
                    record["hard_negatives"] = [negative]
                elif isinstance(record.get("proof"), dict) and isinstance(
                    record["proof"].get("hard_negatives"), list
                ):
                    record["hard_negatives"] = list(record["proof"]["hard_negatives"])
            qid = int(record.get("id", -1))
            if qid in seen:
                issues.append({"kind": "duplicate-id", "id": qid, "shard": shard_name})
            seen.add(qid)
            missing = sorted(required - set(record))
            if missing:
                issues.append({"kind": "record-fields-missing", "id": qid, "fields": missing})
            status = str(record.get("status", ""))
            if status not in allowed:
                issues.append({"kind": "invalid-status", "id": qid, "status": status})
            if qid not in submission:
                issues.append({"kind": "submission-id-missing", "id": qid})
                continue
            current = number(record.get("current_answer"))
            candidate_answer = number(submission[qid].get("answer"))
            recomputed = number(record.get("recomputed_answer"))
            if current is None or candidate_answer is None or not math.isclose(
                current, candidate_answer, rel_tol=0.0, abs_tol=0.0050001
            ):
                issues.append(
                    {
                        "kind": "current-answer-mismatch",
                        "id": qid,
                        "record": current,
                        "candidate": candidate_answer,
                    }
                )
            if status == "ambiguous":
                if require_terminal:
                    issues.append({"kind": "ambiguous-terminal-gate", "id": qid})
            elif recomputed is None:
                issues.append({"kind": "recomputed-answer-missing", "id": qid})
            elif status in {"source_confirmed_no_change", "cleanup_only"} and not math.isclose(
                recomputed, current if current is not None else float("nan"), rel_tol=0.0, abs_tol=0.0050001
            ):
                issues.append(
                    {
                        "kind": "no-change-status-but-answer-differs",
                        "id": qid,
                        "current": current,
                        "recomputed": recomputed,
                    }
                )
            elif status == "source_confirmed_change" and math.isclose(
                recomputed, current if current is not None else float("nan"), rel_tol=0.0, abs_tol=0.0050001
            ):
                issues.append({"kind": "change-status-without-answer-delta", "id": qid})

            if not str(record.get("proof", "")).strip():
                issues.append({"kind": "proof-empty", "id": qid})
            refs = record.get("physical_refs")
            if not isinstance(refs, list) or not refs:
                issues.append({"kind": "physical-refs-empty", "id": qid})
            negatives = record.get("hard_negatives")
            if not isinstance(negatives, list) or not negatives:
                issues.append({"kind": "hard-negatives-empty", "id": qid})
            if status in {"source_confirmed_change", "cleanup_only"} and not record.get(
                "proposed_mutation"
            ):
                issues.append({"kind": "mutation-missing", "id": qid})
            all_records.append(record)

    if len(all_expected) != len(set(all_expected)):
        issues.append({"kind": "manifest-shard-overlap"})
    missing_ids = sorted(set(all_expected) - seen)
    extra_ids = sorted(seen - set(all_expected))
    if missing_ids:
        issues.append({"kind": "expected-ids-missing", "ids": missing_ids})
    if extra_ids:
        issues.append({"kind": "unexpected-ids", "ids": extra_ids})

    order = {qid: index for index, qid in enumerate(all_expected)}
    all_records.sort(key=lambda row: order.get(int(row.get("id", -1)), 10**9))
    counts = Counter(str(row.get("status", "")) for row in all_records)
    return {
        "schema_version": 1,
        "gate": "v275_residual_source_shards",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "candidate": candidate.name,
        "candidate_submission_sha256": actual_hash,
        "shard_hashes": shard_hashes,
        "expected_ids": all_expected,
        "record_count": len(all_records),
        "unique_id_count": len(seen),
        "status_counts": dict(sorted(counts.items())),
        "answer_change_ids": [
            int(row["id"])
            for row in all_records
            if row.get("status") == "source_confirmed_change"
        ],
        "cleanup_ids": [
            int(row["id"]) for row in all_records if row.get("status") == "cleanup_only"
        ],
        "ambiguous_ids": [
            int(row["id"]) for row in all_records if row.get("status") == "ambiguous"
        ],
        "issues": issues,
        "issue_count": len(issues),
        "structurally_valid": not issues,
        "ready_for_terminal_ledger_ingest": not issues and not counts.get("ambiguous", 0),
        "records": all_records,
        "claim_limit": "Structural/arithmetic gate only; parent source review remains required before ledger mutation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-terminal", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.manifest,
        root=ROOT,
        require_terminal=args.require_terminal,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if report["structurally_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
