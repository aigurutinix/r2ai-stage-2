"""Build a resumable causal-lineage sidecar for legacy V297 programs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from kingpro.product.counterfactual_lineage import CounterfactualLineageAnalyzer


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _declared_ids(candidate: Path) -> set[int]:
    ids: set[int] = set()
    audit = candidate / "source_audit.json"
    if audit.is_file():
        for row in json.loads(audit.read_text(encoding="utf-8-sig")):
            if row.get("id") is not None and row.get("sources"):
                ids.add(int(row["id"]))
    for path in (candidate / "data").glob("q*_source_cells.csv"):
        match = re.fullmatch(r"q(\d+)_source_cells", path.stem)
        if match:
            ids.add(int(match.group(1)))
    return ids


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate", type=Path, default=ROOT / "sub_v297_scope2")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "runtime_lineage" / "sub_v297_scope2_counterfactual.json",
    )
    parser.add_argument("--ids", help="optional comma-separated IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    root = args.root.resolve()
    candidate = args.candidate.resolve()
    rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8-sig"))
    by_id = {int(row["id"]): row for row in rows}
    declared = _declared_ids(candidate)
    targets = sorted(set(by_id) - declared)
    if args.ids:
        requested = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
        targets = [question_id for question_id in targets if question_id in requested]
    if args.limit is not None:
        targets = targets[: max(0, args.limit)]

    records: dict[int, dict] = {}
    if args.resume and args.output.is_file():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if Path(str(prior.get("candidate", ""))).name != candidate.name:
            raise ValueError("resume candidate mismatch")
        records = {int(row["question_id"]): row for row in prior.get("records", [])}

    analyzer = CounterfactualLineageAnalyzer(root, candidate)
    started = time.perf_counter()

    def report() -> dict:
        completed = [records[qid] for qid in targets if qid in records]
        reasons = Counter(
            str(row.get("reason") or row.get("status") or "unknown")
            for row in completed
            if row.get("status") != "verified"
        )
        return {
            "schema_version": "counterfactual-cell-lineage/v1",
            "candidate": str(candidate),
            "input_hashes": {
                "submission_sha256": _sha256(candidate / "submission.json"),
                "catalog_sha256": _sha256(root / "build" / "catalog.jsonl"),
                "analyzer_sha256": _sha256(
                    root / "src" / "kingpro" / "product" / "counterfactual_lineage.py"
                ),
            },
            "submission_questions": len(rows),
            "preexisting_lineage_questions": len(declared),
            "target_questions": len(targets),
            "completed_questions": len(completed),
            "verified_questions": sum(row.get("status") == "verified" for row in completed),
            "verified_cells": sum(len(row.get("cells", [])) for row in completed),
            "unresolved_reasons": dict(sorted(reasons.items())),
            "complete": len(completed) == len(targets),
            "records": completed,
            "elapsed_ms_current_process": round((time.perf_counter() - started) * 1000),
            "claim_limit": (
                "A cell is included only when a valid numeric perturbation changes the "
                "replayed result and its complete evidence frame equals exactly one relevant "
                "physical table. This sidecar does not modify V297 or estimate private accuracy."
            ),
        }

    pending = [question_id for question_id in targets if question_id not in records]
    for index, question_id in enumerate(pending, start=1):
        result = analyzer.analyze(by_id[question_id])
        result["question_id"] = question_id
        records[question_id] = result
        print(
            f"[{index}/{len(pending)}] q{question_id}: {result.get('status')} "
            f"cells={len(result.get('cells', []))} exec={result.get('executions', 0)} "
            f"ms={result.get('elapsed_ms', 0)}",
            flush=True,
        )
        if index % max(1, args.checkpoint_every) == 0:
            _atomic_json(args.output, report())
    payload = report()
    _atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
