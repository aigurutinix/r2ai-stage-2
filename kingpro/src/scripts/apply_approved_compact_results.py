"""Apply explicitly approved high-confidence compact keep verdicts to ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path)
    parser.add_argument("--approved-ids", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.task_root.resolve()
    approved = parse_ids(args.approved_ids)
    manifest = {int(item["id"]): item for item in load(root / "manifest.json")}
    ledger_path = args.ledger.resolve()
    ledger = load(ledger_path)
    verdicts = ledger.get("verdicts")
    if not isinstance(verdicts, dict):
        raise SystemExit("ledger missing verdicts object")
    staged: dict[str, dict] = {}
    for qid in approved:
        if str(qid) in verdicts:
            raise SystemExit(f"q{qid} already in ledger")
        item = manifest.get(qid)
        if item is None:
            raise SystemExit(f"q{qid} absent from task manifest")
        task = load(Path(item["task_path"]))
        result = load(Path(item["result_path"]))
        if result.get("verdict") != "keep" or result.get("confidence") != "high":
            raise SystemExit(f"q{qid} is not a high-confidence keep")
        if task.get("intent_source_flags") or task.get("arithmetic_invariant_flags"):
            raise SystemExit(f"q{qid} has unresolved source/invariant flags")
        if any(value.get("status") != "pass" for value in task.get("runtime", {}).values()):
            raise SystemExit(f"q{qid} runtime is not fully passing")
        if task.get("physical", {}).get("status") not in {"pass", "positional_pass"}:
            raise SystemExit(f"q{qid} physical status requires primary adjudication")
        staged[str(qid)] = {
            "status": "source_confirmed_no_change_via_compact_factory",
            "answer": result.get("answer_current"),
            "confidence": "high",
            "one_line_reason": result.get("one_line_reason"),
            "source_refs": result.get("source_refs", []),
            "compact_task": str(Path(item["task_path"]).relative_to(ROOT)),
            "compact_result": str(Path(item["result_path"]).relative_to(ROOT)),
            "source_bundle_hash": task.get("source_bundle_hash"),
            "normalized_ast_hash": task.get("normalized_ast_hash"),
            "mutation": "none",
        }
    if not args.dry_run:
        verdicts.update(staged)
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"approved": approved, "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
