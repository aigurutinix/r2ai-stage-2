"""Render small, self-contained audit tasks from factory records.

Agents read task JSON files and write result JSON files.  The chat response is
only a path acknowledgement, which keeps model/context token use low while all
evidence remains durable on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "build/v240_factory_batch_remaining/records.jsonl"
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"
DEFAULT_OUT = ROOT / "build/v241_compact_agent_tasks"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def compact_task(record: dict) -> dict:
    physical = record.get("physical_audit", {})
    runtime = record.get("runtime", {})
    unused = record.get("unused_evidence") or {}
    return {
        "schema_version": "1.0",
        "id": int(record["id"]),
        "question": record.get("question"),
        "stored_answer": record.get("answer"),
        "priority_score": record.get("priority_score", 0),
        "priority_reasons": record.get("priority_reasons", []),
        "triage": record.get("triage"),
        "relevant_docs": record.get("relevant_docs", []),
        "relevant_tables": record.get("relevant_tables", []),
        "source_metric_keys": record.get("source_metric_keys", []),
        "intent_source_flags": record.get("intent_source_flags", []),
        "arithmetic_invariant_flags": record.get("arithmetic_invariant_flags", []),
        "runtime": {
            name: {
                key: value
                for key, value in payload.items()
                if key in {"status", "result", "stored_answer", "delta", "error"}
            }
            for name, payload in runtime.items()
        },
        "physical": {
            "status": physical.get("status"),
            "counts": physical.get("counts"),
            "failures": physical.get("field_failures", [])[:3],
            "positional_reads": physical.get("reads", [])[:6],
            "unresolved": physical.get("unresolved", [])[:3],
            "artifact": physical.get("artifact"),
        },
        "unused_evidence_count": len(unused.get("unused_indices", [])),
        "unused_evidence": unused.get("unused_evidence", [])[:10],
        "source_bundle_hash": record.get("source_bundle_hash"),
        "normalized_ast_hash": record.get("normalized_ast_hash"),
        "instructions": {
            "mode": "read_only_source_adjudication",
            "required": [
                "verify scope/period/unit/metric/operation against physical source",
                "recompute answer independently",
                "separate answer change from cleanup/tool gap/oracle conflict",
                "write result JSON to result_path; do not edit submission/ledger",
            ],
            "result_schema": {
                "id": "int",
                "verdict": "answer_change|keep|cleanup|tool_gap|oracle_conflict",
                "answer_current": "number|string",
                "answer_proposed": "number|string|null",
                "confidence": "high|medium|low",
                "one_line_reason": "string <= 240 chars",
                "source_refs": "array <= 6",
                "mutation": "object|null",
                "cluster_ids": "array",
            },
            "chat_response": "Return only: DONE q<ID> <result_path>",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    output = args.out.resolve()
    tasks_dir = output / "tasks"
    results_dir = output / "results"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    closed = {int(value) for value in load_json(args.ledger.resolve()).get("verdicts", {})}
    records = [record for record in load_jsonl(args.records.resolve()) if int(record["id"]) not in closed]
    records.sort(key=lambda item: (-int(item.get("priority_score", 0)), int(item["id"])))
    records = records[: max(0, args.limit)]
    manifest = []
    for record in records:
        qid = int(record["id"])
        task = compact_task(record)
        result_path = results_dir / f"q{qid}.json"
        task["result_path"] = str(result_path)
        task_path = tasks_dir / f"q{qid}.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "id": qid,
                "priority_score": task["priority_score"],
                "task_path": str(task_path),
                "result_path": str(result_path),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(manifest), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
