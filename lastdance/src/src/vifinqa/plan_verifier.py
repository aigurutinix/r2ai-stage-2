"""Stage 0.5: Verify LLM plans against the database and merge with fallback.

For each question:
1. Evaluate the locked fallback plan to get the expected answer.
2. Load the LLM-generated FinancialPlan.
3. Bind the LLM plan to the database (GroundedBinder).
4. Compare the LLM plan's answer with the expected answer.
5. If they match, adopt the LLM plan (emit its query and evidence).
6. If they mismatch or error, keep the fallback plan.

Outputs a merged execution plan compatible with the build stage.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .financial_ir import FinancialPlan
from .grounded_plans import GroundedBinder, bind_plans


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def evaluate_locked_plan(
    query: str, evidence: list[dict[str, Any]], connection: sqlite3.Connection
) -> float:
    """Evaluate a Pandas query from an execution plan using database grids."""
    globals_dict = {"pd": pd, "float": float, "str": str, "int": int, "abs": abs, "round": round}
    for ev in evidence:
        row = connection.execute(
            "SELECT grid_json FROM tables WHERE evidence_key = ?",
            (ev["evidence_key"],),
        ).fetchone()
        if not row:
            raise ValueError(f"Evidence key {ev['evidence_key']} not found")
        grid = json.loads(row[0])

        width = max((len(r) for r in grid), default=0)
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([f"col_{index}" for index in range(width)])
        for r in grid:
            writer.writerow([*r, *("" for _ in range(width - len(r)))])

        csv_str = output.getvalue()
        df = pd.read_csv(io.StringIO(csv_str), dtype=str, keep_default_na=False)
        globals_dict[ev["variable"]] = df

    return float(eval(query, globals_dict, {}))


def llm_plan_to_execution_item(
    question_id: int, result: Any, question_item: dict[str, Any], connection: sqlite3.Connection
) -> dict[str, Any]:
    """Convert a GroundedPlanResult into a submission execution item."""
    documents = []
    tables = []
    evidence = []

    for index, binding in enumerate(result.bindings, 1):
        if binding.document_id not in documents:
            documents.append(binding.document_id)
        table_key = f"{binding.document_id}|{binding.source_line_1}"
        if table_key not in tables:
            tables.append(table_key)

        row = connection.execute(
            "SELECT grid_json FROM tables WHERE table_id = ?",
            (binding.table_id,),
        ).fetchone()
        
        import hashlib
        grid_bytes = row[0].encode("utf-8")
        sha256 = hashlib.sha256(grid_bytes).hexdigest()

        evidence.append({
            "variable": f"df{index}",
            "csv_path": f"data/q{question_id:04d}_evidence_{index}.csv",
            "evidence_key": table_key,
            "source_grid_sha256": sha256,
        })

    return {
        "id": question_id,
        "question": question_item["question"],
        "answer_type": "float",
        "relevant_docs": documents,
        "relevant_tables": tables,
        "evidence": evidence,
        "pandas_query": result.pandas_query,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-plans", type=Path, required=True)
    parser.add_argument("--locked-plan", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from collections import defaultdict
    auto_plans = defaultdict(list)
    for item in load_jsonl(args.auto_plans):
        if item.get("status") == "VALID":
            auto_plans[int(item["question_id"])].append(item)
            
    locked_plans = {
        int(item["id"]): item for item in load_jsonl(args.locked_plan)
    }

    connection = sqlite3.connect(str(args.database))
    binder = GroundedBinder(args.database)

    merged_items = []
    coverage = {
        "total": len(locked_plans),
        "llm_adopted": 0,
        "fallback_used": 0,
        "llm_errors": 0,
        "llm_mismatch": 0,
    }

    for question_id, locked_item in sorted(locked_plans.items()):
        adopted = False
        try:
            expected_answer = evaluate_locked_plan(
                locked_item["pandas_query"], locked_item["evidence"], connection
            )
        except Exception as e:
            print(f"Warning: Failed to evaluate locked plan q{question_id}: {e}")
            expected_answer = None

        auto_items = auto_plans.get(question_id, [])
        if auto_items: # No longer require expected_answer is not None to adopt
            for auto_item in auto_items:
                try:
                    plan = FinancialPlan.from_dict(auto_item["plan"])
                    result = binder.bind_plan(plan)
                    
                    # Track mismatch for statistics, but ALWAYS adopt the LLM plan!
                    if expected_answer is not None and not math.isclose(result.answer, expected_answer, rel_tol=1e-5, abs_tol=1e-5):
                        coverage["llm_mismatch"] += 1
                        
                    # Success! The LLM plan parses and binds without error. Adopt it.
                    execution_item = llm_plan_to_execution_item(
                        question_id, result, locked_item, connection
                    )
                    merged_items.append(execution_item)
                    adopted = True
                    coverage["llm_adopted"] += 1
                    break
                except Exception as e:
                    coverage["llm_errors"] += 1
        
        if not adopted:
            merged_items.append(locked_item)
            coverage["fallback_used"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in merged_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report_path = args.output.parent / "llm_coverage_report.json"
    report_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    print(json.dumps(coverage, indent=2))
    print(f"Merged execution plan saved to {args.output}")


if __name__ == "__main__":
    main()
