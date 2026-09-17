#!/usr/bin/env python3
"""Build typed-slot coverage traces and a smaller evidence-aware ranking.

The input is the existing reranker-pair JSONL.  The output ranking is compatible
with ``run.py --ranking``; the trace records the wider read pool and every slot
that it could cover.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifinqa.answers import fold, question_years
from vifinqa.evidence import build_evidence_slots
from vifinqa.jsonl import load_jsonl
from vifinqa.slots import load_line_items, named_line_items, select_slot_tables


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-ranking", type=Path)
    parser.add_argument("--line-items", type=Path, default=root / "data" / "derived" / "line_items.json")
    parser.add_argument("--read-depth", type=int, default=50)
    parser.add_argument("--budget-multiplier", type=int, default=2)
    args = parser.parse_args()

    records = load_jsonl(args.pairs)
    labels = load_line_items(args.line_items)
    base = json.loads(args.base_ranking.read_text(encoding="utf-8")) if args.base_ranking else {}
    rankings: dict[str, list[str]] = {}
    traces: list[dict] = []

    for record in records:
        identifier = str(record["id"])
        candidate_by_id = {candidate["table_id"]: candidate for candidate in record["candidates"]}
        order = list(base.get(identifier, [])) or [candidate["table_id"] for candidate in record["candidates"]]
        order = [table_id for table_id in order if table_id in candidate_by_id]
        read_pool = order[: max(1, args.read_depth)]
        items = named_line_items(record["question"], labels)
        years = required_report_years_from_question(record["question"])
        slots = build_evidence_slots(
            record["question"],
            {"years": years, "scope": None},
            items,
        )
        budget = min(30, max(1, args.budget_multiplier * len(record.get("selected_docs", []))))
        selected, legacy_slots = select_slot_tables(
            {"selected_docs": record.get("selected_docs", []), "candidates": [candidate_by_id[table_id] for table_id in read_pool]},
            read_pool,
            items,
            budget,
        )
        coverage = {
            slot.slot_id: [
                table_id for table_id in read_pool
                if f" {slot.metric} " in f" {fold(candidate_by_id[table_id].get('text', ''))} "
            ]
            for slot in slots
        }
        used_tables = {table_id for table_ids in coverage.values() for table_id in table_ids}
        selected = [table_id for table_id in selected if table_id in used_tables]
        if not selected and read_pool:
            selected = read_pool[:1]
        rankings[identifier] = selected
        traces.append({
            "id": record["id"],
            "read_pool": read_pool,
            "submit_set": selected,
            "budget": budget,
            "metrics": items,
            "typed_slots": [slot.__dict__ if hasattr(slot, "__dict__") else {
                "slot_id": slot.slot_id, "ticker": slot.ticker,
                "requested_year": slot.requested_year, "scope": slot.scope,
                "metric": slot.metric, "role": slot.role, "stage": slot.stage,
            } for slot in slots],
            "slot_table_candidates": coverage,
            "covered_slots": sum(bool(table_ids) for table_ids in coverage.values()),
            "slot_count": len(slots),
            "legacy_slots": legacy_slots,
        })

    write_json(args.output, rankings)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text("".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces), encoding="utf-8")
    slot_count = sum(trace["slot_count"] for trace in traces)
    covered = sum(trace["covered_slots"] for trace in traces)
    print(json.dumps({
        "questions": len(records),
        "slot_count": slot_count,
        "covered_slots": covered,
        "slot_coverage": round(covered / slot_count, 4) if slot_count else 0.0,
        "output": str(args.output),
        "trace": str(args.trace),
    }, ensure_ascii=False, indent=2))


def required_report_years_from_question(question: str) -> list[int]:
    """Use the question years for an offline trace without re-running the gate."""
    return question_years(question)


if __name__ == "__main__":
    main()
