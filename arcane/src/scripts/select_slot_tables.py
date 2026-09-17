#!/usr/bin/env python3
"""Reserve one high-ranked table for each source-derived evidence slot."""

import argparse
import json
import re
from pathlib import Path

from vifinqa.fusion import first_stage
from vifinqa.jsonl import load_jsonl
from vifinqa.slots import (
    append_confidence_item_tables,
    load_line_items,
    named_line_items,
    select_confidence_slot_tables,
    select_slot_tables,
)


def arm_order(record: dict, arm: str) -> list[str]:
    candidates = [
        candidate for candidate in record["candidates"]
        if first_stage(candidate, arm) > 0
    ]
    candidates.sort(key=lambda candidate: -first_stage(candidate, arm))
    return [candidate["table_id"] for candidate in candidates]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def merge_records(paths: list[Path]) -> list[dict]:
    """Union candidate tables from repeated pair exports by question ID."""
    merged: dict[int, dict] = {}
    for path in paths:
        for record in load_jsonl(path):
            current = merged.setdefault(record["id"], {
                key: value for key, value in record.items() if key != "candidates"
            } | {"candidates": []})
            by_id = {candidate["table_id"]: candidate for candidate in current["candidates"]}
            for candidate in record["candidates"]:
                previous = by_id.get(candidate["table_id"])
                if previous is None:
                    by_id[candidate["table_id"]] = dict(candidate)
                    continue
                for key, value in candidate.items():
                    if key not in previous or previous[key] is None:
                        previous[key] = value
                if candidate.get("text") and candidate["text"] != previous.get("text"):
                    previous["text"] = f"{previous.get('text', '')}\n{candidate['text']}"
            current["candidates"] = list(by_id.values())
    records = [merged[identifier] for identifier in sorted(merged)]
    if len(records) != len({record["id"] for record in records}):
        raise ValueError("duplicate question IDs after pair merge")
    return records


def load_scores(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None:
        return {}
    return {record["id"]: record["scores"] for record in load_jsonl(path)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--base-ranking", type=Path)
    parser.add_argument("--starting-ranking", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--line-items", type=Path,
        default=root / "data" / "derived" / "line_items.json",
    )
    parser.add_argument("--arm", choices=("sparse", "dense", "hybrid"), default="sparse")
    parser.add_argument("--budget-multiplier", type=int, default=2)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--confidence-slots", action="store_true")
    parser.add_argument("--logit-gap", type=float, default=4.0)
    parser.add_argument("--max-per-report", type=int, default=2)
    parser.add_argument(
        "--adaptive-third-slot", action="store_true",
        help="allow a third scored table only for multi-report, multi-item, or multi-year questions",
    )
    parser.add_argument(
        "--adaptive-append", action="store_true",
        help="append item-carrying scored tables to a supplied starting ranking",
    )
    args = parser.parse_args()

    labels = load_line_items(args.line_items)
    records = merge_records(args.pairs)
    base = json.loads(args.base_ranking.read_text(encoding="utf-8")) if args.base_ranking else {}
    starting = (json.loads(args.starting_ranking.read_text(encoding="utf-8"))
                if args.starting_ranking else {})
    score_by_id = load_scores(args.scores)
    if (args.confidence_slots or args.adaptive_append) and args.scores is None:
        raise SystemExit("confidence modes require --scores")
    if args.adaptive_append and args.starting_ranking is None:
        raise SystemExit("--adaptive-append requires --starting-ranking")

    rankings = {}
    traces = []
    for record in records:
        identifier = record["id"]
        order = base.get(str(identifier)) or arm_order(record, args.arm)
        start_order = starting.get(str(identifier), [])
        items = named_line_items(record["question"], labels)
        budget = min(30, max(1, args.budget_multiplier * len(record.get("selected_docs", []))))
        if args.adaptive_append:
            years = set(re.findall(r"(?<!\d)20\d{2}(?!\d)", record["question"]))
            high_arity = (
                len(record.get("selected_docs", [])) >= 2
                or len(items) >= 2
                or len(years) >= 2
            )
            selected = (
                append_confidence_item_tables(
                    record, start_order, order, items, score_by_id.get(identifier, {}),
                    args.logit_gap, 3,
                ) if high_arity else list(start_order)
            )
            slots = []
        elif args.confidence_slots:
            high_arity = (
                len(record.get("selected_docs", [])) >= 2
                or len(items) >= 2
                or len(set(re.findall(r"(?<!\d)20\d{2}(?!\d)", record["question"]))) >= 2
            )
            adaptive = args.adaptive_third_slot and high_arity
            selected, slots = select_confidence_slot_tables(
                record, order, items, score_by_id.get(identifier, {}),
                args.logit_gap, 3 if adaptive else args.max_per_report,
                require_item_for_extras=adaptive,
            )
        else:
            selected, slots = select_slot_tables(record, order, items, budget)
        rankings[str(identifier)] = selected
        traces.append({
            "id": identifier,
            "line_items": items,
            "budget": budget,
            "selected": selected,
            "slots": slots,
            "covered_slots": sum(slot["table_id"] is not None for slot in slots),
            "input_count": len(order),
            "dropped": len(order) - len(selected),
        })

    write_json(args.output, rankings)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.trace.with_suffix(args.trace.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    temporary.replace(args.trace)
    slot_count = sum(len(trace["slots"]) for trace in traces)
    covered = sum(trace["covered_slots"] for trace in traces)
    dropped = sum(trace["dropped"] for trace in traces)
    print(json.dumps({
        "questions": len(records),
        "questions_with_line_items": sum(bool(trace["line_items"]) for trace in traces),
        "slots": slot_count,
        "covered_slots": covered,
        "slot_coverage": round(covered / slot_count, 4) if slot_count else 0.0,
        "tables_kept": sum(len(trace["selected"]) for trace in traces),
        "tables_dropped": dropped,
        "questions_changed": sum(trace["selected"] != (base.get(str(trace["id"])) or []) for trace in traces),
        "output": str(args.output),
        "trace": str(args.trace),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
