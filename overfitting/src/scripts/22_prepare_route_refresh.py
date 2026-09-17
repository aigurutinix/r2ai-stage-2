"""Select only questions whose current router contract differs from a baseline."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.extraction.build_store import Store
from vifinqa.router.entities import StockMap
from vifinqa.router.router import route_question
from vifinqa.utils.io import read_jsonl, setup_stdout


def _read_ids(path: str) -> set[int]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("ids", [])
    if not isinstance(value, list):
        raise ValueError("target file must contain a JSON list/object")
    return {int(item) for item in value}


def _signature(route: dict) -> tuple:
    requirements = route.get("evidence_requirements") or []
    requirement_keys = tuple(sorted(
        (str(item.get("requirement_id") or ""),
         str(item.get("metric_key") or ""),
         int(item.get("year") or 0))
        for item in requirements
    ))
    plan = route.get("plan") or {}
    return (
        str(plan.get("op") or "lookup"),
        tuple(route.get("metric_keys") or []),
        requirement_keys,
    )


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(config.QUESTIONS_JSONL))
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    parser.add_argument("--base", required=True)
    parser.add_argument("--target-ids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target_ids = _read_ids(args.target_ids)
    old = {int(row["id"]): row for row in read_jsonl(args.base)}
    questions = {
        int(row["id"]): row for row in read_jsonl(args.questions)
        if int(row["id"]) in target_ids
    }
    if set(questions) != target_ids or not target_ids.issubset(old):
        raise ValueError("target ids are missing from questions or baseline retrieval")

    store = Store(Path(args.store_dir), cache_size=4)
    stock = StockMap(Path(args.code_stock))
    changed, transitions = [], Counter()
    for qid in sorted(target_ids):
        new_route = route_question(
            qid, questions[qid]["question"], stock, store).to_dict()
        old_route = old[qid].get("route") or {}
        if _signature(new_route) == _signature(old_route):
            continue
        changed.append(qid)
        old_op = str((old_route.get("plan") or {}).get("op") or "lookup")
        new_op = str((new_route.get("plan") or {}).get("op") or "lookup")
        transitions[f"{old_op}->{new_op}"] += 1

    payload = {
        "name": "private_v35_changed_router_contracts",
        "count": len(changed),
        "source_target_count": len(target_ids),
        "operation_transitions": dict(transitions),
        "ids": changed,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "ids"},
                     ensure_ascii=False, indent=2))
    print(f"refresh ids -> {args.out}")


if __name__ == "__main__":
    main()
