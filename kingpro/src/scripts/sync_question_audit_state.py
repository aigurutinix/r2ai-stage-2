"""Synchronize current_state audit counters from the authoritative ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"
DEFAULT_STATE = ROOT / "knowledge/vothuong/current_state.json"
DEFAULT_PUBLIC = ROOT / "build/v227_residual_public100_fused_v217.json"
DEFAULT_BATCH = ROOT / "build/v227_residual_batch150_fused_v217.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pending_ids(queue_payload: dict, closed: set[int]) -> list[int]:
    return [int(item["id"]) for item in queue_payload.get("queue", []) if int(item["id"]) not in closed]


def compute_stats(verdicts: dict, public: dict, batch: dict) -> dict:
    closed_order = [int(value) for value in verdicts]
    closed = set(closed_order)
    public_pending = pending_ids(public, closed)
    batch_pending = pending_ids(batch, closed)
    no_change = sum(item.get("mutation") == "none" for item in verdicts.values())
    pending = sum(str(item.get("mutation", "")).startswith("pending_batch") for item in verdicts.values())
    return {
        "total_closed": len(closed),
        "closed_no_change": no_change,
        "closed_pending_batch_cleanup": pending,
        "public100_pending": len(public_pending),
        "batch150_pending": len(batch_pending),
        "latest_closed": closed_order[-12:],
        "next_question": public_pending[0] if public_pending else None,
        "next_batch_question": batch_pending[0] if batch_pending else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--public-queue", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--batch-queue", type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    ledger = load(args.ledger.resolve())
    state_path = args.state.resolve()
    state = load(state_path)
    stats = compute_stats(
        ledger["verdicts"],
        load(args.public_queue.resolve()),
        load(args.batch_queue.resolve()),
    )
    target = state["individual_source_audit"]
    target.update(stats)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
