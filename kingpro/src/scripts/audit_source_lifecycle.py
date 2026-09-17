"""Snapshot the source catalog and detect added/removed/restated reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.governance.lifecycle import (  # noqa: E402
    atomic_write_json,
    build_catalog_snapshot,
    compare_snapshots,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "build" / "catalog.jsonl")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build" / "source_lifecycle" / "latest.json"
    )
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()

    previous_path = args.previous or args.output
    previous = None
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
    current = build_catalog_snapshot(args.catalog)
    current["comparison"] = compare_snapshots(previous, current)
    atomic_write_json(args.output, current)
    summary = {
        key: current[key]
        for key in ("generated_at", "catalog_sha256", "report_count", "table_count")
    }
    summary["comparison"] = {
        key: current["comparison"][key]
        for key in ("has_changes", "review_required")
    }
    summary["change_counts"] = {
        key: len(current["comparison"][key])
        for key in ("added_reports", "removed_reports", "changed_reports")
    }
    summary["output"] = str(args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if current["comparison"]["review_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

