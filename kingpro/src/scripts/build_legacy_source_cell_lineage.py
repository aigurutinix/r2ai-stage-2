"""Convert resolved legacy terminal reads into the common cell-lineage schema.

The compact source manifests cover 810 questions.  The remaining legacy
programs are resolved by ``audit_legacy_query_sources.py``.  This adapter lets
all cell-level semantic oracles run over that second population without
changing a candidate or re-executing submitted programs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def convert(payload: dict) -> dict:
    records = []
    skipped = []
    cell_count = 0
    for record in payload.get("records", []):
        cells = []
        for index, read in enumerate(record.get("terminal_reads", [])):
            row = read.get("source_row")
            column = read.get("source_column")
            table = read.get("source_table")
            if not isinstance(row, int) or not isinstance(column, int) or not table:
                skipped.append(
                    {
                        "id": int(record["id"]),
                        "terminal_read_index": index,
                        "reason": "terminal read lacks a scalar physical row/column",
                    }
                )
                continue
            cells.append(
                {
                    "csv": str(read.get("csv", "")),
                    "source_index": index,
                    "ticker": "",
                    "year": "",
                    "metric_key": "legacy_terminal_read",
                    "source_table": str(table),
                    "row_idx": row,
                    "col_idx": column,
                    "source_label": str(read.get("source_label", "")),
                    "header_path": [],
                    "source_context": "",
                    "raw_manifest": str(read.get("raw", "")),
                    "raw_physical": str(read.get("raw", "")),
                }
            )
        if cells:
            cell_count += len(cells)
            records.append(
                {
                    "id": int(record["id"]),
                    "question": record.get("question", ""),
                    "answer": record.get("answer"),
                    "cells": cells,
                }
            )
    return {
        "kind": "legacy_terminal_read_cell_lineage",
        "source_audit": payload.get("submission", ""),
        "legacy_records_received": len(payload.get("records", [])),
        "records_with_scalar_cells": len(records),
        "physical_cells_resolved": cell_count,
        "skipped_terminal_read_count": len(skipped),
        "skipped_terminal_reads": skipped,
        "policy": "Read-only adapter; source row/column provenance is inherited from the legacy resolver.",
        "records": records,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "legacy_audit",
        nargs="?",
        type=Path,
        default=ROOT / "build" / "v209_legacy_query_sources_v207.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "v210_legacy_source_cell_lineage_v209.json",
    )
    args = parser.parse_args()
    payload = json.loads(args.legacy_audit.read_text(encoding="utf-8"))
    result = convert(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "skipped_terminal_reads"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
