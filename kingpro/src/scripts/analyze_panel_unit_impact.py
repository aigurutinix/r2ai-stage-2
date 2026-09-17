"""Simulate exact-table currency-unit normalization for panel programs.

Panel manifests are intended to contain VND-normalized operands.  This audit
checks the header of every exact source CSV referenced by a panel manifest,
replaces a stale ``scale`` only when that header literally declares
``Nghìn/Ngàn VND`` or ``Triệu VND``, executes the unchanged submitted program,
and reports questions whose result changes.

Read-only: it never edits a candidate.  Findings still require source review
before a builder applies them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from grader_check import _SAFE_BUILTINS, _read_csv


ROOT = Path(__file__).resolve().parents[1]


def plain(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(char for char in text if not unicodedata.combining(char)).lower()


def literal_header_scale(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = []
        for index, row in enumerate(csv.reader(handle)):
            rows.extend(row)
            if index >= 5:
                break
    text = plain(" ".join(rows))
    # Extractors often concatenate the period and unit (``2024Triệu VND``),
    # so the left edge cannot require a word boundary after a digit.
    if re.search(r"(?:^|[^a-z])(?:nghin|ngan)\s*(?:vnd|dong)\b", text):
        return 1_000.0
    if re.search(r"(?:^|[^a-z])trieu\s*(?:vnd|dong)\b", text):
        return 1_000_000.0
    if re.search(r"(?:^|[^a-z])ty\s*(?:vnd|dong)\b", text):
        return 1_000_000_000.0
    if re.search(r"\b(?:vnd|dong)\b", text):
        return 1.0
    return None


def manifest_adjustments(candidate: Path, question_id: int) -> list[dict[str, Any]]:
    manifest_path = candidate / "data" / f"q{question_id}_source_cells.csv"
    if not manifest_path.is_file():
        return []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    adjustments = []
    for index, row in enumerate(rows):
        table_ref = str(row.get("source_table", ""))
        source_csv = str(row.get("source_csv", ""))
        if "|" not in table_ref or not source_csv:
            continue
        document = table_ref.split("|", 1)[0]
        exact_path = ROOT / "build" / "tables" / document / source_csv
        expected = literal_header_scale(exact_path)
        if expected is None:
            continue
        try:
            current = float(row.get("scale", 1.0))
        except (TypeError, ValueError):
            continue
        if abs(current - expected) <= max(1e-9, expected * 1e-9):
            continue
        adjustments.append(
            {
                "row_index": index,
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "metric_key": row.get("metric_key", ""),
                "raw": row.get("raw", ""),
                "table_ref": table_ref,
                "source_csv": source_csv,
                "old_scale": current,
                "new_scale": expected,
            }
        )
    return adjustments


def execute(candidate: Path, row: dict[str, Any], adjustments: list[dict[str, Any]]) -> Any:
    dfs = {
        str(item["variable"]): _read_csv(
            (candidate / str(item["csv_path"])).resolve(), typed=True
        )
        for item in row.get("evidence", [])
    }
    manifest_name = f"q{int(row['id'])}_source_cells.csv"
    for item in row.get("evidence", []):
        if Path(str(item.get("csv_path", ""))).name != manifest_name:
            continue
        frame = dfs[str(item["variable"])]
        if "scale" not in frame.columns:
            continue
        for adjustment in adjustments:
            frame.loc[int(adjustment["row_index"]), "scale"] = float(
                adjustment["new_scale"]
            )
    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": dfs,
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))
    exec(compile(str(row["pandas_query"]), f"<q{row['id']}_unit_sim>", "exec"), namespace)  # noqa: S102
    value = namespace["result"]
    return value.item() if hasattr(value, "item") else value


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    submissions = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    panel_rows = json.loads((candidate / "panel_source_audit.json").read_text(encoding="utf-8"))
    panel_ids = {int(row["id"]) for row in panel_rows}
    changed = []
    simulated = 0
    for row in submissions:
        qid = int(row["id"])
        if qid not in panel_ids:
            continue
        adjustments = manifest_adjustments(candidate, qid)
        if not adjustments:
            continue
        simulated += 1
        try:
            new_answer = execute(candidate, row, adjustments)
        except Exception as exc:  # audit should continue across isolated failures
            changed.append(
                {
                    "id": qid,
                    "question": row.get("question", ""),
                    "old_answer": row.get("answer"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "adjustments": adjustments,
                }
            )
            continue
        try:
            same = abs(float(new_answer) - float(row.get("answer"))) <= 1e-9
        except (TypeError, ValueError):
            same = new_answer == row.get("answer")
        if not same:
            changed.append(
                {
                    "id": qid,
                    "question": row.get("question", ""),
                    "old_answer": row.get("answer"),
                    "new_answer": new_answer,
                    "adjustments": adjustments,
                }
            )

    report = {
        "candidate": candidate.name,
        "panel_questions": len(panel_ids),
        "simulated_questions": simulated,
        "changed_count": len(changed),
        "changed_ids": [item["id"] for item in changed],
        "findings": changed,
        "claim_limit": "Exact-header simulation; review every changed selector before mutation.",
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(args.out.resolve())
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
