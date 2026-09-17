"""Execute safe counterfactuals using nearest-table source-unit evidence.

This combines the strict output-unit contract from
``analyze_document_unit_impact`` with the exact extracted-text locator from
``audit_local_source_units``.  It is read-only triage: explicit multipliers in
the query can already compensate for a manifest scale of one, so every result
still needs query/source review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from analyze_document_unit_impact import execute, query_normalizes_vnd
from audit_local_source_units import (
    DEFAULT_DATA_ROOT,
    extracted_text_index,
    nearest_unit_marker,
    parse_table_ref,
    same_scale,
)


def local_adjustments(
    candidate: Path,
    qid: int,
    index: dict[str, Path],
    cache: dict[str, list[str]],
    *,
    max_distance: int,
    standard_metrics_only: bool,
) -> list[dict[str, Any]]:
    path = candidate / "data" / f"q{qid}_source_cells.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    changes = []
    for row_index, row in enumerate(rows):
        metric_key = str(row.get("metric_key", ""))
        if standard_metrics_only and not re.match(r"^(?:cdkt|kqkd|lctt):", metric_key):
            continue
        parsed = parse_table_ref(str(row.get("source_table", "")))
        if parsed is None:
            continue
        document, table_line = parsed
        source_path = index.get(document)
        if source_path is None:
            continue
        if document not in cache:
            cache[document] = source_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        marker = nearest_unit_marker(
            cache[document], table_line, max_distance=max_distance
        )
        if marker is None or float(marker["factor"]) <= 1.0:
            continue
        try:
            current = float(row.get("scale", 1.0))
        except (TypeError, ValueError):
            continue
        expected = float(marker["factor"])
        if same_scale(current, expected):
            continue
        changes.append(
            {
                "row_index": row_index,
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "metric_key": metric_key,
                "raw": row.get("raw", ""),
                "table_ref": row.get("source_table", ""),
                "source_csv": row.get("source_csv", ""),
                "old_scale": current,
                "new_scale": expected,
                "inference": "nearest-table-marker",
                "unit_marker_line": marker["line"],
                "unit_marker_distance": marker["distance"],
                "unit_marker_text": marker["text"],
            }
        )
    return changes


def explicit_multiplier_mentions(query: str) -> list[str]:
    """Expose likely manual compensation; this is evidence, not auto-dismissal."""

    matches = re.findall(
        r"(?:\*|/)\s*(?:1e[369]|10\s*\*\*\s*[369]|1_000(?:_000(?:_000)?)?|1000(?:000(?:000)?)?)",
        query,
        flags=re.I,
    )
    return sorted(set(re.sub(r"\s+", "", value) for value in matches))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-distance", type=int, default=60)
    parser.add_argument("--all-metrics", action="store_true")
    parser.add_argument("--include-unsafe", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    submissions = json.loads(
        (candidate / "submission.json").read_text(encoding="utf-8")
    )
    index = extracted_text_index(args.data_root.resolve())
    cache: dict[str, list[str]] = {}
    findings = []
    simulated = 0
    for row in submissions:
        if not args.include_unsafe and not query_normalizes_vnd(row):
            continue
        changes = local_adjustments(
            candidate,
            int(row["id"]),
            index,
            cache,
            max_distance=args.max_distance,
            standard_metrics_only=not args.all_metrics,
        )
        if not changes:
            continue
        simulated += 1
        try:
            new_answer = execute(candidate, row, changes)
            if abs(float(new_answer) - float(row.get("answer"))) <= 1e-9:
                continue
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": row.get("question", ""),
                    "old_answer": row.get("answer"),
                    "new_answer": new_answer,
                    "explicit_multiplier_mentions": explicit_multiplier_mentions(
                        str(row.get("pandas_query", ""))
                    ),
                    "adjustments": changes,
                }
            )
        except Exception as exc:
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": row.get("question", ""),
                    "old_answer": row.get("answer"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "adjustments": changes,
                }
            )

    report = {
        "candidate": candidate.name,
        "questions": len(submissions),
        "documents_indexed": len(index),
        "documents_read": len(cache),
        "simulated_questions": simulated,
        "safe_vnd_contract_only": not args.include_unsafe,
        "standard_metrics_only": not args.all_metrics,
        "changed_count": len(findings),
        "changed_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "Nearest-table unit counterfactual only. Explicit query conversions "
            "and the requested output unit must be reviewed before mutation."
        ),
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
