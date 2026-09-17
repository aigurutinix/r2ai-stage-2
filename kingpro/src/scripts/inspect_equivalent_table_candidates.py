"""Render equivalent-table candidates with enough context for manual audit.

This script is read-only.  It joins the alternative-table scan with the
submission, source audit and original BTC table catalog, then prints exact
matching coordinates, headers and rows.  It intentionally does not approve or
modify a candidate: semantic equivalence still requires human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (ROOT / "build" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["table_ref"])] = row
    return rows


def table_context(table_ref: str, catalog: dict[str, dict], raw_values: set[str]) -> dict:
    meta = catalog[table_ref]
    path = ROOT / "build" / "tables" / str(meta["csv_path"])
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        index_col=None,
    )
    matches = []
    for row_idx in range(len(frame)):
        for col_idx, column in enumerate(frame.columns):
            value = str(frame.iloc[row_idx, col_idx]).strip()
            if value in raw_values:
                matches.append(
                    {
                        "raw": value,
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "column": str(column),
                        "row": [str(item) for item in frame.iloc[row_idx].tolist()],
                    }
                )
    return {
        "table_ref": table_ref,
        "page": meta.get("page"),
        "shape": [len(frame), len(frame.columns)],
        "columns": [str(column) for column in frame.columns],
        "matches": matches,
        "catalog_context": str(meta.get("search_text", ""))[:1200],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "sub_top123_candidate_v180_dcm_ure_current_year",
    )
    parser.add_argument(
        "--alternatives",
        type=Path,
        default=ROOT / "outputs" / "v180_equivalent_table_alternatives.json",
    )
    parser.add_argument("--ids", required=True, help="comma-separated question IDs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    requested = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
    submission = {int(row["id"]): row for row in read_json(args.candidate / "submission.json")}
    source_audit = {int(row["id"]): row for row in read_json(args.candidate / "source_audit.json")}
    alternatives = {
        int(row["id"]): row for row in read_json(args.alternatives)["records"]
    }
    catalog = load_catalog()
    records = []
    for question_id in sorted(requested):
        row = submission[question_id]
        audit = source_audit[question_id]
        raw_values = {
            str(source.get("raw", "")).strip()
            for source in audit.get("sources", [])
            if str(source.get("raw", "")).strip()
        }
        alternative = alternatives[question_id]
        records.append(
            {
                "id": question_id,
                "question": row["question"],
                "answer": row.get("answer"),
                "relevant_tables": row.get("relevant_tables", []),
                "source_audit": audit.get("sources", []),
                "alternatives": [
                    table_context(table_ref, catalog, raw_values)
                    for table_ref in alternative["extra"]
                ],
            }
        )
    payload = {"candidate": args.candidate.name, "records": records}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
