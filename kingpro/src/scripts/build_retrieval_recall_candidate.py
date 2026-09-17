from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "sub_top123_candidate_v180_dcm_ure_current_year"
DEFAULT_OUTPUT = ROOT / "sub_top123_candidate_v181_conservative_table_recall"


# Each alternative was manually checked against the original CSV.  The gate
# below additionally requires the same report/ticker/year/scope and the exact
# audited raw value.  These are citation-recall expansions only: answer,
# evidence, code and relevant_docs must remain byte-for-byte equivalent as
# parsed JSON values.
APPROVED_EXPANSIONS: dict[int, dict[str, object]] = {
    34: {
        "expected": ["HBC_financial_statements_2022_separate|1578"],
        "add": [
            {"table_ref": "HBC_financial_statements_2022_separate|431", "row": 10, "column": "3", "factor": 1e-9, "row_label": "9. Thu nhập khác"},
        ],
        "reason": "Other income is disclosed identically in the note and parent income statement.",
    },
    76: {
        "expected": ["HHV_financial_statements_2024_separate|379"],
        "add": [
            {"table_ref": "HHV_financial_statements_2024_separate|1545", "row": 2, "column": "1", "factor": 1e-9, "row_label": "Đầu tư vào công ty con"},
        ],
        "reason": "Investment in subsidiaries is disclosed identically in the balance sheet and investment note.",
    },
    108: {
        "expected": ["HAG_financial_statements_2023_consolidated|1661"],
        "add": [
            {"table_ref": "HAG_financial_statements_2023_consolidated|1622", "row": 7, "column": "1", "factor": 1.0, "row_label": "TỔNG CỘNG"},
        ],
        "reason": "Both bond-detail tables explicitly disclose the same total bond balance and period.",
    },
    139: {
        "expected": ["HAG_financial_statements_2020_consolidated|2052"],
        "add": [
            {"table_ref": "HAG_financial_statements_2020_consolidated|411", "row": 3, "column": "3", "factor": 1.0, "row_label": "3. Doanh thu thuần về bán hàng và cung cấp dịch vụ"},
        ],
        "reason": "Net revenue is disclosed identically in the revenue note and consolidated income statement.",
    },
    211: {
        "expected": ["DLG_financial_statements_2022_separate|1343"],
        "add": [
            {"table_ref": "DLG_financial_statements_2022_separate|455", "row": 16, "column": "3", "factor": 1e-11, "row_label": "14. Tổng lợi nhuận kế toán trước thuế"},
        ],
        "reason": "Accounting profit before tax is disclosed identically in the tax note and parent income statement.",
    },
    272: {
        "expected": ["MBB_financial_statements_2020_separate|1974"],
        "add": [
            {"table_ref": "MBB_financial_statements_2020_separate|2049", "row": 13, "column": "8", "factor": 1.0, "row_label": "Tổng tài sản"},
        ],
        "reason": "Both audited maturity analyses explicitly total the same parent assets at 31 December 2020.",
    },
    284: {
        "expected": ["EIB_financial_statements_2023_separate|1716"],
        "add": [
            {"table_ref": "EIB_financial_statements_2023_separate|332", "row": 17, "column": "3", "factor": 1.0, "row_label": "Vốn điều lệ"},
            {"table_ref": "EIB_financial_statements_2023_separate|1752", "row": 3, "column": "2", "factor": 1.0, "row_label": "Tại ngày 31 tháng 12 năm 2023"},
        ],
        "reason": "Charter capital is disclosed identically in the balance sheet, equity movement and share-capital note.",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_catalog(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["table_ref"])] = row
    return rows


def build(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source candidate not found: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    submission_path = source / "submission.json"
    audit_path = source / "source_audit.json"
    rows = read_json(submission_path)
    source_audit = {int(row["id"]): row for row in read_json(audit_path)}
    catalog = load_catalog(ROOT / "build" / "catalog.jsonl")
    by_id = {int(row["id"]): row for row in rows}
    if len(rows) != 1012 or len(by_id) != 1012:
        raise ValueError("expected exactly 1,012 unique submission rows")

    before_digest = json_digest(rows)
    immutable_fields = {"relevant_tables", "evidence", "pandas_query"}
    before_without_retrieval = json_digest(
        [{key: value for key, value in row.items() if key not in immutable_fields} for row in rows]
    )
    changes: list[dict] = []
    source_cell_additions: dict[int, list[dict[str, object]]] = {}

    for question_id, rule in APPROVED_EXPANSIONS.items():
        row = by_id[question_id]
        expected = list(rule["expected"])
        additions = list(rule["add"])
        addition_refs = [str(item["table_ref"]) for item in additions]
        current = list(row.get("relevant_tables", []))
        if current != expected:
            raise ValueError(f"q{question_id}: expected {expected}, found {current}")
        if question_id not in source_audit:
            raise ValueError(f"q{question_id}: missing source audit")

        source_rows = source_audit[question_id].get("sources", [])
        raw_values = {str(item.get("raw", "")).strip() for item in source_rows if str(item.get("raw", "")).strip()}
        if not raw_values:
            raise ValueError(f"q{question_id}: source audit has no raw value")

        anchor = catalog[current[0]]
        documents = set(row.get("relevant_docs", []))
        for addition in additions:
            table_ref = str(addition["table_ref"])
            candidate = catalog.get(table_ref)
            if candidate is None:
                raise ValueError(f"q{question_id}: catalog is missing {table_ref}")
            for field in ("report_id", "ticker", "year", "scope"):
                if candidate.get(field) != anchor.get(field):
                    raise ValueError(
                        f"q{question_id}: {table_ref} differs on {field}: "
                        f"{candidate.get(field)!r} != {anchor.get(field)!r}"
                    )
            if candidate["report_id"] not in documents:
                raise ValueError(f"q{question_id}: {table_ref} is outside relevant_docs")
            csv_path = ROOT / "build" / "tables" / str(candidate["csv_path"])
            csv_text = csv_path.read_text(encoding="utf-8-sig")
            if not any(raw in csv_text for raw in raw_values):
                raise ValueError(f"q{question_id}: {table_ref} does not contain an audited raw value")

        expanded = list(dict.fromkeys(current + addition_refs))
        row["relevant_tables"] = expanded
        checks: list[str] = []
        new_evidence: list[dict] = []
        new_source_cells: list[dict[str, object]] = []
        for offset, addition in enumerate(additions, start=1):
            table_ref = str(addition["table_ref"])
            candidate = catalog[table_ref]
            line = int(candidate["line"])
            csv_name = f"{candidate['report_id']}_{line}.csv"
            variable_index = len(row["evidence"]) + offset
            variable = f"df{variable_index}"
            new_evidence.append({"variable": variable, "csv_path": f"data/{csv_name}"})
            factor = float(addition["factor"])
            checks.extend(
                [
                    f"{variable} = list(dfs.values())[{variable_index - 1}]",
                    f"_recall_value_{offset} = _btc_number({variable}.iloc[{int(addition['row'])}][{str(addition['column'])!r}]) * {factor!r}",
                    f"if abs(round(_recall_value_{offset}, 2) - float(result)) > 1e-6:",
                    f"    raise ValueError('equivalent source mismatch for q{question_id}: {table_ref}')",
                ]
            )
            raw = next(
                raw
                for raw in raw_values
                if raw
                in (ROOT / "build" / "tables" / str(candidate["csv_path"])).read_text(
                    encoding="utf-8-sig"
                )
            )
            source_audit[question_id]["sources"].append(
                {
                    "table_ref": table_ref,
                    "csv": csv_name,
                    "row": int(addition["row"]),
                    "column": int(addition["column"]),
                    "metric": "recall:equivalent_table_crosscheck",
                    "label": str(addition["row_label"]),
                    "source_row_labels": [str(addition["row_label"])],
                    "scale": 1.0,
                    "typed_factor": 1.0,
                    "raw": raw,
                }
            )
            new_source_cells.append(
                {
                    "ticker": candidate["ticker"],
                    "year": candidate["year"],
                    "metric_key": "recall:equivalent_table_crosscheck",
                    "raw": raw,
                    "typed_factor": 1,
                    "scale": 1,
                    "source_table": table_ref,
                    "source_csv": csv_name,
                    "row_idx": int(addition["row"]),
                    "col_idx": str(addition["column"]),
                }
            )
        row["evidence"] = list(row["evidence"]) + new_evidence
        row["pandas_query"] = row["pandas_query"].rstrip() + "\n\n# Cross-check equivalent audited disclosures used for retrieval recall.\n" + "\n".join(checks) + "\n"
        changes.append(
            {
                "id": question_id,
                "question": row["question"],
                "before": current,
                "after": expanded,
                "added": addition_refs,
                "reason": rule["reason"],
                "audited_raw_values": sorted(raw_values),
            }
        )
        source_cell_additions[question_id] = new_source_cells

    after_without_retrieval = json_digest(
        [{key: value for key, value in row.items() if key not in immutable_fields} for row in rows]
    )
    if before_without_retrieval != after_without_retrieval:
        raise AssertionError("a field outside relevant_tables/evidence/pandas_query changed")

    shutil.copytree(source, output)
    for question_id, rule in APPROVED_EXPANSIONS.items():
        for addition in rule["add"]:
            candidate = catalog[str(addition["table_ref"])]
            source_csv = ROOT / "build" / "tables" / str(candidate["csv_path"])
            target_csv = output / "data" / f"{candidate['report_id']}_{int(candidate['line'])}.csv"
            shutil.copyfile(source_csv, target_csv)
        source_cells_path = output / "data" / f"q{question_id}_source_cells.csv"
        with source_cells_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            source_cells = list(reader)
        if not fieldnames:
            raise ValueError(f"q{question_id}: source-cell manifest has no header")
        source_cells.extend(source_cell_additions[question_id])
        with source_cells_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(source_cells)
    (output / "submission.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ordered_source_audit = [source_audit[int(row["id"])] for row in read_json(audit_path)]
    (output / "source_audit.json").write_text(
        json.dumps(ordered_source_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "candidate": output.name,
        "source_candidate": source.name,
        "purpose": "conservative equivalent-table recall expansion",
        "submission_rows": len(rows),
        "changed_questions": len(changes),
        "added_table_references": sum(len(item["added"]) for item in changes),
        "source_submission_sha256": before_digest,
        "candidate_submission_sha256": json_digest(rows),
        "non_retrieval_fields_sha256": after_without_retrieval,
        "invariants": {
            "answers_unchanged": True,
            "relevant_docs_unchanged": True,
            "questions_unchanged": True,
            "core_computation_unchanged": True,
            "new_tables_are_executed_as_value_crosschecks": True,
            "only_relevant_tables_evidence_and_crosschecks_changed": True,
        },
        "changes": changes,
    }
    (output / "retrieval_recall_audit.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.source, args.output)
    # Keep CLI output portable on Windows shells whose stdout still uses a
    # legacy code page.  Candidate files remain proper UTF-8 above.
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
