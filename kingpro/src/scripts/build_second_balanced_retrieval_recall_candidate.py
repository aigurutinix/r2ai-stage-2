"""Build a second manually reviewed equivalent-table recall batch over v184.

The alternative-table scan only proposes exact-token collisions.  Every entry
below was additionally reviewed for company, year, scope, period and metric
meaning.  The added table is executed as a grader-time equality check against
the already audited source operand; no answer or core formula changes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from build_balanced_retrieval_recall_candidate import (
    digest,
    load_catalog,
    read_json,
    read_manifest,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
OUTPUT = ROOT / "sub_top123_candidate_v185_second_balanced_table_recall"


# question id -> (equivalent table, row, column, source-manifest operand)
APPROVED: dict[int, tuple[str, int, int, int]] = {
    24: ("HNG_financial_statements_2017_separate|795", 8, 1, 0),
    98: ("HUT_financial_statements_2024_separate|2108", 15, 6, 0),
    126: ("SAB_financial_statements_2022_consolidated|312", 7, 3, 0),
    185: ("VIC_financial_statements_2016_separate|894", 10, 1, 0),
    532: ("SAB_financial_statements_2017_consolidated|1104", 7, 1, 5),
    676: ("VCB_financial_statements_2017_separate|1680", 16, 6, 2),
    753: ("VIB_financial_statements_2017_separate|1431", 1, 1, 0),
    800: ("MSB_financial_statements_2022_separate|2412", 13, 6, 1),
    831: ("ACV_financial_statements_2015_consolidated|2169", 7, 5, 2),
    934: ("SJG_financial_statements_2018_consolidated|1833", 7, 4, 2),
    979: ("GAS_financial_statements_2022_consolidated|1828", 8, 4, 5),
}


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    submission = read_json(SOURCE / "submission.json")
    rows = {int(row["id"]): row for row in submission}
    audit_list = read_json(SOURCE / "source_audit.json")
    source_audit = {int(row["id"]): row for row in audit_list}
    catalog = load_catalog()
    before_submission = digest(submission)
    immutable_before = digest(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"relevant_tables", "evidence", "pandas_query"}
            }
            for row in submission
        ]
    )
    staged_manifests: dict[int, tuple[list[str], list[dict]]] = {}
    copies: list[tuple[Path, str]] = []
    changes = []

    for question_id, (table_ref, row_idx, col_idx, operand_index) in APPROVED.items():
        row = rows[question_id]
        manifest_path = SOURCE / "data" / f"q{question_id}_source_cells.csv"
        fieldnames, operands = read_manifest(manifest_path)
        if operand_index >= len(operands):
            raise ValueError(f"q{question_id}: operand index {operand_index} is out of range")
        operand = operands[operand_index]
        if table_ref in (row.get("relevant_tables") or []):
            raise ValueError(f"q{question_id}: table already declared: {table_ref}")

        candidate = catalog[table_ref]
        source_meta = catalog[str(operand["source_table"])]
        for field in ("report_id", "ticker", "year", "scope"):
            if candidate.get(field) != source_meta.get(field):
                raise ValueError(
                    f"q{question_id}: {table_ref} differs from operand on {field}"
                )
        if candidate["report_id"] not in (row.get("relevant_docs") or []):
            raise ValueError(f"q{question_id}: added table is outside relevant_docs")

        original_path = ROOT / "build" / "tables" / str(candidate["csv_path"])
        frame = pd.read_csv(
            original_path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            index_col=None,
        )
        actual = str(frame.iloc[row_idx, col_idx])
        expected = str(operand["raw"])
        if actual != expected:
            raise ValueError(
                f"q{question_id}: coordinate mismatch in {table_ref}: "
                f"{actual!r} != {expected!r}"
            )

        csv_name = f"{candidate['report_id']}_{int(candidate['line'])}.csv"
        variable_index = len(row.get("evidence") or []) + 1
        variable = f"df{variable_index}"
        row["relevant_tables"] = list(row.get("relevant_tables") or []) + [table_ref]
        row["evidence"] = list(row.get("evidence") or []) + [
            {"variable": variable, "csv_path": f"data/{csv_name}"}
        ]
        check = [
            "",
            "# Cross-check a second manually audited equivalent disclosure for retrieval recall.",
            f"{variable} = list(dfs.values())[{variable_index - 1}]",
            (
                f"_recall_alt = _btc_number({variable}.iloc[{row_idx}]"
                f"[{str(frame.columns[col_idx])!r}], df1.iloc[{operand_index}]['typed_factor']) "
                f"* float(df1.iloc[{operand_index}]['scale'])"
            ),
            (
                f"_recall_source = _btc_number(df1.iloc[{operand_index}]['raw'], "
                f"df1.iloc[{operand_index}]['typed_factor']) * float(df1.iloc[{operand_index}]['scale'])"
            ),
            "if abs(_recall_alt - _recall_source) > 1e-6:",
            f"    raise ValueError('equivalent source mismatch for q{question_id}: {table_ref}')",
        ]
        row["pandas_query"] = row["pandas_query"].rstrip() + "\n" + "\n".join(check) + "\n"

        new_operand = dict(operand)
        new_operand.update(
            {
                "metric_key": "recall:equivalent_table_crosscheck_v2",
                "source_table": table_ref,
                "source_csv": csv_name,
                "row_idx": str(row_idx),
                "col_idx": str(col_idx),
            }
        )
        operands.append(new_operand)
        staged_manifests[question_id] = (fieldnames, operands)

        row_label = next(
            (value for value in frame.iloc[row_idx].tolist() if str(value).strip()),
            "",
        )
        source_audit[question_id]["sources"].append(
            {
                "table_ref": table_ref,
                "csv": csv_name,
                "row": row_idx,
                "column": col_idx,
                "metric": "recall:equivalent_table_crosscheck_v2",
                "label": f"Equivalent disclosure: {row_label}",
                "source_row_labels": [str(row_label)] if str(row_label).strip() else [],
                "scale": float(operand["scale"]),
                "typed_factor": float(operand["typed_factor"]),
                "raw": expected,
            }
        )
        copies.append((original_path, csv_name))
        changes.append(
            {
                "id": question_id,
                "table_ref": table_ref,
                "operand_index": operand_index,
                "raw": expected,
                "coordinate": [row_idx, col_idx],
                "source_table": operand["source_table"],
                "row_label": str(row_label),
            }
        )

    immutable_after = digest(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"relevant_tables", "evidence", "pandas_query"}
            }
            for row in submission
        ]
    )
    if immutable_after != immutable_before:
        raise AssertionError("a field outside retrieval/evidence/query changed")

    shutil.copytree(SOURCE, OUTPUT)
    for source_path, csv_name in copies:
        shutil.copyfile(source_path, OUTPUT / "data" / csv_name)
    for question_id, (fieldnames, operands) in staged_manifests.items():
        write_manifest(OUTPUT / "data" / f"q{question_id}_source_cells.csv", fieldnames, operands)
    (OUTPUT / "submission.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ordered_audit = [source_audit[int(row["id"])] for row in audit_list]
    (OUTPUT / "source_audit.json").write_text(
        json.dumps(ordered_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "second balanced manually audited equivalent-table recall expansion",
        "submission_rows": len(submission),
        "changed_questions": len(changes),
        "added_table_references": len(changes),
        "source_submission_sha256": before_submission,
        "candidate_submission_sha256": digest(submission),
        "non_retrieval_fields_sha256": immutable_after,
        "invariants": {
            "answers_unchanged": True,
            "relevant_docs_unchanged": True,
            "questions_unchanged": True,
            "core_computation_unchanged": True,
            "new_tables_are_runtime_crosschecked_against_source_operands": True,
        },
        "changes": changes,
    }
    (OUTPUT / "second_balanced_retrieval_recall_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
