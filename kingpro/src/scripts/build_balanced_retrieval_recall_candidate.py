"""Build a source-proven, balanced table-recall expansion over v181.

Every added table was manually reviewed as an equivalent disclosure of one
existing source operand.  The generated Pandas program reads the added table
and compares it with that operand at runtime, so retrieval metadata never gets
ahead of executable provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v181_conservative_table_recall"
OUTPUT = ROOT / "sub_top123_candidate_v182_balanced_table_recall"


# question id -> (table_ref, row index, column index, source operand index)
# The source operand index addresses q<ID>_source_cells.csv before expansion.
APPROVED: dict[int, tuple[str, int, int, int]] = {
    77: ("VRE_financial_statements_2024_separate|236", 17, 4, 0),
    298: ("DPM_financial_statements_2021_separate|386", 17, 4, 0),
    508: ("STB_financial_statements_2021_separate|1929", 14, 1, 3),
    582: ("MCH_financial_statements_2021_consolidated|926", 16, 1, 1),
    637: ("VPI_financial_statements_2022_separate|1318", 12, 1, 0),
    656: ("FOX_financial_statements_2024_consolidated|792", 3, 1, 0),
    667: ("EIB_financial_statements_2023_separate|1505", 6, 6, 0),
    688: ("DXG_financial_statements_2018_consolidated|268", 12, 3, 0),
    707: ("VGT_financial_statements_2024_consolidated|170", 9, 3, 1),
    740: ("DNH_financial_statements_2023_separate|676", 5, 1, 1),
    741: ("MPC_financial_statements_2019_consolidated|1542", 1, 1, 1),
    752: ("MBB_financial_statements_2022_consolidated|2089", 20, 1, 1),
    758: ("KBC_financial_statements_2025_separate|1048", 7, 1, 0),
    773: ("KBC_financial_statements_2020_separate|1329", 10, 1, 0),
    779: ("MCH_financial_statements_2018_consolidated|1455", 1, 1, 0),
    782: ("VNM_financial_statements_2016_consolidated|1238", 2, 2, 0),
    786: ("HND_financial_statements_2023|385", 2, 3, 1),
    803: ("MBB_financial_statements_2015_separate|1409", 2, 1, 0),
    815: ("OCB_financial_statements_2017_consolidated|206", 7, 3, 0),
    898: ("VRE_financial_statements_2024_separate|888", 7, 1, 2),
    930: ("PC1_financial_statements_2022_separate|924", 3, 1, 0),
    942: ("VPB_financial_statements_2020_separate|1783", 11, 1, 0),
    993: ("VGC_financial_statements_2022_separate|1610", 5, 1, 3),
    1001: ("NAB_financial_statements_2023_consolidated_1|2173", 10, 4, 4),
    1002: ("DPM_financial_statements_2016_separate|283", 8, 2, 3),
    1012: ("GAS_financial_statements_2017_consolidated|779", 16, 1, 4),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    with (ROOT / "build" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            catalog[str(row["table_ref"])] = row
    return catalog


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
                f"q{question_id}: coordinate mismatch in {table_ref}: {actual!r} != {expected!r}"
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
            "# Cross-check a manually audited equivalent disclosure used for retrieval recall.",
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
                "metric_key": "recall:equivalent_table_crosscheck",
                "source_table": table_ref,
                "source_csv": csv_name,
                "row_idx": str(row_idx),
                "col_idx": str(col_idx),
            }
        )
        operands.append(new_operand)
        staged_manifests[question_id] = (fieldnames, operands)

        row_label = next((value for value in frame.iloc[row_idx].tolist() if str(value).strip()), "")
        source_audit[question_id]["sources"].append(
            {
                "table_ref": table_ref,
                "csv": csv_name,
                "row": row_idx,
                "column": col_idx,
                "metric": "recall:equivalent_table_crosscheck",
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
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ordered_audit = [source_audit[int(row["id"])] for row in audit_list]
    (OUTPUT / "source_audit.json").write_text(
        json.dumps(ordered_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "balanced manually audited equivalent-table recall expansion",
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
    (OUTPUT / "balanced_retrieval_recall_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
