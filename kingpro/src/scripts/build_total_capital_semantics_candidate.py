"""Build a v207 component binding ``tổng nguồn vốn`` to code 440 rows.

The accounting identity normally makes total assets (270) equal total capital
and liabilities (440), which can hide a semantic source error.  q617, q673 and
q853 explicitly ask for ``tổng nguồn vốn`` but v206 reads total-assets rows.
This builder switches all six denominator/source operands to the requested
code-440 rows.  q617 also changes numerically because the extracted GAS rows
are not byte-identical; q673/q853 are provenance-only repairs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v206_semantic_batch11"
OUTPUT = ROOT / "sub_top123_candidate_v207_total_capital_semantics_component"

RULES = {
    617: {
        "old_answer": 10.40,
        "new_answer": 10.34,
        "tables": [
            "GAS_financial_statements_2018_consolidated|280",
            "GAS_financial_statements_2015_consolidated|339",
        ],
        "replacements": {
            0: {
                "old_table": "GAS_financial_statements_2018_consolidated|253",
                "table": "GAS_financial_statements_2018_consolidated|280",
                "source_csv": "table_4_line280.csv",
                "row_idx": "32",
                "col_idx": "3",
                "old_raw": "62.614.420.245.293",
                "raw": "62.614.410.245.293",
                "label": "TỔNG CỘNG NGUỒN VỐN (440=300+400)",
            },
            1: {
                "old_table": "GAS_financial_statements_2015_consolidated|297",
                "table": "GAS_financial_statements_2015_consolidated|339",
                "source_csv": "table_4_line339.csv",
                "row_idx": "32",
                "col_idx": "3",
                "old_raw": "56.714.606.287.288",
                "raw": "56.746.605.287.240",
                "label": "TỔNG CỘNG NGUỒN VỐN (440=300+400)",
            },
        },
    },
    673: {
        "old_answer": 39.19,
        "new_answer": 39.19,
        "tables": [
            "SCR_financial_statements_2025_separate|248",
            "SCR_financial_statements_2025_separate|299",
        ],
        "replacements": {
            1: {
                "old_table": "SCR_financial_statements_2025_separate|265",
                "table": "SCR_financial_statements_2025_separate|299",
                "source_csv": "table_6_line299.csv",
                "row_idx": "11",
                "col_idx": "3",
                "old_raw": "11.075.021.385.794",
                "raw": "11.075.021.385.794",
                "label": "TỔNG CỘNG NGUỒN VỐN",
            },
        },
    },
    853: {
        "old_answer": 66.39,
        "new_answer": 66.39,
        "tables": [
            "DNH_financial_statements_2022_separate|273",
            "GEG_financial_statements_2022_separate|275",
            "POW_financial_statements_2022_separate|304",
        ],
        "replacements": {
            1: {
                "old_table": "DNH_financial_statements_2022_separate|255",
                "table": "DNH_financial_statements_2022_separate|273",
                "source_csv": "table_2_line273.csv",
                "row_idx": "22",
                "col_idx": "3",
                "old_raw": "8.256.582.932.950",
                "raw": "8.256.582.932.950",
                "label": "TỔNG CỘNG NGUỒN VỐN(440 = 300 + 400)",
            },
            3: {
                "old_table": "GEG_financial_statements_2022_separate|260",
                "table": "GEG_financial_statements_2022_separate|275",
                "source_csv": "table_5_line275.csv",
                "row_idx": "26",
                "col_idx": "3",
                "old_raw": "7.017.287.244.652",
                "raw": "7.017.287.244.652",
                "label": "TỔNG NGUỒN VỐN",
            },
            5: {
                "old_table": "POW_financial_statements_2022_separate|275",
                "table": "POW_financial_statements_2022_separate|304",
                "source_csv": "table_3_line304.csv",
                "row_idx": "23",
                "col_idx": "4",
                "old_raw": "46.106.801.970.338",
                "raw": "46.106.801.970.338",
                "label": "TỔNG CỘNG NGUỒN VỐN(440=300+400)",
            },
        },
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_source_row(row: dict[str, str], replacement: dict[str, str]) -> None:
    if row.get("source_table") != replacement["old_table"] or row.get("raw") != replacement["old_raw"]:
        raise AssertionError(f"unexpected old source binding: {row}")
    row.update(
        {
            "metric_key": "cdkt:440",
            "raw": replacement["raw"],
            "source_table": replacement["table"],
            "source_csv": replacement["source_csv"],
            "row_idx": replacement["row_idx"],
            "col_idx": replacement["col_idx"],
        }
    )


def patch_manifest(path: Path, replacements: dict[int, dict[str, str]]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, replacement in replacements.items():
        patch_source_row(rows[index], replacement)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return [rows[index] for index in sorted(replacements)]


def patch_audit_source(source: dict, replacement: dict[str, str]) -> None:
    if source.get("table_ref") != replacement["old_table"] or source.get("raw") != replacement["old_raw"]:
        raise AssertionError(f"unexpected old source-audit binding: {source}")
    source.update(
        {
            "table_ref": replacement["table"],
            "csv": replacement["source_csv"],
            "row": int(replacement["row_idx"]),
            "column": int(replacement["col_idx"]),
            "metric": "cdkt:440",
            "label": replacement["label"],
            "source_row_labels": [replacement["label"]],
            "raw": replacement["raw"],
        }
    )


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    old_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    by_id = {int(row["id"]): row for row in rows}
    audit_rows = json.loads((OUTPUT / "source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(row["id"]): row for row in audit_rows}
    changes = []

    for qid, rule in RULES.items():
        row = by_id[qid]
        if float(row["answer"]) != float(rule["old_answer"]):
            raise AssertionError(f"q{qid}: unexpected old answer {row['answer']}")
        old_tables = list(row.get("relevant_tables") or [])
        row["answer"] = float(rule["new_answer"])
        row["relevant_tables"] = list(rule["tables"])
        corrected = patch_manifest(OUTPUT / "data" / f"q{qid}_source_cells.csv", rule["replacements"])

        audit = audit_by_id[qid]
        if float(audit.get("answer")) != float(rule["old_answer"]):
            raise AssertionError(f"q{qid}: unexpected source-audit answer")
        for index, replacement in rule["replacements"].items():
            patch_audit_source(audit["sources"][index], replacement)
        audit["old_answer"] = float(rule["old_answer"])
        audit["answer"] = float(rule["new_answer"])
        audit["note"] = "Bind explicit total-capital wording to the physical total nguồn vốn (code 440) row."
        changes.append(
            {
                "id": qid,
                "old_answer": rule["old_answer"],
                "new_answer": rule["new_answer"],
                "old_relevant_tables": old_tables,
                "new_relevant_tables": rule["tables"],
                "corrected_sources": corrected,
            }
        )

    write_json(OUTPUT / "submission.json", rows)
    write_json(OUTPUT / "source_audit.json", audit_rows)

    changed_ids = [int(row["id"]) for row in rows if row != old_by_id[int(row["id"])]]
    if changed_ids != sorted(RULES):
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    allowed = {"answer", "relevant_tables"}
    for qid in RULES:
        changed_fields = {
            key for key in set(by_id[qid]) | set(old_by_id[qid])
            if by_id[qid].get(key) != old_by_id[qid].get(key)
        }
        expected = {"relevant_tables"} | ({"answer"} if qid == 617 else set())
        if changed_fields != expected or not changed_fields <= allowed:
            raise AssertionError(f"q{qid}: unexpected changed fields {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(RULES),
        "answer_changed_ids": [617],
        "provenance_only_ids": [673, 853],
        "source_binding_changes": 6,
        "changes": changes,
        "recomputation_q617": {
            "beginning_2019_total_capital": 62_614_410_245_293,
            "beginning_2016_total_capital": 56_746_605_287_240,
            "growth_percent": 10.340362966826547,
            "rounded_answer": 10.34,
        },
        "invariants": {
            "only_three_submission_rows_changed": True,
            "questions_unchanged": True,
            "relevant_docs_unchanged": True,
            "evidence_paths_unchanged": True,
            "pandas_queries_unchanged": True,
            "q617_answer_recomputed_from_code_440": True,
            "q673_q853_answers_unchanged": True,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Exact-source component; hold for a larger v207 bundle and full release gate.",
    }
    write_json(OUTPUT / "total_capital_semantics_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
