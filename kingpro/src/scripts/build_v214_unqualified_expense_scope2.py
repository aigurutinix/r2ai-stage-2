"""Build two source-backed table-scope repairs on cumulative v213.

The questions name a generic expense element but do not qualify it as selling
or administrative expense.  Same-template controls q22 and q343 use the
``chi phi san xuat kinh doanh theo yeu to`` disclosure.  q49 and q216 instead
selected the first same-label row from narrower selling/admin disclosures.

Only the two reviewed questions are changed.  v206/v207 and v213 remain
immutable, and the output is a separate ablation candidate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v213_attribute_hardened"
OUTPUT = ROOT / "sub_top123_candidate_v214_scope2_ablation"

PATCHES = {
    49: {
        "old_answer": 63.5,
        "new_answer": 1240.15,
        "old_table": "PC1_financial_statements_2018_consolidated|1534",
        "new_table": "PC1_financial_statements_2018_consolidated|1592",
        "old_code": "_r = df1[df1['0'].str.contains('Chi phí dịch vụ mua ngoài', case=False, na=False, regex=False)]",
        "new_code": "_r = df2[df2['0'].str.contains('Chi phí dịch vụ mua ngoài', case=False, na=False, regex=False)]",
        "source_value": "1.240.153.428.059",
        "source_section": "37. CHI PHÍ SẢN XUẤT KINH DOANH THEO YẾU TỐ",
    },
    216: {
        "old_answer": 49.07,
        "new_answer": 77.96,
        "old_table": "GEG_financial_statements_2019_separate|1055",
        "new_table": "GEG_financial_statements_2019_separate|1101",
        "source_value": "77.961.799.614",
        "source_section": "28 CHI PHÍ SẢN XUẤT KINH DOANH THEO YẾU TỐ",
    },
}


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}

    # q49 already carries both raw source tables as evidence.  Select the
    # general expense-element table instead of the selling-expense table.
    q49 = by_id[49]
    p49 = PATCHES[49]
    if float(q49["answer"]) != p49["old_answer"]:
        raise AssertionError(f"q49 unexpected old answer: {q49['answer']}")
    if q49["relevant_tables"] != [p49["old_table"]]:
        raise AssertionError(f"q49 unexpected old tables: {q49['relevant_tables']}")
    if str(q49["pandas_query"]).count(p49["old_code"]) != 1:
        raise AssertionError("q49 old selector not unique")
    q49["answer"] = p49["new_answer"]
    q49["relevant_tables"] = [p49["new_table"]]
    q49["pandas_query"] = str(q49["pandas_query"]).replace(
        p49["old_code"], p49["new_code"]
    )

    # q216 uses the compact source-cell evidence format.  Retarget that one
    # row to the general expense-element disclosure and preserve its parser.
    q216 = by_id[216]
    p216 = PATCHES[216]
    if float(q216["answer"]) != p216["old_answer"]:
        raise AssertionError(f"q216 unexpected old answer: {q216['answer']}")
    if q216["relevant_tables"] != [p216["old_table"]]:
        raise AssertionError(f"q216 unexpected old tables: {q216['relevant_tables']}")
    q216["answer"] = p216["new_answer"]
    q216["relevant_tables"] = [p216["new_table"]]

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    if changed != sorted(PATCHES):
        raise AssertionError(f"unexpected changed IDs: {changed}")
    expected_fields = {
        49: {"answer", "relevant_tables", "pandas_query"},
        216: {"answer", "relevant_tables"},
    }
    for qid in changed:
        fields = {
            key
            for key in by_id[qid]
            if by_id[qid].get(key) != source_by_id[qid].get(key)
        }
        if fields != expected_fields[qid]:
            raise AssertionError(f"q{qid} unexpected changed fields: {fields}")

    shutil.copytree(SOURCE, OUTPUT)
    _write_json(OUTPUT / "submission.json", rows)

    source_cells_path = OUTPUT / "data" / "q216_source_cells.csv"
    old_cells = source_cells_path.read_text(encoding="utf-8")
    replacements = {
        "note:q216_employee_expense_current_year": "note:q216_general_employee_expense_current_year",
        "49.069.513.900": p216["source_value"],
        "GEG_financial_statements_2019_separate|1055": p216["new_table"],
        "GEG_financial_statements_2019_separate_1055.csv": "GEG_financial_statements_2019_separate_1101.csv",
        ",1,1\n": ",2,1\n",
    }
    new_cells = old_cells
    for old, new in replacements.items():
        if new_cells.count(old) != 1:
            raise AssertionError(f"q216 evidence token not unique: {old!r}")
        new_cells = new_cells.replace(old, new)
    source_cells_path.write_text(new_cells, encoding="utf-8")

    audits = json.loads((OUTPUT / "source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in audits}
    updated_audit_ids: list[int] = []
    for qid, patch in PATCHES.items():
        item = audit_by_id.get(qid)
        if item is None:
            continue
        if float(item["answer"]) != patch["old_answer"]:
            raise AssertionError(f"q{qid} unexpected source-audit answer")
        item["answer"] = patch["new_answer"]
        item["note"] = (
            "Unqualified expense element resolved to the general "
            f"SXKD-by-element disclosure: {patch['new_table']} ({patch['source_section']})."
        )
        updated_audit_ids.append(qid)
    _write_json(OUTPUT / "source_audit.json", audits)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(PATCHES),
        "repairs": {str(qid): patch for qid, patch in PATCHES.items()},
        "source_audit_rows_updated": updated_audit_ids,
        "internal_controls": {
            "q22": "unqualified chi phí nhân công -> SXKD theo yếu tố",
            "q343": "unqualified chi phí nguyên liệu, vật liệu -> SXKD theo yếu tố",
            "q648": "retained: no general table and exact selling-row label",
            "q773": "retained: already uses general KBC disclosure plus VRE source",
            "q789": "retained: prompt says tổng and correctly sums selling + admin",
            "q923": "retained: multi-year source-cell lineage already consistent",
        },
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "claim_limit": "Source-backed ablation; public effect unknown until submitted.",
    }
    _write_json(OUTPUT / "v214_unqualified_expense_scope2_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
