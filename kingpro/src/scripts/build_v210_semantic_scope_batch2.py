"""Build v210 from v209 with two independently source-proven scope repairs.

q769 replaces an ACV investment-property cell with the like-for-like
intangible-PPE land-use-rights cell.  q1003 replaces MSR's consolidated/group
2015 cash-flow column with the parent-company 2015 column.  The v209 lineage
is copied and never modified in place.
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
SOURCE = ROOT / "sub_top123_candidate_v209_q15_board_role_r2"
OUTPUT = ROOT / "sub_top123_candidate_v210_semantic_scope_batch2"

Q769_OLD = 26.89
Q769_NEW = 31.29
VSC_REF = "VSC_financial_statements_2015_consolidated|846"
ACV_OLD_REF = "ACV_financial_statements_2015_consolidated|1641"
ACV_REF = "ACV_financial_statements_2015_consolidated|1571"
ACV_NAME = "ACV_financial_statements_2015_consolidated_1571.csv"
ACV_PHYSICAL = ROOT / "build" / "tables" / "ACV_financial_statements_2015_consolidated" / "table_22_line1571.csv"

Q1003_OLD = 352.70
Q1003_NEW = -106.07
MSR_REF = "MSR_financial_statements_2015_separate|271"
MSR_NAME = "MSR_financial_statements_2015_separate_271.csv"
MSR_PHYSICAL = ROOT / "build" / "tables" / "MSR_financial_statements_2015_separate" / "table_5_line271.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_manifest(path: Path, ticker: str, expected: dict[str, str], updates: dict[str, str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["ticker"] == ticker)
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"unexpected {path.name} {ticker} {key}: {row[key]!r}")
    row.update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patch_audits(path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))

    q769 = next(item for item in records if int(item["id"]) == 769)
    if float(q769["answer"]) != Q769_OLD:
        raise AssertionError("unexpected q769 source-audit answer")
    q769["answer"] = Q769_NEW
    q769["note"] = "Like-for-like VSC/ACV intangible-PPE ending land-use-rights NBV difference, VND billion"
    acv = next(item for item in q769["sources"] if item["table_ref"] == ACV_OLD_REF)
    acv.update(
        {
            "table_ref": ACV_REF,
            "csv": ACV_NAME,
            "row": 12,
            "column": 1,
            "label": "Intangible-PPE land-use-rights ending NBV",
            "source_row_labels": ["GIÁ TRỊ CÒN LẠI", "31/12/2015"],
            "raw": "36.648.986.835",
        }
    )

    q1003 = next(item for item in records if int(item["id"]) == 1003)
    if float(q1003["answer"]) != Q1003_OLD:
        raise AssertionError("unexpected q1003 source-audit answer")
    q1003["answer"] = Q1003_NEW
    q1003["note"] = (
        "total parent-company MSR/HPG/AAA/DCM net operating cash flow, VND billion; "
        "MSR parent-company 2015 is column Công ty/2015 and is in thousand VND"
    )
    msr = next(item for item in q1003["sources"] if item["table_ref"] == MSR_REF)
    if msr["raw"] != "41.706.084" or int(msr["column"]) != 2:
        raise AssertionError("unexpected old q1003 MSR binding")
    msr.update(
        {
            "csv": MSR_NAME,
            "row": 19,
            "column": 4,
            "label": "Lưu chuyển tiền thuần từ hoạt động kinh doanh — Công ty 2015",
            "source_row_labels": ["Lưu chuyển tiền thuần từ hoạt động kinh doanh", "Công ty", "2015"],
            "raw": "(417.067.012)",
        }
    )
    write_json(path, records)


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for required in (SOURCE, ACV_PHYSICAL, MSR_PHYSICAL):
        if not required.exists():
            raise FileNotFoundError(required)
    if "36.648.986.835" not in ACV_PHYSICAL.read_text(encoding="utf-8-sig"):
        raise AssertionError("ACV physical source token moved")
    msr_text = MSR_PHYSICAL.read_text(encoding="utf-8-sig")
    if "(417.067.012)" not in msr_text or "Công ty" not in msr_text:
        raise AssertionError("MSR parent-company source token/header moved")

    shutil.copytree(SOURCE, OUTPUT)
    shutil.copy2(ACV_PHYSICAL, OUTPUT / "data" / ACV_NAME)
    shutil.copy2(MSR_PHYSICAL, OUTPUT / "data" / MSR_NAME)

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source_rows}
    by_id = {int(item["id"]): item for item in rows}

    q769 = by_id[769]
    if float(q769["answer"]) != Q769_OLD or q769["relevant_tables"] != [VSC_REF, ACV_OLD_REF]:
        raise AssertionError("unexpected submitted v209 q769 state")
    q769["answer"] = Q769_NEW
    q769["relevant_tables"] = [VSC_REF, ACV_REF]

    q1003 = by_id[1003]
    if float(q1003["answer"]) != Q1003_OLD or MSR_REF not in q1003["relevant_tables"]:
        raise AssertionError("unexpected submitted v209 q1003 state")
    q1003["answer"] = Q1003_NEW
    write_json(OUTPUT / "submission.json", rows)

    patch_manifest(
        OUTPUT / "data" / "q769_source_cells.csv",
        "ACV",
        {"source_table": ACV_OLD_REF, "raw": "32.243.749.055", "row_idx": "9", "col_idx": "4"},
        {"raw": "36.648.986.835", "source_table": ACV_REF, "source_csv": ACV_NAME, "row_idx": "12", "col_idx": "1"},
    )
    patch_manifest(
        OUTPUT / "data" / "q1003_source_cells.csv",
        "MSR",
        {"source_table": MSR_REF, "raw": "41.706.084", "row_idx": "19", "col_idx": "2"},
        {"raw": "(417.067.012)", "source_csv": MSR_NAME, "row_idx": "19", "col_idx": "4"},
    )
    patch_audits(OUTPUT / "source_audit.json")

    changed_ids = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])]]
    if changed_ids != [769, 1003]:
        raise AssertionError(f"unexpected changed IDs: {changed_ids}")
    if {key for key in q769 if q769.get(key) != source_by_id[769].get(key)} != {"answer", "relevant_tables"}:
        raise AssertionError("unexpected q769 changed fields")
    if {key for key in q1003 if q1003.get(key) != source_by_id[1003].get(key)} != {"answer"}:
        raise AssertionError("unexpected q1003 changed fields")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [769, 1003],
        "repairs": {
            "769": {"old_answer": Q769_OLD, "new_answer": Q769_NEW, "reason": "like-for-like asset class"},
            "1003": {"old_answer": Q1003_OLD, "new_answer": Q1003_NEW, "reason": "parent-company rather than group column"},
        },
        "recomputation": {
            "q769_vnd": {"VSC": 5_355_027_273, "ACV": 36_648_986_835, "rounded_abs_difference_billion": Q769_NEW},
            "q1003_vnd": {"MSR": -417_067_012_000, "HPG": -45_061_179_652, "AAA": 29_663_853_542, "DCM": 326_395_298_043, "rounded_sum_billion": Q1003_NEW},
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Two independent exact-source repairs; public effect unknown until submitted.",
    }
    write_json(OUTPUT / "v210_semantic_scope_batch2_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
