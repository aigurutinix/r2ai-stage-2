"""Build v222 with the source-verified all-fixed-assets repair for q931."""

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
SOURCE = ROOT / "sub_top123_candidate_v221_segment_scope_batch5"
OUTPUT = ROOT / "sub_top123_candidate_v222_total_fixed_assets_scope_batch6"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 931
OLD_TABLES = {
    "GEE_financial_statements_2025_separate|1003",
    "GEX_financial_statements_2025_separate|1309",
    "VGC_financial_statements_2025_separate|1241",
    "SAM_financial_statements_2025_separate|974",
    "PC1_financial_statements_2025_separate|1097",
}
TABLE_FILES = {
    "GEE_financial_statements_2025_separate|419": "table_1_line419.csv",
    "GEX_financial_statements_2025_separate|489": "table_2_line489.csv",
    "VGC_financial_statements_2025_separate|320": "table_5_line320.csv",
    "SAM_financial_statements_2025_separate|285": "table_3_line285.csv",
    "PC1_financial_statements_2025_separate|325": "table_1_line325.csv",
}
# row_idx is the pandas row index, so the physical CSV row is row_idx + 1.
CELLS = [
    # ticker, metric, raw, table, row_idx, col_idx
    ("GEE", "tangible_gross", "9.490.539.932", "GEE_financial_statements_2025_separate|419", 21, 4),
    ("GEE", "tangible_depreciation", "(7.148.669.440)", "GEE_financial_statements_2025_separate|419", 22, 4),
    ("GEE", "intangible_gross", "17.644.588.750", "GEE_financial_statements_2025_separate|419", 24, 4),
    ("GEE", "intangible_amortisation", "(8.011.819.418)", "GEE_financial_statements_2025_separate|419", 25, 4),
    ("GEX", "tangible_gross", "97.983.124.514", "GEX_financial_statements_2025_separate|489", 8, 4),
    ("GEX", "tangible_depreciation", "(40.304.826.813)", "GEX_financial_statements_2025_separate|489", 9, 4),
    ("GEX", "intangible_gross", "6.936.563.538", "GEX_financial_statements_2025_separate|489", 11, 4),
    ("GEX", "intangible_amortisation", "(1.110.025.020)", "GEX_financial_statements_2025_separate|489", 12, 4),
    ("VGC", "tangible_gross", "4.003.424.199.446", "VGC_financial_statements_2025_separate|320", 6, 4),
    ("VGC", "tangible_depreciation", "(2.692.553.107.516)", "VGC_financial_statements_2025_separate|320", 7, 4),
    ("VGC", "finance_lease_gross", "1.524.249.182", "VGC_financial_statements_2025_separate|320", 9, 4),
    ("VGC", "finance_lease_depreciation", "-", "VGC_financial_statements_2025_separate|320", 10, 4),
    ("VGC", "intangible_gross", "177.058.165.646", "VGC_financial_statements_2025_separate|320", 12, 4),
    ("VGC", "intangible_amortisation", "(46.006.504.368)", "VGC_financial_statements_2025_separate|320", 13, 4),
    ("SAM", "tangible_gross", "16.867.945.035", "SAM_financial_statements_2025_separate|285", 22, 3),
    ("SAM", "tangible_depreciation", "(15.168.351.394)", "SAM_financial_statements_2025_separate|285", 23, 3),
    ("SAM", "intangible_gross", "697.830.000", "SAM_financial_statements_2025_separate|285", 25, 3),
    ("SAM", "intangible_amortisation", "(697.830.000)", "SAM_financial_statements_2025_separate|285", 26, 3),
    ("PC1", "tangible_gross", "2.887.974.978.217", "PC1_financial_statements_2025_separate|325", 6, 3),
    ("PC1", "tangible_depreciation", "(1.001.599.075.979)", "PC1_financial_statements_2025_separate|325", 7, 3),
    ("PC1", "intangible_gross", "11.825.866.600", "PC1_financial_statements_2025_separate|325", 9, 3),
    ("PC1", "intangible_amortisation", "(1.603.465.229)", "PC1_financial_statements_2025_separate|325", 10, 3),
]
NEW_ANSWER = 57.15

QUERY = r'''def _number(value):
    s = str(value).strip()
    if s in ('', '-', 'nan'):
        return 0.0
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace('.', '').replace(',', '')
    number = float(s)
    return -abs(number) if negative else number

df1 = list(dfs.values())[0].copy()
df1['value'] = df1['raw'].map(_number)
df1['component'] = df1['metric'].map(
    lambda x: 'depreciation' if ('depreciation' in x or 'amortisation' in x) else 'gross'
)
totals = df1.groupby(['ticker', 'component'])['value'].sum().unstack(fill_value=0.0)
totals['ratio_pct'] = totals['depreciation'].abs() / totals['gross'] * 100
result = round(float(totals['ratio_pct'].mean()), 2)
result = float(result)'''


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def physical_path(table: str) -> Path:
    document = table.split("|", 1)[0]
    return ROOT / "build" / "tables" / document / TABLE_FILES[table]


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    source_rows = load(SOURCE / "submission.json")
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    source_by_id = {int(row["id"]): row for row in source_rows}
    row = {int(item["id"]): item for item in rows}[QID]
    row["answer"] = NEW_ANSWER
    row["relevant_tables"] = list(TABLE_FILES)
    row["evidence"] = [{"variable": "df1", "csv_path": "data/q931_source_cells.csv"}]
    row["pandas_query"] = QUERY

    changed = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])]]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)

    fields = ["ticker", "year", "metric", "raw", "source_table", "source_csv", "row_idx", "col_idx"]
    manifest_rows = []
    copied = set()
    for ticker, metric, raw, table, row_idx, col_idx in CELLS:
        source = physical_path(table)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            physical = list(csv.reader(handle))
        observed = physical[row_idx + 1][col_idx]
        if observed != raw:
            raise AssertionError(f"{table} [{row_idx},{col_idx}]: {observed!r} != {raw!r}")
        manifest_rows.append({
            "ticker": ticker, "year": "2025", "metric": metric, "raw": raw,
            "source_table": table, "source_csv": TABLE_FILES[table],
            "row_idx": row_idx, "col_idx": col_idx,
        })
        if table not in copied:
            shutil.copy2(source, BUILDING / "data" / (table.replace("|", "_") + ".csv"))
            copied.add(table)
    with (BUILDING / "data" / "q931_source_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {str(QID): {"from": source_by_id[QID]["answer"], "to": NEW_ANSWER}},
        "old_tangible_only_tables_removed": sorted(OLD_TABLES),
        "complete_fixed_asset_tables_added": list(TABLE_FILES),
        "per_company_percent": {
            "GEE": 55.8703407515316, "GEX": 39.4729078993011,
            "VGC": 65.4843443464858, "SAM": 90.3244027797604,
            "PC1": 34.5955668990541,
        },
        "source_submission_sha256": digest(SOURCE / "submission.json"),
        "claim_limit": "One source-verified scope repair; leaderboard effect is unmeasured.",
    }
    write(BUILDING / "v222_total_fixed_assets_scope_batch6_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = digest(OUTPUT / "submission.json")
    write(OUTPUT / "v222_total_fixed_assets_scope_batch6_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
