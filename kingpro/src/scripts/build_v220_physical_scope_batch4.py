"""Build v220 with the source-verified consolidated repair for q224."""

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
SOURCE = ROOT / "sub_top123_candidate_v219_internal_scope_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v220_physical_scope_batch4"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 224
NEW_TABLE = "HUT_financial_statements_2024_separate|1624"
NEW_DOCUMENT = "HUT_financial_statements_2024_separate"
NEW_ANSWER = 1200.50
SOURCE_CSV = ROOT / "build" / "tables" / NEW_DOCUMENT / "table_37_line1624.csv"

FIELDS = [
    "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
    "source_table", "source_csv", "row_idx", "col_idx",
]

QUERY = '''def _btc_number(x, typed_factor=1):
    if not isinstance(x, str):
        return float(x) * float(typed_factor)
    s = str(x).strip().replace(' ', ' ')
    if s in ('', '-'):
        return 0.0
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace('%', '').replace('$', '')
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        tail = s.split(',')[-1]
        s = s.replace(',', '.') if len(tail) <= 2 else s.replace(',', '')
    elif '.' in s and len(s.split('.')[-1]) == 3:
        s = s.replace('.', '')
    value = float(s)
    return -abs(value) if negative else value

df1 = list(dfs.values())[0]
v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])
result = float(round(v0 / 1e9, 2))'''


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(SOURCE_CSV)

    source_rows = load(SOURCE / "submission.json")
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}
    row = by_id[QID]
    row["answer"] = NEW_ANSWER
    row["pandas_query"] = QUERY
    row["relevant_docs"] = list(dict.fromkeys([*row["relevant_docs"], NEW_DOCUMENT]))
    row["relevant_tables"] = list(dict.fromkeys([*row["relevant_tables"], NEW_TABLE]))
    row["evidence"] = [{"variable": "df1", "csv_path": "data/q224_source_cells.csv"}]

    changed = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])]]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)

    manifest = {
        "ticker": "HUT", "year": "2024", "metric_key": "note:trade_payables_third_parties",
        "raw": "1.200.498.290.074", "typed_factor": "1.0", "scale": "1.0",
        "source_table": NEW_TABLE, "source_csv": SOURCE_CSV.name,
        "row_idx": "2", "col_idx": "1",
    }
    with (BUILDING / "data" / "q224_source_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(manifest)
    shutil.copy2(SOURCE_CSV, BUILDING / "data" / "HUT_financial_statements_2024_separate_1624.csv")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {str(QID): {"from": source_by_id[QID]["answer"], "to": NEW_ANSWER}},
        "physical_source": {
            "table": NEW_TABLE,
            "row": 2,
            "column": 1,
            "label": "Phải trả người bán là bên thứ ba",
            "raw": manifest["raw"],
            "scope": "BCTC hợp nhất vật lý trong container *_separate bị hoán đổi",
        },
        "old_retrieval_labels_retained": True,
        "source_submission_sha256": digest(SOURCE / "submission.json"),
        "claim_limit": "One source-verified answer repair; leaderboard effect is unmeasured.",
    }
    write(BUILDING / "v220_physical_scope_batch4_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = digest(OUTPUT / "submission.json")
    write(OUTPUT / "v220_physical_scope_batch4_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
