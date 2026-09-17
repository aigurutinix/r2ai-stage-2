"""Build rollbackable v276 from v274 with q638/q613/q627 source fixes."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_v274_q986_nan"
OUTPUT = ROOT / "sub_v276_q638_fix"
HEADER = ["ticker", "year", "metric_key", "raw", "typed_factor", "scale", "source_table", "source_csv", "row_idx", "col_idx"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def source(table_ref: str, csv_name: str, row: int, col: int, metric: str, raw: str, label: str) -> dict:
    return {"table_ref": table_ref, "csv": csv_name, "row": row, "column": col, "metric": metric, "label": label, "source_row_labels": [label], "scale": 1.0, "typed_factor": 1.0, "raw": raw}


def build() -> dict:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    temp = Path(mkdtemp(prefix="v276_q638_", dir=str(ROOT)))
    try:
        staged = temp / OUTPUT.name
        shutil.copytree(SOURCE, staged)
        baseline_sha = sha(SOURCE / "submission.json")
        submission = json.loads((staged / "submission.json").read_text(encoding="utf-8-sig"))
        by_id = {int(row["id"]): row for row in submission}
        if by_id[638]["answer"] != -56.62:
            raise AssertionError("unexpected q638 baseline answer")
        by_id[638]["answer"] = 5.74
        by_id[638]["relevant_tables"] = ["GEX_financial_statements_2025_separate|1801", "GEX_financial_statements_2022_separate|1417"]
        # Preserve query/docs/evidence declarations; only the compact manifest
        # is corrected to the lessee (thuê) table.
        write_csv(staged / "data" / "q638_source_cells.csv", [
            ["GEX", "2025", "note:operating_lease_due_within_one_year", "27.258.108.199", "1.0", "1.0", "GEX_financial_statements_2025_separate|1801", "GEX_financial_statements_2025_separate_1801.csv", "2", "1"],
            ["GEX", "2022", "note:operating_lease_due_within_one_year", "25.779.332.206", "1.0", "1.0", "GEX_financial_statements_2022_separate|1417", "GEX_financial_statements_2022_separate_1417.csv", "2", "1"],
        ])
        write_csv(staged / "data" / "q613_source_cells.csv", [
            ["IJC", "2022", "lctt:40", "(115.274.660.827)", "1.0", "1.0", "IJC_financial_statements_2022_separate|483", "IJC_financial_statements_2022_separate_483.csv", "8", "3"],
            ["IJC", "2018", "lctt:40", "(301.534.111.657)", "1.0", "1.0", "IJC_financial_statements_2018_separate|438", "IJC_financial_statements_2018_separate_438.csv", "8", "3"],
        ])
        write_csv(staged / "data" / "q627_source_cells.csv", [
            ["PVT", "2018", "note:deferred_tax_asset", "41.676.677.160", "1.0", "1.0", "PVT_financial_statements_2018_consolidated|949", "PVT_financial_statements_2018_consolidated_949.csv", "3", "1"],
            ["PVT", "2016", "note:deferred_tax_asset", "13.838.474.530", "1.0", "1.0", "PVT_financial_statements_2016_consolidated|952", "PVT_financial_statements_2016_consolidated_952.csv", "3", "1"],
        ])
        (staged / "submission.json").write_text(json.dumps(list(by_id.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        audit_path = staged / "source_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
        audit_by_id = {int(item["id"]): item for item in audit}
        if 613 in audit_by_id or 627 in audit_by_id:
            raise AssertionError("q613/q627 source audit unexpectedly present")
        audit_by_id[638]["answer"] = 5.74
        audit_by_id[638]["sources"][1] = source("GEX_financial_statements_2022_separate|1417", "GEX_financial_statements_2022_separate_1417.csv", 2, 1, "note:operating_lease_due_within_one_year", "25.779.332.206", "Cam ket thue hoat dong - Tu 1 nam tro xuong")
        audit_by_id[613] = {"id": 613, "old_answer": 186259.45, "answer": 186259.45, "note": "IJC net financing cash-flow difference, 2022 versus 2018, million VND", "sources": [source("IJC_financial_statements_2022_separate|483", "IJC_financial_statements_2022_separate_483.csv", 8, 3, "lctt:40", "(115.274.660.827)", "Net cash flow from financing activities"), source("IJC_financial_statements_2018_separate|438", "IJC_financial_statements_2018_separate_438.csv", 8, 3, "lctt:40", "(301.534.111.657)", "Net cash flow from financing activities")]}
        audit_by_id[627] = {"id": 627, "old_answer": 27838.2, "answer": 27838.2, "note": "PVT deferred-tax asset difference at year-end 2018 versus 2016, million VND", "sources": [source("PVT_financial_statements_2018_consolidated|949", "PVT_financial_statements_2018_consolidated_949.csv", 3, 1, "note:deferred_tax_asset", "41.676.677.160", "Deferred tax asset"), source("PVT_financial_statements_2016_consolidated|952", "PVT_financial_statements_2016_consolidated_952.csv", 3, 1, "note:deferred_tax_asset", "13.838.474.530", "Deferred tax asset")]}
        audit_path.write_text(json.dumps(sorted(audit_by_id.values(), key=lambda item: int(item["id"])), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if sha(staged / "submission.json") == baseline_sha:
            raise AssertionError("q638 answer change did not persist")
        (staged / "v276_q638_fix_audit.json").write_text(json.dumps({"candidate": OUTPUT.name, "baseline": SOURCE.name, "answer_changes": [{"id": 638, "from": -56.62, "to": 5.74}], "lineage_only_ids": [613, 627], "changed_submission_fields": ["q638.answer", "q638.relevant_tables"], "query_docs_evidence_unchanged": True, "automatic_promotion": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(staged), str(OUTPUT))
        return {"candidate": OUTPUT.name, "baseline": SOURCE.name, "answer_changes": [{"id": 638, "from": -56.62, "to": 5.74}], "lineage_only_ids": [613, 627], "submission_baseline_sha256": baseline_sha, "submission_candidate_sha256": sha(OUTPUT / "submission.json")}
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
