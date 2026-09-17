"""Build v206: eleven source-verified semantic and provenance repairs over v205.

Six questions change answers after a direct source recheck.  Five additional
questions keep their answers but replace component/note evidence with the cash
flow statement row whose wording exactly matches the question.  Every changed
program reads a compact runtime manifest tied back to a physical BTC table cell.
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
SOURCE = ROOT / "sub_top123_candidate_v205_q118_vcb_general_provision"
OUTPUT = ROOT / "sub_top123_candidate_v206_semantic_batch11"


def source(table_ref: str, row: int, column: int, metric: str, label: str,
           ticker: str, year: int, typed_factor: float = 1.0) -> dict:
    return {
        "table_ref": table_ref,
        "row": row,
        "column": column,
        "metric": metric,
        "label": label,
        "ticker": ticker,
        "year": year,
        "typed_factor": typed_factor,
        "scale": 1.0,
    }


FIXES = {
    11: {
        "old": 21_704_473.0,
        "new": 39_849_011.0,
        "note": "BID consolidated 2016 deposits at other credit institutions; not the cash-equivalent subset.",
        "sources": [source(
            "BID_financial_statements_2016_consolidated|979", 2, 1,
            "note:deposits_at_other_credit_institutions", "Tiền gửi tại các TCTD khác",
            "BID", 2016,
        )],
        "formula": "direct_million",
    },
    50: {
        "old": 187_902.0,
        "new": 2_720_958.0,
        "note": "ABB parent 2023 gross VAMC special-bond balance; not its provision.",
        "sources": [source(
            "ABB_financial_statements_2023_separate|1538", 1, 1,
            "note:vamc_special_bond_gross", "Trái phiếu đặc biệt do VAMC phát hành",
            "ABB", 2023,
        )],
        "formula": "direct_million",
    },
    69: {
        "old": 66_076_449.0,
        "new": 259_236_746.0,
        "note": "SHB consolidated 2019 customer deposits, Total column; not the due-within-one-month bucket.",
        "sources": [source(
            "SHB_financial_statements_2019_consolidated|2372", 17, 8,
            "note:customer_deposits_total", "Tiền gửi của khách hàng - Tổng cộng",
            "SHB", 2019,
        )],
        "formula": "direct_million",
    },
    131: {
        "old": 24.08,
        "new": 24.08,
        "note": "Answer-neutral provenance repair to the exact 2025 ending-cash cash-flow row.",
        "sources": [source(
            "HUT_financial_statements_2025_separate|400", 10, 4,
            "lctt:70", "Tiền và tương đương tiền cuối năm",
            "HUT", 2025,
        )],
        "formula": "direct_billion",
    },
    154: {
        "old": 3_111.62,
        "new": 3_111.62,
        "note": "Answer-neutral provenance repair to the exact 2022 opening-cash cash-flow row.",
        "sources": [source(
            "NLG_financial_statements_2022_consolidated|295", 8, 3,
            "lctt:60", "Tiền và tương đương tiền đầu năm",
            "NLG", 2022,
        )],
        "formula": "direct_billion",
    },
    177: {
        "old": 40_802.32,
        "new": 40_802.32,
        "note": "Answer-neutral provenance repair from a net-debt subtraction row to the exact ending-cash cash-flow row.",
        "sources": [source(
            "FIT_financial_statements_2018_consolidated|369", 39, 2,
            "lctt:70", "Tiền và tương đương tiền cuối kỳ",
            "FIT", 2018,
        )],
        "formula": "direct_vnd_to_million",
    },
    197: {
        "old": 944_860.0,
        "new": 877_765.0,
        "note": "VIB consolidated 2018 ending total customer-loan provision; not the prior-year opening balance.",
        "sources": [source(
            "VIB_financial_statements_2018_consolidated|1066", 7, 3,
            "note:customer_loan_provision_total_ending", "Số dư cuối năm - Tổng cộng",
            "VIB", 2018, typed_factor=1000.0,
        )],
        "formula": "direct_million",
    },
    278: {
        "old": 1.74,
        "new": 1.74,
        "note": "Answer-neutral provenance repair to the exact 2023 ending-cash cash-flow row.",
        "sources": [source(
            "VIF_financial_statements_2023_consolidated|378", 10, 3,
            "lctt:70", "Tiền và tương đương tiền cuối năm",
            "VIF", 2023,
        )],
        "formula": "direct_hundred_billion",
    },
    337: {
        "old": 81.57,
        "new": 81.57,
        "note": "Answer-neutral provenance repair to the exact 2023 ending-cash cash-flow row.",
        "sources": [source(
            "BAF_financial_statements_2023_separate|322", 9, 3,
            "lctt:70", "Tiền và các khoản tương đương tiền cuối năm",
            "BAF", 2023,
        )],
        "formula": "direct_billion",
    },
    639: {
        "old": -61.48,
        "new": -8.03,
        "note": "PC1 construction-activity WIP growth, using the qualified construction row rather than broad WIP.",
        "sources": [
            source(
                "PC1_financial_statements_2020_consolidated|1199", 7, 1,
                "note:construction_wip_ending_2020", "Hoạt động xây lắp - 31/12/2020",
                "PC1", 2020,
            ),
            source(
                "PC1_financial_statements_2020_consolidated|1199", 7, 2,
                "note:construction_wip_opening_2020", "Hoạt động xây lắp - 01/01/2020",
                "PC1", 2020,
            ),
        ],
        "formula": "growth_percent",
    },
}


PARSER = """def _btc_number(x, typed_factor=1):
    if not isinstance(x, str):
        return float(x) * float(typed_factor)
    s = str(x).strip().replace('\u00a0', ' ')
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
        s = s.replace(',', '.' if len(tail) <= 2 else '')
    elif '.' in s:
        tail = s.split('.')[-1]
        if len(tail) == 3:
            s = s.replace('.', '')
    value = float(s)
    return -abs(value) if negative else value

df1 = list(dfs.values())[0]
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def physical_csv(table_ref: str) -> Path:
    document, line = table_ref.rsplit("|", 1)
    matches = sorted((ROOT / "build" / "tables" / document).glob(f"*line{line}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{table_ref}: expected one physical CSV, found {len(matches)}")
    return matches[0]


def read_source_cell(spec: dict) -> str:
    path = physical_csv(spec["table_ref"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    row = int(spec["row"]) + 1  # first physical row is the CSV header
    column = int(spec["column"])
    try:
        return str(rows[row][column]).strip()
    except IndexError as exc:
        raise IndexError(f"{spec['table_ref']} row={row - 1} col={column}") from exc


def q588_sources() -> list[dict]:
    result = []
    for year, table_ref in (
        (2023, "SAB_financial_statements_2023_separate|1112"),
        (2019, "SAB_financial_statements_2019_separate|913"),
    ):
        path = physical_csv(table_ref)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))[1:]
        for row_index, row in enumerate(rows):
            if len(row) < 2:
                continue
            raw = row[1].strip()
            if not raw or raw == "-" or not raw.replace(".", "").isdigit():
                continue
            result.append(source(
                table_ref, row_index, 1,
                f"note:related_party_other_short_term_receivable:{year}:{row_index}",
                row[0].strip(), "SAB", year,
            ))
    return result


def query(formula: str) -> str:
    if formula == "direct_million":
        body = "v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])\nresult = round(v0, 2)"
    elif formula == "direct_billion":
        body = "v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])\nresult = round(v0 / 1e9, 2)"
    elif formula == "direct_vnd_to_million":
        body = "v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])\nresult = round(v0 / 1e6, 2)"
    elif formula == "direct_hundred_billion":
        body = "v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])\nresult = round(v0 / 1e11, 2)"
    elif formula == "growth_percent":
        body = "v0 = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])\nv1 = _btc_number(df1.iloc[1]['raw'], df1.iloc[1]['typed_factor']) * float(df1.iloc[1]['scale'])\nresult = round((v0 / v1 - 1) * 100, 2)"
    elif formula == "q588_difference":
        body = """_values = df1.apply(lambda row: _btc_number(row['raw'], row['typed_factor']) * float(row['scale']), axis=1)
_years = df1['year'].astype(int)
_sum_2023 = float(_values[_years == 2023].sum())
_sum_2019 = float(_values[_years == 2019].sum())
result = round(abs(_sum_2023 - _sum_2019) / 1e9, 2)"""
    else:
        raise KeyError(formula)
    return PARSER + body + "\n"


def write_manifest(question_id: int, specs: list[dict]) -> list[dict]:
    fieldnames = (
        "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
        "source_table", "source_csv", "row_idx", "col_idx",
    )
    manifest_rows = []
    audit_sources = []
    for spec in specs:
        raw = read_source_cell(spec)
        source_csv = physical_csv(spec["table_ref"]).name
        manifest_rows.append({
            "ticker": spec["ticker"],
            "year": spec["year"],
            "metric_key": spec["metric"],
            "raw": raw,
            "typed_factor": spec["typed_factor"],
            "scale": spec["scale"],
            "source_table": spec["table_ref"],
            "source_csv": source_csv,
            "row_idx": spec["row"],
            "col_idx": spec["column"],
        })
        audit_sources.append({
            "table_ref": spec["table_ref"],
            "csv": source_csv,
            "row": spec["row"],
            "column": spec["column"],
            "metric": spec["metric"],
            "label": spec["label"],
            "source_row_labels": [spec["label"]],
            "scale": spec["scale"],
            "typed_factor": spec["typed_factor"],
            "raw": raw,
        })
    path = OUTPUT / "data" / f"q{question_id}_source_cells.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    return audit_sources


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)

    fixes = dict(FIXES)
    fixes[588] = {
        "old": 16.53,
        "new": 108.61,
        "note": "SAB parent related-party other short-term receivables: sum every disclosed counterparty in 2023 and 2019, then take the absolute difference.",
        "sources": q588_sources(),
        "formula": "q588_difference",
    }

    shutil.copytree(SOURCE, OUTPUT)
    old_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    by_id = {int(row["id"]): row for row in rows}

    source_audit_path = OUTPUT / "source_audit.json"
    audit_rows = json.loads(source_audit_path.read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in audit_rows}

    fix_reports = {}
    for question_id in sorted(fixes):
        fix = fixes[question_id]
        row = by_id[question_id]
        if float(row["answer"]) != float(fix["old"]):
            raise AssertionError(f"q{question_id} unexpected old answer {row['answer']}")
        old_tables = list(row.get("relevant_tables", []))
        tables = list(dict.fromkeys(spec["table_ref"] for spec in fix["sources"]))
        audit_sources = write_manifest(question_id, fix["sources"])

        row["answer"] = fix["new"]
        row["relevant_tables"] = tables
        row["evidence"] = [{"variable": "df1", "csv_path": f"data/q{question_id}_source_cells.csv"}]
        row["pandas_query"] = query(fix["formula"])

        audit_by_id[question_id] = {
            "id": question_id,
            "old_answer": fix["old"],
            "answer": fix["new"],
            "note": fix["note"],
            "sources": audit_sources,
        }
        fix_reports[str(question_id)] = {
            "old_answer": fix["old"],
            "new_answer": fix["new"],
            "old_relevant_tables": old_tables,
            "new_relevant_tables": tables,
            "source_count": len(audit_sources),
            "note": fix["note"],
        }

    write_json(OUTPUT / "submission.json", rows)
    ordered_audit = [audit_by_id[int(row["id"])] for row in rows if int(row["id"]) in audit_by_id]
    write_json(source_audit_path, ordered_audit)

    changed_ids = [int(row["id"]) for row in rows if row != old_by_id[int(row["id"])]]
    expected_ids = sorted(fixes)
    if changed_ids != expected_ids:
        raise AssertionError(f"unexpected changed IDs: {changed_ids}; expected {expected_ids}")
    allowed = {"answer", "relevant_tables", "evidence", "pandas_query"}
    for question_id in expected_ids:
        changed_fields = {
            key for key in set(by_id[question_id]) | set(old_by_id[question_id])
            if by_id[question_id].get(key) != old_by_id[question_id].get(key)
        }
        if not changed_fields <= allowed or "pandas_query" not in changed_fields:
            raise AssertionError(f"q{question_id} unexpected fields: {sorted(changed_fields)}")
        if float(fixes[question_id]["old"]) != float(fixes[question_id]["new"]) and "answer" not in changed_fields:
            raise AssertionError(f"q{question_id} answer was expected to change")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": expected_ids,
        "answer_changed_ids": [qid for qid in expected_ids if fixes[qid]["old"] != fixes[qid]["new"]],
        "provenance_only_ids": [qid for qid in expected_ids if fixes[qid]["old"] == fixes[qid]["new"]],
        "fixes": fix_reports,
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Eleven source-verified changes; leaderboard effect remains unmeasured.",
    }
    write_json(OUTPUT / "v206_semantic_batch11_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
