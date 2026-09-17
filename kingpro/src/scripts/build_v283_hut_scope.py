"""Build rollbackable V283 with only q98/q714 physical-masthead scope fixes."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_v276_q638_fix"
OUTPUT = ROOT / "sub_v283_hut_scope"
CATALOG = ROOT / "build" / "catalog.jsonl"
HEADER = [
    "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
    "source_table", "source_csv", "row_idx", "col_idx",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def audit_source(
    table_ref: str,
    csv_name: str,
    row: int,
    column: int,
    metric: str,
    raw: str,
    label: str,
) -> dict:
    return {
        "table_ref": table_ref,
        "csv": csv_name,
        "row": row,
        "column": column,
        "metric": metric,
        "label": label,
        "source_row_labels": [label],
        "scale": 1.0,
        "typed_factor": 1.0,
        "raw": raw,
    }


def physical_proof() -> dict:
    catalog = {
        row["table_ref"]: row
        for row in (
            json.loads(line)
            for line in CATALOG.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    checks = [
        ("HUT_financial_statements_2024_consolidated|325", 12, 4, "146.469.679.444", "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG"),
        ("HUT_financial_statements_2024_separate|399", 6, 4, "874.739.630.652", "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT"),
        ("HUT_financial_statements_2024_separate|399", 7, 4, "706.004.285.205", "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT"),
    ]
    records = []
    for table_ref, row_idx, col_idx, expected, masthead in checks:
        entry = catalog[table_ref]
        csv_path = ROOT / "build" / "tables" / entry["csv_path"]
        frame = pd.read_csv(
            csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
        actual = str(frame.iloc[row_idx, col_idx]).strip()
        if actual != expected:
            raise AssertionError(f"physical token mismatch {table_ref}: {actual!r}")
        report_path = (
            ROOT / "data" / "financial_statements" / entry["ticker"]
            / str(entry["year"]) / entry["report_id"]
            / f"{entry['report_id']}_extracted.txt"
        )
        lines = report_path.read_text(encoding="utf-8").splitlines()
        context = " | ".join(
            line.strip()
            for line in lines[max(0, int(entry["line"]) - 16) : int(entry["line"])]
            if line.strip()
        )
        if masthead.casefold() not in context.casefold():
            raise AssertionError(f"physical masthead mismatch {table_ref}")
        records.append(
            {
                "table_ref": table_ref,
                "catalog_declared_scope": entry["scope"],
                "physical_masthead": masthead,
                "csv_path": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "csv_sha256": sha(csv_path),
                "report_path": str(report_path.relative_to(ROOT)).replace("\\", "/"),
                "report_sha256": sha(report_path),
                "row_idx": row_idx,
                "col_idx": col_idx,
                "raw": actual,
            }
        )
    return {
        "container_swap": {
            "HUT_financial_statements_2024_consolidated": "physical RIENG",
            "HUT_financial_statements_2024_separate": "physical HOP NHAT",
        },
        "checks": records,
    }


def build() -> dict:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {OUTPUT}")
    proof = physical_proof()
    temp = Path(mkdtemp(prefix="v283_hut_scope_", dir=str(ROOT)))
    try:
        staged = temp / OUTPUT.name
        shutil.copytree(SOURCE, staged)
        submission_path = staged / "submission.json"
        rows = json.loads(submission_path.read_text(encoding="utf-8-sig"))
        by_id = {int(row["id"]): row for row in rows}
        if float(by_id[98]["answer"]) != 3177.37 or float(by_id[714]["answer"]) != 238.89:
            raise AssertionError("unexpected V276 baseline answers")
        original_queries = {qid: by_id[qid]["pandas_query"] for qid in (98, 714)}
        original_evidence = {
            qid: json.loads(json.dumps(by_id[qid]["evidence"])) for qid in (98, 714)
        }

        by_id[98]["answer"] = 146.47
        by_id[98]["relevant_docs"] = ["HUT_financial_statements_2024_consolidated"]
        by_id[98]["relevant_tables"] = ["HUT_financial_statements_2024_consolidated|325"]
        by_id[714]["answer"] = 168.74
        by_id[714]["relevant_docs"] = ["HUT_financial_statements_2024_separate"]
        by_id[714]["relevant_tables"] = ["HUT_financial_statements_2024_separate|399"]

        write_csv(
            staged / "data" / "q98_source_cells.csv",
            [[
                "HUT", "2024", "cdkt:140", "146.469.679.444", "1.0", "1.0",
                "HUT_financial_statements_2024_consolidated|325",
                "table_0_line325.csv", "12", "4",
            ]],
        )
        write_csv(
            staged / "data" / "q714_source_cells.csv",
            [
                [
                    "HUT", "2024", "kqkd:21", "874.739.630.652", "1.0", "1.0",
                    "HUT_financial_statements_2024_separate|399",
                    "table_3_line399.csv", "6", "4",
                ],
                [
                    "HUT", "2024", "kqkd:22", "706.004.285.205", "1.0", "1.0",
                    "HUT_financial_statements_2024_separate|399",
                    "table_3_line399.csv", "7", "4",
                ],
            ],
        )
        submission_path.write_text(
            json.dumps(list(by_id.values()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        audit_path = staged / "source_audit.json"
        audits = json.loads(audit_path.read_text(encoding="utf-8-sig"))
        audit_by_id = {int(item["id"]): item for item in audits}
        audit_by_id[98] = {
            "id": 98,
            "old_answer": 3177.37,
            "answer": 146.47,
            "note": "HUT parent net inventory; physical RIENG masthead overrides swapped consolidated container name",
            "sources": [
                audit_source(
                    "HUT_financial_statements_2024_consolidated|325",
                    "table_0_line325.csv", 12, 4, "cdkt:140",
                    "146.469.679.444", "Hàng tồn kho (code 140)",
                )
            ],
        }
        audit_by_id[714] = {
            "id": 714,
            "old_answer": 238.89,
            "answer": 168.74,
            "note": "HUT consolidated net finance result; physical HOP NHAT masthead overrides swapped separate container name",
            "sources": [
                audit_source(
                    "HUT_financial_statements_2024_separate|399",
                    "table_3_line399.csv", 6, 4, "kqkd:21",
                    "874.739.630.652", "Doanh thu hoạt động tài chính",
                ),
                audit_source(
                    "HUT_financial_statements_2024_separate|399",
                    "table_3_line399.csv", 7, 4, "kqkd:22",
                    "706.004.285.205", "Chi phí tài chính",
                ),
            ],
        }
        audit_path.write_text(
            json.dumps(
                sorted(audit_by_id.values(), key=lambda item: int(item["id"])),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if any(by_id[qid]["pandas_query"] != original_queries[qid] for qid in (98, 714)):
            raise AssertionError("target query changed")
        if any(by_id[qid]["evidence"] != original_evidence[qid] for qid in (98, 714)):
            raise AssertionError("target evidence binding changed")

        candidate_audit = {
            "schema_version": "v283-hut-scope-candidate/v1",
            "candidate": OUTPUT.name,
            "baseline": SOURCE.name,
            "answer_changes": [
                {"id": 98, "from": 3177.37, "to": 146.47},
                {"id": 714, "from": 238.89, "to": 168.74},
            ],
            "changed_submission_fields": {
                "98": ["answer", "relevant_docs", "relevant_tables"],
                "714": ["answer", "relevant_docs", "relevant_tables"],
            },
            "queries_unchanged": True,
            "evidence_bindings_unchanged": True,
            "compact_manifests_updated": [98, 714],
            "source_audit_updated": [98, 714],
            "physical_masthead_proof": proof,
            "automatic_promotion": False,
            "packaged": False,
            "submitted": False,
            "baseline_submission_sha256": sha(SOURCE / "submission.json"),
            "candidate_submission_sha256": sha(submission_path),
        }
        (staged / "v283_hut_scope_audit.json").write_text(
            json.dumps(candidate_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), str(OUTPUT))
        return candidate_audit
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(build(), ensure_ascii=False, indent=2))
