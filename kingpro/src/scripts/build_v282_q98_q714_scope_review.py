"""Build a deterministic, read-only physical-scope review for HUT q98/q714."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "sub_v276_q638_fix"
FINDING = ROOT / "build" / "v281_semantic_direction_v276_final.json"
CATALOG = ROOT / "build" / "catalog.jsonl"
OUTPUT = ROOT / "build" / "v282_q98_q714_scope_review.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(raw: str) -> Decimal:
    text = str(raw).strip()
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    match = re.search(r"[0-9][0-9.,]*", text)
    if match is None:
        raise ValueError(f"no number in {raw!r}")
    value = Decimal(match.group(0).replace(".", "").replace(",", "."))
    return -abs(value) if negative else value


def rounded(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def spec(
    role: str,
    table_ref: str,
    row_idx: int,
    col_idx: int,
    raw: str,
    label: str,
    expected_masthead: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "table_ref": table_ref,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "expected_raw": raw,
        "expected_label": label,
        "expected_masthead": expected_masthead,
    }


REVIEW_SPECS: dict[int, dict[str, Any]] = {
    98: {
        "question_scope": "công ty mẹ",
        "physical_scope": "riêng",
        "formula": "inventory_closing_vnd / 1e9",
        "sources": [
            spec(
                "inventory_closing_parent_2024",
                "HUT_financial_statements_2024_consolidated|325",
                12,
                4,
                "146.469.679.444",
                "Hàng tồn kho",
                "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG",
            )
        ],
        "hard_negatives": [
            {
                **spec(
                    "wrong_scope_consolidated_inventory",
                    "HUT_financial_statements_2024_separate|328",
                    15,
                    4,
                    "3.177.372.538.020",
                    "Hàng tồn kho",
                    "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
                ),
                "why_wrong": "The filename says separate, but the physical masthead is consolidated; q98 explicitly asks công ty mẹ.",
                "counterfactual_answer": 3177.37,
            },
            {
                **spec(
                    "wrong_period_parent_opening_inventory",
                    "HUT_financial_statements_2024_consolidated|325",
                    12,
                    5,
                    "6.092.398.912",
                    "Hàng tồn kho",
                    "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG",
                ),
                "why_wrong": "Column 5 is Số đầu năm, not the requested 2024 closing balance.",
                "counterfactual_answer": 6.09,
            },
        ],
        "proposed_mutation": {
            "answer": {"from": 3177.37, "to": 146.47},
            "relevant_docs": {
                "remove": ["HUT_financial_statements_2024_separate"],
                "add": ["HUT_financial_statements_2024_consolidated"],
                "note": "Container names are swapped; the added container has the RIÊNG masthead.",
            },
            "relevant_tables": {
                "remove": ["HUT_financial_statements_2024_separate|328"],
                "add": ["HUT_financial_statements_2024_consolidated|325"],
            },
            "source_cells": [
                {"raw": "146.469.679.444", "row_idx": 12, "col_idx": 4}
            ],
            "pandas_formula": "result = round(146469679444 / 1e9, 2)",
        },
    },
    714: {
        "question_scope": "unqualified; default consolidated",
        "physical_scope": "hợp nhất",
        "formula": "(finance_revenue_2024 - finance_expense_2024) / 1e9",
        "sources": [
            spec(
                "finance_revenue_consolidated_2024",
                "HUT_financial_statements_2024_separate|399",
                6,
                4,
                "874.739.630.652",
                "Doanh thu hoạt động tài chính",
                "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT",
            ),
            spec(
                "finance_expense_consolidated_2024",
                "HUT_financial_statements_2024_separate|399",
                7,
                4,
                "706.004.285.205",
                "Chi phí tài chính",
                "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT",
            ),
        ],
        "hard_negatives": [
            {
                **spec(
                    "wrong_scope_finance_revenue_separate_2024",
                    "HUT_financial_statements_2024_consolidated|376",
                    6,
                    4,
                    "576.356.569.368",
                    "Doanh thu hoạt động tài chính",
                    "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH RIÊNG",
                ),
                "paired_wrong_raw": "337.464.727.127",
                "paired_source": spec(
                    "wrong_scope_finance_expense_separate_2024",
                    "HUT_financial_statements_2024_consolidated|376",
                    7,
                    4,
                    "337.464.727.127",
                    "Chi phí tài chính",
                    "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH RIÊNG",
                ),
                "why_wrong": "The container says consolidated, but its physical masthead is RIÊNG. The unqualified query follows the declared/default consolidated scope.",
                "counterfactual_answer": 238.89,
            },
            {
                **spec(
                    "wrong_formula_gross_finance_revenue_only",
                    "HUT_financial_statements_2024_separate|399",
                    6,
                    4,
                    "874.739.630.652",
                    "Doanh thu hoạt động tài chính",
                    "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT",
                ),
                "why_wrong": "Kết quả thuần requires finance revenue minus finance expense; revenue alone is gross, not net.",
                "counterfactual_answer": 874.74,
            },
            {
                **spec(
                    "wrong_period_consolidated_finance_revenue_2023",
                    "HUT_financial_statements_2024_separate|399",
                    6,
                    5,
                    "376.200.206.685",
                    "Doanh thu hoạt động tài chính",
                    "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT",
                ),
                "paired_wrong_raw": "405.905.966.955",
                "paired_source": spec(
                    "wrong_period_consolidated_finance_expense_2023",
                    "HUT_financial_statements_2024_separate|399",
                    7,
                    5,
                    "405.905.966.955",
                    "Chi phí tài chính",
                    "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT",
                ),
                "why_wrong": "Column 5 is Năm trước (2023), not 2024.",
                "counterfactual_answer": -29.71,
            },
        ],
        "proposed_mutation": {
            "answer": {"from": 238.89, "to": 168.74},
            "relevant_docs": {
                "remove": ["HUT_financial_statements_2024_consolidated"],
                "add": ["HUT_financial_statements_2024_separate"],
                "note": "Container names are swapped; the added container has the HỢP NHẤT masthead.",
            },
            "relevant_tables": {
                "remove": ["HUT_financial_statements_2024_consolidated|376"],
                "add": ["HUT_financial_statements_2024_separate|399"],
            },
            "source_cells": [
                {"raw": "874.739.630.652", "row_idx": 6, "col_idx": 4},
                {"raw": "706.004.285.205", "row_idx": 7, "col_idx": 4},
            ],
            "pandas_formula": "result = round((874739630652 - 706004285205) / 1e9, 2)",
        },
    },
}


def main() -> int:
    submission_path = CANDIDATE / "submission.json"
    submission = {
        int(row["id"]): row
        for row in json.loads(submission_path.read_text(encoding="utf-8"))
        if int(row["id"]) in REVIEW_SPECS
    }
    if set(submission) != set(REVIEW_SPECS):
        raise ValueError("candidate rows missing")
    finding_payload = json.loads(FINDING.read_text(encoding="utf-8"))
    finding_rows = finding_payload.get("records", finding_payload)
    finding_ids = {int(row["id"]) for row in finding_rows}
    if not set(REVIEW_SPECS).issubset(finding_ids):
        raise ValueError("scope finding rows missing")
    catalog = {
        entry["table_ref"]: entry
        for entry in (
            json.loads(line)
            for line in CATALOG.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    def resolve(source: dict[str, Any]) -> tuple[dict[str, Any], Decimal]:
        entry = catalog[source["table_ref"]]
        csv_path = ROOT / "build" / "tables" / entry["csv_path"]
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        raw = str(frame.iloc[source["row_idx"], source["col_idx"]]).strip()
        row_text = " | ".join(str(value).strip() for value in frame.iloc[source["row_idx"]])
        if raw != source["expected_raw"]:
            raise ValueError(f"stale raw at {source['table_ref']}: {raw!r}")
        if source["expected_label"].casefold() not in row_text.casefold():
            raise ValueError(f"stale label at {source['table_ref']}: {row_text!r}")
        report_path = (
            ROOT
            / "data"
            / "financial_statements"
            / entry["ticker"]
            / str(entry["year"])
            / entry["report_id"]
            / f"{entry['report_id']}_extracted.txt"
        )
        lines = report_path.read_text(encoding="utf-8").splitlines()
        start = max(0, int(entry["line"]) - 16)
        context = " | ".join(line.strip() for line in lines[start : int(entry["line"])] if line.strip())
        if source["expected_masthead"].casefold() not in context.casefold():
            raise ValueError(
                f"physical masthead mismatch at {source['table_ref']}: {context!r}"
            )
        value = number(raw)
        return (
            {
                "role": source["role"],
                "table_ref": source["table_ref"],
                "container_report_id": entry["report_id"],
                "catalog_declared_scope": entry["scope"],
                "physical_masthead": source["expected_masthead"],
                "ticker": entry["ticker"],
                "year": str(entry["year"]),
                "page": int(entry["page"]),
                "line": int(entry["line"]),
                "catalog_csv": entry["csv_path"],
                "resolved_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "csv_sha256": sha256(csv_path),
                "resolved_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
                "report_sha256": sha256(report_path),
                "row_idx": source["row_idx"],
                "col_idx": source["col_idx"],
                "row_text": row_text,
                "raw": raw,
                "parsed_vnd": str(value),
                "unit": "VND",
                "masthead_context": context,
            },
            value,
        )

    records: list[dict[str, Any]] = []
    for question_id in sorted(REVIEW_SPECS):
        review = REVIEW_SPECS[question_id]
        row = submission[question_id]
        physical_refs: list[dict[str, Any]] = []
        values: list[Decimal] = []
        for source in review["sources"]:
            resolved, value = resolve(source)
            physical_refs.append(resolved)
            values.append(value)
        checked_negatives: list[dict[str, Any]] = []
        for negative in review["hard_negatives"]:
            resolved, _value = resolve(negative)
            paired = None
            if "paired_source" in negative:
                paired, _paired_value = resolve(negative["paired_source"])
            checked_negatives.append(
                {
                    **resolved,
                    "why_wrong": negative["why_wrong"],
                    **(
                        {"paired_wrong_raw": negative["paired_wrong_raw"]}
                        if "paired_wrong_raw" in negative
                        else {}
                    ),
                    **({"paired_physical_ref": paired} if paired is not None else {}),
                    "counterfactual_answer": negative["counterfactual_answer"],
                }
            )
        exact = values[0] / Decimal(10) ** 9 if question_id == 98 else (values[0] - values[1]) / Decimal(10) ** 9
        recomputed = rounded(exact)
        current = float(row["answer"])
        if recomputed == current:
            raise ValueError(f"q{question_id} expected a physical-scope delta")
        result_lines = [
            line.strip()
            for line in str(row["pandas_query"]).splitlines()
            if line.strip().startswith("result =")
        ]
        records.append(
            {
                "id": question_id,
                "question": row["question"],
                "decision": "answer_change",
                "status": "source_confirmed_change",
                "current_answer": current,
                "recomputed_answer": recomputed,
                "scope_resolution": {
                    "question_scope": review["question_scope"],
                    "resolved_physical_scope": review["physical_scope"],
                    "rule": (
                        "Explicit công ty mẹ requires a RIÊNG masthead."
                        if question_id == 98
                        else "Unqualified issuer metric uses the declared/default consolidated scope; verify HỢP NHẤT on the physical masthead."
                    ),
                    "container_names_swapped": True,
                    "masthead_overrides_filename": True,
                },
                "proof": {
                    "formula": review["formula"],
                    "exact_unrounded": str(exact),
                    "rounding": "two decimal places; half-up (not a tie)",
                    "candidate_result_lines": result_lines,
                    "company": "HUT / CTCP Tasco",
                    "year": 2024,
                    "unit": "VND converted to billion VND by division by 1e9",
                },
                "physical_refs": physical_refs,
                "hard_negatives": checked_negatives,
                "proposed_mutation": review["proposed_mutation"],
            }
        )

    counts = Counter(record["status"] for record in records)
    report = {
        "schema_version": "v282-hut-scope-container-review/v1",
        "mode": "read_only",
        "inputs": {
            "candidate_submission": {
                "path": str(submission_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(submission_path),
            },
            "scope_findings": {
                "path": str(FINDING.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(FINDING),
            },
            "catalog_sha256": sha256(CATALOG),
        },
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "answer_change_ids": [record["id"] for record in records],
        "container_swap_proof": {
            "HUT_financial_statements_2024_consolidated": "physical mastheads are RIÊNG",
            "HUT_financial_statements_2024_separate": "physical mastheads are HỢP NHẤT",
        },
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reloaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert reloaded["record_count"] == 2
    assert reloaded["answer_change_ids"] == [98, 714]
    assert reloaded["counts"] == {"source_confirmed_change": 2}
    assert [(record["id"], record["current_answer"], record["recomputed_answer"]) for record in reloaded["records"]] == [
        (98, 3177.37, 146.47),
        (714, 238.89, 168.74),
    ]
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "counts": reloaded["counts"],
                "changes": [
                    {"id": record["id"], "from": record["current_answer"], "to": record["recomputed_answer"]}
                    for record in reloaded["records"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
