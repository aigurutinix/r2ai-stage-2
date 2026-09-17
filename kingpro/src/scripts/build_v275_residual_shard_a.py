"""Build the deterministic, read-only source adjudication for residual shard A.

The candidate, catalog and physical tables are inputs only.  Every declared
operand is re-read from ``build/tables`` through ``build/catalog.jsonl`` before
the result is emitted.  The script intentionally does not edit the candidate
or the durable review ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "sub_v274_q986_nan"
OUTPUT = ROOT / "build" / "v275_residual_shard_a.json"
IDS = (615, 621, 638, 639, 651, 665, 684, 702, 721, 752, 881)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(raw: str) -> Decimal:
    text = str(raw).strip().replace("\xa0", "")
    if text in {"", "-"}:
        return Decimal(0)
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    token = re.search(r"[0-9][0-9.,]*", text)
    if token is None:
        raise ValueError(f"no number in {raw!r}")
    value = Decimal(token.group(0).replace(".", "").replace(",", "."))
    return -abs(value) if negative else value


def rounded(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def operand(
    role: str,
    table_ref: str,
    row: int,
    column: int,
    expected_raw: str,
    unit: str,
    *,
    expected_label: str | None = None,
    label_col: int = 0,
    context: str | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "table_ref": table_ref,
        "row_idx": row,
        "col_idx": column,
        "expected_raw": expected_raw,
        "source_unit": unit,
        "expected_label": expected_label,
        "label_col": label_col,
        "context": context,
    }


SPECS: dict[int, dict[str, Any]] = {
    615: {
        "status": "source_confirmed_no_change",
        "formula": "(ending_2020 / ending_2017 - 1) * 100",
        "operands": [
            operand("ending_2020", "DXG_financial_statements_2020_consolidated|1007", 1, 2, "9.620.347.821.442", "VND", expected_label="Bắt động sản dở dang"),
            operand("ending_2017", "DXG_financial_statements_2017_consolidated|939", 1, 2, "2.965.209.292.615", "VND", expected_label="Bất động sản dở dang"),
        ],
        "hard_negatives": [
            {
                **operand("wrong_period_beginning_2020", "DXG_financial_statements_2020_consolidated|1007", 1, 3, "6.442.659.029.053", "VND", expected_label="Bắt động sản dở dang"),
                "why_wrong": "Column 3 is Số đầu năm (31/12/2019), not the requested 2020 ending balance.",
                "counterfactual_answer": 117.28,
            }
        ],
    },
    621: {
        "status": "source_confirmed_no_change",
        "formula": "abs(fuel_cost_2024 - fuel_cost_2021) / 1e12",
        "operands": [
            operand("fuel_cost_2024", "VJC_financial_statements_2024_separate|1719", 1, 1, "23.858.693.067.798", "VND", expected_label="Chi phí nhiên liệu"),
            operand("fuel_cost_2021", "VJC_financial_statements_2021_separate|1384", 1, 1, "3.059.363.382.827", "VND", expected_label="Chi phí nhiên liệu"),
        ],
        "hard_negatives": [
            {
                **operand("wrong_year_2023", "VJC_financial_statements_2024_separate|1719", 1, 2, "21.242.888.445.391", "VND", expected_label="Chi phí nhiên liệu"),
                "why_wrong": "Column 2 is Năm 2023, while the question asks 2024 and 2021.",
                "counterfactual_answer": 18.18,
            }
        ],
    },
    638: {
        "status": "source_confirmed_change",
        "formula": "(lease_commitment_2025 / lease_commitment_2022 - 1) * 100",
        "operands": [
            operand(
                "lease_commitment_2025",
                "GEX_financial_statements_2025_separate|1801",
                2,
                1,
                "27.258.108.199",
                "VND",
                expected_label="Từ 1 năm trở xuống",
                context="Raw report lines 1797-1801: Cam kết thuê hoạt động; Công ty thuê đất.",
            ),
            operand(
                "lease_commitment_2022",
                "GEX_financial_statements_2022_separate|1417",
                2,
                1,
                "25.779.332.206",
                "VND",
                expected_label="Từ 1 năm trở xuống",
                context="Raw report lines 1415-1417: Công ty thuê đất theo các hợp đồng thuê hoạt động.",
            ),
        ],
        "hard_negatives": [
            {
                **operand(
                    "wrong_direction_lease_receivable_2022",
                    "GEX_financial_statements_2022_separate|1423",
                    2,
                    1,
                    "62.834.689.635",
                    "VND",
                    expected_label="Từ 1 năm trở xuống",
                    context="Raw report lines 1419-1423: Cam kết cho thuê hoạt động; Công ty cho thuê văn phòng.",
                ),
                "why_wrong": "This is cho thuê (future lease receipts), not thuê (future minimum lease payments) asked by the question.",
                "counterfactual_answer": -56.62,
            }
        ],
        "proposed_mutation": {
            "answer": {"from": -56.62, "to": 5.74},
            "relevant_tables": {
                "remove": ["GEX_financial_statements_2022_separate|1423"],
                "add": ["GEX_financial_statements_2022_separate|1417"],
            },
            "source_operand": {
                "replace_raw": "62.834.689.635",
                "with_raw": "25.779.332.206",
                "row_idx": 2,
                "col_idx": 1,
            },
            "pandas_formula": "result = round((27258108199 / 25779332206 - 1) * 100, 2)",
            "reason": "The 2022 operand crossed from thuê to the adjacent but opposite-direction cho thuê disclosure.",
        },
    },
    639: {
        "status": "source_confirmed_no_change",
        "formula": "(construction_wip_2020 / construction_wip_2019 - 1) * 100",
        "operands": [
            operand("construction_wip_2020", "PC1_financial_statements_2020_consolidated|1199", 7, 1, "395.317.620.609", "VND", expected_label="+ Hoạt động xây lắp"),
            operand("construction_wip_2019", "PC1_financial_statements_2020_consolidated|1199", 7, 2, "429.830.478.898", "VND", expected_label="+ Hoạt động xây lắp"),
        ],
        "hard_negatives": [
            {
                **operand("wrong_broad_total_2020", "PC1_financial_statements_2020_consolidated|1199", 26, 1, "468.867.078.164", "VND"),
                "why_wrong": "The total mixes real estate, construction, industrial and other WIP instead of the qualified hoạt động xây lắp row.",
                "paired_wrong_raw": "1.217.101.489.013",
                "counterfactual_answer": -61.48,
            }
        ],
    },
    651: {
        "status": "source_confirmed_no_change",
        "formula": "(interest_expense_2017 - interest_expense_2018) / interest_expense_2017 * 100",
        "operands": [
            operand("interest_expense_2017", "MML_financial_statements_2017_consolidated|213", 8, 3, "491.384.792.041", "VND", expected_label="Trong đó: Chi phí lãi vay"),
            operand("interest_expense_2018", "MML_financial_statements_2018_consolidated|225", 8, 3, "235.444.066.977", "VND", expected_label="Trong đó: Chi phí lãi vay"),
        ],
        "hard_negatives": [
            {
                **operand("restated_comparative_2017", "MML_financial_statements_2018_consolidated|225", 8, 4, "422.368.615.574", "VND", expected_label="Trong đó: Chi phí lãi vay"),
                "why_wrong": "This is the 2017 comparative printed in the 2018 report; the benchmark pair uses each requested report year's current-period value. It is a real restatement/reclassification alternative and must not be mixed silently.",
                "counterfactual_answer": 44.26,
            }
        ],
    },
    665: {
        "status": "source_confirmed_no_change",
        "formula": "intangible_fixed_assets_2016 / total_assets_2016 * 100",
        "operands": [
            operand("intangible_fixed_assets_2016", "NVL_financial_statements_2016_consolidated|170", 10, 3, "28.642.968.853", "VND", expected_label="Tài sản cố định vô hình", label_col=1),
            operand("total_assets_2016", "NVL_financial_statements_2016_consolidated|170", 26, 3, "36.527.075.713.997", "VND", expected_label="TỔNG TÀI SẢN", label_col=1),
        ],
        "hard_negatives": [
            {
                **operand("wrong_year_intangible_2015", "NVL_financial_statements_2016_consolidated|170", 10, 4, "27.706.140.244", "VND", expected_label="Tài sản cố định vô hình", label_col=1),
                "why_wrong": "Column 4 is 2015, not the requested 31/12/2016 balance.",
                "paired_wrong_raw": "26.570.408.635.043",
                "counterfactual_answer": 0.1,
            }
        ],
    },
    684: {
        "status": "source_confirmed_no_change",
        "formula": "general_provision_2016 / total_customer_loan_provision_2016 * 100",
        "operands": [
            operand("general_provision_2016", "SHB_financial_statements_2016_separate|1479", 1, 1, "1.018.726", "million VND", expected_label="Dự phòng chung (i)"),
            operand("total_customer_loan_provision_2016", "SHB_financial_statements_2016_separate|1479", 3, 1, "1.691.202", "million VND"),
        ],
        "hard_negatives": [
            {
                **operand("wrong_numerator_specific_provision", "SHB_financial_statements_2016_separate|1479", 2, 1, "672.476", "million VND", expected_label="Dự phòng cụ thể (ii)"),
                "why_wrong": "This is specific provision, the complement of the requested general provision. The blank total row is independently 1,018,726 + 672,476 = 1,691,202.",
                "counterfactual_answer": 39.76,
            }
        ],
    },
    702: {
        "status": "source_confirmed_no_change",
        "formula": "net_other_income_2018 / 1e9",
        "operands": [
            operand("net_other_income_2018", "CEO_financial_statements_2018_separate|246", 14, 3, "(5.014.399.788)", "VND", expected_label="13. Lợi nhuận khác (40 = 31 - 32)"),
        ],
        "hard_negatives": [
            {
                **operand("wrong_gross_other_income", "CEO_financial_statements_2018_separate|246", 12, 3, "24.274.993.826", "VND", expected_label="11. Thu nhập khác"),
                "why_wrong": "Row 31 is gross other income and ignores other expense; 'thuần' maps to row 40 = row 31 - row 32.",
                "counterfactual_answer": 24.27,
            }
        ],
    },
    721: {
        "status": "source_confirmed_no_change",
        "formula": "share_investment_ending / associate_joint_venture_investment_ending * 100",
        "operands": [
            operand("share_investment_ending", "VIF_financial_statements_2020_consolidated|1262", 2, 1, "15.996.208.039", "VND", expected_label="Đầu tư vào cổ phiếu (i)", context="Section 16.2 Đầu tư góp vốn vào đơn vị khác."),
            operand("associate_joint_venture_investment_ending", "VIF_financial_statements_2020_consolidated|1162", 2, 1, "1.141.390.360.287", "VND", expected_label="Đầu tư vào công ty liên kết", context="Section 16.1; balance-sheet code 252 is Đầu tư vào công ty liên doanh, liên kết."),
        ],
        "hard_negatives": [
            {
                **operand("wrong_denominator_all_long_term_investments", "VIF_financial_statements_2020_consolidated|1162", 4, 1, "1.160.207.138.431", "VND", expected_label="TỔNG CỘNG"),
                "why_wrong": "The total also includes investments in other entities, so it is broader than investments in associates/joint ventures.",
                "counterfactual_answer": 1.38,
            }
        ],
    },
    752: {
        "status": "source_confirmed_no_change",
        "formula": "abs(abb_net_service_income_2022 - mbb_net_service_income_2022)",
        "operands": [
            operand("abb_net_service_income_2022", "ABB_financial_statements_2022_consolidated|296", 6, 2, "232.042", "million VND", expected_label="Lãi thuần từ hoạt động dịch vụ"),
            operand("mbb_net_service_income_2022", "MBB_financial_statements_2022_consolidated|410", 6, 2, "4.135.568", "million VND", expected_label="Lãi thuần từ hoạt động dịch vụ"),
            operand("mbb_equivalent_crosscheck", "MBB_financial_statements_2022_consolidated|2089", 20, 1, "4.135.568", "million VND", expected_label="Lãi thuần từ hoạt động dịch vụ"),
        ],
        "hard_negatives": [
            {
                **operand("wrong_year_abb_2021", "ABB_financial_statements_2022_consolidated|296", 6, 3, "352.239", "million VND", expected_label="Lãi thuần từ hoạt động dịch vụ"),
                "why_wrong": "Column 3 is Năm trước (2021); pairing prior-year columns would answer a different question.",
                "paired_wrong_raw": "4.367.378",
                "counterfactual_answer": 4015139.0,
            }
        ],
    },
    881: {
        "status": "source_confirmed_no_change",
        "formula": "mean(abs(accumulated_depreciation_y) / gross_cost_y for y in requested_years) * 100",
        "operands": [
            operand("gross_2016", "HND_financial_statements_2016|127", 18, 3, "22.141.526.552.885", "VND", expected_label="Nguyên giá"),
            operand("depreciation_2016", "HND_financial_statements_2016|127", 19, 3, "(8.001.667.854.893)", "VND", expected_label="Giá trị hao mòn lũy kế"),
            operand("gross_2018", "HND_financial_statements_2018|122", 17, 3, "22.058.473.317.440", "VND", expected_label="Nguyên giá"),
            operand("depreciation_2018", "HND_financial_statements_2018|122", 18, 3, "(11.731.433.309.660)", "VND", expected_label="Giá trị hao mòn lũy kế"),
            operand("gross_2019", "HND_financial_statements_2019|214", 18, 3, "22.079.164.840.230", "VND", expected_label="Nguyên giá"),
            operand("depreciation_2019", "HND_financial_statements_2019|214", 19, 3, "(13.520.488.721.292)", "VND", expected_label="Giá trị hao mòn lũy kế"),
            operand("gross_2020", "HND_financial_statements_2020|208", 19, 3, "22.083.494.486.346", "VND", expected_label="Nguyên giá"),
            operand("depreciation_2020", "HND_financial_statements_2020|208", 20, 3, "(15.298.798.199.853)", "VND", expected_label="Giá trị hao mòn lũy kế"),
            operand("gross_2021", "HND_financial_statements_2021|205", 21, 3, "22.125.917.998.980", "VND", expected_label="Nguyên giá"),
            operand("depreciation_2021", "HND_financial_statements_2021|205", 22, 3, "(16.599.466.811.506)", "VND", expected_label="Giá trị hao mòn lũy kế"),
        ],
        "hard_negatives": [
            {
                "role": "wrong_period_opening_columns",
                "table_refs": ["HND_financial_statements_2016|127", "HND_financial_statements_2018|122", "HND_financial_statements_2019|214", "HND_financial_statements_2020|208", "HND_financial_statements_2021|205"],
                "col_idx": 4,
                "why_wrong": "Column 4 is each report's opening/prior-year balance, not the requested year-end balance in column 3.",
                "counterfactual_answer": 51.18,
            }
        ],
    },
}


def calculate(question_id: int, values: dict[str, Decimal]) -> tuple[Decimal, list[Decimal] | None]:
    if question_id == 615:
        return (values["ending_2020"] / values["ending_2017"] - 1) * 100, None
    if question_id == 621:
        return abs(values["fuel_cost_2024"] - values["fuel_cost_2021"]) / Decimal(10) ** 12, None
    if question_id == 638:
        return (values["lease_commitment_2025"] / values["lease_commitment_2022"] - 1) * 100, None
    if question_id == 639:
        return (values["construction_wip_2020"] / values["construction_wip_2019"] - 1) * 100, None
    if question_id == 651:
        return (values["interest_expense_2017"] - values["interest_expense_2018"]) / values["interest_expense_2017"] * 100, None
    if question_id == 665:
        return values["intangible_fixed_assets_2016"] / values["total_assets_2016"] * 100, None
    if question_id == 684:
        return values["general_provision_2016"] / values["total_customer_loan_provision_2016"] * 100, None
    if question_id == 702:
        return values["net_other_income_2018"] / Decimal(10) ** 9, None
    if question_id == 721:
        return values["share_investment_ending"] / values["associate_joint_venture_investment_ending"] * 100, None
    if question_id == 752:
        if values["mbb_net_service_income_2022"] != values["mbb_equivalent_crosscheck"]:
            raise ValueError("q752 MBB equivalent tables disagree")
        return abs(values["abb_net_service_income_2022"] - values["mbb_net_service_income_2022"]), None
    if question_id == 881:
        ratios = [
            abs(values[f"depreciation_{year}"]) / values[f"gross_{year}"] * 100
            for year in (2016, 2018, 2019, 2020, 2021)
        ]
        return sum(ratios) / Decimal(len(ratios)), ratios
    raise KeyError(question_id)


def main() -> int:
    submission_path = CANDIDATE / "submission.json"
    catalog_path = ROOT / "build" / "catalog.jsonl"
    rows = {
        int(row["id"]): row
        for row in json.loads(submission_path.read_text(encoding="utf-8"))
        if int(row["id"]) in IDS
    }
    if set(rows) != set(IDS):
        raise ValueError(f"missing candidate rows: {sorted(set(IDS) - set(rows))}")
    catalog = {
        entry["table_ref"]: entry
        for entry in (
            json.loads(line)
            for line in catalog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    records: list[dict[str, Any]] = []
    for question_id in sorted(IDS):
        row = rows[question_id]
        spec = SPECS[question_id]
        physical_refs: list[dict[str, Any]] = []
        values: dict[str, Decimal] = {}
        for source in spec["operands"]:
            entry = catalog[source["table_ref"]]
            csv_path = ROOT / "build" / "tables" / entry["csv_path"]
            frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            raw = str(frame.iloc[source["row_idx"], source["col_idx"]]).strip()
            label = str(frame.iloc[source["row_idx"], source.get("label_col", 0)]).strip()
            if raw != source["expected_raw"]:
                raise ValueError(f"q{question_id} stale raw at {source['table_ref']}: {raw!r}")
            if source.get("expected_label") and source["expected_label"].casefold() not in label.casefold():
                raise ValueError(f"q{question_id} stale label at {source['table_ref']}: {label!r}")
            parsed = number(raw)
            values[source["role"]] = parsed
            physical_refs.append(
                {
                    "role": source["role"],
                    "table_ref": source["table_ref"],
                    "report_id": entry["report_id"],
                    "ticker": entry["ticker"],
                    "year": str(entry["year"]),
                    "scope": entry["scope"],
                    "page": int(entry["page"]),
                    "line": int(entry["line"]),
                    "catalog_csv": entry["csv_path"],
                    "resolved_path": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                    "csv_sha256": sha256(csv_path),
                    "row_idx": source["row_idx"],
                    "col_idx": source["col_idx"],
                    "column_label": str(frame.columns[source["col_idx"]]),
                    "row_label": label,
                    "raw": raw,
                    "parsed_value": float(parsed),
                    "source_unit": source["source_unit"],
                    "context": source.get("context"),
                    "declared_by_candidate": source["table_ref"] in row.get("relevant_tables", []),
                }
            )
        hard_negative_checks = 0
        for negative in spec["hard_negatives"]:
            if "table_ref" in negative:
                entry = catalog[negative["table_ref"]]
                csv_path = ROOT / "build" / "tables" / entry["csv_path"]
                frame = pd.read_csv(
                    csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
                )
                raw = str(
                    frame.iloc[negative["row_idx"], negative["col_idx"]]
                ).strip()
                if raw != negative["expected_raw"]:
                    raise ValueError(
                        f"q{question_id} stale hard-negative raw at "
                        f"{negative['table_ref']}: {raw!r}"
                    )
                label = str(
                    frame.iloc[
                        negative["row_idx"], negative.get("label_col", 0)
                    ]
                ).strip()
                if (
                    negative.get("expected_label")
                    and negative["expected_label"].casefold() not in label.casefold()
                ):
                    raise ValueError(
                        f"q{question_id} stale hard-negative label at "
                        f"{negative['table_ref']}: {label!r}"
                    )
                hard_negative_checks += 1
            else:
                # q881 uses the same five physical tables but the adjacent
                # opening-balance column.  Prove the alternative column exists
                # and is explicitly headed as an opening/prior-period balance.
                for table_ref in negative.get("table_refs", []):
                    entry = catalog[table_ref]
                    csv_path = ROOT / "build" / "tables" / entry["csv_path"]
                    frame = pd.read_csv(
                        csv_path,
                        dtype=str,
                        keep_default_na=False,
                        encoding="utf-8-sig",
                    )
                    column = int(negative["col_idx"])
                    if column >= len(frame.columns):
                        raise ValueError(
                            f"q{question_id} hard-negative column missing at {table_ref}"
                        )
                    header = str(frame.iloc[0, column]).casefold()
                    if not any(token in header for token in ("1/1/", "1/01/", "đầu")):
                        raise ValueError(
                            f"q{question_id} hard-negative column is not opening balance "
                            f"at {table_ref}: {header!r}"
                        )
                    hard_negative_checks += 1
        exact, component_values = calculate(question_id, values)
        recomputed = rounded(exact)
        current = float(row["answer"])
        status = spec["status"]
        if status == "source_confirmed_no_change" and recomputed != current:
            raise ValueError(f"q{question_id} no-change verdict disagrees: {current} vs {recomputed}")
        if status == "source_confirmed_change" and recomputed == current:
            raise ValueError(f"q{question_id} change verdict has no delta")
        result_lines = [
            line.strip()
            for line in str(row.get("pandas_query", "")).splitlines()
            if line.strip().startswith("result =")
        ]
        proof: dict[str, Any] = {
            "company_check": sorted({item["ticker"] for item in physical_refs}),
            "year_check": sorted({item["year"] for item in physical_refs}),
            "scope_check": sorted({item["scope"] for item in physical_refs}),
            "formula": spec["formula"],
            "exact_unrounded": str(exact),
            "rounding": "two decimal places; half-up (not a tie for any shard-A result)",
            "candidate_result_lines": result_lines,
            "unit_check": sorted({item["source_unit"] for item in physical_refs}),
            "hard_negatives": spec["hard_negatives"],
            "hard_negative_physical_checks": hard_negative_checks,
        }
        if component_values is not None:
            proof["component_percentages"] = [str(value) for value in component_values]
        records.append(
            {
                "id": question_id,
                "question": row["question"],
                "status": status,
                "current_answer": current,
                "recomputed_answer": recomputed,
                "proof": proof,
                "physical_refs": physical_refs,
                "proposed_mutation": spec.get("proposed_mutation"),
            }
        )

    counts = Counter(record["status"] for record in records)
    report = {
        "schema_version": "v275-residual-source-adjudication/v1",
        "mode": "read_only",
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"),
            "submission_sha256": sha256(submission_path),
        },
        "catalog_sha256": sha256(catalog_path),
        "requested_ids": list(IDS),
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "answer_change_ids": [record["id"] for record in records if record["status"] == "source_confirmed_change"],
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Self-check the serialized artifact rather than trusting the in-memory object.
    reloaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if reloaded["record_count"] != len(IDS):
        raise ValueError("serialized record count mismatch")
    if sorted(record["id"] for record in reloaded["records"]) != sorted(IDS):
        raise ValueError("serialized ID set mismatch")
    if reloaded["counts"] != dict(sorted(counts.items())):
        raise ValueError("serialized status counts mismatch")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "counts": reloaded["counts"], "answer_change_ids": reloaded["answer_change_ids"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
