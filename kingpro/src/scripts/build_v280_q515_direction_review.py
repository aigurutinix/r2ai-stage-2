"""Build the fail-closed, read-only adjudication of q515's direction finding."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "sub_v276_q638_fix"
FINDING = ROOT / "build" / "v279_semantic_direction_v276.json"
CATALOG = ROOT / "build" / "catalog.jsonl"
OUTPUT = ROOT / "build" / "v280_q515_direction_review.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_number(raw: str) -> Decimal:
    text = str(raw).strip()
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    match = re.search(r"[0-9][0-9.,]*", text)
    if match is None:
        raise ValueError(f"no numeric token in {raw!r}")
    value = Decimal(match.group(0).replace(".", "").replace(",", "."))
    return -abs(value) if negative else value


def source(
    role: str,
    table_ref: str,
    row_idx: int,
    col_idx: int,
    raw: str,
    label: str,
    header: str,
    unit: str = "VND",
) -> dict[str, Any]:
    return {
        "role": role,
        "table_ref": table_ref,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "expected_raw": raw,
        "expected_label": label,
        "expected_header": header,
        "unit": unit,
    }


SOURCES = [
    source("selector_total_liabilities_2020", "VAB_financial_statements_2020_consolidated|327", 15, 2, "80.805.422.197.489", "TỔNG NỘ PHẢI TRẢ", "31/12/2020"),
    source("selector_total_liabilities_2024", "VAB_financial_statements_2024_consolidated|312", 12, 2, "110.975.359.140.135", "TỔNG NỘ PHẢI TRẢ", "31/12/2024"),
    source("selector_total_liabilities_2025", "VAB_financial_statements_2025_consolidated|333", 13, 2, "130.330.504.529.167", "TỔNG NỘ PHẢI TRẢ", "Số cuối năm"),
    source("unselected_target_materials_tools_2024", "VAB_financial_statements_2024_consolidated|1336", 7, 1, "3.609.504.412", "- Vật liệu và công cụ", "31/12/2024"),
    source("selected_target_materials_tools_2025", "VAB_financial_statements_2025_consolidated|1323", 1, 1, "4.552.242.279", "- Vật liệu và công cụ", "Số cuối năm"),
]

HARD_NEGATIVES = [
    {
        **source("wrong_selector_total_liabilities_and_equity_2025", "VAB_financial_statements_2025_consolidated|333", 22, 2, "140.485.531.667.485", "TỔNG NỘ PHẢI TRẢ VÀ VỐN CHỦ SỞ HỮU", "Số cuối năm"),
        "why_wrong": "This broader balance-sheet total includes equity; the selector explicitly asks total liabilities only.",
    },
    {
        **source("wrong_period_total_liabilities_opening_2025", "VAB_financial_statements_2025_consolidated|333", 13, 3, "110.975.359.140.135", "TỔNG NỘ PHẢI TRẢ", "Số đầu năm"),
        "why_wrong": "The opening column is the prior-period balance, not the requested 2025 ending balance.",
    },
    {
        **source("wrong_target_broad_other_assets_total_2025", "VAB_financial_statements_2025_consolidated|1323", 4, 1, "903.401.169.785", "Cộng", "Số cuối năm"),
        "why_wrong": "The total includes deferred costs and other assets; the target is the material-and-tools component only.",
        "counterfactual_billion_vnd": 903.4,
    },
    {
        **source("wrong_period_materials_tools_opening_2025", "VAB_financial_statements_2025_consolidated|1323", 1, 2, "3.609.504.412", "- Vật liệu và công cụ", "Số đầu năm"),
        "why_wrong": "This is the 2025 opening/2024 ending balance, not the selected year's ending target.",
        "counterfactual_billion_vnd": 3.61,
    },
]


def main() -> int:
    submission_path = CANDIDATE / "submission.json"
    rows = json.loads(submission_path.read_text(encoding="utf-8"))
    row = next(item for item in rows if int(item["id"]) == 515)
    finding_payload = json.loads(FINDING.read_text(encoding="utf-8"))
    finding_rows = finding_payload.get("records", finding_payload)
    finding = next(item for item in finding_rows if int(item["id"]) == 515)
    if str(finding["question"]) != str(row["question"]):
        raise ValueError("finding/candidate question mismatch")

    catalog = {
        item["table_ref"]: item
        for item in (
            json.loads(line)
            for line in CATALOG.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    def resolve(spec: dict[str, Any]) -> tuple[dict[str, Any], Decimal]:
        entry = catalog[spec["table_ref"]]
        csv_path = ROOT / "build" / "tables" / entry["csv_path"]
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        raw = str(frame.iloc[spec["row_idx"], spec["col_idx"]]).strip()
        label = str(frame.iloc[spec["row_idx"], 0]).strip()
        header = str(frame.iloc[0, spec["col_idx"]]).strip()
        if raw != spec["expected_raw"]:
            raise ValueError(f"stale raw at {spec['table_ref']}: {raw!r}")
        if spec["expected_label"].casefold() not in label.casefold():
            raise ValueError(f"stale label at {spec['table_ref']}: {label!r}")
        if spec["expected_header"].casefold() not in header.casefold():
            raise ValueError(f"stale header at {spec['table_ref']}: {header!r}")
        value = parse_number(raw)
        return (
            {
                "role": spec["role"],
                "table_ref": spec["table_ref"],
                "report_id": entry["report_id"],
                "ticker": entry["ticker"],
                "year": str(entry["year"]),
                "scope": entry["scope"],
                "page": int(entry["page"]),
                "line": int(entry["line"]),
                "catalog_csv": entry["csv_path"],
                "resolved_path": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "csv_sha256": sha256(csv_path),
                "row_idx": spec["row_idx"],
                "col_idx": spec["col_idx"],
                "row_label": label,
                "column_header": header,
                "raw": raw,
                "parsed_value": str(value),
                "unit": spec["unit"],
            },
            value,
        )

    physical_refs: list[dict[str, Any]] = []
    values: dict[str, Decimal] = {}
    for spec in SOURCES:
        resolved, value = resolve(spec)
        physical_refs.append(resolved)
        values[spec["role"]] = value
    checked_negatives: list[dict[str, Any]] = []
    for negative in HARD_NEGATIVES:
        resolved, _value = resolve(negative)
        checked_negatives.append(
            {
                **resolved,
                "why_wrong": negative["why_wrong"],
                **(
                    {"counterfactual_billion_vnd": negative["counterfactual_billion_vnd"]}
                    if "counterfactual_billion_vnd" in negative
                    else {}
                ),
            }
        )

    selector_values = {
        2020: values["selector_total_liabilities_2020"],
        2024: values["selector_total_liabilities_2024"],
        2025: values["selector_total_liabilities_2025"],
    }
    selected_year = max(selector_values, key=selector_values.__getitem__)
    if selected_year != 2025:
        raise ValueError(f"unexpected selected year: {selected_year}")
    target_vnd = values["selected_target_materials_tools_2025"]
    exact_billion = target_vnd / Decimal(10) ** 9
    recomputed = float(
        exact_billion.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )
    if recomputed != float(row["answer"]):
        raise ValueError(f"answer mismatch: {row['answer']} vs {recomputed}")

    # Replay the submitted branch directly from its manifest values. This is a
    # separate DAG check, not the physical-source recomputation above.
    manifest = pd.read_csv(
        CANDIDATE / "data" / "q515_source_cells.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    manifest_values = [parse_number(raw) for raw in manifest["raw"]]
    branch_result = (
        manifest_values[4] / Decimal(10) ** 9
        if manifest_values[2] == max(manifest_values[:3])
        else manifest_values[3] / Decimal(10) ** 9
    )
    if branch_result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) != Decimal("4.55"):
        raise ValueError("submitted program DAG replay mismatch")

    report = {
        "schema_version": "v280-q515-direction-review/v1",
        "mode": "read_only",
        "inputs": {
            "candidate_submission": {
                "path": str(submission_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(submission_path),
            },
            "direction_finding": {
                "path": str(FINDING.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(FINDING),
            },
            "catalog": {
                "path": str(CATALOG.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(CATALOG),
            },
        },
        "id": 515,
        "question": row["question"],
        "verdict": "false_positive",
        "answer_change": False,
        "cleanup_only": False,
        "current_answer": float(row["answer"]),
        "recomputed_answer": recomputed,
        "finding_adjudication": {
            "trigger": "Directional/sign audit associated the word 'nợ' with a negative-valued metric.",
            "decision": "False positive: total liabilities are a positive balance-sheet stock. 'Cao nhất' means the maximum positive closing balance; no sign inversion or absolute-value rewrite is appropriate.",
            "selector_metric": "total_liabilities",
            "target_metric": "materials_and_tools",
            "roles_are_distinct": True,
        },
        "program_dag": {
            "stage_1_selector": {
                "operation": "argmax",
                "metric": "total_liabilities_closing_vnd",
                "candidates": {str(year): str(value) for year, value in selector_values.items()},
                "selected_year": selected_year,
            },
            "stage_2_target": {
                "metric": "materials_and_tools_closing_vnd",
                "selected_year": selected_year,
                "value_vnd": str(target_vnd),
                "conversion": "value_vnd / 1e9",
                "exact_billion_vnd": str(exact_billion),
                "rounding": "two decimal places; half-up (not a tie)",
                "answer_billion_vnd": recomputed,
            },
            "submitted_branch_replay_billion_vnd": str(branch_result),
            "completeness_note": "The submitted code includes an unnecessary 2024 fallback and does not load a 2020 target. This cannot affect the fixed-source result because the independently verified selector uniquely chooses 2025; it is not the alleged direction error.",
        },
        "physical_refs": physical_refs,
        "hard_negatives": checked_negatives,
        "metadata_observation": {
            "2024_target_context": "The packet inherited the preceding 10.2 heading, but the physical CSV row 6 starts 10.3 Tài sản Có khác and row 7 is the requested component. Source value and answer remain correct.",
            "impact": "none",
        },
        "proposed_mutation": None,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reloaded = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert reloaded["id"] == 515
    assert reloaded["verdict"] == "false_positive"
    assert reloaded["current_answer"] == reloaded["recomputed_answer"] == 4.55
    assert len(reloaded["physical_refs"]) == 5
    assert len(reloaded["hard_negatives"]) == 4
    assert reloaded["program_dag"]["stage_1_selector"]["selected_year"] == 2025
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "verdict": reloaded["verdict"],
                "current_answer": 4.55,
                "recomputed_answer": 4.55,
                "selected_year": 2025,
                "physical_refs": 5,
                "hard_negatives": 4,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
