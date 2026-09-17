"""Audit selected cells against semantically competing sibling columns.

Financial tables often put two plausible values on the same row: voting vs
economic-interest rates, original cost vs carrying amount, or maturity buckets
vs a total.  Row-label audits cannot distinguish them.  This read-only oracle
uses the physical column header path and only reports a mismatch when a sibling
column in the *same row* contains the requested concept more clearly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from audit_legacy_source_scope import fold
    from audit_source_cell_semantics import header_path, is_financial_number, parsed_table
except ModuleNotFoundError:  # imported as ``scripts.audit_column_header_contracts``
    from scripts.audit_legacy_source_scope import fold
    from scripts.audit_source_cell_semantics import (
        header_path,
        is_financial_number,
        parsed_table,
    )


ROOT = Path(__file__).resolve().parents[1]

# A contract fires only when the question asks for ``question_cues``, the
# selected header misses ``wanted_header_cues``, and a numeric sibling column
# on the same row contains a wanted cue.  The sibling requirement makes these
# substantially safer than free-text context matching.
CONTRACTS = [
    {
        "name": "parent_company_column",
        "question_cues": ("cong ty me", "bao cao rieng"),
        "wanted_header_cues": ("cong ty",),
        "forbidden_selected_cues": ("tap doan", "hop nhat"),
    },
    {
        "name": "voting_rights",
        "question_cues": ("quyen bieu quyet", "ty le bieu quyet"),
        "wanted_header_cues": ("bieu quyet",),
    },
    {
        "name": "economic_interest",
        "question_cues": ("ty le loi ich", "loi ich kinh te"),
        "wanted_header_cues": ("loi ich",),
    },
    {
        "name": "ownership_rate",
        "question_cues": ("ty le so huu",),
        "wanted_header_cues": ("so huu",),
    },
    {
        "name": "original_cost",
        "question_cues": ("nguyen gia", "gia goc"),
        "wanted_header_cues": ("nguyen gia", "gia goc"),
    },
    {
        "name": "accumulated_depreciation",
        "question_cues": ("hao mon luy ke", "khau hao luy ke"),
        "wanted_header_cues": ("hao mon", "khau hao luy ke"),
    },
    {
        "name": "carrying_amount",
        "question_cues": ("gia tri con lai", "gia tri ghi so"),
        "wanted_header_cues": ("gia tri con lai", "gia tri ghi so"),
    },
    {
        "name": "fair_value",
        "question_cues": ("gia tri hop ly",),
        "wanted_header_cues": ("gia tri hop ly",),
    },
    {
        "name": "principal",
        "question_cues": ("tien goc", "du no goc"),
        "wanted_header_cues": ("goc",),
    },
    {
        "name": "interest",
        "question_cues": ("tien lai", "lai phai thu", "lai phai tra"),
        "wanted_header_cues": ("lai",),
    },
    {
        "name": "amount",
        "question_cues": ("so tien", "gia tri"),
        "wanted_header_cues": ("so tien", "gia tri", "vnd", "dong"),
        "forbidden_selected_cues": ("ty le", "%"),
    },
    {
        "name": "percentage",
        "question_cues": ("bao nhieu phan tram", "bao nhieu %", "ty le", "ty trong"),
        "wanted_header_cues": ("ty le", "%", "ty trong"),
        "forbidden_selected_cues": ("so tien", "gia tri"),
    },
    {
        "name": "total_bucket",
        # ``tong gia tri con lai cua <asset class>`` commonly qualifies one
        # component column rather than asking for the table-wide Total column.
        # Keep this contract for explicit aggregate nouns only.
        "question_cues": ("tong so",),
        "wanted_header_cues": ("tong cong", "tong so", "tong", "cong"),
        "forbidden_selected_cues": (
            "duoi 1 thang",
            "den 1 thang",
            "tu 1 den 3 thang",
            "tren 12 thang",
            "sau 12 thang",
        ),
    },
    {
        "name": "current_term",
        "question_cues": ("ngan han",),
        "wanted_header_cues": ("ngan han", "duoi 12 thang", "den 12 thang"),
        "forbidden_selected_cues": ("dai han", "tren 12 thang", "sau 12 thang"),
    },
    {
        "name": "noncurrent_term",
        "question_cues": ("dai han",),
        "wanted_header_cues": ("dai han", "tren 12 thang", "sau 12 thang"),
        "forbidden_selected_cues": ("ngan han", "duoi 12 thang", "den 12 thang"),
    },
    {
        "name": "domestic",
        "question_cues": ("trong nuoc", "noi dia"),
        "wanted_header_cues": ("trong nuoc", "noi dia"),
        "forbidden_selected_cues": ("nuoc ngoai", "xuat khau"),
    },
    {
        "name": "foreign",
        "question_cues": ("nuoc ngoai", "xuat khau"),
        "wanted_header_cues": ("nuoc ngoai", "xuat khau"),
        "forbidden_selected_cues": ("trong nuoc", "noi dia"),
    },
]


def contains_any(text: str, cues: tuple[str, ...]) -> bool:
    """Match folded semantic cues as complete phrases, not substrings.

    Substring matching turns ``ngan hang`` (bank) into a false match for
    ``ngan han`` (current/short-term).  Most cues are folded ASCII phrases,
    while a few are punctuation markers such as ``%``; boundaries are only
    required on alphanumeric cue edges so both forms remain supported.
    """

    for cue in cues:
        if not cue:
            continue
        prefix = r"(?<![0-9a-z])" if cue[0].isalnum() else ""
        suffix = r"(?![0-9a-z])" if cue[-1].isalnum() else ""
        if re.search(prefix + re.escape(cue) + suffix, text):
            return True
    return False


def question_matches_contract(contract: dict, folded: str, original: str) -> bool:
    """Evaluate question cues while preserving accent-sensitive distinctions."""

    if contract["name"] == "total_bucket":
        # Folding merges Vietnamese ``tổng cộng`` and ``Tổng Công ty`` into
        # the same ASCII phrase.  Match the former on the original text and
        # keep ``tổng số`` as the unambiguous folded cue.
        return "tổng cộng" in original.casefold() or contains_any(
            folded, contract["question_cues"]
        )
    return contains_any(folded, contract["question_cues"])


def sibling_headers(source_table: str, row: int) -> list[dict]:
    frame = parsed_table(source_table)
    siblings = []
    for column in range(len(frame.columns)):
        raw = frame.iloc[row, column]
        if not is_financial_number(raw):
            continue
        path = header_path(frame, column, row)
        siblings.append(
            {
                "column": column,
                "raw": str(raw),
                "header_path": path,
                "header_folded": fold(" | ".join(path)),
            }
        )
    return siblings


def audit(lineage_path: Path) -> dict:
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    findings = []
    checked_cells = 0
    cells_with_numeric_siblings = 0
    for record in lineage.get("records", []):
        original_question = str(record.get("question", ""))
        question = fold(original_question)
        record_cells = record.get("cells", [])
        direct_lookup = len(record_cells) == 1
        for index, cell in enumerate(record_cells):
            checked_cells += 1
            source_table = str(cell.get("source_table", ""))
            row = int(cell.get("row_idx", -1))
            selected_column = int(cell.get("col_idx", -1))
            try:
                siblings = sibling_headers(source_table, row)
            except Exception:
                continue
            if len(siblings) < 2:
                continue
            cells_with_numeric_siblings += 1
            selected = next(
                (item for item in siblings if item["column"] == selected_column), None
            )
            if selected is None:
                continue
            selected_header = selected["header_folded"]
            selected_semantics = fold(
                f"{selected_header} | {selected.get('raw', '')}"
            )
            alternatives = [item for item in siblings if item["column"] != selected_column]

            document = source_table.split("|", 1)[0]
            if "_consolidated" in document and contains_any(
                selected_header, ("cong ty",)
            ):
                group_columns = [
                    item
                    for item in alternatives
                    if contains_any(item["header_folded"], ("tap doan", "hop nhat"))
                ]
                if group_columns and "cong ty me" not in question and "bao cao rieng" not in question:
                    findings.append(
                        {
                            "id": int(record["id"]),
                            "cell_index": index,
                            "contract": "consolidated_group_column",
                            "question": record.get("question"),
                            "answer": record.get("answer"),
                            "source_table": source_table,
                            "row": row,
                            "selected": {
                                key: selected[key]
                                for key in ("column", "raw", "header_path")
                            },
                            "better_siblings": [
                                {
                                    key: item[key]
                                    for key in ("column", "raw", "header_path")
                                }
                                for item in group_columns
                            ],
                            "claim_limit": "same-row group/company review; requested reporting scope must be confirmed",
                        }
                    )

            for contract in CONTRACTS:
                question_cues = contract["question_cues"]
                wanted = contract["wanted_header_cues"]
                forbidden = contract.get("forbidden_selected_cues", ())
                if contract["name"] in {"amount", "percentage", "total_bucket"} and not direct_lookup:
                    continue
                if not question_matches_contract(
                    contract, question, original_question
                ):
                    continue
                # The generic amount contract is only meaningful when the
                # selected column explicitly looks like a percentage.  A bare
                # numeric header plus an arbitrary VND sibling is not evidence
                # that the chosen amount column is wrong.
                if contract["name"] == "amount" and not contains_any(
                    selected_semantics, forbidden
                ):
                    continue
                if contains_any(selected_semantics, wanted) and not contains_any(
                    selected_semantics, forbidden
                ):
                    continue
                matching = [
                    item
                    for item in alternatives
                    if contains_any(
                        fold(f"{item['header_folded']} | {item.get('raw', '')}"),
                        wanted,
                    )
                    and not contains_any(
                        fold(f"{item['header_folded']} | {item.get('raw', '')}"),
                        forbidden,
                    )
                ]
                # If a header band is blank/generic in all columns, this audit
                # has no semantic basis to choose a sibling.
                if not matching:
                    continue
                findings.append(
                    {
                        "id": int(record["id"]),
                        "cell_index": index,
                        "contract": contract["name"],
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "source_table": source_table,
                        "row": row,
                        "selected": {
                            key: selected[key]
                            for key in ("column", "raw", "header_path")
                        },
                        "better_siblings": [
                            {key: item[key] for key in ("column", "raw", "header_path")}
                            for item in matching
                        ],
                        "claim_limit": "same-row sibling review; recompute full program before mutation",
                    }
                )
    unique_ids = sorted({row["id"] for row in findings})
    return {
        "kind": "column_header_semantic_contracts",
        "lineage": str(lineage_path),
        "checked_cells": checked_cells,
        "cells_with_numeric_siblings": cells_with_numeric_siblings,
        "finding_count": len(findings),
        "question_count": len(unique_ids),
        "question_ids": unique_ids,
        "policy": "Read-only. Same-row sibling evidence is necessary but not sufficient for repair.",
        "findings": findings,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lineage",
        nargs="?",
        type=Path,
        default=ROOT / "build" / "v210_source_cell_lineage_v209.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "v210_column_header_contracts_v209.json",
    )
    args = parser.parse_args()
    payload = audit(args.lineage.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "findings"}, ensure_ascii=False, indent=2))
    if payload["findings"]:
        print(json.dumps({"findings": payload["findings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
