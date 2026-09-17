from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_selected_cell_collisions.py"
SPEC = importlib.util.spec_from_file_location("audit_selected_cell_collisions", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(MODULE)


def _record(qid: int, question: str, table: str, raw: str, context: str) -> dict:
    return {
        "id": qid,
        "question": question,
        "answer": 1.0,
        "cells": [
            {
                "source_table": table,
                "row_idx": 0,
                "col_idx": 1,
                "source_label": "Số dư cuối năm",
                "source_context": context,
                "raw_physical": raw,
            }
        ],
    }


def test_selected_cells_find_scope_collision_without_opening_table_corpus() -> None:
    payload = {
        "records": [
            _record(
                10,
                "Số dư cuối năm của khoản phải thu ngắn hạn là bao nhiêu?",
                "AAA_financial_statements_2024_consolidated|100",
                "100",
                "Thuyết minh phải thu dài hạn | Số dư cuối năm",
            ),
            _record(
                11,
                "Số dư cuối năm của khoản phải thu ngắn hạn là bao nhiêu?",
                "AAA_financial_statements_2024_consolidated|200",
                "200",
                "Thuyết minh phải thu ngắn hạn | Số dư cuối năm",
            ),
        ]
    }

    result = MODULE.audit_payloads([payload])

    assert result["unique_physical_cells"] == 2
    assert result["reports_indexed"] == 1
    q10 = [row for row in result["findings"] if row["id"] == 10]
    assert q10
    assert q10[0]["priority"] >= 10
    assert "short_term" in " ".join(q10[0]["reasons"])


def test_equal_values_and_other_reports_are_not_collisions() -> None:
    payload = {
        "records": [
            _record(1, "Số dư cuối năm?", "AAA_report|1", "100", "Số dư cuối năm"),
            _record(2, "Số dư cuối năm?", "AAA_report|2", "100", "Số dư cuối năm"),
            _record(3, "Số dư cuối năm?", "BBB_report|1", "999", "Số dư cuối năm"),
        ]
    }

    result = MODULE.audit_payloads([payload])

    assert result["different_value_comparisons"] == 0
    assert result["finding_count"] == 0


def test_two_operands_used_by_same_question_are_not_competing_sources() -> None:
    record = {
        "id": 9,
        "question": "Chênh lệch số dư cuối năm giữa hai bảng là bao nhiêu?",
        "answer": 50.0,
        "cells": [
            {
                "source_table": "AAA_report|1",
                "row_idx": 0,
                "col_idx": 1,
                "source_label": "Số dư cuối năm",
                "source_context": "Năm nay",
                "raw_physical": "150",
            },
            {
                "source_table": "AAA_report|2",
                "row_idx": 0,
                "col_idx": 1,
                "source_label": "Số dư cuối năm",
                "source_context": "Năm trước",
                "raw_physical": "100",
            },
        ],
    }

    result = MODULE.audit_payloads([{"records": [record]}])

    assert result["different_value_comparisons"] == 0
    assert result["finding_count"] == 0


def test_coverage_reports_missing_lineage_and_unusable_cells() -> None:
    payload = {
        "records": [
            {
                "id": 1,
                "question": "Một câu hỏi",
                "cells": [
                    {
                        "source_table": "AAA_report|1",
                        "row_idx": 0,
                        "col_idx": 1,
                        "source_label": "",
                        "raw_physical": "100",
                    }
                ],
            }
        ]
    }

    result = MODULE.audit_payloads([payload], expected_ids={1, 2})

    assert result["missing_lineage_question_ids"] == [2]
    assert result["question_ids_without_usable_numeric_cells"] == [1, 2]
    assert result["skipped_cell_counts"] == {"missing_label": 1}
