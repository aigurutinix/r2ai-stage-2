from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cross_report_comparatives import (  # noqa: E402
    accounting_number,
    candidate_coordinates,
    unit_scale,
)


def test_accounting_number_preserves_parenthesized_negative():
    assert accounting_number("(3.036.974)") == -3036974


def test_unit_scale_recognizes_vietnamese_units():
    assert unit_scale("Năm trước Triệu đồng") == 1e6
    assert unit_scale("2023 Nghìn VND") == 1e3
    assert unit_scale("2023 VND") == 1


def test_candidate_coordinates_requires_same_code_and_explicit_year():
    frame = pd.DataFrame(
        [
            ["CHỈ TIÊU", "Mã số", "Năm 2024 VND", "Năm 2023 VND"],
            ["TỔNG CỘNG TÀI SẢN", "270", "12.000", "10.000"],
            ["TỔNG NGUỒN VỐN", "440", "12.000", "10.000"],
        ]
    )
    rows = candidate_coordinates(
        frame,
        code="270",
        requested_year=2023,
        table_context="Bảng cân đối kế toán",
    )
    assert len(rows) == 1
    assert rows[0]["raw"] == "10.000"
    assert rows[0]["label"] == "TỔNG CỘNG TÀI SẢN"


def test_candidate_coordinates_rejects_current_year_column():
    frame = pd.DataFrame(
        [
            ["CHỈ TIÊU", "Mã số", "Năm 2024 VND", "Năm 2023 VND"],
            ["TỔNG CỘNG TÀI SẢN", "270", "12.000", "10.000"],
        ]
    )
    rows = candidate_coordinates(
        frame,
        code="270",
        requested_year=2024,
        table_context="Bảng cân đối kế toán",
    )
    assert len(rows) == 1
    assert rows[0]["raw"] == "12.000"


def test_candidate_coordinates_requires_year_in_selected_header_lineage():
    frame = pd.DataFrame(
        [
            ["CHá»ˆ TIÃŠU", "MÃ£ sá»‘", "Thuyáº¿t minh", "NÄƒm 2023 VND"],
            ["HÃ ng tá»“n kho", "140", "9", "10.000"],
        ]
    )
    rows = candidate_coordinates(
        frame,
        code="140",
        requested_year=2023,
        table_context="Báº£ng cÃ¢n Ä‘á»‘i káº¿ toÃ¡n nÄƒm 2023",
    )
    assert [row["raw"] for row in rows] == ["10.000"]
