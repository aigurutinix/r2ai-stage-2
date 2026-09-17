from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_source_cell_semantics.py"
SPEC = importlib.util.spec_from_file_location("audit_source_cell_semantics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_context_detects_general_specific_provision_collision() -> None:
    assert module.context_conflicts(
        "Số dư cuối kỳ dự phòng chung cho vay khách hàng là bao nhiêu?",
        "Biến động dự phòng cụ thể cho vay khách hàng",
    ) == [
        "question asks 'du phong chung' but source context only signals 'du phong cu the'"
    ]


def test_context_accepts_matching_provision_scope() -> None:
    assert module.context_conflicts(
        "Số dư cuối kỳ dự phòng chung cho vay khách hàng là bao nhiêu?",
        "Biến động dự phòng chung cho vay khách hàng",
    ) == []


def test_context_skips_multi_role_question_and_prefers_direct_row_semantics() -> None:
    assert module.context_conflicts(
        "Số dư ròng phải thu và phải trả là bao nhiêu?",
        "Phải thu",
    ) == []
    assert module.context_conflicts(
        "Số phải trả sau 12 tháng là bao nhiêu?",
        "Quyền phải thu ở đoạn lân cận",
        "Số phải trả sau 12 tháng note:payables_after_12_months",
    ) == []


def test_document_scope_detects_unqualified_question_using_separate_table() -> None:
    assert module.document_scope_conflicts(
        "Quỹ đầu tư phát triển cuối năm của DPM là bao nhiêu?",
        "DPM_financial_statements_2015_consolidated|272",
        "BÁO CÁO TÀI CHÍNH RIÊNG | BẢNG CÂN ĐỐI KẾ TOÁN",
    ) == ["unqualified consolidated question uses a physical separate table"]


def test_document_scope_detects_parent_question_using_consolidated_table() -> None:
    assert module.document_scope_conflicts(
        "Hàng tồn kho của công ty mẹ HUT là bao nhiêu?",
        "HUT_financial_statements_2024_separate|328",
        "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
    ) == ["question asks parent/separate scope but physical table is consolidated"]


def test_document_scope_accepts_physical_scope_even_in_swapped_container() -> None:
    assert module.document_scope_conflicts(
        "Quỹ đầu tư phát triển cuối năm của DPM là bao nhiêu?",
        "DPM_financial_statements_2015_consolidated|1909",
        "BÁO CÁO TÀI CHÍNH HỢP NHẤT | BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
    ) == []
    assert module.document_scope_conflicts(
        "Hàng tồn kho của công ty mẹ HUT là bao nhiêu?",
        "HUT_financial_statements_2024_consolidated|325",
        "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG",
    ) == []
    assert module.document_scope_conflicts(
        "Báo cáo công ty mẹ năm 2023 và báo cáo hợp nhất năm 2024",
        "DTK_financial_statements_2024_consolidated|318",
        "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
    ) == []


def test_header_path_stops_before_first_financial_data_row() -> None:
    frame = pd.DataFrame([
        ["", "Thuyết minh", "Năm nay Triệu đồng", "Năm trước Triệu đồng"],
        ["Thu nhập lãi", "25", "33.587.667", "30.476.971"],
        ["Chi phí lãi", "26", "(22.061.113)", "(21.296.283)"],
        ["Chi phí dự phòng", "", "(3.036.974)", "(2.500.000)"],
    ])
    assert module.header_path(frame, 2, 3) == ["Năm nay Triệu đồng"]


def test_header_path_preserves_multi_level_merged_ancestry() -> None:
    frame = pd.DataFrame([
        ["Triệu VND", "Quá hạn", "Trong hạn", "Trong hạn"],
        ["Triệu VND", "Trên 3 tháng", "Đến 1 tháng", "Tổng cộng"],
        ["Tài sản", "", "", ""],
        ["Tiền gửi khách hàng", "-", "66.076.449", "259.236.746"],
    ])
    assert module.header_path(frame, 3, 3) == ["Trong hạn", "Tổng cộng"]


def test_year_or_date_header_is_not_financial_data() -> None:
    assert not module.is_financial_number("31/12/2024 VND")
    assert not module.is_financial_number("Năm 2024")
    assert module.is_financial_number("(5.946.334.636.629)")
