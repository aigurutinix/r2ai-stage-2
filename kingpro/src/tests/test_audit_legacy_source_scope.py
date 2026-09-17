from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_legacy_source_scope.py"
SPEC = importlib.util.spec_from_file_location("audit_legacy_source_scope", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def rules(question: str, header: str, context: str = "", label: str = "Chỉ tiêu") -> set[str]:
    return {
        item.rule for item in module.classify(
            question=question,
            source_table="DOC|10",
            source_label=label,
            raw="1",
            source_header=header,
            context=context,
        )
    }


def test_ending_question_rejects_flow_header() -> None:
    assert "ending_question_flow_table" in rules(
        "Lãi vay phải trả cuối năm 2023 là bao nhiêu?",
        "0,1,2 | ,Năm nay,Năm trước | Lãi vay phải trả,26,-",
    )


def test_ending_question_accepts_balance_header() -> None:
    assert "ending_question_flow_table" not in rules(
        "Lãi vay phải trả cuối năm 2023 là bao nhiêu?",
        "0,1,2 | ,Số cuối năm,Số đầu năm | Lãi vay phải trả,350,284",
    )


def test_cashflow_ending_row_is_valid_despite_year_columns() -> None:
    assert "ending_question_flow_table" not in rules(
        "Tiền và tương đương tiền cuối năm 2024 là bao nhiêu?",
        "CHỈ TIÊU,Số năm nay,Số năm trước",
        label="Tiền và tương đương tiền cuối năm",
    )


def test_unqualified_question_flags_purchase_date_fair_value() -> None:
    assert "unqualified_question_acquisition_table" in rules(
        "Tổng chi phí xây dựng cơ bản dở dang cuối năm là bao nhiêu?",
        "Giá trị hợp lý được xác định tạm thời tại ngày mua",
        "Giao dịch hợp nhất kinh doanh | Mua Công ty Vicentra",
    )


def test_qualified_acquisition_question_is_not_flagged() -> None:
    assert "unqualified_question_acquisition_table" not in rules(
        "Giá trị hợp lý tại ngày mua của tài sản là bao nhiêu?",
        "Giá trị hợp lý được xác định tại ngày mua",
        "Hợp nhất kinh doanh",
    )


def test_collateral_subset_requires_question_qualifier() -> None:
    assert "unqualified_question_collateral_subset" in rules(
        "Trái phiếu Chính phủ là bao nhiêu?",
        "0,1,2 | ,Số cuối năm,Số đầu năm",
        context="Tài sản cầm cố, thế chấp",
        label="Trái phiếu Chính phủ",
    )


def test_acquisition_word_in_unrelated_movement_column_is_not_enough() -> None:
    assert "unqualified_question_acquisition_table" not in rules(
        "Thuế phải nộp đầu năm là bao nhiêu?",
        "1/1/2021,Số phải nộp,Tăng phải thu do mua công ty con",
        context="23. Thuế và các khoản phải nộp Nhà nước",
        label="Thuế thu nhập doanh nghiệp",
    )


def test_collateral_negation_is_not_a_restricted_subset() -> None:
    assert "unqualified_question_collateral_subset" not in rules(
        "Cho vay khách hàng là tổ chức cuối năm là bao nhiêu?",
        "0,1 | Cho vay khách hàng là tổ chức,100",
        context="Mức rủi ro tín dụng tối đa chưa tính đến tài sản thế chấp",
        label="Cho vay khách hàng là tổ chức",
    )


def test_unqualified_current_year_rejects_opening_comparative_table() -> None:
    assert "current_year_question_opening_table" in rules(
        "Tỷ lệ cam kết ngoại bảng trên tổng tài sản năm 2019 là bao nhiêu?",
        "Chỉ tiêu,Quá hạn,Không chịu lãi,Tổng",
        context="Bảng dưới đây tóm tắt rủi ro lãi suất tại ngày 01 tháng 01 năm 2019",
        label="Tổng Tài sản",
    )


def test_explicit_opening_question_accepts_opening_comparative_table() -> None:
    assert "current_year_question_opening_table" not in rules(
        "Tổng tài sản đầu năm 2019 là bao nhiêu?",
        "Chỉ tiêu,Quá hạn,Không chịu lãi,Tổng",
        context="Bảng dưới đây tóm tắt rủi ro lãi suất tại ngày 01 tháng 01 năm 2019",
        label="Tổng Tài sản",
    )


def test_table_with_both_date_column_groups_is_not_assumed_opening() -> None:
    assert "current_year_question_opening_table" not in rules(
        "Giá gốc hàng gửi đi bán cuối năm 2019 là bao nhiêu?",
        "0,1,2,3,4 | ,Giá gốc,Dự phòng,Giá gốc,Dự phòng",
        context="06. Hàng tồn kho | 31/12/2019 | 01/01/2019",
        label="Hàng gửi đi bán",
    )
