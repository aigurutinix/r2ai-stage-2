from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_exact_value_semantic_alternatives import coordinate_semantics  # noqa: E402


def test_high_confidence_when_candidate_row_recovers_question_phrase():
    frame = pd.DataFrame(
        {
            "0": ["Chi phí hoa hồng môi giới"],
            "Năm 2023 VND": ["42.745.665.072"],
        }
    )
    findings = coordinate_semantics(
        question="Chi phí hoa hồng môi giới năm 2023 là bao nhiêu?",
        raw="42.745.665.072",
        year=2023,
        current_text="Chi phí tài chính",
        current_label="Giá vốn dịch vụ",
        alternative_frame=frame,
        alternative_text="Chi tiết chi phí hoa hồng môi giới",
    )
    assert findings
    assert findings[0]["confidence"] == "high"
    assert "hoa hong" in findings[0]["recovered_label_bigrams"]


def test_reject_equal_value_under_wrong_year():
    frame = pd.DataFrame(
        {
            "0": ["Chi phí hoa hồng môi giới"],
            "Năm 2022 VND": ["42.745.665.072"],
        }
    )
    findings = coordinate_semantics(
        question="Chi phí hoa hồng môi giới năm 2023 là bao nhiêu?",
        raw="42.745.665.072",
        year=2023,
        current_text="Chi phí tài chính",
        current_label="Giá vốn dịch vụ",
        alternative_frame=frame,
        alternative_text="Chi tiết chi phí hoa hồng môi giới",
    )
    assert findings == []


def test_reject_context_only_gain_when_candidate_label_is_weaker():
    frame = pd.DataFrame(
        {
            "0": ["Tổng cộng"],
            "Năm 2023 VND": ["42.745.665.072"],
        }
    )
    findings = coordinate_semantics(
        question="Chi phí hoa hồng môi giới năm 2023 là bao nhiêu?",
        raw="42.745.665.072",
        year=2023,
        current_text="Chi phí tài chính",
        current_label="Giá vốn dịch vụ",
        alternative_frame=frame,
        alternative_text="Chi tiết chi phí hoa hồng môi giới",
    )
    assert findings == []


def test_reject_cross_statement_duplicate_for_metric_family():
    frame = pd.DataFrame(
        {
            "0": ["Lợi nhuận trước thuế Điều chỉnh cho các khoản"],
            "Năm 2023 VND": ["7.792.728.743.173"],
        }
    )
    findings = coordinate_semantics(
        question="Lợi nhuận trước thuế năm 2023 là bao nhiêu?",
        raw="7.792.728.743.173",
        year=2023,
        current_text="Báo cáo kết quả hoạt động kinh doanh",
        current_label="Tổng lợi nhuận kế toán trước thuế",
        alternative_frame=frame,
        alternative_text="Báo cáo lưu chuyển tiền tệ - lưu chuyển tiền từ hoạt động kinh doanh",
        metric_key="kqkd:50",
    )
    assert findings == []


def test_reject_cross_statement_duplicate_when_family_only_in_ocr_header():
    frame = pd.DataFrame(
        [
            ["Mã số", "CHỈ TIÊU", "Năm 2016 VND"],
            ["", "III. LU'U CHUYỀN TIỀN TỪ HOẠT ĐỘNG TÀI CHÍNH", "III. LU'U CHUYỀN TIỀN TỪ HOẠT ĐỘNG TÀI CHÍNH"],
            ["70", "Tiền và tương đương tiền cuối năm", "287.578.924.583"],
        ]
    )
    findings = coordinate_semantics(
        question="Số dư tiền và tương đương tiền ngắn hạn cuối năm 2016 là bao nhiêu?",
        raw="287.578.924.583",
        year=2016,
        current_text="Báo cáo tình hình tài chính",
        current_label="Tiền và các khoản tương đương tiền",
        alternative_frame=frame,
        alternative_text="Năm 2016 VND",
        metric_key="cdkt:110",
    )
    assert findings == []


def test_reject_placeholder_as_exact_value():
    frame = pd.DataFrame({"0": ["Tiền gửi và vay các TCTD khác"], "Năm 2021": ["-"]})
    findings = coordinate_semantics(
        question="Tiền vay các bên liên quan năm 2021 là bao nhiêu?",
        raw="-",
        year=2021,
        current_text="Tiền vay tại Ngân hàng",
        current_label="Tiền vay tại Ngân hàng",
        alternative_frame=frame,
        alternative_text="Tiền gửi và vay các TCTD khác",
        metric_key="note:related_party_loans",
    )
    assert findings == []


def test_reject_ending_shares_replaced_by_average_shares():
    frame = pd.DataFrame(
        {
            "0": ["Cổ phiếu phổ thông đang lưu hành bình quân trong kỳ"],
            "Năm 2020": ["268.631.965"],
        }
    )
    findings = coordinate_semantics(
        question="Số lượng cổ phiếu phổ thông đang lưu hành cuối năm 2020 là bao nhiêu?",
        raw="268.631.965",
        year=2020,
        current_text="Số lượng cổ phiếu đang lưu hành",
        current_label="Số lượng cổ phiếu đang lưu hành",
        alternative_frame=frame,
        alternative_text="Lãi cơ bản trên cổ phiếu",
        metric_key="note:ending_common_shares_outstanding",
    )
    assert findings == []


def test_reject_total_replaced_by_component():
    frame = pd.DataFrame(
        {
            "0": ["Giá vốn dịch vụ môi giới bất động sản"],
            "Năm 2023": ["89.144.410.598"],
        }
    )
    findings = coordinate_semantics(
        question="Tổng chi phí hoa hồng môi giới bất động sản năm 2023 là bao nhiêu?",
        raw="89.144.410.598",
        year=2023,
        current_text="Chi phí hoa hồng môi giới bất động sản",
        current_label="Cộng",
        alternative_frame=frame,
        alternative_text="Chi tiết giá vốn dịch vụ môi giới bất động sản",
        metric_key="note:brokerage_commission",
    )
    assert findings == []


def test_colocated_operands_downgrade_exact_duplicate_to_review():
    frame = pd.DataFrame(
        {
            "0": ["Vay dài hạn"],
            "Số cuối năm": ["8.356.636.117.641"],
        }
    )
    findings = coordinate_semantics(
        question="Tỷ trọng vay USD trong tổng vay dài hạn cuối năm là bao nhiêu?",
        raw="8.356.636.117.641",
        year=2024,
        current_text="Chi tiết dư nợ theo loại tiền",
        current_label="",
        alternative_frame=frame,
        alternative_text="Vay dài hạn",
        metric_key="note:total_long_term_loans",
        current_table_cell_count=2,
    )
    assert findings
    assert findings[0]["confidence"] == "review"
    assert findings[0]["context_rule"] == "preserve-co-located-operands"
