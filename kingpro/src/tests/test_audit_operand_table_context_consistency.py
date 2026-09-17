from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_operand_table_context_consistency import classify_context  # noqa: E402


FAMILIES = {
    "investment_property": ("bat dong san dau tu",),
    "intangible_fixed_assets": ("tai san co dinh vo hinh", "tscd vo hinh"),
    "tangible_fixed_assets": ("tai san co dinh huu hinh", "tscd huu hinh"),
}


def test_q769_investment_property_is_classified_from_title() -> None:
    family, _ = classify_context(
        {"header_path": ["31/12/2015 VND"], "source_context": "14. BẤT ĐỘNG SẢN ĐẦU TƯ"},
        {"section_title": "14. BẤT ĐỘNG SẢN ĐẦU TƯ", "search_text": "- Quyền sử dụng đất"},
        FAMILIES,
    )
    assert family == "investment_property"


def test_transfer_from_investment_property_does_not_override_tangible_title() -> None:
    family, _ = classify_context(
        {
            "header_path": ["Tổng", "NGUYÊN GIÁ"],
            "source_context": "12. TĂNG, GIẢM TÀI SẢN CỐ ĐỊNH HỮU HÌNH",
        },
        {
            "section_title": "12. TĂNG, GIẢM TÀI SẢN CỐ ĐỊNH HỮU HÌNH",
            "search_text": "Tăng từ bất động sản đầu tư | Số dư cuối năm",
        },
        FAMILIES,
    )
    assert family == "tangible_fixed_assets"


def test_weak_incidental_phrase_is_not_enough_to_classify() -> None:
    family, _ = classify_context(
        {"header_path": ["Tổng"], "source_context": "Báo cáo tài chính"},
        {"section_title": "", "search_text": "Chuyển từ xây dựng cơ bản dở dang"},
        FAMILIES,
    )
    assert family is None


def test_previous_section_noise_yields_to_last_actual_heading() -> None:
    family, _ = classify_context(
        {
            "header_path": ["Tổng cộng"],
            "source_context": (
                "Các thay đổi khác về TSCĐ hữu hình | "
                "A.7.11. Tình hình tăng, giảm TSCĐ vô hình"
            ),
        },
        {
            "section_title": (
                "Các thay đổi khác về TSCĐ hữu hình | "
                "A.7.11. Tình hình tăng, giảm TSCĐ vô hình"
            ),
            "search_text": "Giá trị còn lại của TSCĐ vô hình",
        },
        FAMILIES,
    )
    assert family == "intangible_fixed_assets"
