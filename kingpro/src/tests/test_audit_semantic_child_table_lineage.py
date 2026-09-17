from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_semantic_child_table_lineage import (  # noqa: E402
    additive_components,
    is_total_row_label,
    semantic_child_candidate,
)


def commission_detail() -> pd.DataFrame:
    return pd.DataFrame([
        ["", "Năm 2022", "Năm 2021"],
        ["", "VND", "VND"],
        ["Hoa hồng cho nhân viên", "271.127.387.123", "287.996.978.216"],
        ["Hoa hồng cho người mua, cộng tác viên", "9.981.134.594", "11.335.753.031"],
        ["Hoa hồng cho Đơn vị liên kết", "279.802.607.788", "10.895.832.716"],
        ["Cộng", "560.911.129.505", "310.228.563.963"],
    ])


def test_detects_semantically_exact_child_total() -> None:
    finding = semantic_child_candidate(
        question="Tổng chi phí hoa hồng môi giới bất động sản cao nhất vào năm nào?",
        year="2022",
        raw="560.911.129.505",
        current_search_text="Giá vốn dịch vụ môi giới bất động sản | Cộng",
        current_label="Giá vốn dịch vụ môi giới bất động sản",
        alternative_frame=commission_detail(),
        alternative_search_text="Chi tiết giá vốn theo nội dung chi phí | Hoa hồng cho nhân viên",
    )
    assert finding is not None
    assert finding["candidate_row"] == 5
    assert finding["candidate_column"] == 1
    assert {"hoa", "hong"} <= set(finding["missing_question_terms_recovered"])
    assert finding["confidence"] == "high"
    assert "hoa hong" in finding["recovered_repeated_component_phrases"]


def test_rejects_equal_value_in_comparative_year_column() -> None:
    assert semantic_child_candidate(
        question="Tổng chi phí hoa hồng môi giới bất động sản năm 2022 là bao nhiêu?",
        year="2022",
        raw="310.228.563.963",
        current_search_text="Giá vốn dịch vụ môi giới bất động sản",
        current_label="Giá vốn dịch vụ môi giới bất động sản",
        alternative_frame=commission_detail(),
        alternative_search_text="Chi tiết chi phí hoa hồng môi giới bất động sản",
    ) is None


def test_rejects_non_additive_total() -> None:
    frame = commission_detail()
    frame.iloc[5, 1] = "999"
    assert additive_components(frame, 5, 1) is None


def test_rejects_child_without_new_question_semantics() -> None:
    assert semantic_child_candidate(
        question="Giá vốn dịch vụ môi giới bất động sản năm 2022 là bao nhiêu?",
        year="2022",
        raw="560.911.129.505",
        current_search_text="Giá vốn dịch vụ môi giới bất động sản",
        current_label="Giá vốn dịch vụ môi giới bất động sản",
        alternative_frame=commission_detail(),
        alternative_search_text="Chi tiết giá vốn dịch vụ môi giới bất động sản",
    ) is None


def test_total_label_does_not_match_contributor_word() -> None:
    assert is_total_row_label("Cộng")
    assert is_total_row_label("TỔNG CỘNG")
    assert not is_total_row_label("Hoa hồng cho người mua, cộng tác viên")
