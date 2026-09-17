from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_total_row_component_semantics import (  # noqa: E402
    additive_total_candidate,
    financial_number,
    finding_is_covered_by_selected_cells,
    question_requests_total,
    total_targets,
)


def test_financial_number_preserves_parenthesized_sign() -> None:
    assert financial_number("(1.234.567)") == -1234567


def test_question_requires_explicit_total_intent() -> None:
    assert question_requests_total("Tổng dự phòng rủi ro cuối năm là bao nhiêu?")
    assert not question_requests_total("Số dư tại Việt Nam là bao nhiêu?")
    assert not question_requests_total("Công ty mẹ Tổng Công ty Khí Việt Nam")
    assert total_targets("Tổng dự phòng rủi ro của ngân hàng là bao nhiêu?") == [
        "du phong rui ro"
    ]


def test_detects_blank_additive_total_after_selected_component() -> None:
    frame = pd.DataFrame([
        ["", "Số cuối năm"],
        ["Dự phòng rủi ro cho vay khách hàng tại Việt Nam", "39.850.765"],
        ["Dự phòng rủi ro cho vay khách hàng tại nước ngoài", "618.295"],
        ["", "40.469.060"],
    ])
    finding = additive_total_candidate(
        frame,
        1,
        1,
        "Tổng dự phòng rủi ro cho vay khách hàng là bao nhiêu?",
    )
    assert finding is not None
    assert finding["candidate_row"] == 3
    assert finding["candidate_raw"] == "40.469.060"
    assert [row["row"] for row in finding["component_rows"]] == [1, 2]


def test_detects_explicit_total_label() -> None:
    frame = pd.DataFrame([
        ["Tiền gửi trong nước", "100"],
        ["Tiền gửi nước ngoài", "20"],
        ["Tổng cộng", "120"],
    ])
    finding = additive_total_candidate(frame, 0, 1, "Tổng tiền gửi là bao nhiêu?")
    assert finding is not None
    assert finding["candidate_label"] == "Tổng cộng"


def test_does_not_flag_non_additive_following_row() -> None:
    frame = pd.DataFrame([
        ["Trong nước", "100"],
        ["Nước ngoài", "20"],
        ["", "130"],
    ])
    assert additive_total_candidate(frame, 0, 1, "Tổng giá trị là bao nhiêu?") is None


def test_does_not_flag_when_selected_row_is_already_total() -> None:
    frame = pd.DataFrame([
        ["Tổng cộng", "120"],
        ["", "120"],
    ])
    assert additive_total_candidate(frame, 0, 1, "Tổng giá trị là bao nhiêu?") is None


def test_does_not_replace_an_explicitly_requested_component() -> None:
    frame = pd.DataFrame([
        ["Vay dài hạn ngân hàng", "100"],
        ["Nợ thuê tài chính", "20"],
        ["Tổng cộng", "120"],
    ])
    question = "Tổng số dư vay dài hạn ngân hàng là bao nhiêu?"
    assert additive_total_candidate(frame, 0, 1, question) is None


def test_keeps_currency_qualifier_after_non_entity_cua_phrase() -> None:
    frame = pd.DataFrame([
        ["Tiền gửi không kỳ hạn bằng VND", "100"],
        ["Tiền gửi không kỳ hạn bằng ngoại tệ", "20"],
        ["Tiền gửi có kỳ hạn bằng VND", "300"],
        ["", "420"],
    ])
    question = (
        "Tổng tiền gửi không kỳ hạn cùng tiền gửi có kỳ hạn của các tổ chức "
        "tín dụng khác bằng VND là bao nhiêu?"
    )
    assert additive_total_candidate(frame, 0, 1, question) is None


def test_does_not_mix_selector_total_with_unrelated_output_cell() -> None:
    frame = pd.DataFrame([
        ["Chi phí khấu hao", "10"],
        ["Chi phí khác", "20"],
        ["Tổng cộng", "30"],
    ])
    question = "Chi phí khấu hao trong năm có tổng nợ vay cao nhất là bao nhiêu?"
    assert additive_total_candidate(frame, 0, 1, question) is None


def test_suppresses_when_program_already_reads_all_components() -> None:
    finding = {
        "candidate_row": 3,
        "component_rows": [{"row": 1}, {"row": 2}],
    }
    selected = {("DOC|1", 1, 1), ("DOC|1", 2, 1)}
    assert finding_is_covered_by_selected_cells(finding, "DOC|1", 1, selected)
