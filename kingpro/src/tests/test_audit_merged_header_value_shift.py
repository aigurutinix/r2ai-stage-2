from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_merged_header_value_shift.py"
SPEC = importlib.util.spec_from_file_location("audit_merged_header_value_shift", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def shifted_frame() -> pd.DataFrame:
    return pd.DataFrame([
        ["Mã số", "", "Thuyết minh", "Kỳ báo cáo", "Kỳ báo cáo"],
        ["Mã số", "", "Thuyết minh", "2023 VND", "2022 VND"],
        ["20", "CFO", "9.517", "5.053", "5.053"],
    ])


def test_detects_duplicated_prior_value_and_left_shift() -> None:
    finding = module.shifted_left_candidate(shifted_frame(), 2, 3, 2023)
    assert finding is not None
    assert finding["selected_raw"] == "5.053"
    assert finding["candidate_raw"] == "9.517"
    assert finding["candidate_column"] == 2


def test_requires_requested_year_in_selected_header() -> None:
    assert module.shifted_left_candidate(shifted_frame(), 2, 3, 2024) is None


def test_equal_left_value_is_not_actionable() -> None:
    frame = shifted_frame()
    frame.iloc[2, 2] = "5.053"
    assert module.shifted_left_candidate(frame, 2, 3, 2023) is None


def test_no_right_duplicate_is_not_actionable() -> None:
    frame = shifted_frame()
    frame.iloc[2, 4] = "4.000"
    assert module.shifted_left_candidate(frame, 2, 3, 2023) is None


def test_adjacent_dated_scope_is_not_a_shift() -> None:
    frame = pd.DataFrame([
        ["", "Mã số", "Thuyết minh", "Tập đoàn", "Tập đoàn", "Công ty", "Công ty"],
        ["", "Mã số", "Thuyết minh", "31/12/2017", "1/1/2017", "31/12/2017", "1/1/2017"],
        ["Nợ dài hạn", "330", "", "10.260", "11.306", "541", "541"],
    ])
    assert module.shifted_left_candidate(frame, 2, 5, 2017) is None
    assert module.undated_left_amount_candidate(frame, 2, 5, 2017) is None


def test_broader_hint_accepts_large_undated_left_amount() -> None:
    frame = shifted_frame()
    frame.iloc[2, 2] = "9.517.000"
    frame.iloc[2, 3] = "5.053.000"
    frame.iloc[2, 4] = "4.000.000"
    finding = module.undated_left_amount_candidate(frame, 2, 3, 2023)
    assert finding is not None
    assert finding["confidence"] == "review"
