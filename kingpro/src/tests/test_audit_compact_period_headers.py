from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_compact_period_headers import (  # noqa: E402
    is_explicit_period_descriptor,
    nearest_period_descriptor,
    report_year_from_table,
)


def test_report_year_comes_from_document_id() -> None:
    assert report_year_from_table("ABC_financial_statements_2024_separate|123") == 2024


def test_recognizes_explicit_and_relative_period_headers() -> None:
    assert is_explicit_period_descriptor("Năm nay Triệu đồng")
    assert is_explicit_period_descriptor("31/12/2024 VND")
    assert is_explicit_period_descriptor("Năm 2023")
    assert not is_explicit_period_descriptor("Số cuối năm")
    assert not is_explicit_period_descriptor("202.220.754.280")


def test_finds_closest_explicit_header() -> None:
    frame = pd.DataFrame([
        ["", "Năm 2024", "Năm 2023"],
        ["Doanh thu", "100.000", "90.000"],
    ])
    assert nearest_period_descriptor(frame, 1, 1) == "nam 2024"
    assert nearest_period_descriptor(frame, 1, 2) == "nam 2023"


def test_ignores_year_like_value_on_accounting_data_row() -> None:
    frame = pd.DataFrame([
        ["", "Năm 2024", "Năm 2023"],
        ["Số nhân viên", "2023", "150.000.000"],
        ["Chi phí", "120.000.000", "100.000.000"],
    ])
    assert nearest_period_descriptor(frame, 2, 1) == "nam 2024"
