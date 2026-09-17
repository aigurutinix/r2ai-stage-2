from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_local_source_units.py"
SPEC = importlib.util.spec_from_file_location("audit_local_source_units", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_recognizes_vietnamese_currency_markers() -> None:
    assert module.unit_factor_from_marker("Ngàn VND") == 1_000.0
    assert module.unit_factor_from_marker("Đơn vị tính: nghìn đồng") == 1_000.0
    assert module.unit_factor_from_marker("Đơn vị tính: Triệu VND") == 1_000_000.0
    assert module.unit_factor_from_marker("Đơn vị tính: VND") == 1.0


def test_does_not_treat_long_table_body_as_unit_marker() -> None:
    lines = [
        "Đơn vị tính: VND",
        "<table>" + (" giao dịch bằng VND " * 100) + "</table>",
    ]
    marker = module.nearest_unit_marker(lines, 2)
    assert marker is not None
    assert marker["factor"] == 1.0
    assert marker["line"] == 1


def test_reads_unit_from_table_header_without_confusing_ngan_han() -> None:
    table = (
        "<table><tr><td></td><td>31.12.2022Triệu VND</td></tr>"
        "<tr><td>Ngắn hạn</td><td>263.259.964</td></tr></table>"
    )
    assert module.unit_factor_from_table_header(table) == 1_000_000.0


def test_vnd_followed_by_ratio_header_is_not_billion_vnd() -> None:
    table = (
        "<table><tr><td>Số vốn góp VND</td>"
        "<td>Tỷ lệ trên vốn điều lệ</td></tr>"
        "<tr><td>420.000.000.000</td><td>42%</td></tr></table>"
    )
    assert module.unit_factor_from_table_header(table) is None


def test_company_followed_by_vnd_is_not_billion_vnd() -> None:
    table = (
        "<table><tr><td>Tổng vốn chủ sở hữu của Công ty VND</td>"
        "<td>Chênh lệch tỷ giá hối đoái VND</td></tr>"
        "<tr><td>5.643.794.616.826</td><td>143.433.871.620</td></tr></table>"
    )
    assert module.unit_factor_from_table_header(table) is None


def test_reads_unit_from_second_header_row() -> None:
    table = (
        "<table><tr><td>Số cuối năm</td><td>Thời hạn</td></tr>"
        "<tr><td>Ngàn VND</td><td></td></tr>"
        "<tr><td>2.083.992.733</td><td>2021</td></tr></table>"
    )
    assert module.unit_factor_from_table_header(table) == 1_000.0


def test_nearest_marker_wins_when_report_changes_units() -> None:
    lines = [
        "Đơn vị tính: VND",
        "<table><tr><td>old</td></tr></table>",
        "Ngàn VND",
        "<table><tr><td>new</td></tr></table>",
    ]
    marker = module.nearest_unit_marker(lines, 4)
    assert marker is not None
    assert marker["factor"] == 1_000.0
    assert marker["line"] == 3


def test_real_q915_hag_table_has_thousand_vnd_marker() -> None:
    path = (
        ROOT
        / "data/financial_statements/HAG/2016"
        / "HAG_financial_statements_2016_separate"
        / "HAG_financial_statements_2016_separate_extracted.txt"
    )
    marker = module.nearest_unit_marker(
        path.read_text(encoding="utf-8").splitlines(), 281
    )
    assert marker is not None
    assert marker["factor"] == 1_000.0
    assert marker["line"] == 279


def test_real_q746_sources_are_both_base_vnd() -> None:
    cases = [
        (
            ROOT
            / "data/financial_statements/DXS/2024"
            / "DXS_financial_statements_2024_separate"
            / "DXS_financial_statements_2024_separate_extracted.txt",
            425,
        ),
        (
            ROOT
            / "data/financial_statements/KHG/2024"
            / "KHG_financial_statements_2024_separate"
            / "KHG_financial_statements_2024_separate_extracted.txt",
            216,
        ),
    ]
    for path, line in cases:
        marker = module.nearest_unit_marker(
            path.read_text(encoding="utf-8").splitlines(), line
        )
        assert marker is not None
        assert marker["factor"] == 1.0
