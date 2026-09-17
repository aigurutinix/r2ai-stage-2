from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "audit_audited_period_columns.py"
SPEC = importlib.util.spec_from_file_location("audited_period_columns", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_current_header_maps_to_report_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(path, "nam nay vnd") == (
        2024,
        "current_year_header",
    )


def test_prior_header_maps_to_previous_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(path, "nam truoc trieu dong") == (
        2023,
        "prior_year_header",
    )


def test_opening_header_maps_to_previous_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(path, "so dau nam vnd") == (
        2023,
        "opening_header",
    )


def test_explicit_date_takes_precedence() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(path, "tai ngay 31/12/2022 vnd") == (
        2022,
        "explicit_header_year",
    )


def test_legal_citation_does_not_invent_period() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(
        path,
        "mau b09-dn ban hanh theo thong tu 200/2014/tt-btc ngay 22/12/2014",
    ) is None


def test_account_name_containing_prior_year_is_not_a_period_header() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(
        path,
        "loi nhuan sau thue chua phan phoi nam truoc 100 200",
    ) is None


def test_opening_metric_uses_report_context_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(
        path,
        "vnd so dau nam 100 200",
        "note:opening_outstanding_shares",
    ) == (2024, "opening_report_context")


def test_first_january_balance_maps_to_previous_year_end() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(path, "1/1/2024 vnd") == (
        2023,
        "explicit_header_year",
    )


def test_prior_metric_keeps_report_context_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.expected_column_year(
        path,
        "tai ngay 31/12/2023 vnd",
        "note:recorded_derivatives_prior",
    ) == (2024, "prior_operand_report_context")


def test_nearest_stacked_period_header_wins(tmp_path: Path) -> None:
    path = tmp_path / "ABC_financial_statements_2024_separate_100.csv"
    path.write_text(
        "0,1\n"
        ",Năm 2023\n"
        "Doanh thu,100\n"
        ",Năm 2024\n"
        "Doanh thu,200\n",
        encoding="utf-8",
    )
    assert module.cell_period_descriptor(path, 3, 1) == "nam 2024"


def test_joined_current_year_header_is_recognized(tmp_path: Path) -> None:
    path = tmp_path / "ABC_financial_statements_2024_separate_100.csv"
    path.write_text(
        "0,1\n"
        ",Năm nayTriệu đồng\n"
        "Số đầu năm,Số đầu năm\n"
        "Số trích lập trong năm,Số trích lập trong năm\n"
        "Khoản mục,100\n",
        encoding="utf-8",
    )
    assert module.cell_period_descriptor(path, 3, 1) == "nam naytrieu dong"


def test_counterparty_lifecycle_date_is_not_a_table_period(tmp_path: Path) -> None:
    path = tmp_path / "ABC_financial_statements_2024_separate_100.csv"
    path.write_text(
        "0,1\n"
        ",2024VND\n"
        "Công ty ABC (đến ngày 9 tháng 11 năm 2025),"
        "Công ty ABC (đến ngày 9 tháng 11 năm 2025)\n"
        "Doanh thu,100\n",
        encoding="utf-8",
    )
    assert module.cell_period_descriptor(path, 2, 1) == "2024vnd"
