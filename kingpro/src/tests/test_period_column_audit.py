from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_period_columns.py"
SPEC = importlib.util.spec_from_file_location("period_column_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_same_report_year_rejects_prior_column() -> None:
    path = Path("EIB_financial_statements_2022_separate_1732.csv")
    mismatch = module.year_role_mismatch(
        "Tổng quỹ lương năm 2022 của công ty mẹ EIB là bao nhiêu?",
        path,
        "nam truoc trieu dong",
    )
    assert mismatch == (2022, 2022, "prior")


def test_same_report_year_accepts_current_column() -> None:
    path = Path("GEG_financial_statements_2019_separate_1055.csv")
    assert module.year_role_mismatch(
        "Chi phí nhân viên của công ty mẹ GEG trong năm 2019 là bao nhiêu?",
        path,
        "nam nay vnd",
    ) is None


def test_following_report_accepts_prior_comparative_column() -> None:
    path = Path("HBC_financial_statements_2023_separate_1578.csv")
    assert module.year_role_mismatch(
        "Thu nhập khác của HBC trong năm 2022 là bao nhiêu?",
        path,
        "nam truoc vnd",
    ) is None


def test_multi_year_question_is_not_classified() -> None:
    path = Path("STB_financial_statements_2022_separate_1005.csv")
    assert module.year_role_mismatch(
        "Tăng trưởng từ năm 2016 đến năm 2022 là bao nhiêu?",
        path,
        "nam truoc",
    ) is None


def test_exact_date_header_rejects_different_requested_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.year_role_mismatch(
        "Số dư cuối năm 2023 của ABC là bao nhiêu?",
        path,
        "tai ngay 31/12/2024 vnd",
    ) == (2023, 2024, "explicit:2024")


def test_exact_date_header_accepts_requested_year() -> None:
    path = Path("ABC_financial_statements_2024_separate_100.csv")
    assert module.year_role_mismatch(
        "Số dư cuối năm 2023 của ABC là bao nhiêu?",
        path,
        "tai ngay 31/12/2023 vnd",
    ) is None


def test_ambiguous_multi_year_header_falls_back_to_role() -> None:
    assert module.descriptor_explicit_year(
        "31/12/2024 01/01/2024 31/12/2023"
    ) is None


def test_legal_citation_year_is_not_a_data_period() -> None:
    descriptor = (
        "mau b09-dn ban hanh theo thong tu 200/2014/tt-btc "
        "ngay 22/12/2014 cua bo tai chinh"
    )
    assert module.descriptor_explicit_year(descriptor) is None
