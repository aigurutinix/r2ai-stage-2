from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_structural_risks.py"
SPEC = importlib.util.spec_from_file_location("requested_year_source_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_explicit_ending_date_rejects_previous_report_year() -> None:
    question = (
        "Tổng số tiền thuê tối thiểu của HND đến ngày 31 tháng 12 năm 2025 "
        "là bao nhiêu?"
    )
    assert module.ending_year_source_mismatch(
        question, ["HND_financial_statements_2024"]
    ) == "2025"


def test_explicit_ending_date_accepts_matching_report_year() -> None:
    question = "Số dư HND cuối năm 2025 là bao nhiêu?"
    assert module.ending_year_source_mismatch(
        question, ["HND_financial_statements_2025"]
    ) is None


def test_opening_date_allows_previous_report_year() -> None:
    question = "Số dư phải thu của HHV đến ngày 01/01/2022 là bao nhiêu?"
    assert module.ending_year_source_mismatch(
        question, ["HHV_financial_statements_2021_consolidated"]
    ) is None


def test_company_aliases_prefer_longest_non_overlapping_name() -> None:
    aliases = {
        "POW": (
            "tong cong ty dien luc dau khi viet nam ctcp",
            "dien luc dau khi viet nam ctcp",
        ),
        "GAS": ("tong cong ty khi viet nam ctcp", "khi viet nam ctcp"),
        "GEG": ("ctcp dien gia lai", "dien gia lai"),
    }
    question = (
        "Trung bình của CTCP Điện Gia Lai và Tổng Công ty Điện lực "
        "Dầu khí Việt Nam - CTCP là bao nhiêu?"
    )
    assert module.company_names_in_question(question, aliases) == {"GEG", "POW"}


def test_company_aliases_keep_separate_company_occurrences() -> None:
    aliases = {
        "HAG": ("ctcp hoang anh gia lai", "hoang anh gia lai"),
        "HNG": (
            "ctcp nong nghiep quoc te hoang anh gia lai",
            "nong nghiep quoc te hoang anh gia lai",
        ),
    }
    question = (
        "So sánh CTCP Hoàng Anh Gia Lai với CTCP Nông nghiệp Quốc tế "
        "Hoàng Anh Gia Lai."
    )
    assert module.company_names_in_question(question, aliases) == {"HAG", "HNG"}


def test_ticker_token_inside_another_company_name_is_not_a_second_entity() -> None:
    aliases = {
        "FPT": ("ctcp fpt", "fpt"),
        "FTS": ("ctcp chung khoan fpt", "chung khoan fpt"),
    }
    question = "Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023?"
    assert module.standalone_ticker_tokens(question, {"FPT", "FTS"}, aliases) == set()
    assert module.company_names_in_question(question, aliases) == {"FTS"}


def test_ticker_token_outside_company_name_is_retained() -> None:
    aliases = {
        "FPT": ("ctcp fpt", "fpt"),
        "FTS": ("ctcp chung khoan fpt", "chung khoan fpt"),
    }
    question = "So sánh CTCP Chứng khoán FPT (FTS) với FPT."
    assert module.standalone_ticker_tokens(question, {"FPT", "FTS"}, aliases) == {
        "FPT",
        "FTS",
    }
