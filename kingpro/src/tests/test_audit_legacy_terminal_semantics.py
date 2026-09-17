from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_legacy_terminal_semantics.py"
SPEC = importlib.util.spec_from_file_location("audit_legacy_terminal_semantics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(module)


def test_generic_ending_label_uses_section_scope() -> None:
    reasons = module.context_conflicts(
        "Số dư cuối kỳ dự phòng chung cho các khoản cho vay khách hàng là bao nhiêu?",
        ["Biến động dự phòng cụ thể cho các khoản cho vay khách hàng như sau:"],
    )
    assert reasons == [
        "question asks 'du phong chung' but table context only signals 'du phong cu the'"
    ]


def test_matching_section_scope_is_clean() -> None:
    assert module.context_conflicts(
        "Số dư cuối kỳ dự phòng chung cho các khoản cho vay khách hàng là bao nhiêu?",
        ["Biến động dự phòng chung cho các khoản cho vay khách hàng như sau:"],
    ) == []


def test_bidirectional_tax_scope_conflict() -> None:
    reasons = module.context_conflicts(
        "Chi phí thuế thu nhập doanh nghiệp hoãn lại là bao nhiêu?",
        ["Chi phí thuế thu nhập doanh nghiệp hiện hành"],
    )
    assert any("hoan lai" in reason and "hien hanh" in reason for reason in reasons)


def test_asset_balance_inside_provision_movement_is_flagged() -> None:
    reasons = module.structural_context_conflicts(
        "So du trai phieu dac biet do VAMC phat hanh den 31/12/2023?",
        ["Trai phieu dac biet do VAMC phat hanh"],
        ["Bien dong du phong rui ro chung khoan dau tu trong nam"],
        ["ABB_financial_statements_2023_separate|1575"],
    )
    assert reasons == [
        "unqualified balance question reads a row inside a provision-movement table"
    ]


def test_credit_risk_classification_balance_is_not_a_provision_false_positive() -> None:
    assert module.structural_context_conflicts(
        "So du no co kha nang mat von nam 2019?",
        ["No co kha nang mat von"],
        ["Du phong rui ro tin dung"],
        ["ABC_financial_statements_2019_consolidated|10"],
    ) == []


def test_current_report_year_inside_prior_year_table_is_flagged() -> None:
    reasons = module.structural_context_conflicts(
        "Tong so du du phong cuoi nam 2018 la bao nhieu?",
        ["So du cuoi nam"],
        ["Thay doi du phong rui ro trong nam truoc nhu sau"],
        ["VIB_financial_statements_2018_consolidated|1070"],
    )
    assert reasons == [
        "current report-year question reads a table explicitly scoped to the prior year"
    ]
