from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_same_label_value_collisions.py"
SPEC = importlib.util.spec_from_file_location("audit_same_label_value_collisions", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(MODULE)


def test_prior_year_scope_loses_to_current_scope() -> None:
    score, reasons = MODULE.collision_priority(
        "Tong so du du phong cuoi nam 2018?",
        "Thay doi du phong trong nam truoc nhu sau",
        "Thay doi du phong trong nam nhu sau",
    )
    assert score >= 10
    assert reasons


def test_asset_balance_loses_provision_movement_scope() -> None:
    score, reasons = MODULE.collision_priority(
        "So du trai phieu VAMC cuoi nam?",
        "Bien dong du phong rui ro chung khoan dau tu",
        "Trai phieu dac biet do VAMC phat hanh",
    )
    assert score >= 8
    assert reasons


def test_risk_classification_is_exempt_from_generic_provision_rule() -> None:
    score, _ = MODULE.collision_priority(
        "So du no co kha nang mat von?",
        "Bien dong du phong rui ro tin dung",
        "Phan tich chat luong no",
    )
    assert score == 0


def test_near_duplicate_label_ignores_footnote_suffix() -> None:
    assert MODULE.similar_label(
        "Trai phieu dac biet do VAMC phat hanh",
        "Trai phieu dac biet do VAMC phat hanh (a)",
    )


def test_specific_activity_row_beats_broad_wip_total() -> None:
    score, reasons = MODULE.collision_priority(
        "Ty le tang chi phi san xuat kinh doanh do dang trong hoat dong xay lap?",
        "Hang ton kho",
        "Chi tiet chi phi san xuat kinh doanh do dang",
        "Chi phi san xuat kinh doanh do dang",
        "Hoat dong xay lap",
    )
    assert score >= 6
    assert any("qualifiers" in reason for reason in reasons)


def test_vietnamese_short_term_question_uses_scope_signal() -> None:
    score, reasons = MODULE.collision_priority(
        "Số dư khoản phải thu ngắn hạn cuối năm là bao nhiêu?",
        "Thuyết minh phải thu dài hạn",
        "Thuyết minh phải thu ngắn hạn",
        "Số dư cuối năm",
        "Số dư cuối năm",
    )
    assert score >= 10
    assert any("short_term" in reason for reason in reasons)
