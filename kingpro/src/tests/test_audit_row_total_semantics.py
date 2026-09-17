from decimal import Decimal

from scripts.audit_row_total_semantics import (
    asks_for_total,
    explicit_total_rows,
    is_child_label,
    is_total_row,
    local_subtotal_evidence,
)


def test_total_cue_does_not_confuse_cong_ty_with_total():
    assert asks_for_total("Tong so tien cua cong ty A")
    assert not asks_for_total("Chi phi tai chinh cua cong ty A")


def test_total_row_is_strict():
    assert is_total_row("TONG CONG")
    assert is_total_row("Tong so")
    assert not is_total_row("Tong tai san")
    assert not is_total_row("Nguoi mua tra tien truoc")


def test_child_label_requires_visible_hierarchy_marker():
    assert is_child_label("- Khach hang A")
    assert is_child_label("1. Khach hang A")
    assert not is_child_label("Nguoi mua tra tien truoc")


def test_local_subtotal_evidence_sums_contiguous_children():
    rows = [
        ["label", "2023"],
        ["Nguoi mua tra tien truoc", "30"],
        ["- A", "10"],
        ["- B", "20"],
        ["Ben lien quan", "5"],
        ["TONG CONG", "35"],
    ]
    evidence = local_subtotal_evidence(rows, 1, 1)
    assert evidence["selected_equals_child_sum"] is True
    assert Decimal(evidence["child_sum"]) == Decimal(30)
    assert explicit_total_rows(rows, 1)[0]["value"] == "35"
