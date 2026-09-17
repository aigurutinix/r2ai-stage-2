from vifinqa.slots import (
    named_line_items,
    select_confidence_slot_tables,
    select_slot_tables,
)


def test_named_line_items_keeps_longest_explicit_metric():
    labels = ("doanh thu thuan", "doanh thu", "nam 2024", "cong ty co phan")

    assert named_line_items(
        "Doanh thu thuần của Công ty Cổ phần ABC năm 2024 là bao nhiêu?", labels,
    ) == ["doanh thu thuan"]


def test_slot_selection_reserves_one_exact_carrier_per_report_and_item():
    record = {
        "selected_docs": ["R1", "R2"],
        "candidates": [
            {"table_id": "R1|1", "text": "Chi phí khác"},
            {"table_id": "R1|2", "text": "Doanh thu thuần"},
            {"table_id": "R2|1", "text": "Doanh thu thuần"},
            {"table_id": "R2|2", "text": "Tài sản"},
        ],
    }

    ranking, slots = select_slot_tables(
        record, ["R1|1", "R2|2", "R1|2", "R2|1"], ["doanh thu thuan"], 2,
    )

    assert ranking == ["R1|2", "R2|1"]
    assert [slot["table_id"] for slot in slots] == ["R1|2", "R2|1"]


def test_slot_selection_falls_back_to_base_order_without_items():
    record = {
        "selected_docs": ["R1"],
        "candidates": [{"table_id": "R1|1", "text": "Bảng"}],
    }

    ranking, slots = select_slot_tables(record, ["R1|1"], [], 2)

    assert ranking == ["R1|1"]
    assert slots == []


def test_confidence_slots_keep_floors_and_drop_low_confidence_padding():
    record = {
        "selected_docs": ["R1", "R2"],
        "candidates": [
            {"table_id": "R1|1", "text": "Doanh thu"},
            {"table_id": "R1|2", "text": "Chi phí"},
            {"table_id": "R1|3", "text": "Tài sản"},
            {"table_id": "R2|1", "text": "Doanh thu"},
            {"table_id": "R2|2", "text": "Chi phí"},
        ],
    }
    ranking, slots = select_confidence_slot_tables(
        record,
        ["R1|1", "R2|1", "R1|2", "R2|2", "R1|3"],
        ["doanh thu"],
        {"R1|1": 0.9, "R1|2": 0.01, "R2|1": 0.8, "R2|2": 0.01},
        logit_gap=4,
        max_per_report=2,
    )

    assert ranking == ["R1|1", "R2|1"]
    assert [slot["table_id"] for slot in slots] == ["R1|1", "R2|1"]


def test_confidence_slots_allow_scored_extra_but_not_unscored_extra():
    record = {
        "selected_docs": ["R1"],
        "candidates": [
            {"table_id": "R1|1", "text": "Bảng"},
            {"table_id": "R1|2", "text": "Bảng phụ"},
            {"table_id": "R1|3", "text": "Bảng khác"},
        ],
    }
    ranking, _ = select_confidence_slot_tables(
        record, ["R1|1", "R1|2", "R1|3"], [],
        {"R1|2": 0.8}, logit_gap=4, max_per_report=2,
    )

    assert ranking == ["R1|1", "R1|2"]


def test_confidence_slots_can_require_item_for_adaptive_extra():
    record = {
        "selected_docs": ["R1"],
        "candidates": [
            {"table_id": "R1|1", "text": "Doanh thu"},
            {"table_id": "R1|2", "text": "Chi phí"},
            {"table_id": "R1|3", "text": "Doanh thu phụ"},
        ],
    }
    ranking, _ = select_confidence_slot_tables(
        record, ["R1|1", "R1|2", "R1|3"], ["doanh thu"],
        {"R1|1": 0.9, "R1|2": 0.8, "R1|3": 0.7},
        max_per_report=3, require_item_for_extras=True,
    )

    assert ranking == ["R1|1", "R1|3"]
