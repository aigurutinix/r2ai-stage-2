from vifinqa.evidence import (
    EvidenceCell,
    build_evidence_slots,
    complete_slots,
    operation_hint,
    select_best_cells,
)


def test_operation_hint_stays_lookup_for_plain_metric():
    assert operation_hint("Doanh thu năm 2024 là bao nhiêu?") == "lookup"
    assert operation_hint("Tốc độ tăng trưởng doanh thu năm 2024") == "growth"


def test_growth_builds_prior_and_current_slots():
    slots = build_evidence_slots(
        "Tốc độ tăng trưởng doanh thu năm 2024?",
        {"tickers": ["ABC"], "years": [2024], "scope": "consolidated"},
        ["doanh thu"],
    )

    assert [(slot.requested_year, slot.role) for slot in slots] == [
        (2023, "prior"), (2024, "current"),
    ]
    assert all(slot.ticker == "ABC" for slot in slots)


def test_slot_selection_reports_incomplete_evidence():
    slots = build_evidence_slots(
        "Tốc độ tăng trưởng doanh thu năm 2024?",
        {"tickers": ["ABC"], "years": [2024]},
        ["doanh thu"],
    )
    candidates = {
        slots[0].slot_id: [EvidenceCell(slots[0].slot_id, "R|1", 3, 2, 10.0, 0.8, 2023)],
    }
    selected = select_best_cells(slots, candidates)
    assert len(selected) == 1
    assert not complete_slots(slots, selected)
