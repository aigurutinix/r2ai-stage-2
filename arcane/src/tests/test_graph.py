"""Tests for the slot -> row -> table evidence path."""

from vifinqa.graph import (
    NOTE_REF,
    SOURCE_TABLE,
    Edge,
    EvidencePath,
    EvidenceSlot,
    RowAnchor,
    anchor_edges,
    dedupe_edges,
    evidence_slots,
    row_anchors,
    select_paths,
    table_paths,
)


GRID = [
    ["TÀI SẢN", "Mã số", "Thuyết minh", "31/12/2016"],
    ["I. Tiền và các khoản tương đương tiền", "110", "4", "1.880.612.291.229"],
    ["II. Đầu tư tài chính ngắn hạn", "120", "5", "1.000.000.000"],
    ["III. Các khoản phải thu ngắn hạn", "130", "6", "3.000.000.000"],
]

REPORT = """BẢNG CÂN ĐỐI KẾ TOÁN
<table><tr><td>TÀI SẢN</td></tr></table>
4. TIỀN VÀ CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN
<table><tr><td>Tiền mặt tại quỹ</td></tr></table>
"""

ITEM = "tien va cac khoan tuong duong tien"


def anchor_of(items):
    return row_anchors(GRID, "SAB_2016", "SAB_2016|2", items)


def test_anchor_matches_only_the_named_row():
    anchors = anchor_of([ITEM])
    assert [a.row for a in anchors] == [1]
    assert anchors[0].line_item == ITEM


def test_anchor_carries_the_code_and_note_from_the_row():
    anchor = anchor_of([ITEM])[0]
    assert (anchor.code, anchor.note) == ("110", "4")


def test_anchor_keeps_the_raw_label_for_provenance():
    assert anchor_of([ITEM])[0].label == "I. Tiền và các khoản tương đương tiền"


def test_anchor_matches_a_label_split_across_columns():
    grid = [
        ["Nhóm", "Chỉ tiêu", "Mã số", "Thuyết minh", "2023"],
        ["I.", "Tiền và các khoản tương đương tiền", "110", "4", "1.000"],
    ]
    anchors = row_anchors(grid, "R", "R|1", [ITEM])
    assert [(anchor.row, anchor.code, anchor.note) for anchor in anchors] == [(1, "110", "4")]


def test_no_anchors_without_line_items():
    assert anchor_of([]) == []


def test_several_items_anchor_several_rows():
    anchors = anchor_of([ITEM, "dau tu tai chinh ngan han"])
    assert [a.row for a in anchors] == [1, 2]


def test_source_table_edge_is_always_emitted():
    edges = anchor_edges(anchor_of([ITEM])[0], REPORT, source_line=2)
    assert edges[0].kind == SOURCE_TABLE
    assert edges[0].table_id == "SAB_2016|2"


def test_note_edge_points_at_the_note_table():
    edges = anchor_edges(anchor_of([ITEM])[0], REPORT, source_line=2)
    assert [(e.kind, e.table_id) for e in edges[1:]] == [(NOTE_REF, "SAB_2016|4")]


def test_note_edge_is_skipped_when_notes_are_disabled():
    edges = anchor_edges(anchor_of([ITEM])[0], REPORT, source_line=2, follow_notes=False)
    assert [e.kind for e in edges] == [SOURCE_TABLE]


def test_note_edge_never_returns_to_its_own_table():
    edges = anchor_edges(anchor_of([ITEM])[0], REPORT, source_line=4)
    assert [e.kind for e in edges] == [SOURCE_TABLE]


def test_row_without_a_note_yields_only_its_source_table():
    anchor = RowAnchor("SAB_2016", "SAB_2016|2", 1, "Tiền", ITEM, "110", "")
    assert [e.kind for e in anchor_edges(anchor, REPORT, source_line=2)] == [SOURCE_TABLE]


def test_dedupe_keeps_the_first_reason_for_a_table():
    anchor = RowAnchor("R", "R|1", 0, "x", ITEM)
    edges = [Edge("R|9", SOURCE_TABLE, anchor), Edge("R|9", NOTE_REF, anchor)]
    assert [e.kind for e in dedupe_edges(edges, set())] == [SOURCE_TABLE]


def test_dedupe_drops_tables_already_selected():
    anchor = RowAnchor("R", "R|1", 0, "x", ITEM)
    edges = [Edge("R|1", SOURCE_TABLE, anchor), Edge("R|2", NOTE_REF, anchor)]
    assert [e.table_id for e in dedupe_edges(edges, {"R|1"})] == ["R|2"]


def test_slots_cover_every_report_and_item():
    slots = evidence_slots(["R1", "R2"], ["cash", "debt"])
    assert [(slot.report_id, slot.line_item) for slot in slots] == [
        ("R1", "cash"), ("R1", "debt"), ("R2", "cash"), ("R2", "debt"),
    ]


def test_table_paths_never_cross_reports():
    slot = EvidenceSlot("OTHER", ITEM)
    assert table_paths(slot, "SAB_2016|2", GRID, REPORT, 2) == []


def test_table_paths_keep_the_full_three_hop_trace():
    slot = EvidenceSlot("SAB_2016", ITEM)
    paths = table_paths(slot, "SAB_2016|2", GRID, REPORT, 2)
    assert [(path.slot, path.anchor.row, path.edge.kind) for path in paths] == [
        (slot, 1, SOURCE_TABLE), (slot, 1, NOTE_REF),
    ]


def test_path_selection_covers_slots_before_second_evidence():
    slot_a, slot_b = EvidenceSlot("R", "cash"), EvidenceSlot("R", "debt")
    anchor_a = RowAnchor("R", "R|1", 0, "cash", "cash")
    anchor_b = RowAnchor("R", "R|2", 0, "debt", "debt")
    paths = [
        EvidencePath(slot_a, anchor_a, Edge("R|1", SOURCE_TABLE, anchor_a)),
        EvidencePath(slot_a, anchor_a, Edge("R|9", NOTE_REF, anchor_a)),
        EvidencePath(slot_b, anchor_b, Edge("R|2", SOURCE_TABLE, anchor_b)),
    ]
    selected = select_paths(paths, set(), 2)
    assert [path.edge.table_id for path in selected] == ["R|1", "R|2"]


def test_path_selection_skips_existing_source_and_takes_its_note():
    slot = EvidenceSlot("R", "cash")
    anchor = RowAnchor("R", "R|1", 0, "cash", "cash")
    paths = [
        EvidencePath(slot, anchor, Edge("R|1", SOURCE_TABLE, anchor)),
        EvidencePath(slot, anchor, Edge("R|9", NOTE_REF, anchor)),
    ]
    assert [path.edge.table_id for path in select_paths(paths, {"R|1"}, 1)] == ["R|9"]
