"""Tests for the declared statement-to-note link."""

from vifinqa.notes import (
    code_column,
    linked_note_lines,
    note_column,
    note_table_line,
    row_label,
    row_note,
)


BALANCE_SHEET = [
    ["TÀI SẢN", "Mã số", "Thuyết minh", "31/12/2016", "1/1/2016"],
    ["I. Tiền và các khoản tương đương tiền", "110", "4", "1.880.612.291.229", "6.406.079.584.088"],
    ["II. Đầu tư tài chính ngắn hạn", "120", "5", "1.000.000.000", "2.000.000.000"],
    ["III. Các khoản phải thu ngắn hạn", "130", "6", "3.000.000.000", "4.000.000.000"],
]

REPORT = """BẢNG CÂN ĐỐI KẾ TOÁN
<table><tr><td>TÀI SẢN</td></tr></table>
some prose about accounting policies
4. TIỀN VÀ CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN
<table><tr><td>Tiền mặt tại quỹ</td></tr></table>
5. ĐẦU TƯ TÀI CHÍNH NGẮN HẠN
<table><tr><td>Tiền gửi có kỳ hạn</td></tr></table>
"""


def test_code_column_is_detected_by_repeated_codes():
    assert code_column(BALANCE_SHEET) == 1


def test_code_column_absent_when_no_column_repeats():
    assert code_column([["Chỉ tiêu", "2023"], ["Doanh thu", "100.000"]]) is None


def test_code_column_prefers_an_explicit_header_in_a_short_table():
    assert code_column([["Chỉ tiêu", "Mã số", "Thuyết minh"], ["Tiền", "110", "4"]]) == 1


def test_note_column_uses_its_header_instead_of_position():
    grid = [["Chỉ tiêu", "Mã số", "2023", "Thuyết minh"], ["Tiền", "110", "1.000", "4"]]
    assert note_column(grid, 1) == 3


def test_row_label_keeps_split_text_cells_and_drops_numbers():
    row = ["I.", "Tiền và tương đương tiền", "110", "4", "1.000.000"]
    assert row_label(row, 2, 3) == "I. Tiền và tương đương tiền"


def test_row_note_reads_the_cell_right_of_the_code():
    assert row_note(BALANCE_SHEET[1], 1) == "4"


def test_row_note_rejects_a_non_reference_cell():
    assert row_note(["Tiền", "110", "1.880.612.291.229"], 1) is None


def test_row_note_handles_a_short_row():
    assert row_note(["Tiền", "110"], 1) is None


def test_note_table_line_points_at_the_table_under_the_heading():
    assert note_table_line(REPORT, "4") == 5


def test_note_table_line_accepts_a_roman_prefixed_reference():
    assert note_table_line(REPORT, "V.04") == 5


def test_note_table_line_preserves_a_decimal_note_number():
    report = "5.1. Chi phí lãi vay\n<table><tr><td>a</td></tr></table>\n"
    assert note_table_line(report, "5.1") == 2


def test_note_table_line_returns_none_for_an_absent_note():
    assert note_table_line(REPORT, "9") is None


def test_note_table_line_ignores_a_numbered_line_of_prose():
    """A heading is short; a numbered sentence in a paragraph is not a note."""
    prose = "4. " + "x" * 200 + "\n<table><tr><td>a</td></tr></table>\n"
    assert note_table_line(prose, "4") is None


def test_note_table_line_does_not_fall_through_to_the_next_note():
    report = "4. Không có bảng ở đây\n5. Có bảng\n<table><tr><td>a</td></tr></table>\n"
    assert note_table_line(report, "4") is None


def test_link_follows_only_rows_the_question_names():
    links = linked_note_lines(REPORT, BALANCE_SHEET, ["tien va cac khoan tuong duong tien"], 2)
    assert links == [("4", 5)]


def test_link_is_empty_without_line_items():
    assert linked_note_lines(REPORT, BALANCE_SHEET, [], 2) == []


def test_link_drops_a_note_resolving_to_its_own_table():
    assert linked_note_lines(
        REPORT, BALANCE_SHEET, ["tien va cac khoan tuong duong tien"], 5,
    ) == []


def test_link_follows_several_named_items_in_source_order():
    links = linked_note_lines(
        REPORT, BALANCE_SHEET,
        ["tien va cac khoan tuong duong tien", "dau tu tai chinh ngan han"], 2,
    )
    assert links == [("4", 5), ("5", 7)]
