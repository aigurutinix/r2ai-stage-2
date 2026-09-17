"""Unit tests for the answer stage: cell binding and arithmetic planning."""

from vifinqa.answers import (
    EvidenceValue, answer_plan, bound_cells, header_years, strict_number,
    source_scale, year_matched_cells,
)


HEADERS = ["Chỉ tiêu", "Số đầu năm", "31/12/2022", "31/12/2023"]
ROWS = [
    HEADERS,
    ["Doanh thu thuần", "1.000", "1.200", "1.400"],
    ["Giá vốn hàng bán", "(800)", "900.500", "1.100"],
    ["Chi phí quản lý doanh nghiệp", "", "062 Chi phí", "n/a"],
]


def test_header_years_reads_glued_ocr_labels():
    assert header_years("31.12.2022Triệu VND") == {2022}
    assert header_years("Tỷ lệ khấu trừ tối đa") == set()


def test_year_matched_cells_only_returns_requested_periods():
    cells = list(ROWS[1])
    matched = year_matched_cells(cells, HEADERS, [2022, 2023])
    assert matched == [(2, 1200.0, 2022), (3, 1400.0, 2023)]
    assert year_matched_cells(cells, HEADERS, []) == []


def test_strict_number_rejects_numbers_inside_text():
    assert strict_number("(800)") == -800.0
    assert strict_number("1.100") == 1100.0
    assert strict_number("99,999%") == 99.999
    assert strict_number("3,50 - 9,00") is None
    assert strict_number("062 Chi phí") is None
    assert strict_number("31/12/2017Triệu VND") is None
    assert strict_number("") is None


def test_source_scale_reads_currency_units_but_not_percent_cells():
    assert source_scale("Đơn vị tính: Triệu VND") == 1e6
    assert source_scale("Ngàn VND") == 1e3
    assert source_scale("Đơn vị tính: Triệu VND", "12,5%") == 1.0


def test_bound_cells_binds_label_rows_and_their_year_columns():
    rows = [list(row) for row in ROWS]
    tokens = {"doanh", "thu"}
    bound = bound_cells(rows, HEADERS, tokens, [2022, 2023])
    assert (1, 2) in bound and (1, 3) in bound
    # The expense row shares no token with the question.
    assert all(index != 3 for index, _ in bound)
    # A row without a readable numeric cell contributes nothing.
    assert all(index != 2 or column != 1 for index, column in bound)


def test_bound_cells_without_tokens_binds_nothing():
    assert bound_cells([list(row) for row in ROWS], HEADERS, set(), [2022]) == []


def value(variable="df0", row=1, column=3, raw=1400.0, year=2023):
    return EvidenceValue(variable, row, column, raw, f"R|{row}", year)


def test_plan_growth_reads_two_year_columns_in_order():
    values = [value(column=2, raw=1200.0, year=2022), value(column=3, raw=1400.0, year=2023)]
    answer, expression = answer_plan(
        "Tăng trưởng doanh thu năm 2023 so với năm 2022 là bao nhiêu phần trăm?", values)
    assert answer == 16.67
    assert "df0.iloc[0, 3]" in expression and "df0.iloc[0, 2]" in expression


def test_plan_sums_every_named_year_of_one_row():
    values = [
        value(column=2, raw=1200.0, year=2022),
        value(column=3, raw=1400.0, year=2023),
        value("df1", 5, 1, 999.0),
    ]
    answer, _ = answer_plan(
        "Tổng doanh thu trong các năm 2022 và 2023 là bao nhiêu triệu đồng?", values)
    assert answer == round(2600 / 1e6, 2)


def test_plan_difference_is_newest_minus_oldest():
    values = [value(column=2, raw=1200.0, year=2022), value(column=3, raw=1400.0, year=2023)]
    answer, _ = answer_plan(
        "Chênh lệch doanh thu giữa năm 2023 và năm 2022 là bao nhiêu triệu đồng?", values)
    assert answer == round(200 / 1e6, 2)


def test_plan_average_over_a_year_span():
    values = [value(column=c, raw=v, year=y) for c, v, y in
              ((1, 1000.0, None), (2, 1200.0, 2022), (3, 1400.0, 2023))]
    answer, _ = answer_plan(
        "Doanh thu bình quân giai đoạn 2022-2023 là bao nhiêu triệu đồng?",
        [values[1], values[2]])
    assert answer == round(1300 / 1e6, 2)


def test_company_boilerplate_does_not_read_as_an_operator():
    # "công ty" folds to "cong ty"; before the boilerplate strip its "cong"
    # triggered the sum branch on a plain lookup.
    lookup = ("Số dư cho vay khách hàng ngành Thương mại của công ty mẹ "
              "Ngân hàng TMCP Á Châu (ACB) cuối năm 2022 là bao nhiêu triệu đồng?")
    answer, expression = answer_plan(lookup, [value()])
    assert answer == round(1400 / 1e6, 2)
    assert "+" not in expression


def test_plan_emits_the_same_source_unit_conversion_it_uses_for_the_answer():
    # The CSV cell contains 2,500 while its source table declares triệu đồng.
    # EvidenceValue.value is canonical VND; the emitted query must reconstruct it.
    scaled = EvidenceValue("df0", 1, 1, 2_500_000_000.0, "R|1", source_scale=1e6)
    answer, expression = answer_plan("Doanh thu là bao nhiêu tỷ đồng?", [scaled])
    assert answer == 2.5
    assert "* 1e+06" in expression


def test_ratio_orders_operands_by_question_metric_not_table_order():
    values = [
        EvidenceValue("df0", 1, 1, 100.0, "R|1", label="Nợ vay ngắn hạn"),
        EvidenceValue("df1", 1, 1, 10.0, "R|2", label="Chi phí lãi vay"),
    ]
    answer, _ = answer_plan(
        "Tỷ lệ chi phí lãi vay trên nợ vay ngắn hạn là bao nhiêu %?", values,
    )

    assert answer == 10.0


def test_year_superlative_returns_the_winning_year():
    values = [
        EvidenceValue("df0", 1, 1, 100.0, "R|1", year=2022),
        EvidenceValue("df1", 1, 1, 140.0, "R|2", year=2024),
        EvidenceValue("df2", 1, 1, 120.0, "R|3", year=2023),
    ]
    answer, expression = answer_plan(
        "Năm nào có doanh thu cao nhất trong các năm 2022, 2023 và 2024?", values,
    )

    assert answer == 2024.0
    assert "[1]" in expression and "2024" in expression


def test_year_superlative_then_lookup_uses_the_winning_metric_year():
    values = [
        EvidenceValue("df0", 1, 1, 100.0, "R|2022", year=2022, label="Doanh thu thuần"),
        EvidenceValue("df1", 1, 1, 140.0, "R|2024", year=2024, label="Doanh thu thuần"),
        EvidenceValue("df0", 2, 1, 1.2, "R|2022", year=2022, label="Hệ số thanh toán hiện hành"),
        EvidenceValue("df1", 2, 1, 1.8, "R|2024", year=2024, label="Hệ số thanh toán hiện hành"),
    ]
    answer, _ = answer_plan(
        "Tại năm mà doanh thu thuần cao nhất trong giai đoạn 2022-2024, "
        "hệ số thanh toán hiện hành là bao nhiêu lần?", values,
    )
    assert answer == 1.8


def test_multi_year_max_amount_scopes_to_requested_metric():
    values = [
        EvidenceValue("df0", 1, 1, 100_000_000.0, "R|2022", year=2022, label="Giá trị còn lại tài sản cố định vô hình"),
        EvidenceValue("df1", 1, 1, 140_000_000.0, "R|2024", year=2024, label="Giá trị còn lại tài sản cố định vô hình"),
    ]
    answer, _ = answer_plan(
        "Giá trị còn lại của tài sản cố định vô hình lớn nhất trong các năm "
        "2022 và 2024 là bao nhiêu triệu đồng?", values,
    )
    assert answer == 140.0


def test_numeric_expression_handles_parenthesized_negative_cells():
    negative = EvidenceValue("df0", 1, 1, -193_422_721_494.0, "R|1")
    _, expression = answer_plan("Giá trị là bao nhiêu đồng?", [negative])
    assert ".replace('--', '-')" in expression
