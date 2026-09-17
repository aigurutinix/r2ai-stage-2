"""Scope and operation classification for the derived-question composer.

`eligible` decides which questions leave the single-cell path, so its boundaries
are pinned. Half these cases assert that a question is *rejected* — a wrong
inclusion silently replaces a good single-cell answer with a composition built
from the wrong operands.
"""

from dataclasses import replace

from vifin.answering.compose import classify, eligible
from vifin.query.parse import ParsedQuestion


def q(text: str, **kwargs) -> ParsedQuestion:
    base = ParsedQuestion(id=1, question=text, tickers=["ABC"], years=[2022, 2023],
                          target_unit="ty")
    return replace(base, **kwargs)


def test_operation_words_are_recognised():
    assert classify("Chênh lệch doanh thu giữa năm 2022 và 2023") == "diff"
    assert classify("Tốc độ tăng trưởng doanh thu thuần") == "growth"
    assert classify("Doanh thu cao nhất qua các năm") == "max"
    assert classify("Doanh thu thấp nhất qua các năm") == "min"
    assert classify("Doanh thu trung bình qua các năm") == "avg"
    assert classify("Tổng doanh thu tích lũy qua các năm") == "sum"


def test_growth_beats_diff_when_both_words_appear():
    # "tăng trưởng ... so với" is a growth rate, not a subtraction.
    assert classify("Tăng trưởng doanh thu năm 2023 so với năm 2022") == "growth"


def test_metric_name_lookalikes_are_not_operations():
    assert classify("Lỗ chênh lệch tỷ giá năm 2023") is None
    assert classify("Tỷ lệ quyền biểu quyết tại công ty con") is None
    assert classify("Thu nhập bình quân tháng/người") is None
    assert classify("Chênh lệch đánh giá lại tài sản") is None


def test_plain_lookup_is_out_of_scope():
    assert eligible(q("Doanh thu thuần năm 2023")) is None


def test_multi_company_reduces_over_companies():
    assert eligible(q("Doanh thu cao nhất", tickers=["ABC", "DEF"],
                      years=[2023])) == ("max", "ticker")


def test_one_company_one_year_is_out_of_scope():
    assert eligible(q("Chênh lệch doanh thu", years=[2023])) is None


def test_no_year_at_all_is_out_of_scope():
    assert eligible(q("Doanh thu cao nhất", tickers=["ABC", "DEF"], years=[])) is None


def test_diff_requires_exactly_two_years():
    assert eligible(q("Chênh lệch doanh thu", years=[2021, 2022, 2023])) is None
    assert eligible(q("Chênh lệch doanh thu", years=[2022, 2023])) == ("diff", "year")
    # Two companies, one year: the two operands are the companies.
    assert eligible(q("Chênh lệch doanh thu", tickers=["ABC", "DEF"],
                      years=[2023])) == ("diff", "ticker")


def test_series_ops_accept_many_years():
    assert eligible(q("Doanh thu cao nhất", years=[2020, 2021, 2022, 2023])) == ("max", "year")


def test_money_op_needs_a_currency_unit_but_growth_does_not():
    assert eligible(q("Doanh thu cao nhất", target_unit="phan_tram")) is None
    assert eligible(q("Tăng trưởng doanh thu", target_unit="phan_tram")) == ("growth", "year")
