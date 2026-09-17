import math

import pytest

from vifinqa.utils.viet_num import parse_vn_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234.567", 1_234_567.0),
        ("12,5", 12.5),
        ("5,832", 5.832),
        ("0,9237", 0.9237),
        ("0.123", 0.123),
        ("0.9237", 0.9237),
        ("24.643,992", 24_643.992),
        ("(1.839,25)", -1_839.25),
        ("1,234,567.89", 1_234_567.89),
        ("66,67%", 66.67),
    ],
)
def test_parse_vn_number_locale_formats(raw, expected):
    assert math.isclose(parse_vn_number(raw), expected, rel_tol=0, abs_tol=1e-9)


@pytest.mark.parametrize("raw", [None, "", "-", "không có", float("nan")])
def test_parse_vn_number_rejects_non_numbers(raw):
    assert parse_vn_number(raw) is None
