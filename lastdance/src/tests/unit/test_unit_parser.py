"""Unit tests for unit parsing."""

from vifinqa.semantic_parser import parse_unit

def test_parse_unit_vnd():
    assert parse_unit("cho biết bằng tỷ đồng") == "VND_1e9"
    assert parse_unit("tính bằng triệu đồng") == "VND_1e6"
    assert parse_unit("đơn vị nghìn đồng") == "VND_1e3"

def test_parse_unit_percent():
    assert parse_unit("tỷ lệ phần trăm") == "percent"

def test_parse_unit_ratio():
    assert parse_unit("gấp bao nhiêu lần") == "ratio"
