"""Unit tests for scope parsing."""

from vifinqa.semantic_parser import parse_scope

def test_parse_scope_consolidated():
    assert parse_scope("Tài sản hợp nhất của FPT") == "consolidated"

def test_parse_scope_separate():
    assert parse_scope("Lợi nhuận công ty mẹ VNM") == "separate"

def test_parse_scope_any():
    assert parse_scope("Doanh thu của FPT") == "any"
