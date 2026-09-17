"""Unit tests for entity parsing."""

from vifinqa.semantic_parser import parse_entities, EntityRef

def test_parse_ticker():
    entities = parse_entities("Doanh thu của FPT năm 2023")
    assert len(entities) == 1
    assert entities[0].ticker == "FPT"

def test_parse_full_name():
    entities = parse_entities("Lợi nhuận của Ngân hàng TMCP Ngoại thương Việt Nam")
    assert len(entities) == 1
    assert entities[0].ticker == "VCB"
