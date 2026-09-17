import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.analyze_document_unit_impact import (
    output_multiplier,
    query_normalizes_vnd,
    table_unit,
)


def test_output_multiplier_handles_vietnamese_d_stroke() -> None:
    assert output_multiplier("bao nhiêu nghìn tỷ đồng?") == 1_000_000_000_000
    assert output_multiplier("bao nhiêu trăm tỷ VNĐ?") == 100_000_000_000
    assert output_multiplier("bao nhiêu tỷ đồng?") == 1_000_000_000
    assert output_multiplier("bao nhiêu triệu đồng?") == 1_000_000
    assert output_multiplier("bao nhiêu nghìn đồng?") == 1_000


def test_safe_contract_requires_result_division_by_requested_unit() -> None:
    # Source and requested unit are both thousand VND: scale=1 is intentional.
    raw_thousand = {
        "question": "Chi phí là bao nhiêu nghìn đồng?",
        "pandas_query": "result = round(v0, 2)",
    }
    assert not query_normalizes_vnd(raw_thousand)

    normalized_billion = {
        "question": "Tổng dòng tiền là bao nhiêu tỷ đồng?",
        "pandas_query": "result = round((v0 + v1) / 1e9, 2)",
    }
    assert query_normalizes_vnd(normalized_billion)


def test_table_unit_accepts_ascii_ocr_banner_but_not_bare_vnd(tmp_path: Path) -> None:
    thousand = tmp_path / "thousand.csv"
    thousand.write_text("0,1\n,2015Nghin VND\n", encoding="utf-8")
    assert table_unit(thousand, include_base=False) == 1_000

    bare = tmp_path / "bare.csv"
    bare.write_text("currency,value\nVND,123\n", encoding="utf-8")
    assert table_unit(bare, include_base=False) is None
