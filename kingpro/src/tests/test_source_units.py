from __future__ import annotations

from pathlib import Path

from kingpro.financial.source_units import (
    build_local_unit_factors,
    nearest_unit_marker,
    unit_factor_from_marker,
)
from kingpro.financial.statement_cube import _parse_table


def test_explicit_local_marker_recognizes_vietnamese_units() -> None:
    assert unit_factor_from_marker("Đơn vị tính: VND") == 1.0
    assert unit_factor_from_marker("Ngàn VND") == 1_000.0
    assert unit_factor_from_marker("Triệu đồng") == 1_000_000.0
    assert unit_factor_from_marker("Tỷ VND") == 1_000_000_000.0
    assert unit_factor_from_marker("Tỷ lệ tăng trưởng") is None


def test_nearest_marker_is_table_local() -> None:
    lines = ["Đơn vị: Triệu VND", "prose", "Ngàn VND", "<table><tr></tr></table>"]
    marker = nearest_unit_marker(lines, 4)
    assert marker == {"factor": 1_000.0, "line": 3, "distance": 1, "text": "Ngàn VND"}


def test_build_map_uses_exact_report_and_table_line(tmp_path: Path) -> None:
    report = "AAA_financial_statements_2024_consolidated"
    folder = tmp_path / report
    folder.mkdir()
    folder.joinpath(f"{report}_extracted.txt").write_text(
        "heading\nNgàn VND\n<table><tr><td>x</td></tr></table>\n",
        encoding="utf-8",
    )
    factors, stats = build_local_unit_factors(
        [{"report_id": report, "table_ref": f"{report}|3", "line": 3}],
        tmp_path,
    )
    assert factors == {f"{report}|3": 1_000.0}
    assert stats["local_unit_tables"] == 1


def test_explicit_marker_overrides_full_vnd_magnitude_heuristic(tmp_path: Path) -> None:
    table_ref = "AAA_financial_statements_2024_consolidated|3"
    csv_path = tmp_path / "table.csv"
    csv_path.write_text(
        "0,1,2,3,4\n"
        "Mã số,CHỈ TIÊU,Thuyết minh,Năm nay,Năm trước\n"
        "01,Doanh thu bán hàng,27,5.000.000.000,4.000.000.000\n"
        "11,Giá vốn hàng bán,28,(3.000.000.000),(2.000.000.000)\n"
        "20,Lợi nhuận gộp,,2.000.000.000,2.000.000.000\n"
        "25,Chi phí bán hàng,,(500.000.000),(400.000.000)\n"
        "26,Chi phí quản lý,,(300.000.000),(250.000.000)\n",
        encoding="utf-8",
    )
    entry = {
        "ticker": "AAA",
        "year": "2024",
        "report_id": "AAA_financial_statements_2024_consolidated",
        "table_ref": table_ref,
        "csv_path": csv_path.name,
    }
    parsed = _parse_table(entry, tmp_path, {table_ref: 1_000.0})
    assert parsed is not None
    _kind, cells = parsed
    revenue = next(cell for cell in cells if cell.metric_key == "kqkd:01")
    assert revenue.raw == "5.000.000.000"
    assert revenue.scale == 1_000.0
    assert revenue.value == 5_000_000_000_000.0
