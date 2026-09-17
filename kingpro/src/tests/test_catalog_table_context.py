from __future__ import annotations

import pandas as pd

from kingpro.corpus.build_catalog import build_search_text
from kingpro.corpus.table_context import (
    header_paths,
    inferred_blank_totals,
    is_financial_number,
    section_ancestors,
    table_context,
)


def q312_frame(total_2022: str = "216.674.089.381") -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["", "31/12/2022", "31/12/2022", "1/1/2022", "1/1/2022"],
            ["", "Nguyên tệ", "Tương đương VND", "Nguyên tệ", "Tương đương VND"],
            ["USD", "9.102.109", "214.174.589.504", "5.544.215", "125.741.402.737"],
            ["EUR", "101.106", "2.499.499.877", "101.038", "2.565.581.147"],
            ["", "", total_2022, "", "128.306.983.884"],
        ]
    )


def test_q312_preserves_multi_level_headers_and_proves_blank_total() -> None:
    frame = q312_frame()

    assert "31/12/2022 > Tương đương VND" in header_paths(frame)
    assert "1/1/2022 > Tương đương VND" in header_paths(frame)
    assert inferred_blank_totals(frame) == (
        "Tổng cộng > 31/12/2022 > Tương đương VND",
        "Tổng cộng > 1/1/2022 > Tương đương VND",
    )


def test_non_additive_blank_row_is_not_called_a_total() -> None:
    frame = q312_frame(total_2022="216.674.089.380")
    # The opening-period column still proves its own total, but the altered
    # ending-period value must not inherit that label.
    inferred = inferred_blank_totals(frame)
    assert "Tổng cộng > 31/12/2022 > Tương đương VND" not in inferred
    assert "Tổng cộng > 1/1/2022 > Tương đương VND" in inferred


def test_year_and_date_cells_remain_in_the_header_band() -> None:
    assert is_financial_number("2022") is False
    assert is_financial_number("31/12/2022") is False
    assert is_financial_number("214.174.589.504") is True

    frame = pd.DataFrame(
        [
            ["", "2022", "31/12/2022"],
            ["", "Nguyên tệ", "Tương đương VND"],
            ["USD", "9.102.109", "214.174.589.504"],
        ]
    )
    assert "2022 > Nguyên tệ" in header_paths(frame)
    assert "31/12/2022 > Tương đương VND" in header_paths(frame)


def test_section_ancestor_prefers_closest_subsection_over_previous_prose() -> None:
    lines = [
        "32. Các khoản mục ngoài Bảng cân đối kế toán",
        "(a) Tài sản thuê ngoài",
        "Các khoản tiền thuê tối thiểu phải trả cho hợp đồng thuê hoạt động.",
        "<table><tr><td>old</td></tr></table>",
        "(b) Ngoại tệ các loại",
        "<table><tr><td>target</td></tr></table>",
    ]

    section = section_ancestors(lines, table_line=6)
    assert section == (
        "32. Các khoản mục ngoài Bảng cân đối kế toán | (b) Ngoại tệ các loại"
    )
    assert "thuê hoạt động" not in section
    assert "(a)" not in section


def test_build_search_text_contains_only_source_derived_structural_context() -> None:
    frame = q312_frame()
    context = table_context(frame)
    text = build_search_text(
        "VGT",
        "Tập đoàn Dệt May Việt Nam",
        "2022",
        "hợp nhất",
        frame,
        section_title=(
            "32. Các khoản mục ngoài Bảng cân đối kế toán | (b) Ngoại tệ các loại"
        ),
        context=context,
    )

    assert "Header:" in text
    assert "31/12/2022 | Nguyên tệ | Tương đương VND | 1/1/2022" in text
    assert "Chỉ tiêu: USD | EUR" in text
    assert "Dòng tổng suy luận: Tổng cộng" in text
    # Full ancestry remains available as structured metadata for reranking and
    # citations even though BM25 receives a deduplicated bag of path atoms.
    assert "31/12/2022 > Tương đương VND" in context["header_text"]
    assert (
        "Tổng cộng > 31/12/2022 > Tương đương VND"
        in context["inferred_row_text"]
    )
    assert "question" not in text.casefold()
    assert "answer" not in text.casefold()
