from __future__ import annotations

from kingpro.retrieval import bm25_index
from kingpro.retrieval.bm25_index import extract_all_facets


def test_unique_legal_name_fragments_resolve_common_short_names() -> None:
    facets = extract_all_facets(
        "Trong nhóm Hoà Phát, Hoa Sen, Nam Kim, Đô thị Kinh Bắc, "
        "Minh Phú, Vicem Hà Tiên, Văn Phú và Hải Phát năm 2024"
    )
    assert set(facets["tickers"]) >= {
        "HPG", "HSG", "NKG", "KBC", "MPC", "HT1", "VPI", "HPX"
    }


def test_brand_aliases_resolve_binh_son_and_pvtrans() -> None:
    facets = extract_all_facets(
        "Trong nhóm Bình Sơn, PLX và PVTrans năm 2024"
    )
    assert set(facets["tickers"]) == {"BSR", "PLX", "PVT"}


def test_brand_alias_survives_unicode_dashes_in_company_name() -> None:
    facets = extract_all_facets(
        "Tổng Công ty Cổ phần Bia – Rượu – Nước giải khát Sài Gòn năm 2017"
    )
    assert "SAB" in facets["tickers"]


def test_short_brand_alias_does_not_shadow_longer_legal_name() -> None:
    high_tech_only = extract_all_facets(
        "CTCP Masan High-Tech Materials năm 2024"
    )
    both = extract_all_facets(
        "Tập đoàn Masan và CTCP Masan High-Tech Materials năm 2024"
    )
    assert set(high_tech_only["tickers"]) == {"MSR"}
    assert set(both["tickers"]) >= {"MSN", "MSR"}


def test_range_accepts_optional_nam_after_connector() -> None:
    facets = extract_all_facets(
        "Trong giai đoạn từ năm 2022 đến năm 2025 của KBC"
    )
    assert facets["years"] == ["2022", "2023", "2024", "2025"]


def test_sang_connector_expands_intermediate_years() -> None:
    facets = extract_all_facets(
        "Mức thay đổi của HPG từ năm 2022 sang năm 2024"
    )
    assert facets["years"] == ["2022", "2023", "2024"]


def test_generic_dau_khi_phrase_does_not_invent_a_company() -> None:
    facets = extract_all_facets("Các doanh nghiệp ngành Dầu khí năm 2024")
    assert facets["tickers"] == []


def test_unique_scope_backfill_fails_closed_for_ambiguous_reports() -> None:
    previous = bm25_index._CACHE.pop("facet_report_index", None)
    rows = [
        {"ticker": "AAA", "year": "2024", "scope": "hợp nhất", "table_ref": "one|1"},
        {"ticker": "BBB", "year": "2024", "scope": "hợp nhất", "table_ref": "first|1"},
        {"ticker": "BBB", "year": "2024", "scope": "hợp nhất", "table_ref": "second|1"},
    ]
    try:
        unique = bm25_index._unique_report_for_facet(rows, "AAA", "2024", "hợp nhất")
        ambiguous = bm25_index._unique_report_for_facet(rows, "BBB", "2024", "hợp nhất")
    finally:
        bm25_index._CACHE.pop("facet_report_index", None)
        if previous is not None:
            bm25_index._CACHE["facet_report_index"] = previous
    assert unique is not None and unique["table_ref"] == "one|1"
    assert ambiguous is None


def test_facet_report_inventory_keeps_one_representative_per_physical_filing() -> None:
    previous = bm25_index._CACHE.pop("facet_report_index", None)
    rows = [
        {"ticker": "NAB", "year": "2022", "scope": "hợp nhất", "table_ref": "first|1"},
        {"ticker": "NAB", "year": "2022", "scope": "hợp nhất", "table_ref": "first|2"},
        {"ticker": "NAB", "year": "2022", "scope": "hợp nhất", "table_ref": "second|1"},
    ]
    try:
        reports = bm25_index._reports_for_facet(rows, "NAB", "2022", "hợp nhất")
    finally:
        bm25_index._CACHE.pop("facet_report_index", None)
        if previous is not None:
            bm25_index._CACHE["facet_report_index"] = previous
    assert [row["table_ref"] for row in reports] == ["first|1", "second|1"]


def test_scope_neutral_report_is_unique_only_by_ticker_and_year_group() -> None:
    previous = bm25_index._CACHE.pop("facet_report_index", None)
    rows = [
        {"ticker": "HND", "year": "2021", "scope": "không rõ", "table_ref": "neutral|1"},
        {"ticker": "EVF", "year": "2021", "scope": "không rõ", "table_ref": "first|1"},
        {"ticker": "EVF", "year": "2021", "scope": "không rõ", "table_ref": "second|1"},
    ]
    try:
        unique = bm25_index._unique_report_for_facet(rows, "HND", "2021", "không rõ")
        ambiguous = bm25_index._unique_report_for_facet(rows, "EVF", "2021", "không rõ")
    finally:
        bm25_index._CACHE.pop("facet_report_index", None)
        if previous is not None:
            bm25_index._CACHE["facet_report_index"] = previous
    assert unique is not None and unique["table_ref"] == "neutral|1"
    assert ambiguous is None


def test_explicit_separate_scope_phrases_are_normalized() -> None:
    separate_questions = [
        "Tổng tài sản trên BCTC riêng của MBB năm 2023 là bao nhiêu?",
        "Giá trị trên báo cáo tài chính riêng của NVL năm 2023 là bao nhiêu?",
        "Số liệu trên báo cáo riêng lẻ của GAS năm 2023 là bao nhiêu?",
        "Xét riêng khối ngân hàng mẹ, MBB năm 2023 có tổng tài sản bao nhiêu?",
        "Số dư tại công ty mẹ của DTK năm 2023 là bao nhiêu?",
    ]
    for question in separate_questions:
        assert extract_all_facets(question)["scope"] == "công ty mẹ"


def test_bare_rieng_does_not_force_separate_scope() -> None:
    facets = extract_all_facets(
        "Xét riêng chỉ tiêu doanh thu hợp nhất của FPT năm 2024 là bao nhiêu?"
    )
    assert facets["scope"] == "hợp nhất"


def test_scope_cue_explicitness_distinguishes_default_from_requested_scope() -> None:
    assert bm25_index.has_explicit_scope_cue("BCTC riêng của GAS năm 2024")
    assert bm25_index.has_explicit_scope_cue("Tổng tài sản hợp nhất GAS năm 2024")
    assert not bm25_index.has_explicit_scope_cue("Tổng tài sản GAS năm 2024")


def test_competitive_implicit_scope_requires_material_score_margin_and_no_explicit_cue() -> None:
    assert bm25_index.is_competitive_implicit_scope(
        "Tỷ lệ sở hữu của HPG năm 2023",
        10.0,
        12.3,
        minimum_ratio=1.22,
    )
    assert not bm25_index.is_competitive_implicit_scope(
        "Tỷ lệ sở hữu của HPG năm 2023",
        10.0,
        12.1,
        minimum_ratio=1.22,
    )


def test_ownership_scope_competition_is_semantically_bounded() -> None:
    facets = {"tickers": ["HPG"], "years": ["2023"], "analytic": False}
    assert bm25_index.is_competitive_ownership_scope(
        "Tỷ lệ sở hữu công ty con của HPG năm 2023",
        facets,
        10.0,
        12.1,
        minimum_ratio=1.2,
    )
    assert not bm25_index.is_competitive_ownership_scope(
        "Doanh thu của HPG năm 2023",
        facets,
        10.0,
        20.0,
        minimum_ratio=1.2,
    )
    assert not bm25_index.is_competitive_ownership_scope(
        "Tỷ lệ sở hữu trên BCTC hợp nhất HPG năm 2023",
        facets,
        10.0,
        20.0,
        minimum_ratio=1.2,
    )


def test_count_scope_fallback_requires_multi_entity_single_period_count() -> None:
    eligible = {"tickers": ["HDG", "GEG", "DNH"], "years": ["2025"], "analytic": True}
    assert bm25_index.needs_count_scope_fallback(
        "Có bao nhiêu trong số HDG, GEG và DNH đạt ngưỡng năm 2025?", eligible
    )
    assert not bm25_index.needs_count_scope_fallback(
        "Công ty nào trong HDG, GEG và DNH đạt ngưỡng năm 2025?", eligible
    )
    assert not bm25_index.needs_count_scope_fallback(
        "Có bao nhiêu HDG, GEG và DNH đạt ngưỡng năm 2024 và 2025?",
        {**eligible, "years": ["2024", "2025"]},
    )


def test_catalog_universe_scan_requires_unnamed_broad_analytic_filter() -> None:
    eligible = {"tickers": [], "years": ["2015", "2016"], "analytic": True}
    assert bm25_index.needs_catalog_universe_scan(
        "Trong các công ty có hàng tồn kho năm 2016 giảm so với năm 2015", eligible
    )
    assert not bm25_index.needs_catalog_universe_scan(
        "Trong các công ty HPG và HSG có hàng tồn kho giảm",
        {**eligible, "tickers": ["HPG", "HSG"]},
    )
    assert not bm25_index.needs_catalog_universe_scan(
        "Doanh thu các công ty năm 2016", eligible
    )


def test_catalog_universe_inventory_keeps_one_report_representative() -> None:
    rows = [
        {"ticker": "AAA", "year": "2015", "scope": "hợp nhất", "table_ref": "a|1"},
        {"ticker": "AAA", "year": "2015", "scope": "hợp nhất", "table_ref": "a|2"},
        {"ticker": "AAA", "year": "2016", "scope": "hợp nhất", "table_ref": "b|1"},
        {"ticker": "AAA", "year": "2016", "scope": "công ty mẹ", "table_ref": "c|1"},
    ]
    reports = bm25_index.catalog_universe_reports(rows, ["2015", "2016"], "hợp nhất")
    assert [row["table_ref"] for row in reports] == ["a|1", "b|1"]


def test_common_evidence_year_requires_complete_cross_entity_support() -> None:
    rows = [
        {"ticker": "MSN", "year": "2023", "scope": "hợp nhất", "score": 12.0},
        {"ticker": "VNM", "year": "2023", "scope": "hợp nhất", "score": 4.0},
        {"ticker": "OGC", "year": "2023", "scope": "hợp nhất", "score": 9.0},
        {"ticker": "MSN", "year": "2024", "scope": "hợp nhất", "score": 8.0},
        {"ticker": "VNM", "year": "2024", "scope": "hợp nhất", "score": 8.0},
        {"ticker": "OGC", "year": "2024", "scope": "hợp nhất", "score": 8.0},
        {"ticker": "MSN", "year": "2025", "scope": "hợp nhất", "score": 20.0},
        {"ticker": "VNM", "year": "2025", "scope": "công ty mẹ", "score": 20.0},
    ]
    assert bm25_index.select_common_evidence_year(
        rows, ["MSN", "VNM", "OGC"], "hợp nhất"
    ) == "2024"
    assert bm25_index.select_common_evidence_year(rows, ["MSN"], "hợp nhất") is None
    assert not bm25_index.is_competitive_implicit_scope(
        "Tỷ lệ sở hữu trên BCTC hợp nhất HPG năm 2023",
        10.0,
        20.0,
        minimum_ratio=1.22,
    )


def test_opening_period_backfill_is_bounded_to_one_prior_year() -> None:
    assert bm25_index.explicit_opening_previous_years(
        "Trung bình hàng tồn kho đầu năm và cuối năm của FPT giai đoạn 2021-2024",
        ["2021", "2022", "2023", "2024"],
    ) == ["2020"]
    assert bm25_index.explicit_opening_previous_years(
        "Số dư đến ngày 01/01/2022 của HHV", ["2022"]
    ) == ["2021"]
    assert bm25_index.explicit_opening_previous_years(
        "Tăng trưởng từ đầu năm 2016 đến đầu năm 2019 của GAS", ["2016", "2019"]
    ) == ["2015", "2018"]


def test_non_opening_language_does_not_expand_years() -> None:
    assert bm25_index.explicit_opening_previous_years(
        "Đầu tư tài chính của FPT năm 2024", ["2024"]
    ) == []
    assert bm25_index.explicit_opening_previous_years(
        "Số dư đầu năm 2023 của SAB là bao nhiêu?", ["2023"]
    ) == []


def test_explicit_mixed_scope_is_bound_to_its_year_segment() -> None:
    question = (
        "DTK qua báo cáo công ty mẹ các năm 2022, 2023 và "
        "báo cáo hợp nhất các năm 2024, 2025"
    )
    assert bm25_index.extract_year_scope_overrides(question) == {
        "2022": "công ty mẹ",
        "2023": "công ty mẹ",
        "2024": "hợp nhất",
        "2025": "hợp nhất",
    }


def test_single_scope_does_not_create_year_overrides() -> None:
    assert bm25_index.extract_year_scope_overrides(
        "Báo cáo tài chính hợp nhất của FPT giai đoạn 2022-2024"
    ) == {}


def test_average_balance_metrics_require_one_prior_report_year() -> None:
    assert bm25_index.financial_formula_previous_years(
        "ROE của DCM năm 2017 là bao nhiêu?", ["2017"]
    ) == ["2016"]
    assert bm25_index.financial_formula_previous_years(
        "Tỷ suất sinh lời trên vốn chủ sở hữu của DCM năm 2017", ["2017"]
    ) == ["2016"]
    assert bm25_index.financial_formula_previous_years(
        "Vòng quay vốn chủ sở hữu của HSG năm 2019", ["2019"]
    ) == ["2018"]
    assert bm25_index.financial_formula_previous_years(
        "Số ngày tồn kho của HPG giai đoạn 2022-2024", ["2022", "2023", "2024"]
    ) == ["2021"]


def test_non_balance_ratio_does_not_add_formula_year() -> None:
    assert bm25_index.financial_formula_previous_years(
        "Biên lợi nhuận gộp của HPG năm 2024", ["2024"]
    ) == []
    assert bm25_index.financial_formula_previous_years(
        "Hàng tồn kho bình quân năm 2021 và năm 2022 của HPG", ["2021", "2022"]
    ) == []
    assert bm25_index.financial_formula_previous_years(
        "Tỷ lệ dự phòng hàng tồn kho bình quân qua các năm 2021, 2024", ["2021", "2024"]
    ) == []


def test_period_growth_scan_requires_its_baseline_year() -> None:
    assert bm25_index.growth_scan_previous_years(
        "Trong giai đoạn 2020-2024, năm có tốc độ tăng doanh thu cao nhất của HPG",
        ["2020", "2021", "2022", "2023", "2024"],
    ) == ["2019"]
    assert bm25_index.growth_scan_previous_years(
        "Giai đoạn 2022-2025, năm doanh thu bị sụt giảm sâu nhất so với năm trước",
        ["2022", "2023", "2024", "2025"],
    ) == ["2021"]


def test_two_year_growth_comparison_does_not_expand_as_period_scan() -> None:
    assert bm25_index.growth_scan_previous_years(
        "Doanh thu năm 2024 tăng so với năm 2023", ["2023", "2024"]
    ) == []


def test_multi_entity_comparative_selection_requires_one_baseline_year() -> None:
    assert bm25_index.comparative_selection_previous_years(
        "Năm 2025, trong nhóm ASM, DBC và MSN, công ty có tăng trưởng "
        "doanh thu cao nhất là công ty nào?",
        ["2025"],
        entity_count=3,
    ) == ["2024"]
    assert bm25_index.comparative_selection_previous_years(
        "Trong nhóm HPG, HSG và NKG năm 2024, doanh thu tăng trên 3% "
        "so với kỳ so sánh ở bao nhiêu công ty?",
        ["2024"],
        entity_count=3,
    ) == ["2023"]


def test_comparative_selection_backfill_is_bounded_to_multiple_entities() -> None:
    question = "Tăng trưởng doanh thu của HPG năm 2024 so với năm 2023 là bao nhiêu?"
    assert bm25_index.comparative_selection_previous_years(
        question, ["2023", "2024"], entity_count=1
    ) == []
    assert bm25_index.comparative_selection_previous_years(
        "Doanh thu của HPG, HSG và NKG năm 2024 là bao nhiêu?",
        ["2024"],
        entity_count=3,
    ) == []
    assert bm25_index.comparative_selection_previous_years(
        "Trong nhóm HPG, HSG và NKG, tăng trưởng doanh thu năm 2024 "
        "so với năm 2023 là bao nhiêu?",
        ["2023", "2024"],
        entity_count=3,
    ) == []
    assert bm25_index.comparative_selection_previous_years(
        "Chênh lệch doanh thu giữa HPG và HSG năm 2024 là bao nhiêu?",
        ["2024"],
        entity_count=2,
    ) == []


def test_multi_entity_comparative_series_adds_one_source_dependency_year() -> None:
    assert bm25_index.comparative_series_previous_years(
        "Trong nhóm HPG, HSG và NKG, xét tăng trưởng doanh thu từ năm 2023 "
        "sang năm 2024, công ty nào cao nhất?",
        ["2023", "2024"],
        entity_count=3,
    ) == ["2022"]
    assert bm25_index.comparative_series_previous_years(
        "HPG có doanh thu năm 2023 và 2024 là bao nhiêu?",
        ["2023", "2024"],
        entity_count=1,
    ) == []


def test_sparse_single_entity_series_fills_only_internal_predecessors() -> None:
    assert bm25_index.sparse_series_predecessor_years(
        ["2016", "2018", "2019", "2020"], entity_count=1, analytic=True
    ) == ["2017"]
    assert bm25_index.sparse_series_predecessor_years(
        ["2015", "2021", "2023", "2025"], entity_count=1, analytic=True
    ) == ["2020", "2022", "2024"]
    assert bm25_index.sparse_series_predecessor_years(
        ["2021", "2022", "2023"], entity_count=1, analytic=True
    ) == []
    assert bm25_index.sparse_series_predecessor_years(
        ["2021", "2023", "2025"], entity_count=2, analytic=True
    ) == []


def test_ambiguous_report_variants_are_retained_only_for_long_single_entity_series() -> None:
    eligible = {
        "tickers": ["NAB"],
        "years": ["2021", "2022", "2023", "2024"],
        "analytic": True,
    }
    multi_entity = {**eligible, "tickers": ["NAB", "OCB"]}
    short_series = {**eligible, "years": ["2023", "2024"]}
    plain_lookup = {**eligible, "analytic": False}

    assert bm25_index.series_report_variant_limit(
        eligible,
        1,
        question="Số dư tại năm có giá trị cho vay lớn nhất là bao nhiêu?",
        retain_ambiguous_series_reports=True,
    ) == 2
    assert bm25_index.series_report_variant_limit(
        multi_entity,
        1,
        question="Số dư tại năm có giá trị cho vay lớn nhất là bao nhiêu?",
        retain_ambiguous_series_reports=True,
    ) == 1
    assert bm25_index.series_report_variant_limit(
        short_series,
        1,
        question="Số dư tại năm có giá trị cho vay lớn nhất là bao nhiêu?",
        retain_ambiguous_series_reports=True,
    ) == 1
    assert bm25_index.series_report_variant_limit(
        plain_lookup,
        1,
        question="Số dư tại năm có giá trị cho vay lớn nhất là bao nhiêu?",
        retain_ambiguous_series_reports=True,
    ) == 1
    assert bm25_index.series_report_variant_limit(
        eligible,
        1,
        question="Số dư tại năm có giá trị cho vay lớn nhất là bao nhiêu?",
        retain_ambiguous_series_reports=False,
    ) == 1
    assert bm25_index.series_report_variant_limit(
        eligible,
        1,
        question="Lãi thuần cao nhất trong các năm là bao nhiêu?",
        retain_ambiguous_series_reports=True,
    ) == 1
    assert bm25_index.comparative_series_previous_years(
        "Doanh thu HPG, HSG và NKG năm 2023 và 2024 là bao nhiêu?",
        ["2023", "2024"],
        entity_count=3,
    ) == []


def test_semantic_year_rules_do_not_cascade() -> None:
    question = (
        "Trong giai đoạn 2021-2025, năm có tốc độ tăng doanh thu cao nhất "
        "có ROE tính trên vốn chủ sở hữu bình quân là bao nhiêu?"
    )
    assert bm25_index.expand_retrieval_years(
        question,
        ["2021", "2022", "2023", "2024", "2025"],
        formula=True,
        growth_scan=True,
    ) == ["2020", "2021", "2022", "2023", "2024", "2025"]


def test_curated_long_name_and_bank_brand_aliases_do_not_cross_contaminate() -> None:
    real_estate = extract_all_facets(
        "Tập đoàn Đất Xanh, CTCP Địa ốc Sài Gòn Thương Tín, VINGROUP và C.E.O năm 2017"
    )
    banks = extract_all_facets(
        "Ngân hàng TMCP Phương Đông, Ngân hàng TMCP Á Châu và "
        "Ngân hàng TMCP Sài Gòn Thương Tín năm 2021"
    )
    assert set(real_estate["tickers"]) >= {"DXG", "SCR", "VIC", "CEO"}
    assert "STB" not in real_estate["tickers"]
    assert set(banks["tickers"]) >= {"OCB", "ACB", "STB"}


def test_curated_product_and_bank_short_names_resolve() -> None:
    facets = extract_all_facets(
        "Sabeco, Sản xuất Kinh doanh Xuất nhập khẩu Dịch vụ và Đầu tư Tân Bình, "
        "Saigonbank, Nông nghiệp Quốc tế HAGL, "
        "Eximbank, Dịch vụ Hoàng Huy và công ty mẹ TKV năm 2024"
    )
    assert set(facets["tickers"]) >= {"SAB", "PRT", "SGB", "HNG", "EIB", "HHS", "DTK"}
