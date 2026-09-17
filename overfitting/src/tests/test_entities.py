from pathlib import Path

import pandas as pd
import pytest

from vifinqa.router.entities import StockMap, parse_question


@pytest.fixture()
def stock_map(tmp_path: Path) -> StockMap:
    path = tmp_path / "code_stock.csv"
    pd.DataFrame(
        [
            ("FPT", "Công ty Cổ phần FPT"),
            ("FTS", "Công ty Cổ phần Chứng khoán FPT"),
            ("VNM", "Công ty Cổ phần Sữa Việt Nam"),
            ("GAS", "Tổng Công ty Khí Việt Nam - CTCP"),
            ("POW", "Tổng Công ty Điện lực Dầu khí Việt Nam - CTCP"),
            ("DTK", "Tổng Công ty Điện lực TKV - CTCP"),
            ("OCB", "Ngân hàng TMCP Phương Đông"),
            ("EIB", "Ngân hàng TMCP Xuất Nhập khẩu Việt Nam"),
            ("MSN", "CTCP Tập đoàn Masan"),
            ("MCH", "CTCP Hàng tiêu dùng Masan"),
            ("DBC", "CTCP Tập đoàn Dabaco Việt Nam"),
            ("MPC", "CTCP Tập đoàn Thủy sản Minh Phú"),
            ("ASM", "CTCP Tập đoàn Sao Mai"),
            ("OGC", "CTCP Tập đoàn Đại Dương"),
            ("QNS", "CTCP Đường Quảng Ngãi"),
            ("BSR", "CTCP Lọc hóa Dầu Bình Sơn"),
            ("PLX", "Tập đoàn Xăng dầu Việt Nam"),
            ("PVT", "Tổng CTCP Vận tải Dầu khí"),
            ("GEX", "CTCP Tập đoàn GELEX"),
            ("GEE", "CTCP Điện lực GELEX"),
            ("HHV", "CTCP Đầu tư Hạ tầng Giao thông Đèo Cả"),
            ("NKG", "Cong ty Co phan Thep Nam Kim"),
            ("HPG", "Cong ty Co phan Tap doan Hoa Phat"),
            ("DPM", "Tong Cong ty Phan bon va Hoa chat Dau khi - CTCP"),
            ("HT1", "Cong ty Co phan Xi mang Vicem Ha Tien"),
            ("HAG", "Cong ty Co phan Hoang Anh Gia Lai"),
            ("HNG", "Cong ty Co phan Nong nghiep Quoc te Hoang Anh Gia Lai"),
            ("DCM", "Cong ty Co phan Phan bon Dau khi Ca Mau"),
            ("PRT", "Tong Cong ty San xuat Xuat nhap khau Binh Duong - CTCP"),
            ("DXS", "Cong ty Co phan Dich vu Bat dong san Dat Xanh"),
            ("SAM", "Cong ty Co phan SAM Holdings"),
        ],
        columns=["Mã CK", "Tên công ty"],
    ).to_csv(path, index=False)
    return StockMap(path)


def test_finds_multiple_company_names_in_mention_order(stock_map):
    parsed = parse_question(
        "So sánh Công ty Cổ phần Sữa Việt Nam và Tổng Công ty Khí Việt Nam "
        "trong năm 2024.",
        stock_map,
    )
    assert parsed.tickers == ["VNM", "GAS"]
    assert parsed.ticker_source == "explicit_name"


def test_paraphrased_tan_binh_name_maps_to_prt(stock_map):
    parsed = parse_question(
        "Trong số CTCP Sản xuất Kinh doanh Xuất nhập khẩu Dịch vụ và Đầu tư "
        "Tân Bình và DCM, doanh thu năm 2022 là bao nhiêu?",
        stock_map,
    )

    assert parsed.tickers == ["PRT", "DCM"]


def test_short_nam_kim_name_maps_to_nkg(stock_map):
    parsed = parse_question(
        "Trong nhóm Hoà Phát, Hoa Sen và Nam Kim năm 2024, công ty nào có "
        "biên lợi nhuận cao nhất?",
        stock_map,
    )

    assert "NKG" in parsed.tickers


def test_scenario_trade_names_map_to_distinct_tickers(stock_map):
    oil = parse_question(
        "Trong các doanh nghiệp Bình Sơn, Tập đoàn Xăng dầu - PLX và PVTrans "
        "năm 2024, hệ số nào cao nhất?",
        stock_map,
    )
    gelex = parse_question(
        "Trong nhóm Điện lực Gelex, Tập đoàn Gelex, HHV và SAM năm 2023, "
        "biên lợi nhuận nào cao nhất?",
        stock_map,
    )

    assert oil.tickers == ["BSR", "PLX", "PVT"]
    assert gelex.tickers[:2] == ["GEE", "GEX"]


def test_finds_lowercase_tickers(stock_map):
    parsed = parse_question("so sánh vnm với gas năm 2023", stock_map)
    assert parsed.tickers == ["VNM", "GAS"]
    assert parsed.ticker_source == "explicit"


def test_finds_trade_names_used_in_count_groups(stock_map):
    parsed = parse_question(
        "Trong số các công ty Masan, Đại Dương và Vinamilk có lợi nhuận sau "
        "thuế dương, doanh nghiệp có biên lợi nhuận ròng cao nhất là đơn vị nào?",
        stock_map,
    )
    assert parsed.tickers == ["MSN", "OGC", "VNM"]
    assert parsed.ticker_source == "explicit_name"


def test_finds_short_food_company_names(stock_map):
    parsed = parse_question(
        "Năm 2024, nhóm Minh Phú, Dabaco, Sao Mai và Đường Quảng Ngãi có bao "
        "nhiêu doanh nghiệp tăng doanh thu?",
        stock_map,
    )
    assert parsed.tickers == ["MPC", "DBC", "ASM", "QNS"]
    assert parsed.ticker_source == "explicit_name"


def test_long_company_alias_masks_embedded_other_ticker(stock_map):
    parsed = parse_question(
        "Lợi nhuận của Công ty Cổ phần Chứng khoán FPT năm 2023 là bao nhiêu?",
        stock_map,
    )
    assert parsed.tickers == ["FTS"]


def test_parenthetical_target_wins_over_named_counterparty(stock_map):
    parsed = parse_question(
        "Cong ty Hoang Anh Gia Lai bao lanh khoan vay cho Cong ty Co phan "
        "Nong nghiep Quoc te Hoang Anh Gia Lai (HNG) nam 2023 bao nhieu?",
        stock_map,
    )
    assert parsed.tickers == ["HNG"]


def test_lowercase_vietnamese_word_is_not_misread_as_ticker(stock_map):
    parsed = parse_question(
        "Tien chi de mua sam tai san co dinh cua Cong ty Co phan Dich vu "
        "Bat dong san Dat Xanh nam 2021 la bao nhieu?",
        stock_map,
    )
    assert parsed.tickers == ["DXS"]


def test_parent_company_scope_excludes_earlier_counterparty_alias(stock_map):
    parsed = parse_question(
        "So du phai thu tu Cong ty CP Bao bi Dau khi Viet Nam cua cong ty me "
        "DCM nam 2023 la bao nhieu?",
        stock_map,
    )
    assert parsed.tickers == ["DCM"]


def test_repeated_parent_company_anchors_keep_all_comparison_targets(stock_map):
    parsed = parse_question(
        "Thu nhập của công ty mẹ Ngân hàng TMCP Xuất Nhập khẩu Việt Nam "
        "trừ thu nhập của công ty mẹ Ngân hàng TMCP Phương Đông năm 2025",
        stock_map,
    )
    assert parsed.tickers == ["EIB", "OCB"]


@pytest.mark.parametrize(
    ("question", "years"),
    [
        ("VNM trong giai đoạn 2021-2024", [2021, 2022, 2023, 2024]),
        ("VNM từ năm 2022 đến năm 2025", [2022, 2023, 2024, 2025]),
        ("VNM các năm 2018 và 2020", [2018, 2020]),
    ],
)
def test_year_ranges_are_expanded_but_year_lists_are_not(question, years, stock_map):
    assert parse_question(question, stock_map).years == years


def test_hundred_billion_question_unit(stock_map):
    parsed = parse_question(
        "Lợi nhuận của VNM năm 2024 là bao nhiêu trăm tỷ đồng?", stock_map
    )
    assert parsed.unit_scale == 1e11
    assert parsed.unit_name == "trăm tỷ đồng"
    assert parsed.output_type == "number"


def test_parenthetical_unit_is_not_misread_as_count(stock_map):
    parsed = parse_question(
        "Giá vốn của VNM năm 2024 là bao nhiêu? (đơn vị: đồng)", stock_map
    )
    assert parsed.output_type == "number"
    assert parsed.unit_name == "đồng"


def test_requested_money_unit_beats_percent_condition(stock_map):
    parsed = parse_question(
        "Trong các năm có tỷ lệ lợi nhuận trên doanh thu lớn hơn 10%, "
        "doanh thu thấp nhất của VNM là bao nhiêu tỷ đồng?",
        stock_map,
    )
    assert parsed.output_type == "number"
    assert parsed.unit_scale == 1e9
    assert parsed.unit_name == "tỷ đồng"


def test_first_requested_unit_beats_later_condition_unit(stock_map):
    parsed = parse_question(
        "Chi phí có thể tăng tối đa bao nhiêu phần trăm trước khi hệ số của "
        "VNM giảm về 2,0 lần?",
        stock_map,
    )
    assert parsed.output_type == "percent"


def test_count_of_units_is_distinct_from_parenthetical_unit(stock_map):
    parsed = parse_question(
        "Có bao nhiêu đơn vị gồm VNM và GAS có hệ số dưới 1,5 lần?", stock_map
    )
    assert parsed.output_type == "count"


@pytest.mark.parametrize(
    ("question", "output_type", "unit_name", "is_percent"),
    [
        ("ROE của VNM năm 2024 là bao nhiêu phần trăm?", "percent", "%", True),
        (
            "Biên lợi nhuận của VNM tăng bao nhiêu điểm phần trăm năm 2024?",
            "percentage_point",
            "điểm phần trăm",
            False,
        ),
        (
            "Tỷ lệ CFO trên LNST của VNM năm 2024 là bao nhiêu lần?",
            "ratio",
            "lần",
            False,
        ),
        ("Doanh thu VNM cao nhất vào năm nào?", "year", "năm", False),
        ("Có bao nhiêu ngân hàng OCB và EIB tăng trưởng dương?", "count", "số lượng", False),
    ],
)
def test_output_type_has_precedence_over_metric_words(
    question, output_type, unit_name, is_percent, stock_map
):
    parsed = parse_question(question, stock_map)
    assert parsed.output_type == output_type
    assert parsed.unit_name == unit_name
    assert parsed.is_percent is is_percent


def test_answer_tail_turn_unit_overrides_percent_filter_metric(stock_map):
    parsed = parse_question(
        "Trong nhom co ty trong tai san dai han cao nhat, vong quay tong tai san "
        "cua VNM nam 2024 la bao nhieu vong?",
        stock_map,
    )
    assert parsed.output_type == "ratio"
    assert parsed.unit_name == "vòng"


def test_fertilizer_trade_names_do_not_fuzzy_match_phu_nhuan(stock_map):
    parsed = parse_question(
        "Trong hai doanh nghiệp Đạm Phú Mỹ và Đạm Cà Mau, doanh nghiệp nào "
        "có biên lợi nhuận gộp cao hơn?",
        stock_map,
    )

    assert parsed.tickers == ["DPM", "DCM"]


def test_vicem_ha_tien_trade_name_maps_to_ht1(stock_map):
    parsed = parse_question(
        "So sánh Hòa Phát, Đạm Cà Mau và Vicem Hà Tiên năm 2024.",
        stock_map,
    )

    assert parsed.tickers == ["HPG", "DCM", "HT1"]


def test_short_dau_khi_ca_mau_name_maps_to_dcm(stock_map):
    parsed = parse_question(
        "Trong nhóm Dầu Khí Cà Mau, DPM, GVR, Hòa Phát và Vicem Hà Tiên.",
        stock_map,
    )

    assert parsed.tickers == ["DCM", "DPM", "HPG", "HT1"]


@pytest.mark.parametrize(
    ("question", "unit_scale", "unit_name"),
    [
        ("VNM co bao nhieu trieu co phieu?", 1e6, "triệu cổ phiếu"),
        ("Khoan vay cua VNM la bao nhieu trieu USD?", 1e6, "triệu USD"),
    ],
)
def test_non_money_quantity_units_are_not_labeled_as_vnd(
    question, unit_scale, unit_name, stock_map
):
    parsed = parse_question(question, stock_map)
    assert parsed.output_type == "number"
    assert parsed.unit_scale == unit_scale
    assert parsed.unit_name == unit_name


def test_count_output_supports_in_population_wording(stock_map):
    parsed = parse_question(
        "Co bao nhieu trong so cac ngan hang OCB va EIB co tang truong duong?",
        stock_map,
    )
    assert parsed.output_type == "count"


def test_count_metric_uses_condition_not_bao_nhieu_prefix(stock_map):
    parsed = parse_question(
        "Co bao nhieu cong ty VNM va GAS co dong tien thuan tu hoat dong "
        "kinh doanh lon hon 1 nghin ty dong trong nam 2025?",
        stock_map,
    )
    assert parsed.output_type == "count"
    assert parsed.metric_norm == "dong tien thuan tu hoat dong kinh doanh"
    assert parsed.metric_variants[0] == parsed.metric_norm
    assert parsed.metric_keys == ["cfo"]


def test_router_exposes_canonical_derived_metric_key(stock_map):
    parsed = parse_question(
        "He so thanh toan nhanh cua VNM nam 2024 la bao nhieu lan?",
        stock_map,
    )
    assert "quick_ratio" in parsed.metric_keys


def test_count_metric_keeps_share_count_line_item(stock_map):
    parsed = parse_question(
        "Co bao nhieu cong ty FPT va FTS co so luong co phieu dang luu hanh "
        "vuot 350 trieu co phieu vao cuoi nam 2021?",
        stock_map,
    )
    assert parsed.output_type == "count"
    assert "so luong co phieu dang luu hanh" in parsed.metric_variants


def test_full_separate_financial_statement_phrase_selects_company_only(stock_map):
    parsed = parse_question(
        "Tiền trả trước cho người bán trên báo cáo tài chính riêng của VNM "
        "năm 2024 là bao nhiêu?",
        stock_map,
    )

    assert parsed.doc_type == "separate"
