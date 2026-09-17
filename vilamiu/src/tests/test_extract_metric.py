"""`extract_metric` feeds every label match in the pipeline, so it is pinned.

Half these cases are guards against over-stripping. The function is shared by
the single-cell branch (its strongest one) and by any derived composition, so a
regex that trims one word too many costs answers in both.
"""

from vifin.answering.lookup import extract_metric


def test_canonical_metric_before_owner_clause():
    assert extract_metric(
        "Lãi tiền gửi năm 2018 của công ty mẹ VJC là bao nhiêu triệu đồng?"
    ) == "Lãi tiền gửi"


def test_year_list_clause_is_stripped_whole():
    # The clause contains commas of its own; stopping at the first one left the
    # metric starting with a year.
    assert extract_metric(
        "Trong các năm 2020, 2021 và 2022, số dư vay ngân hàng của ABC là bao nhiêu tỷ đồng?"
    ) == "vay ngân hàng"


def test_company_first_phrasing():
    assert extract_metric(
        "CRE có tổng doanh thu bán hàng và cung cấp dịch vụ trong năm 2019 "
        "là bao nhiêu nghìn tỷ đồng?"
    ) == "tổng doanh thu bán hàng và cung cấp dịch vụ"


def test_entity_marker_before_verb():
    assert extract_metric(
        "Doanh nghiệp CTCP Tập đoàn C.E.O ghi nhận lãi tiền gửi, tiền cho vay "
        "bao nhiêu tỷ đồng trong năm 2023?"
    ) == "lãi tiền gửi, tiền cho vay"


def test_leading_operation_noun_stripped():
    assert extract_metric(
        "Chênh lệch vốn chủ sở hữu giữa cuối năm 2025 và cuối năm 2024 của ABC là bao nhiêu?"
    ) == "vốn chủ sở hữu"


def test_operation_word_kept_when_it_names_the_account():
    # "chênh lệch tỷ giá" is a line item, not a subtraction.
    assert extract_metric(
        "Lỗ chênh lệch tỷ giá của CTCP Đường Quảng Ngãi trong năm 2023 là bao nhiêu tỷ đồng?"
    ) == "Lỗ chênh lệch tỷ giá"
    assert extract_metric(
        "Chênh lệch tỷ giá hối đoái của ABC năm 2020 là bao nhiêu?"
    ) == "Chênh lệch tỷ giá hối đoái"


def test_metric_containing_co_is_not_truncated():
    # "có" here is part of the account name, and no entity marker precedes it.
    assert extract_metric(
        "Tiền gửi có kỳ hạn của Ngân hàng TMCP Á Châu năm 2022 là bao nhiêu tỷ đồng?"
    ) == "Tiền gửi có kỳ hạn"


def test_scope_words_removed():
    assert extract_metric(
        "Giá vốn bán điện của công ty mẹ CTCP Thủy điện Đa Nhim năm 2019 là bao nhiêu?"
    ) == "Giá vốn bán điện"


def test_no_year_survives_a_leading_period_clause():
    assert extract_metric("Cuối năm 2024, tổng tài sản của ABC là bao nhiêu?") == "tổng tài sản"


def test_cua_naming_the_account_is_not_a_split_point():
    # "của quyền sử dụng đất" names the asset; the owner clause is the later one.
    assert extract_metric(
        "Giá trị còn lại của quyền sử dụng đất của CTCP ABC cuối năm 2025 là bao nhiêu?"
    ) == "Giá trị còn lại của quyền sử dụng đất"
    assert extract_metric(
        "Giá vốn của dịch vụ đã cung cấp của Công ty X năm 2020 là bao nhiêu tỷ đồng?"
    ) == "Giá vốn của dịch vụ đã cung cấp"


def test_cua_before_ticker_is_a_split_point():
    assert extract_metric(
        "Tiền gửi của khách hàng của VCB cuối năm 2019 là bao nhiêu tỷ đồng?"
    ) == "Tiền gửi của khách hàng"


def test_no_entity_after_any_cua_keeps_the_whole_head():
    assert extract_metric(
        "Giá trị còn lại của bất động sản đầu tư là bao nhiêu tỷ đồng?"
    ) == "Giá trị còn lại của bất động sản đầu tư"


def test_leading_period_phrase_with_metric_behind_it():
    assert extract_metric(
        "Số dư cuối năm 2015 của Quỹ bình ổn giá xăng dầu là bao nhiêu tỷ đồng?"
    ) == "Quỹ bình ổn giá xăng dầu"


def test_period_word_mid_phrase_does_not_truncate_the_head():
    # Regression guard: this collapsed to "còn lại" in an earlier revision.
    assert extract_metric(
        "Giá trị còn lại cuối năm Tài sản cố định vô hình của ABC là bao nhiêu?"
    ) == "Giá trị còn lại cuối năm Tài sản cố định vô hình"


def test_short_metric_survives():
    assert extract_metric("Tiền của ABC năm 2020 là bao nhiêu tỷ đồng?") == "Tiền"
