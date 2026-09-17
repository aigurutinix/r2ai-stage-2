from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_total_bucket_semantics.py"
SPEC = importlib.util.spec_from_file_location("audit_total_bucket_semantics", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(MODULE)


ROWS = [
    ["0", "1", "2", "3", "4"],
    ["Triệu VND", "Quá hạn", "Trong hạn", "Trong hạn", "Trong hạn"],
    ["Triệu VND", "Dưới 3 tháng", "Đến 1 tháng", "Trên 5 năm", "Tổng cộng"],
    ["Tiền gửi của khách hàng", "-", "66.076.449", "3.102", "259.236.746"],
]


def test_unqualified_balance_read_from_bucket_is_flagged() -> None:
    finding = MODULE.total_bucket_conflict(
        "Tiền gửi của khách hàng cuối năm 2019 là bao nhiêu triệu đồng?",
        ROWS,
        3,
        2,
    )
    assert finding is not None
    assert finding["selected_raw"] == "66.076.449"
    assert finding["total_raw"] == "259.236.746"


def test_explicit_bucket_question_is_clean() -> None:
    assert MODULE.total_bucket_conflict(
        "Tiền gửi của khách hàng trong hạn đến 1 tháng là bao nhiêu?",
        ROWS,
        3,
        2,
    ) is None


def test_total_column_is_clean() -> None:
    assert MODULE.total_bucket_conflict(
        "Tổng tiền gửi của khách hàng là bao nhiêu?",
        ROWS,
        3,
        4,
    ) is None


def test_equal_category_and_total_value_is_clean() -> None:
    rows = [row[:] for row in ROWS]
    rows[3][2] = rows[3][4]
    assert MODULE.total_bucket_conflict(
        "Tá»•ng tiá»n gá»­i cá»§a khÃ¡ch hÃ ng lÃ  bao nhiÃªu?", rows, 3, 2
    ) is None


def test_equal_category_and_total_value_can_be_audited_as_latent_mismatch() -> None:
    rows = [row[:] for row in ROWS]
    rows[3][2] = rows[3][4]
    finding = MODULE.total_bucket_conflict(
        "Tổng tiền gửi của khách hàng là bao nhiêu?",
        rows,
        3,
        2,
        include_equal_values=True,
    )
    assert finding is not None
    assert finding["selected_raw"] == finding["total_raw"]


def test_merged_bucket_and_total_unit_headers_are_detected() -> None:
    rows = [
        ["0", "1", "2", "3"],
        ["", "Trong nướcTriệu đồng", "Nước ngoàiTriệu đồng", "Tổng cộngTriệu đồng"],
        ["Phát hành giấy tờ có giá", "25.820.307", "-", "25.820.307"],
    ]
    finding = MODULE.total_bucket_conflict(
        "Tổng giấy tờ có giá phát hành cuối năm 2022 là bao nhiêu triệu đồng?",
        rows,
        2,
        1,
        include_equal_values=True,
    )
    assert finding is not None
    assert finding["selected_header"].startswith("Trong nước")
    assert finding["total_header"].startswith("Tổng cộng")


def test_shared_period_does_not_qualify_a_geography_bucket() -> None:
    rows = [
        ["0", "1", "2", "3"],
        ["", "Trong nướcTriệu đồng", "Nước ngoàiTriệu đồng", "Tổng cộngTriệu đồng"],
        [
            "Công nợ tại ngày 31 tháng 12 năm 2022",
            "Công nợ tại ngày 31 tháng 12 năm 2022",
            "Công nợ tại ngày 31 tháng 12 năm 2022",
            "Công nợ tại ngày 31 tháng 12 năm 2022",
        ],
        ["Phát hành giấy tờ có giá", "25.820.307", "-", "25.820.307"],
    ]
    assert MODULE.total_bucket_conflict(
        "Tổng giấy tờ có giá phát hành cuối năm 2022 là bao nhiêu triệu đồng?",
        rows,
        3,
        1,
        include_equal_values=True,
    ) is not None


def test_named_paid_column_with_merged_unit_is_clean() -> None:
    rows = [
        ["0", "1", "2"],
        ["", "Số đã nộpTriệu VND", "Tổng cộngTriệu VND"],
        ["Thuế TNDN", "(4.588.752)", "1.900.141"],
    ]
    assert MODULE.total_bucket_conflict(
        "Thuế TNDN đã nộp trong năm là bao nhiêu triệu đồng?",
        rows,
        2,
        1,
        include_equal_values=True,
    ) is None


def test_named_ocr_merged_account_column_is_clean() -> None:
    rows = [
        ["0", "1", "2"],
        ["", "Quỹ dựphòngtài chínhTriệu VND", "Tổng cộngTriệu VND"],
        ["Số dư cuối năm", "7.660.332", "70.955.961"],
    ]
    assert MODULE.total_bucket_conflict(
        "Tỷ trọng quỹ dự phòng tài chính trong vốn chủ sở hữu là bao nhiêu?",
        rows,
        2,
        1,
        include_equal_values=True,
    ) is None


def test_explicit_business_category_is_clean() -> None:
    rows = [
        ["0", "1", "2"],
        ["Triá»‡u VND", "Quá»¹ dá»± phÃ²ng tÃ i chÃ­nh", "Tá»•ng cá»™ng"],
        ["Cuá»‘i nÄƒm", "4.744.306", "44.900.909"],
    ]
    assert MODULE.total_bucket_conflict(
        "Tá»· trá»ng Quá»¹ dá»± phÃ²ng tÃ i chÃ­nh trong vá»‘n chá»§ sá»Ÿ há»¯u cuá»‘i nÄƒm",
        rows,
        2,
        1,
    ) is None


def test_compact_spacing_in_explicit_maturity_bucket_is_clean() -> None:
    rows = [
        ["0", "1", "2"],
        ["Triá»‡u VND", "Tá»« 1 Ä‘áº¿n3 thÃ¡ng", "Tá»•ng cá»™ng"],
        ["TÃ i sáº£n", "27.401.037", "181.190.363"],
    ]
    assert MODULE.total_bucket_conflict(
        "Tá»· trá»ng tÃ i sáº£n cÃ³ ká»³ háº¡n 1-3 thÃ¡ng trÃªn tá»•ng tÃ i sáº£n",
        rows,
        2,
        1,
    ) is None


def test_explicit_segment_acronym_is_clean() -> None:
    rows = [
        ["0", "1", "2"],
        ["VND", "Thu phÃ­ tráº¡m BOT", "Tá»•ng cá»™ng"],
        ["TÃ i sáº£n bá»™ pháº­n", "35.317", "36.201"],
    ]
    assert MODULE.total_bucket_conflict(
        "Tá»· trá»ng tÃ i sáº£n bá»™ pháº­n BOT trÃªn tá»•ng tÃ i sáº£n",
        rows,
        2,
        1,
    ) is None


def test_year_columns_without_total_header_are_clean() -> None:
    rows = [
        ["0", "1", "2"],
        ["", "31/12/2020", "01/01/2020"],
        ["Hàng tồn kho", "100", "80"],
    ]
    assert MODULE.total_bucket_conflict(
        "Hàng tồn kho cuối năm 2020 là bao nhiêu?", rows, 2, 1
    ) is None


def test_source_row_is_resolved_by_captured_label_and_value() -> None:
    read = {
        "source_row": 2,
        "source_column": 2,
        "source_label": "Tiền gửi của khách hàng",
        "raw": "66.076.449",
    }
    assert MODULE.resolve_source_row(ROWS, read) == 3


def test_compact_source_row_accounts_for_catalog_header() -> None:
    source = {
        "row_idx": "2",
        "col_idx": "2",
        "raw": "66.076.449",
    }
    assert MODULE.resolve_compact_source_row(ROWS, source) == 3


def test_compact_source_row_falls_back_to_nearest_raw_match() -> None:
    rows = [
        ["0", "1"],
        ["Header", "NÄƒm 2020"],
        ["Other", "100"],
        ["Wanted", "250"],
    ]
    source = {"row_idx": "0", "col_idx": "1", "raw": "250"}
    assert MODULE.resolve_compact_source_row(rows, source) == 3


def test_financial_instrument_header_is_not_mistaken_for_total() -> None:
    assert not MODULE.is_total_header("Công cụ tài chính phái sinh")
    assert MODULE.is_total_header("Trong hạn | Tổng cộng")


def test_explicit_sum_of_buckets_is_suppressed() -> None:
    base = {
        "id": 71,
        "source_table": "GEG|1",
        "source_label": "Bán hàng và cung cấp dịch vụ ra bên ngoài",
        "total_column": 4,
        "total_raw": "2.998.867.342.581",
    }
    findings = [
        {**base, "selected_column": 1, "selected_raw": "2.935.428.348.323"},
        {**base, "selected_column": 2, "selected_raw": "63.438.994.258"},
    ]
    assert MODULE.suppress_additive_total_findings(findings) == []
