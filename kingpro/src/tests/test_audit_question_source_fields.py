from scripts.audit_question_source_fields import (
    column_semantic_context,
    fold,
    header_matches_year,
    infer_document_scope,
    semantic_ancestor_label,
    semantic_label_ok,
    split_header_column,
    year_header,
)


def test_document_scope_accepts_numbered_extraction_suffixes() -> None:
    assert infer_document_scope("MBB_financial_statements_2022_separate_1") == "separate"
    assert infer_document_scope("POW_financial_statements_2025_consolidated_1") == "consolidated"
    assert infer_document_scope("STB_financial_statements_2021_consolidated") == "consolidated"
    assert infer_document_scope("VPB_financial_statements_2023") == "unknown"


def test_total_assets_rejects_accounting_identity_total_sources() -> None:
    assert semantic_label_ok("cdkt:270", "tong tai san") is True
    assert semantic_label_ok("cdkt:270", "tong cong tai san") is True
    assert semantic_label_ok("cdkt:270", "tong cong nguon von") is False


def test_depreciation_and_long_term_borrowing_note_semantics() -> None:
    assert semantic_label_ok(
        "note:depreciation_amortisation_expense",
        "chi phi khau hao va hao mon",
    ) is True
    assert semantic_label_ok(
        "note:depreciation_amortisation_expense",
        "tong cong chi phi theo yeu to",
    ) is False
    assert semantic_label_ok(
        "note:long_term_borrowings",
        "vay va no thue tai chinh dai han",
    ) is True
    assert semantic_label_ok("note:long_term_borrowings", "vay ngan han") is False


def test_blank_loan_provision_total_uses_section_ancestor() -> None:
    rows = [
        ["0", "1", "2"],
        [
            "Chi tiết dự phòng các khoản cho vay ngắn hạn",
            "31/12/2024VND",
            "01/01/2024VND",
        ],
        ["Công ty CP Nhà Hòa Bình", "75.075.867.681", "75.075.867.661"],
        ["Công ty CP Chứng khoán Sen Vàng", "1.429.181.347", "1.429.181.347"],
        ["Ông Lê Anh Dũng", "4.359.635.693", "4.359.635.693"],
        ["", "80.864.684.721", "80.864.684.701"],
    ]

    assert semantic_ancestor_label(
        rows,
        row_idx=5,
        metric_key="note:short_term_loan_provision",
    ) == "Chi tiết dự phòng các khoản cho vay ngắn hạn"


def test_q612_prefers_local_loan_header_over_global_tax_header() -> None:
    rows = [
        ["0", "1", "2", "3"],
        ["Thuế", "Tại ngày 1.1.2021 Phải thu VND", "", ""],
        ["Thuế TNDN", "100", "", ""],
        ["", "", "", ""],
        ["16. VAY VÀ NỢ THUÊ TÀI CHÍNH", "", "", ""],
        ["(a) Ngắn hạn", "", "", ""],
        ["", "Tại ngày 31.12.2021 VND", "Tại ngày 1.1.2021 VND", ""],
        ["Vay ngân hàng", "3.773.154.733.117", "2.468.663.849.028", ""],
    ]

    assert year_header(rows, 2, selected_row_idx=7) == "Tại ngày 1.1.2021 VND"
    assert header_matches_year("Tại ngày 1.1.2021 VND", 2020, 2021) is True


def test_gross_sales_code01_rejects_net_revenue_row() -> None:
    assert semantic_label_ok(
        "kqkd:01", "doanh thu ban hang va cung cap dich vu"
    ) is True
    assert semantic_label_ok(
        "kqkd:01", "doanh thu thuan ve ban hang va cung cap dich vu"
    ) is False
    assert semantic_label_ok(
        "kqkd:01", "doanh thu ve ban hang va cung cap dich vu"
    ) is True


def test_outstanding_share_child_uses_semantic_ancestor() -> None:
    rows = [
        ["0", "1"],
        ["Cổ phiếu đã phát hành", ""],
        ["Cổ phiếu phổ thông", "1.200.662.193"],
        ["Cổ phiếu đang lưu hành", ""],
        ["Cổ phiếu phổ thông", "1.200.139.398"],
    ]

    assert semantic_ancestor_label(
        rows,
        row_idx=4,
        metric_key="note:ending_common_shares_outstanding",
    ) == "Cổ phiếu đang lưu hành"


def test_current_income_tax_expense_semantics() -> None:
    assert semantic_label_ok(
        "note:current_income_tax_expense_million",
        "chi phi thue thu nhap hien hanh",
    ) is True
    assert semantic_label_ok(
        "note:current_income_tax_expense_million",
        "du phong thua thieu trong nhung nam truoc",
    ) is False


def test_construction_in_progress_balance_sheet_semantics() -> None:
    assert semantic_label_ok(
        "cdkt:242",
        "1 chi phi xay dung co ban do dang",
    ) is True
    assert semantic_label_ok("cdkt:242", "tai san do dang dai han") is False


def test_cash_and_cash_equivalents_total_semantics() -> None:
    assert semantic_label_ok(
        "cdkt:110",
        "i tien va cac khoan tuong duong tien",
    ) is True
    assert semantic_label_ok("cdkt:110", "tien mat") is False


def test_split_header_right_aligns_value_columns() -> None:
    body = [
        ["0", "1", "2", "3", "4"],
        ["Net cash flow", "20", "", "9.685", "7.887"],
    ]
    header = [
        ["0", "1", "2", "3"],
        ["Code", "Note", "2024 VND", "2023 VND"],
    ]

    mapped = split_header_column(body, header, 3)

    assert mapped == 2
    assert year_header(header, mapped) == "2024 VND"


def test_short_term_supplier_advances_semantics() -> None:
    assert semantic_label_ok(
        "note:short_term_supplier_advances",
        "2 tra truoc cho nguoi ban ngan han",
    ) is True
    assert semantic_label_ok(
        "note:short_term_supplier_advances",
        "nguoi mua tra tien truoc ngan han",
    ) is False


def test_supplier_payable_matrix_binds_counterparty_and_metric() -> None:
    assert semantic_label_ok(
        "note:q338_pvoil_supplier_payable",
        "tong cong ty dau viet nam cong ty co phan phai tra nha cung cap",
    ) is True
    assert semantic_label_ok(
        "note:q338_pvoil_supplier_payable",
        "tong cong ty dau viet nam cong ty co phan phai thu khach hang",
    ) is False


def test_unlabeled_subtotal_uses_section_ancestor_not_last_component() -> None:
    rows = [
        ["0", "1", "2"],
        ["", "2018Triệu VND", "2017Triệu VND"],
        [
            "Chi phí thuế thu nhập hiện hành",
            "Chi phí thuế thu nhập hiện hành",
            "Chi phí thuế thu nhập hiện hành",
        ],
        ["Năm hiện hành", "726.873", "599.980"],
        ["Dự phòng (thừa)/thiếu trong những năm trước", "(181)", "30.981"],
        ["", "726.692", "630.961"],
    ]

    assert semantic_ancestor_label(
        rows,
        row_idx=5,
        metric_key="note:current_income_tax_expense_million",
    ) == "Chi phí thuế thu nhập hiện hành"


def test_matrix_note_combines_column_metric_and_row_period() -> None:
    rows = [
        ["0", "1", "2"],
        ["", "Chi phíthuế mặt bằngTriệu VND", "Chi phísửa chữa lớnTriệu VND"],
        ["Số dư đầu năm", "268.105", "91.450"],
        ["Số dư cuối năm", "258.051", "111.029"],
    ]
    column = column_semantic_context(rows, 1)
    context = f"Số dư cuối năm | {column}"

    assert column == "Chi phíthuế mặt bằngTriệu VND"
    assert semantic_label_ok("note:premises_tax_ending", fold(context)) is True
    assert header_matches_year("Số dư cuối năm", 2019, 2019) is True
