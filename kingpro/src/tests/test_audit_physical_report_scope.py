from scripts.audit_physical_report_scope import line_scope


def test_incidental_report_reference_is_not_a_masthead() -> None:
    assert line_scope(
        "Thông tin về lãi trên cổ phiếu được trình bày trên Báo cáo tài chính hợp nhất."
    ) == ""


def test_anchored_physical_mastheads_are_authoritative() -> None:
    assert line_scope("BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG") == "separate"
    assert line_scope("BÁO CÁO LƯU CHUYỂN TIỀN TỆ HỢP NHẤT") == "consolidated"
    assert line_scope("BẢN THUYẾT MINH BÁO CÁO TÀI CHÍNH RIÊNG (tiếp theo)") == "separate"
