from scripts.audit_source_cell_semantics import semantic_direction_conflicts


def test_q638_lessor_source_conflicts_with_lessee_question() -> None:
    question = "Tăng trưởng cam kết thuê hoạt động ngắn hạn là bao nhiêu?"
    source = "Cam kết cho thuê hoạt động | Công ty cho thuê văn phòng"
    reasons = semantic_direction_conflicts(question, source)
    assert reasons == ["lease_role: question=lessee source=lessor"]


def test_q638_correct_lessee_source_is_clear() -> None:
    question = "Tăng trưởng cam kết thuê hoạt động ngắn hạn là bao nhiêu?"
    source = "Công ty thuê đất; tiền thuê tối thiểu phải trả trong tương lai"
    assert semantic_direction_conflicts(question, source) == []


def test_lessor_question_rejects_lessee_disclosure() -> None:
    question = "Cam kết cho thuê hoạt động cuối năm là bao nhiêu?"
    source = "Công ty thuê đất theo hợp đồng thuê hoạt động"
    assert semantic_direction_conflicts(question, source) == [
        "lease_role: question=lessor source=lessee"
    ]


def test_combined_mua_ban_is_not_split_into_trade_roles() -> None:
    question = "Thu nhập từ mua bán chứng khoán đầu tư là bao nhiêu?"
    source = "Thu nhập từ mua bán chứng khoán đầu tư"
    assert semantic_direction_conflicts(question, source) == []


def test_direct_payable_row_overrides_receivable_sibling_ancestry() -> None:
    question = "Số phải trả sau 12 tháng là bao nhiêu?"
    ancestry = "Quyền phải thu từ hợp đồng cho công ty liên kết vay"
    assert semantic_direction_conflicts(
        question, ancestry, "Số phải trả sau 12 tháng note:payables_after_12_months"
    ) == []


def test_selector_direction_does_not_contaminate_unrelated_target_cell() -> None:
    question = "Vật liệu tại năm có tổng nợ phải trả cao nhất là bao nhiêu?"
    ancestry = "Các khoản lãi phí phải thu | Tài sản có khác"
    assert semantic_direction_conflicts(
        question, ancestry, "Vật liệu và công cụ note:materials_and_tools"
    ) == []
