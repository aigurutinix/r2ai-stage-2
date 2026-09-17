from scripts.audit_role_bound_label_collisions import roles


def test_role_parser_distinguishes_governance_groups() -> None:
    assert roles("Thù lao thành viên HĐQT") == {"board"}
    assert roles("Thu nhập Ban Tổng Giám đốc") == {"management"}
    assert roles("Lương của Ban Kiểm soát") == {"supervisory"}


def test_generic_person_question_has_no_forced_role() -> None:
    assert roles("Thu nhập của bà Chu Thị Bình") == set()
