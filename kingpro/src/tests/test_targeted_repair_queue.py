from scripts.build_targeted_repair_queue import family, plausible_counterfactual


def test_company_name_tong_cong_ty_is_not_an_aggregation_cue():
    question = "Chi phí của Tổng công ty A năm 2024 là bao nhiêu?"
    assert family(question) == "direct_lookup"


def test_tru_di_is_an_ordered_difference_cue():
    question = "Chi phí của Công ty A trừ đi chi phí của Tổng công ty B là bao nhiêu?"
    assert family(question) == "difference_sign"


def test_hieu_giua_is_an_ordered_difference_cue():
    question = "Hiệu giữa trái phiếu của ngân hàng A và ngân hàng B là bao nhiêu?"
    assert family(question) == "difference_sign"


def test_trai_phieu_cua_is_not_mistaken_for_hieu_cua():
    question = "Tổng giá trị trái phiếu của HAG cuối năm là bao nhiêu?"
    assert family(question) == "aggregation_total"


def test_plausible_counterfactual_rejects_scale_explosions():
    assert plausible_counterfactual({"baseline_answer": 10, "alternative_answer": 12})
    assert not plausible_counterfactual({"baseline_answer": 10, "alternative_answer": 10000})
