from types import SimpleNamespace

from vifinqa.retrieval.retrieve import (_apply_quota,
                                        _operating_lease_schedule_score,
                                        _retrieval_metric_variants)
from vifinqa.retrieval.shortlist import (candidate_matches_metric,
                                         candidate_matches_requirement,
                                         requirement_linking_variants,
                                         requirement_specificity_key)
from vifinqa.router.evidence import build_evidence_requirements, evidence_coverage


def _route(question, metric="doanh thu thuan", variants=None, facts=None):
    return SimpleNamespace(
        question=question,
        metric_norm=metric,
        metric_variants=variants or [metric],
        plan={"facts": facts or []},
    )


def test_quick_ratio_retrieval_variants_include_formula_components():
    route = _route(
        "Có bao nhiêu doanh nghiệp vừa có hệ số thanh toán nhanh lớn hơn 1 lần?"
    )

    variants = _retrieval_metric_variants(route)

    assert "tai san ngan han" in variants
    assert "hang ton kho" in variants
    assert "no ngan han" in variants


def test_debt_to_equity_retrieval_variants_include_formula_components():
    route = _route(
        "Doanh nghiệp nào có hệ số nợ phải trả trên vốn chủ sở hữu nhỏ hơn 1 lần?"
    )

    variants = _retrieval_metric_variants(route)

    assert "no phai tra" in variants
    assert "von chu so huu" in variants


def test_retrieval_variants_include_plan_fact_metrics():
    route = _route(
        "Tính tỷ lệ CFO trên doanh thu thuần của VNM",
        facts=[
            {"ticker": "VNM", "year": 2024, "doc_type": "consolidated",
             "metric": "luu chuyen tien thuan tu hoat dong kinh doanh"},
            {"ticker": "VNM", "year": 2024, "doc_type": "consolidated",
             "metric": "doanh thu thuan"},
        ],
    )

    variants = _retrieval_metric_variants(route)

    assert "luu chuyen tien thuan tu hoat dong kinh doanh" in variants
    assert "doanh thu thuan" in variants


def test_nested_ranking_builds_every_entity_operand_requirement():
    question = (
        "Năm 2016, hệ số thanh toán hiện hành của doanh nghiệp có hệ số "
        "nợ phải trả trên vốn chủ sở hữu cao nhất là bao nhiêu lần?"
    )

    requirements = build_evidence_requirements(
        question,
        ["VNM", "DLG", "MBS"],
        [2016],
        "consolidated",
    )

    assert len(requirements) == 12
    keys = {requirement.metric_key for requirement in requirements}
    assert keys == {"current_assets", "current_liabilities", "liabilities", "equity"}


def test_inventory_days_requires_prior_year_opening_inventory():
    requirements = build_evidence_requirements(
        "Mức tăng của tỷ lệ 365 lần hàng tồn kho bình quân đầu kỳ và cuối kỳ "
        "trên giá vốn hàng bán từ năm 2021 đến 2022",
        ["HPG"], [2021, 2022], "consolidated",
    )
    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert ("inventory", 2020) in periods
    assert ("inventory", 2021) in periods
    assert ("inventory", 2022) in periods
    assert ("cost_of_goods_sold", 2020) not in periods
    assert ("cost_of_goods_sold", 2021) in periods
    assert ("cost_of_goods_sold", 2022) in periods


def test_average_balance_formula_requires_prior_year_stock():
    requirements = build_evidence_requirements(
        "ROE năm 2024 được tính bằng lợi nhuận sau thuế chia cho vốn chủ sở "
        "hữu bình quân đầu và cuối kỳ",
        ["AAA"], [2024], "consolidated",
    )
    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert ("net_profit", 2024) in periods
    assert ("equity", 2023) in periods
    assert ("equity", 2024) in periods


def test_decomposed_quick_ratio_builds_all_formula_operands():
    requirements = build_evidence_requirements(
        "Tài sản ngắn hạn trừ hàng tồn kho rồi chia cho nợ ngắn hạn năm 2024",
        ["AAA"], [2024], "consolidated",
    )

    keys = {requirement.metric_key for requirement in requirements}

    assert {"current_assets", "inventory", "current_liabilities"} <= keys


def test_filter_aggregate_ratio_wording_builds_all_operands():
    requirements = build_evidence_requirements(
        "Trong các công ty có tỷ lệ tài sản ngắn hạn trên nợ ngắn hạn thấp "
        "hơn 1 lần, trung bình tỷ lệ lưu chuyển tiền thuần từ hoạt động kinh "
        "doanh trên nợ ngắn hạn là bao nhiêu lần?",
        ["ACV", "DLG"], [2024], "consolidated",
    )
    keys = {requirement.metric_key for requirement in requirements}

    assert keys == {"current_assets", "current_liabilities", "cfo"}


def test_filter_aggregate_cfo_alias_is_canonical():
    requirements = build_evidence_requirements(
        "Các công ty duy trì dòng tiền hoạt động dương ở cả năm 2024 và 2025 "
        "có tăng trưởng doanh thu thuần bình quân là bao nhiêu phần trăm?",
        ["AAA", "BBB"], [2024, 2025], "consolidated",
    )
    keys = {requirement.metric_key for requirement in requirements}

    assert {"cfo", "net_revenue"} <= keys


def test_decomposed_inventory_days_keeps_opening_period():
    requirements = build_evidence_requirements(
        "Giá trị hàng tồn kho bình quân năm 2021 và 2022 chia cho giá vốn "
        "hàng bán năm 2022 rồi nhân 365",
        ["AAA"], [2021, 2022], "consolidated",
    )

    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert ("inventory", 2020) in periods
    assert ("inventory", 2021) in periods
    assert ("inventory", 2022) in periods


def test_single_year_growth_adds_only_metric_previous_period():
    requirements = build_evidence_requirements(
        "Năm 2025, trong nhóm có tốc độ tăng trưởng doanh thu thuần cao hơn "
        "trung vị; hệ số khả năng thanh toán lãi vay của doanh nghiệp có biên "
        "lợi nhuận gộp cao nhất là bao nhiêu lần?",
        ["AAA", "BBB"], [2025], "consolidated",
    )
    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert ("net_revenue", 2024) in periods
    assert ("gross_profit", 2024) not in periods
    assert ("pretax_profit", 2024) not in periods


def test_multi_year_growth_adds_opening_period_for_every_candidate_year():
    requirements = build_evidence_requirements(
        "Trong giai đoạn 2021 đến 2023, năm đầu tiên tăng trưởng doanh thu "
        "thuần âm là năm nào?",
        ["AAA"], [2021, 2022, 2023], "consolidated",
    )
    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert {("net_revenue", year) for year in range(2020, 2024)} <= periods


def test_next_year_projection_adds_target_evidence_after_candidate_interval():
    requirements = build_evidence_requirements(
        "Trong giai đoạn 2021 đến 2023, năm có hệ số thanh toán nhanh thấp "
        "nhất; tỷ lệ CFO trên nợ ngắn hạn ở năm ngay sau năm đó là bao nhiêu?",
        ["AAA"], [2021, 2022, 2023], "consolidated",
    )
    periods = {
        (requirement.metric_key, requirement.year)
        for requirement in requirements
    }

    assert ("cfo", 2024) in periods
    assert ("current_liabilities", 2024) in periods


def test_quota_prefers_best_exact_table_for_each_requirement_when_budget_allows():
    route = SimpleNamespace(
        plan={"facts": [{}, {}]},
        evidence_requirements=[
            {"requirement_id": "AAA|2024|gross_profit"},
            {"requirement_id": "AAA|2024|net_revenue"},
        ],
    )
    candidates = [
        {
            "report_id": "AAA_2024", "table_pos": 1, "score": 10.0,
            "requirement_hits": [
                "AAA|2024|gross_profit", "AAA|2024|net_revenue"
            ],
            "requirement_scores": {
                "AAA|2024|gross_profit": 80.0,
                "AAA|2024|net_revenue": 80.0,
            },
        },
        {
            "report_id": "AAA_2024", "table_pos": 2, "score": 100.0,
            "requirement_hits": ["AAA|2024|gross_profit"],
            "requirement_scores": {"AAA|2024|gross_profit": 95.0},
        },
        {
            "report_id": "AAA_2024", "table_pos": 3, "score": 90.0,
            "requirement_hits": ["AAA|2024|net_revenue"],
            "requirement_scores": {"AAA|2024|net_revenue": 95.0},
        },
    ]

    selected = _apply_quota(candidates, route, depth=2)

    assert [candidate["table_pos"] for candidate in selected] == [2, 3]
    assert evidence_coverage(route.evidence_requirements, selected)["complete"]


def test_requirement_match_rejects_lexically_similar_wrong_metric():
    candidate = SimpleNamespace(label="Phải trả ngắn hạn khác", code="319")

    assert not candidate_matches_metric(candidate, "current_assets")


def test_requirement_preserves_and_enforces_named_counterparty():
    requirements = build_evidence_requirements(
        "Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai của HNG",
        ["HNG"], [2017], "separate",
        ["vay dai han voi hoang anh gia lai"],
    )
    requirement = requirements[0].to_dict()
    parent = SimpleNamespace(label="Vay dài hạn", code="")
    child = SimpleNamespace(
        label="Công ty Cổ phần HoàngAnh Gia Lai Công ty mẹ Vay dài hạn", code="")

    assert requirement["metric_variants"][0] == "vay dai han voi hoang anh gia lai"
    assert not candidate_matches_requirement(parent, requirement)
    assert candidate_matches_requirement(child, requirement)


def test_requirement_cache_fingerprint_only_tracks_named_detail():
    base = {"metric_key": "borrowings_long_term"}
    hag = {**base, "metric_variants": ["vay dai han voi hoang anh gia lai"]}
    alpha = {**base, "metric_variants": ["vay dai han voi cong ty alpha"]}
    generic_a = {**base, "metric_variants": ["vay dai han cuoi nam 2024"]}
    generic_b = {**base, "metric_variants": ["cac khoan vay dai han"]}

    assert requirement_specificity_key(hag) != requirement_specificity_key(alpha)
    assert requirement_specificity_key(generic_a) == ()
    assert requirement_specificity_key(generic_b) == ()


def test_generic_requirement_linking_drops_noisy_source_phrase():
    requirement = {
        "metric_key": "voting_rate",
        "metric_variants": [
            "tong so phai tra sau 12 thang cua cong ty co ty le quyen bieu quyet",
            "ty le quyen bieu quyet",
        ],
    }

    variants = requirement_linking_variants(requirement)

    assert variants[0] == "ty le quyen bieu quyet"
    assert all("phai tra sau 12 thang" not in value for value in variants)


def test_named_detail_requirement_linking_keeps_source_phrase():
    requirement = {
        "metric_key": "borrowings_long_term",
        "metric_variants": [
            "vay dai han voi hoang anh gia lai",
            "vay dai han",
        ],
    }

    assert requirement_linking_variants(requirement)[0] == (
        "vay dai han voi hoang anh gia lai"
    )


def test_v7_period_formula_requirements_include_prior_year_operands():
    accrual = build_evidence_requirements(
        "Tỷ số dồn tích năm 2021 là bao nhiêu phần trăm?",
        ["AAA"], [2021], "consolidated",
    )
    leverage = build_evidence_requirements(
        "Đòn bẩy kinh doanh năm 2024 cao nhất là bao nhiêu lần?",
        ["AAA"], [2024], "consolidated",
    )

    assert {(item.metric_key, item.year) for item in accrual} == {
        ("net_profit", 2021), ("cfo", 2021),
        ("total_assets", 2020), ("total_assets", 2021),
    }
    assert {(item.metric_key, item.year) for item in leverage} == {
        ("operating_profit", 2023), ("operating_profit", 2024),
        ("net_revenue", 2023), ("net_revenue", 2024),
    }


def test_v9_note_ratio_requires_parent_and_child_for_every_period():
    requirements = build_evidence_requirements(
        "Tỷ trọng trung bình dư nợ cho vay ngành bất động sản của OCB tại "
        "các mốc 31/12/2020 và 31/12/2024 là bao nhiêu phần trăm?",
        ["OCB"], [2020, 2024], "separate",
    )

    assert {(item.metric_key, item.year) for item in requirements} == {
        ("real_estate_customer_loans", 2020),
        ("customer_loans", 2020),
        ("real_estate_customer_loans", 2024),
        ("customer_loans", 2024),
    }


def test_v9_three_operand_note_ratio_retrieves_all_denominator_parts():
    requirements = build_evidence_requirements(
        "Tổng nợ vay gấp tổng tiền mặt và tiền gửi ngân hàng của HNG cuối "
        "năm 2020 bao nhiêu lần?",
        ["HNG"], [2020], "separate",
    )

    assert {item.metric_key for item in requirements} == {
        "borrowings_total", "cash_on_hand", "bank_deposits",
    }


def test_v9_implicit_deposit_interest_share_retrieves_total_expense():
    requirements = build_evidence_requirements(
        "Tỷ trọng chi phí lãi tiền gửi của HDB năm 2025 là bao nhiêu phần trăm?",
        ["HDB"], [2025], "consolidated",
    )

    assert {item.metric_key for item in requirements} == {
        "deposit_interest_expense", "bank_interest_expense",
    }


def test_lease_schedule_requirement_selects_requested_direction():
    requirement = {
        "metric_key": "operating_lease_commitments",
        "metric_label": "cam ket cho thue hoat dong",
        "metric_variants": [
            "tien thue phai thu trong tuong lai",
            "cam ket cho thue hoat dong",
        ],
    }
    grid = str([
        ["", "Số cuối năm"],
        ["Đến 1 năm", "10.000.000"],
        ["Trên 1 năm đến 5 năm", "20.000.000"],
        ["TỔNG CỘNG", "30.000.000"],
    ])

    assert _operating_lease_schedule_score({
        "context": "Tập đoàn là bên cho thuê văn phòng",
        "grid_json": grid,
    }, requirement) == 96.0
    assert _operating_lease_schedule_score({
        "context": "Tập đoàn là bên đi thuê, tiền thuê phải trả",
        "grid_json": grid,
    }, requirement) is None
