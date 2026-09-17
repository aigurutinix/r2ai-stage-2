import io

import pandas as pd

from vifinqa.codegen.exact_growth import try_exact_growth_answer


def _table(var, year, value, *, label="Doanh thu thuần", code="10"):
    rows = [{
        "row": 3, "label": label, "code": code, "col": 3,
        "col_name": f"Năm {year}", "value": value, "unit_scale": 1e9,
    }]
    return {
        "var": var,
        "report_id": f"AAA_financial_statements_{year}_consolidated",
        "report_year": year, "table_pos": 5,
        "context": "Báo cáo kết quả hoạt động kinh doanh",
        "grid_json": "[]", "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def _requirement(year, metric_key="net_revenue", metric_label="doanh thu thuan"):
    return {
        "requirement_id": f"AAA|{year}|{metric_key}",
        "ticker": "AAA", "year": year, "doc_type": "consolidated",
        "metric_key": metric_key, "metric_label": metric_label,
        "metric_variants": [metric_label], "statement": "income_statement",
    }


def _route(*years, output_type="percent"):
    return {
        "question": "Tăng trưởng doanh thu thuần của AAA là bao nhiêu phần trăm?",
        "tickers": ["AAA"], "years": list(years),
        "doc_type": "consolidated", "unit_scale": 1.0,
        "output_type": output_type, "plan": {"op": "growth_pct"},
        "evidence_requirements": [_requirement(year) for year in years],
    }


def test_exact_growth_uses_earliest_and_latest_period_and_replays():
    tables = [
        _table("df1", 2021, 100.0),
        _table("df2", 2022, 120.0),
        _table("df3", 2023, 150.0),
    ]

    answer = try_exact_growth_answer(_route(2021, 2022, 2023), tables)

    assert answer.ok, answer.detail
    assert answer.answer == 50.0
    frames = {
        table["var"]: pd.read_csv(io.StringIO(table["csv_text"]))
        for table in tables
    }
    assert eval(answer.pandas_query, frames) == 50.0
    assert answer.tier == "vas_growth_current"


def test_exact_growth_normalizes_accounting_expense_signs():
    route = _route(2022, 2023)
    route["question"] = "Chi phí bán hàng tăng trưởng bao nhiêu phần trăm?"
    route["evidence_requirements"] = [
        _requirement(year, "selling_expense", "chi phi ban hang")
        for year in (2022, 2023)
    ]
    tables = [
        _table("df1", 2022, -100.0, label="Chi phí bán hàng", code="25"),
        _table("df2", 2023, -150.0, label="Chi phí bán hàng", code="25"),
    ]

    answer = try_exact_growth_answer(route, tables)

    assert answer.ok, answer.detail
    assert answer.answer == 50.0


def test_exact_growth_returns_positive_decrease_magnitude():
    route = _route(2017, 2018)
    route["question"] = "Chi phí lãi vay giảm bao nhiêu % từ 2017 đến 2018?"
    route["evidence_requirements"] = [
        _requirement(year, "interest_expense", "chi phi lai vay")
        for year in (2017, 2018)
    ]
    tables = [
        _table("df1", 2017, -120.0, label="Chi phí lãi vay", code="23"),
        _table("df2", 2018, -90.0, label="Chi phí lãi vay", code="23"),
    ]

    answer = try_exact_growth_answer(route, tables)

    assert answer.ok, answer.detail
    assert answer.answer == 25.0
    frames = {
        table["var"]: pd.read_csv(io.StringIO(table["csv_text"]))
        for table in tables
    }
    assert eval(answer.pandas_query, frames) == 25.0


def test_exact_growth_refuses_zero_base_and_detail_parent_mismatch():
    zero = try_exact_growth_answer(
        _route(2022, 2023),
        [_table("df1", 2022, 0.0), _table("df2", 2023, 50.0)],
    )
    detail_route = _route(2022, 2023)
    detail_route["question"] = (
        "Tăng trưởng doanh thu thuần từ hoạt động xây dựng là bao nhiêu phần trăm?"
    )
    detail = try_exact_growth_answer(
        detail_route,
        [_table("df1", 2022, 100.0), _table("df2", 2023, 120.0)],
    )

    assert not zero.ok
    assert "base is zero" in zero.detail
    assert not detail.ok
    assert "misses detail=hoat dong xay dung" in detail.detail


def test_exact_growth_refuses_mixed_metrics_or_entities():
    mixed_metric = _route(2022, 2023)
    mixed_metric["evidence_requirements"][1]["metric_key"] = "gross_profit"
    mixed_entity = _route(2022, 2023)
    mixed_entity["evidence_requirements"][1]["ticker"] = "BBB"

    assert not try_exact_growth_answer(mixed_metric, []).ok
    assert not try_exact_growth_answer(mixed_entity, []).ok
