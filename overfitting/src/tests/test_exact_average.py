import io

import pandas as pd

from vifinqa.codegen.exact_average import try_exact_average_answer


def _table(var, report_id, year, value):
    rows = [{
        "row": 3, "label": "Doanh thu thuần", "code": "10", "col": 3,
        "col_name": f"Năm {year}", "value": value, "unit_scale": 1e9,
    }]
    return {
        "var": var, "report_id": report_id, "report_year": year,
        "table_pos": 5, "context": "Báo cáo kết quả hoạt động kinh doanh",
        "grid_json": "[]", "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def _requirement(year):
    return {
        "requirement_id": f"AAA|{year}|net_revenue",
        "ticker": "AAA", "year": year, "doc_type": "consolidated",
        "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
        "metric_variants": ["doanh thu thuan"],
        "statement": "income_statement",
    }


def _route(*years, output_type="number"):
    return {
        "question": "Doanh thu thuần trung bình của AAA là bao nhiêu tỷ đồng?",
        "tickers": ["AAA"], "years": list(years),
        "doc_type": "consolidated", "unit_scale": 1e9,
        "output_type": output_type, "plan": {"op": "average"},
        "evidence_requirements": [_requirement(year) for year in years],
    }


def test_exact_average_resolves_one_metric_across_years_and_replays():
    tables = [
        _table("df1", "AAA_financial_statements_2022_consolidated", 2022, 100.0),
        _table("df2", "AAA_financial_statements_2023_consolidated", 2023, 160.0),
        _table("df3", "AAA_financial_statements_2024_consolidated", 2024, 220.0),
    ]

    answer = try_exact_average_answer(_route(2022, 2023, 2024), tables)

    assert answer.ok, answer.detail
    assert answer.answer == 160.0
    frames = {
        table["var"]: pd.read_csv(io.StringIO(table["csv_text"]))
        for table in tables
    }
    assert eval(answer.pandas_query, frames) == 160.0
    assert answer.tier == "vas_average_current"


def test_exact_average_refuses_derived_percent_output():
    answer = try_exact_average_answer(
        _route(2023, 2024, output_type="percent"), [])

    assert not answer.ok
    assert "unsupported average output=percent" in answer.detail


def test_exact_average_refuses_mixed_metrics():
    route = _route(2023, 2024)
    route["evidence_requirements"][1]["metric_key"] = "gross_profit"

    answer = try_exact_average_answer(route, [])

    assert not answer.ok
    assert "average metrics=" in answer.detail
