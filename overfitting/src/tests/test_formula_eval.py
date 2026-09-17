from vifinqa.validation.gen_formula_eval import (
    Cell,
    _is_nonclosing_date,
    build_cases,
)


EXPECTED_CLASSES = {
    "growth_pct",
    "gross_margin",
    "debt_equity",
    "margin_difference",
    "margin_change",
    "count_threshold",
    "count_years",
    "count_multi_condition",
    "ranking_ratio",
    "nested_ranking",
    "temporal_count",
    "average_margin_change",
}


def _fake_cell(ticker, year, metric, value, row):
    report_id = f"{ticker}_financial_statements_{year}_consolidated"
    return Cell(
        ticker=ticker,
        year=year,
        doc_type="consolidated",
        metric=metric,
        report_id=report_id,
        table_pos=1,
        line_no=10,
        row=row,
        col=1,
        label=metric.replace("_", " "),
        value_vnd=float(value) * 1e9,
    )


def _facts():
    facts = {}
    tickers = ("AAA", "BBB", "CCC", "DDD")
    metrics = (
        "net_revenue", "gross_profit", "operating_cash_flow",
        "current_assets", "inventory", "total_assets", "liabilities",
        "current_liabilities", "equity",
    )
    for ticker_index, ticker in enumerate(tickers):
        for year in (2022, 2023, 2024):
            period = year - 2022
            revenue = 100.0 + 10 * ticker_index + 20 * period
            gross = (35.0 - 8 * period if ticker == "AAA"
                     else 20.0 + 3 * period + ticker_index)
            inventory = (10.0 + 8 * period if ticker == "AAA"
                         else 20.0 - 2 * period + ticker_index)
            values = {
                "net_revenue": revenue,
                "gross_profit": gross,
                "operating_cash_flow": 10.0,
                "current_assets": 80.0 if ticker == "AAA" else 130.0,
                "inventory": inventory,
                "total_assets": 100.0 + 50 * ticker_index + 25 * period,
                "liabilities": 80.0 + 30 * ticker_index,
                "current_liabilities": 100.0,
                "equity": 100.0,
            }
            for row, metric in enumerate(metrics, 1):
                cell = _fake_cell(ticker, year, metric, values[metric], row)
                facts[(ticker, year, metric)] = cell
    return facts


def test_stock_gold_prefers_closing_date_in_same_year():
    assert _is_nonclosing_date("01/01/2024", 2024)
    assert _is_nonclosing_date("01/07/2024", 2024)
    assert not _is_nonclosing_date("31/12/2024", 2024)
    assert not _is_nonclosing_date("Năm 2024", 2024)


def test_formula_eval_builds_every_supported_class_with_replay_metadata():
    names = {ticker: f"Company {ticker}" for ticker in ("AAA", "BBB", "CCC", "DDD")}
    questions, gold = build_cases(_facts(), names, per_class=1, seed=7)

    assert {item["klass"] for item in gold.values()} == EXPECTED_CLASSES
    assert [question["id"] for question in questions] == list(
        range(1, len(questions) + 1))
    assert all(item["relevant_docs"] for item in gold.values())
    assert all(item["relevant_tables"] for item in gold.values())
    assert all(item["operands"] for item in gold.values())
