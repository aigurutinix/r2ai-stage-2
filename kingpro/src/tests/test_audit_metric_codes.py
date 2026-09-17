from scripts.audit_metric_codes import (
    DIRECT_CODE_FILTER,
    RULES,
    evidence_csv_to_table_ref,
    fold,
    infer_legacy_query_rows,
)


def test_evidence_csv_to_table_ref_uses_final_numeric_suffix() -> None:
    assert evidence_csv_to_table_ref(
        "data/OCB_financial_statements_2022_separate_2_1430.csv"
    ) == "OCB_financial_statements_2022_separate_2|1430"
    assert evidence_csv_to_table_ref("data/q325_source_cells.csv") is None


def test_direct_code_filter_parses_legacy_dataframe_program() -> None:
    query = "_r = df5[df5['0'].astype(str).str.strip() == '60']"
    match = DIRECT_CODE_FILTER.search(query)
    assert match is not None
    assert match.group("var") == "df5"
    assert match.group("code") == "60"


def test_legacy_inference_disambiguates_cashflow_code_60_from_npat() -> None:
    submission = {
        325: {
            "question": "Lợi nhuận sau thuế của công ty mẹ TTF năm 2025 là bao nhiêu?",
            "evidence": [
                {"variable": "df1", "csv_path": "data/a.csv"},
                {"variable": "df2", "csv_path": "data/b.csv"},
                {"variable": "df3", "csv_path": "data/c.csv"},
                {"variable": "df4", "csv_path": "data/d.csv"},
                {
                    "variable": "df5",
                    "csv_path": "data/TTF_financial_statements_2025_separate_281.csv",
                },
            ],
            "pandas_query": "_r = df5[df5['0'].astype(str).str.strip() == '60']",
        }
    }
    rows = infer_legacy_query_rows(submission, set())
    assert len(rows) == 1
    metrics = {source["metric"] for source in rows[0]["sources"]}
    assert "lctt:60" in metrics
    assert all(metric.startswith("lctt:") for metric in metrics)

    npat_rule = next(rule for rule in RULES if rule.name == "net_profit_after_tax")
    assert npat_rule.matches_question(fold(submission[325]["question"]))
    assert not npat_rule.accepts(metrics, fold(" ".join(metrics)))
