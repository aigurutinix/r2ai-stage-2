from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_explicit_cohort_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_explicit_cohort_coverage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_program_calls_only_accept_literal_source_values() -> None:
    program = """
_source_value('AAA', 2024, 'kqkd:10')
_source_value(ticker, 2025, 'kqkd:60')
_source_value('BBB', '2023', 'cdkt:400')
"""
    assert MODULE._program_calls(program) == [
        ("AAA", "2024", "kqkd:10"),
        ("BBB", "2023", "cdkt:400"),
    ]


def test_year_asymmetry_requires_majority_coverage() -> None:
    tickers = ["AAA", "BBB", "CCC"]
    pairs = {
        ("AAA", "2024"),
        ("BBB", "2024"),
        ("AAA", "2023"),
    }
    assert MODULE._year_asymmetry(tickers, pairs) == [
        {
            "year": "2024",
            "present_tickers": ["AAA", "BBB"],
            "missing_tickers": ["CCC"],
            "coverage": 2,
            "cohort_size": 3,
        }
    ]


def test_report_pair_parser_ignores_non_financial_ids() -> None:
    assert MODULE._pair_from_report("AAA_financial_statements_2024_consolidated") == (
        "AAA",
        "2024",
    )
    assert MODULE._pair_from_report("factsheet_2024") is None
