from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from audit_positional_semantics import (  # noqa: E402
    _evaluate_dataframe_aliases,
    _resolve_derived_frame,
    _safe_dataframe_expression,
    positional_reads,
)
from audit_direct_units import (  # noqa: E402
    direct_reads,
    requested_currency_unit,
    source_unit,
)


class PositionalSemanticAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "Chỉ tiêu": ["Không liên quan", "Doanh thu mục tiêu"],
                "2023": ["1", "250"],
            }
        )

    def test_filtered_alias_is_resolved_and_evaluated(self) -> None:
        query = """
filtered_df = df1[df1.iloc[:, 0].str.contains('mục tiêu', case=False, na=False, regex=False)]
result = filtered_df.iloc[0, 1]
"""
        self.assertEqual(positional_reads(query), [("filtered_df", "df1", 0, 1)])

        environment = _evaluate_dataframe_aliases(query, {"df1": self.frame})
        filtered = _resolve_derived_frame("filtered_df", environment)

        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertEqual(filtered.iloc[0, 0], "Doanh thu mục tiêu")
        self.assertEqual(filtered.iloc[0, 1], "250")

    def test_inline_filter_is_inspected_against_filtered_rows(self) -> None:
        expression = "df1[df1.iloc[:, 0].str.contains('mục tiêu', case=False, na=False, regex=False)]"
        query = f"result = {expression}.iloc[0, 1]"

        self.assertEqual(positional_reads(query), [(expression, "df1", 0, 1)])
        resolved = _resolve_derived_frame(expression, {"df1": self.frame})

        self.assertIsInstance(resolved, pd.DataFrame)
        self.assertEqual(resolved.iloc[0, 0], "Doanh thu mục tiêu")

    def test_private_attributes_and_non_pandas_calls_are_rejected(self) -> None:
        private = __import__("ast").parse("df1.__class__", mode="eval").body
        import_call = __import__("ast").parse("__import__('os')", mode="eval").body

        self.assertFalse(_safe_dataframe_expression(private, {"df1"}))
        self.assertFalse(_safe_dataframe_expression(import_call, {"df1"}))
        self.assertIsNone(_resolve_derived_frame("df1.__class__", {"df1": self.frame}))


class DirectUnitAuditTests(unittest.TestCase):
    def test_explicit_units_are_detected_but_bare_vnd_is_ambiguous(self) -> None:
        self.assertEqual(requested_currency_unit("bao nhiêu trăm tỷ đồng?"), 100_000_000_000)
        self.assertEqual(
            requested_currency_unit("vượt 10.000 tỷ đồng thì giá trị là bao nhiêu triệu đồng?"),
            1_000_000,
        )
        self.assertEqual(source_unit("Số cuối năm - Triệu VND"), 1_000_000)
        self.assertIsNone(source_unit("Số cuối năm - VND"))

    def test_direct_filter_read_is_extracted_from_ast(self) -> None:
        query = """
_r = df2[df2['0'].str.contains('Doanh thu', case=False, na=False, regex=False)]
_v = _r['3'].values[0]
result = float(_v)
"""
        reads = direct_reads(query)

        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].dataframe, "df2")
        self.assertEqual(reads[0].pattern, "Doanh thu")
        self.assertFalse(reads[0].regex)
        self.assertEqual(reads[0].column, "3")


if __name__ == "__main__":
    unittest.main()
