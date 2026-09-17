from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from kingpro.retrieval.table_reranker import (
    label_column,
    label_match_score,
    labels_from_csv,
    rerank_tables,
)
from kingpro.product.service import ProductService, RefusalPolicy


class TableRerankerTests(unittest.TestCase):
    def test_label_column_skips_leading_financial_codes(self) -> None:
        frame = pd.DataFrame(
            {
                "Mã số": ["100", "110", "111", "112"],
                "TÀI SẢN": [
                    "A. TÀI SẢN NGẮN HẠN",
                    "I. Tiền và các khoản tương đương tiền",
                    "1. Tiền",
                    "2. Các khoản tương đương tiền",
                ],
                "Số cuối năm": ["10", "8", "2", "6"],
            }
        )
        self.assertEqual(label_column(frame), 1)

    def test_exact_financial_label_scores_above_unrelated_label(self) -> None:
        question = "Tiền và các khoản tương đương tiền của SAB cuối năm 2016 là bao nhiêu?"
        exact = label_match_score(question, ["I. Tiền và các khoản tương đương tiền"])
        unrelated = label_match_score(question, ["Phải trả người bán ngắn hạn"])
        self.assertGreater(exact, unrelated)
        self.assertGreater(exact, 0.8)

    def test_rerank_reads_actual_label_column_and_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.csv"
            distractor = root / "distractor.csv"
            pd.DataFrame(
                {
                    "0": ["100", "110"],
                    "1": ["TÀI SẢN NGẮN HẠN", "Tiền và các khoản tương đương tiền"],
                    "2": ["1", "2"],
                }
            ).to_csv(target, index=False, encoding="utf-8-sig")
            pd.DataFrame({"0": ["Phải trả người bán", "Nợ vay"]}).to_csv(
                distractor, index=False, encoding="utf-8-sig"
            )
            labels_from_csv.cache_clear()
            hits = [
                {"table_ref": "D|1", "score": 4.0},
                {"table_ref": "T|2", "score": 3.5},
            ]
            catalog = {
                "D|1": {"csv_path": distractor.name},
                "T|2": {"csv_path": target.name},
            }
            ranked = rerank_tables(
                "Tiền và các khoản tương đương tiền cuối năm là bao nhiêu?",
                hits,
                catalog,
                root,
                label_weight=2.0,
            )
            self.assertEqual([row["table_ref"] for row in ranked], ["T|2", "D|1"])
            self.assertGreater(ranked[0]["label_match_score"], ranked[1]["label_match_score"])

    def test_product_retrieval_uses_wide_pool_then_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = root / "build" / "tables"
            tables.mkdir(parents=True)
            pd.DataFrame({"0": ["Nợ vay", "Phải trả người bán"]}).to_csv(
                tables / "d.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(
                {"0": ["110"], "1": ["Tiền và các khoản tương đương tiền"]}
            ).to_csv(tables / "t.csv", index=False, encoding="utf-8-sig")
            catalog = {
                "SAB_2016|1": {
                    "table_ref": "SAB_2016|1", "report_id": "SAB_2016",
                    "ticker": "SAB", "year": "2016", "scope": "công ty mẹ",
                    "csv_path": "d.csv", "search_text": "Mã số",
                },
                "SAB_2016|2": {
                    "table_ref": "SAB_2016|2", "report_id": "SAB_2016",
                    "ticker": "SAB", "year": "2016", "scope": "công ty mẹ",
                    "csv_path": "t.csv", "search_text": "Mã số",
                },
            }
            service = ProductService(
                root=root,
                llm_fn=lambda _system, _user: "result = 0",
                policy=RefusalPolicy(base_tables=1, max_tables=1, table_rerank_pool=40),
            )
            service._catalog = catalog
            facets = {
                "tickers": ["SAB"], "years": ["2016"],
                "scope": "công ty mẹ", "analytic": False,
            }
            hits = [
                {"table_ref": "SAB_2016|1", "score": 4.0},
                {"table_ref": "SAB_2016|2", "score": 3.5},
            ]
            with patch(
                "kingpro.product.service.retrieve_decomposed",
                return_value=[{"table_ref": "SAB_2016|1", "ticker": "SAB", "year": "2016"}],
            ), patch(
                "kingpro.product.service.tables_in_reports", return_value=hits
            ) as table_search:
                _facets, _docs, ranked, _checks = service._retrieve(
                    "Tiền và các khoản tương đương tiền của SAB cuối năm 2016?", facets
                )
            self.assertEqual(table_search.call_args.kwargs["n"], 40)
            self.assertEqual([row["table_ref"] for row in ranked], ["SAB_2016|2"])
            self.assertIn("Nhãn dòng đã xác minh", ranked[0]["search_text"])

    def test_rerank_uses_section_header_and_proven_blank_total_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distractor = root / "distractor.csv"
            target = root / "target.csv"
            pd.DataFrame({"0": ["USD", "EUR"]}).to_csv(
                distractor, index=False, encoding="utf-8-sig"
            )
            pd.DataFrame({"0": ["USD", "EUR", ""]}).to_csv(
                target, index=False, encoding="utf-8-sig"
            )
            labels_from_csv.cache_clear()
            hits = [
                {"table_ref": "D|1", "score": 4.0},
                {"table_ref": "T|2", "score": 4.0},
            ]
            catalog = {
                "D|1": {"csv_path": distractor.name},
                "T|2": {
                    "csv_path": target.name,
                    "section_title": "(b) Ngoại tệ các loại",
                    "header_text": "31/12/2022 > Tương đương VND",
                    "inferred_row_text": (
                        "Tổng cộng > 31/12/2022 > Tương đương VND"
                    ),
                },
            }

            ranked = rerank_tables(
                "Tổng số dư tương đương đồng ngoại tệ cuối năm 2022 là bao nhiêu?",
                hits,
                catalog,
                root,
                label_weight=2.0,
            )

            self.assertEqual(ranked[0]["table_ref"], "T|2")
            self.assertGreater(
                ranked[0]["label_match_score"], ranked[1]["label_match_score"]
            )
            self.assertIn("Ngoại tệ các loại", ranked[0]["retrieval_label_text"])


if __name__ == "__main__":
    unittest.main()
