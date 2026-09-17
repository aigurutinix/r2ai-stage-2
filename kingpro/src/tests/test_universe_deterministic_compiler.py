from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.answering.sandbox import run_pandas_code
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler
from kingpro.product.service import ProductService
from kingpro.retrieval.bm25_index import extract_all_facets


REGISTRY = (
    ROOT
    / "sub_top123_candidate_v217_missing_panel_operand_batch3"
    / "submission.json"
)


class UniverseDeterministicCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.row = next(row for row in rows if int(row["id"]) == 464)
        cls.compiler = DeterministicFinancialCompiler(ROOT)

    def test_q464_compiles_and_replays_from_bound_source_cells(self) -> None:
        question = str(self.row["question"])
        compiled = self.compiler.compile(question, extract_all_facets(question))
        self.assertIsNotNone(compiled)
        assert compiled is not None

        self.assertEqual(
            compiled.metric,
            "universe:inventory_decline->cfo_margin_pct",
        )
        self.assertTrue(
            math.isclose(
                compiled.answer,
                float(self.row["answer"]),
                rel_tol=0,
                abs_tol=0.0050001,
            )
        )
        self.assertGreaterEqual(len(compiled.source_cells), 100)
        self.assertGreaterEqual(
            len({cell["ticker"] for cell in compiled.source_cells}),
            40,
        )
        self.assertNotIn(str(float(self.row["answer"])), compiled.pandas_query)

        replay = run_pandas_code(
            compiled.pandas_query,
            compiled.csv_paths,
            timeout=5,
        )
        self.assertTrue(replay["ok"], replay)
        self.assertTrue(
            math.isclose(
                float(replay["result"]),
                compiled.answer,
                rel_tol=0,
                abs_tol=1e-9,
            )
        )

    def test_threshold_is_parsed_instead_of_hard_coded(self) -> None:
        question = (
            "Trong các công ty có hàng tồn kho năm 2016 giảm ít nhất 11% "
            "so với năm 2015, tỷ số CFO trên doanh thu thuần năm 2016 "
            "cao nhất là bao nhiêu phần trăm?"
        )
        compiled = self.compiler.compile(question, extract_all_facets(question))
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertIn("<= -11", compiled.pandas_query)
        replay = run_pandas_code(
            compiled.pandas_query,
            compiled.csv_paths,
            timeout=5,
        )
        self.assertTrue(replay["ok"], replay)
        self.assertTrue(
            math.isclose(
                float(replay["result"]),
                compiled.answer,
                rel_tol=0,
                abs_tol=1e-9,
            )
        )

    def test_unknown_or_explicit_company_queries_still_fail_closed(self) -> None:
        explicit = (
            "Trong các công ty VNM có hàng tồn kho năm 2016 giảm ít nhất 10% "
            "so với năm 2015, tỷ lệ CFO/doanh thu thuần năm 2016 cao nhất "
            "là bao nhiêu phần trăm?"
        )
        self.assertIsNone(
            self.compiler._compile_universe_inventory_decline_cfo_margin(
                explicit,
                extract_all_facets(explicit),
            )
        )
        unknown = (
            "Trong các công ty có doanh thu năm 2016 giảm ít nhất 10% so với "
            "năm 2015, tỷ lệ lợi nhuận cao nhất là bao nhiêu phần trăm?"
        )
        self.assertIsNone(
            self.compiler.compile(unknown, extract_all_facets(unknown))
        )

    def test_product_service_routes_safe_broad_paraphrase_before_ticker_guard(self) -> None:
        question = (
            "Trong các công ty có hàng tồn kho năm 2016 giảm ít nhất 10% "
            "so với năm 2015, tỷ số CFO trên doanh thu thuần năm 2016 "
            "cao nhất là bao nhiêu phần trăm?"
        )
        with tempfile.TemporaryDirectory() as temporary:
            empty_registry = Path(temporary) / "submission.json"
            empty_registry.write_text("[]", encoding="utf-8")
            service = ProductService(
                root=ROOT,
                replay_submission=empty_registry,
                llm_fn=lambda _system, _user: "result = 0",
            )
            response = service.ask(question)

        self.assertEqual(response["status"], "answered", response)
        self.assertEqual(
            response["verification"]["mode"],
            "deterministic_compiler",
        )
        self.assertEqual(
            response["verification"]["compiler_metric"],
            "universe:inventory_decline->cfo_margin_pct",
        )
        self.assertTrue(
            math.isclose(
                float(response["answer"]),
                float(self.row["answer"]),
                rel_tol=0,
                abs_tol=0.0050001,
            )
        )


if __name__ == "__main__":
    unittest.main()
