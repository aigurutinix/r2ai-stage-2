from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.answering.sandbox import run_pandas_code
from kingpro.financial.statement_cube import FinancialCube, StatementCell
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler
from kingpro.product.service import ProductService, RefusalPolicy


class DeterministicCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        table_dir = self.root / "build" / "tables" / "VNM_financial_statements_2023_consolidated"
        table_dir.mkdir(parents=True)
        (self.root / "data").mkdir()
        (self.root / "data" / "code_stock.csv").write_text(
            "Mã CK,Tên công ty\nVNM,CTCP Sữa Việt Nam\n", encoding="utf-8-sig"
        )
        csv_path = table_dir / "table_1_line100.csv"
        csv_path.write_text(
            "Chỉ tiêu,2023\n"
            "Tài sản ngắn hạn,200\n"
            "Nợ ngắn hạn,100\n"
            "Tổng tài sản,500\n"
            "Doanh thu thuần,1000\n"
            "Giá vốn hàng bán,400\n"
            "Doanh thu hoạt động tài chính,10\n"
            "Chi phí tài chính,50\n"
            "Chi phí bán hàng,20\n"
            "Chi phí quản lý doanh nghiệp,80\n"
            "Lợi nhuận trước thuế,200\n"
            "Lợi nhuận sau thuế,100\n"
            "Vay ngắn hạn,25\n"
            "Vốn chủ sở hữu,250\n"
            "Chi phí thuế TNDN hiện hành,15\n"
            "Thu nhập khác,12\n"
            "Chi phí khác,20\n"
            "Đầu tư tài chính ngắn hạn,30\n"
            "Tiền và tương đương tiền,60\n"
            "Vốn góp của chủ sở hữu,70\n",
            encoding="utf-8-sig",
        )
        self.table_ref = "VNM_financial_statements_2023_consolidated|100"
        catalog = {
            "table_ref": self.table_ref,
            "report_id": "VNM_financial_statements_2023_consolidated",
            "ticker": "VNM",
            "year": "2023",
            "scope": "hợp nhất",
            "line": 100,
            "page": 12,
            "csv_path": "VNM_financial_statements_2023_consolidated/table_1_line100.csv",
            "search_text": "Chỉ tiêu: Tài sản ngắn hạn; Nợ ngắn hạn; Tổng tài sản",
        }
        (self.root / "build" / "catalog.jsonl").write_text(
            json.dumps(catalog, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cube = FinancialCube()
        for metric_key, label, row_idx, raw in (
            ("cdkt:100", "Tài sản ngắn hạn", 0, "200"),
            ("cdkt:310", "Nợ ngắn hạn", 1, "100"),
            ("cdkt:270", "Tổng tài sản", 2, "500"),
            ("kqkd:10", "Doanh thu thuần", 3, "1000"),
            ("kqkd:11", "Giá vốn hàng bán", 4, "400"),
            ("kqkd:21", "Doanh thu hoạt động tài chính", 5, "10"),
            ("kqkd:22", "Chi phí tài chính", 6, "50"),
            ("kqkd:25", "Chi phí bán hàng", 7, "20"),
            ("kqkd:26", "Chi phí quản lý doanh nghiệp", 8, "80"),
            ("kqkd:50", "Lợi nhuận trước thuế", 9, "200"),
            ("kqkd:60", "Lợi nhuận sau thuế", 10, "100"),
            ("cdkt:320", "Vay ngắn hạn", 11, "25"),
            ("cdkt:400", "Vốn chủ sở hữu", 12, "250"),
            ("kqkd:51", "Chi phí thuế TNDN hiện hành", 13, "15"),
            ("kqkd:31", "Thu nhập khác", 14, "12"),
            ("kqkd:32", "Chi phí khác", 15, "20"),
            ("cdkt:120", "Đầu tư tài chính ngắn hạn", 16, "30"),
            ("cdkt:110", "Tiền và tương đương tiền", 17, "60"),
            ("cdkt:411", "Vốn góp của chủ sở hữu", 18, "70"),
        ):
            cube.put_first(
                StatementCell(
                    ticker="VNM",
                    year="2023",
                    scope="consolidated",
                    metric_key=metric_key,
                    ma_so=metric_key.split(":", 1)[1],
                    label=label,
                    value=float(raw) * 1_000_000,
                    raw=raw,
                    table_ref=self.table_ref,
                    csv_path=str(csv_path),
                    row_idx=row_idx,
                    col_idx=1,
                    scale=1_000_000,
                )
            )

        previous_dir = self.root / "build" / "tables" / "VNM_financial_statements_2022_consolidated"
        previous_dir.mkdir(parents=True)
        previous_csv = previous_dir / "table_1_line100.csv"
        previous_csv.write_text(
            "Chỉ tiêu,2022\n"
            "Vốn chủ sở hữu,150\n"
            "Doanh thu thuần,800\n"
            "Tài sản ngắn hạn,120\n"
            "Nợ ngắn hạn,80\n",
            encoding="utf-8-sig",
        )
        for metric_key, label, row_idx, raw in (
            ("cdkt:400", "Vốn chủ sở hữu", 0, "150"),
            ("kqkd:10", "Doanh thu thuần", 1, "800"),
            ("cdkt:100", "Tài sản ngắn hạn", 2, "120"),
            ("cdkt:310", "Nợ ngắn hạn", 3, "80"),
        ):
            cube.put_first(
                StatementCell(
                    ticker="VNM",
                    year="2022",
                    scope="consolidated",
                    metric_key=metric_key,
                    ma_so=metric_key.split(":", 1)[1],
                    label=label,
                    value=float(raw) * 1_000_000,
                    raw=raw,
                    table_ref="VNM_financial_statements_2022_consolidated|100",
                    csv_path=str(previous_csv),
                    row_idx=row_idx,
                    col_idx=1,
                    scale=1_000_000,
                )
            )
        cube.write_jsonl(self.root / "build" / "statement_cube.jsonl")
        self.compiler = DeterministicFinancialCompiler(self.root)
        self.facets = {
            "tickers": ["VNM"],
            "years": ["2023"],
            "scope": "hợp nhất",
            "analytic": False,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_raw_metric_compiles_from_source_cell_and_replays(self) -> None:
        compiled = self.compiler.compile(
            "Tổng tài sản của VNM năm 2023 là bao nhiêu triệu đồng?", self.facets
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertEqual(compiled.metric, "total_assets")
        self.assertEqual(compiled.answer, 500.0)
        self.assertNotIn("result = 500", compiled.pandas_query)
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
        self.assertTrue(replay["ok"], replay)
        self.assertEqual(replay["result"], 500.0)
        self.assertEqual(compiled.table_refs, [self.table_ref])
        self.assertEqual(
            compiled.submission_evidence(),
            [{"variable": "df1", "table_ref": self.table_ref}],
        )
        with self.assertRaisesRegex(ValueError, "cardinality mismatch"):
            replace(compiled, table_refs=[]).submission_evidence()

    def test_compiled_query_fails_closed_when_source_rows_go_stale(self) -> None:
        compiled = self.compiler.compile(
            "Tổng tài sản của VNM năm 2023 là bao nhiêu triệu đồng?",
            self.facets,
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        source = Path(next(iter(compiled.csv_paths.values())))
        lines = source.read_text(encoding="utf-8-sig").splitlines()
        # Swap two complete data rows after compilation. A positional-only
        # program would silently read another financial metric here.
        lines[1], lines[3] = lines[3], lines[1]
        source.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
        self.assertFalse(replay["ok"], replay)
        self.assertIsNone(replay["result"])

    def test_standard_cash_and_total_sources_aliases_compile_but_beginning_balance_is_blocked(self) -> None:
        cases = (
            ("Tiền và tương đương tiền cuối năm 2023 của VNM là bao nhiêu triệu đồng?", "cash", 60.0),
            ("Tổng nguồn vốn của VNM cuối năm 2023 là bao nhiêu triệu đồng?", "total_assets", 500.0),
            ("Lợi nhuận thuần sau thuế của VNM năm 2023 là bao nhiêu triệu đồng?", "npat", 100.0),
            ("Lợi nhuận kế toán sau thuế TNDN của VNM năm 2023 là bao nhiêu triệu đồng?", "npat", 100.0),
            ("Giá vốn bán hàng của VNM năm 2023 là bao nhiêu triệu đồng?", "cogs", 400.0),
            ("Số dư vay ngắn hạn của VNM cuối năm 2023 là bao nhiêu triệu đồng?", "short_term_borrowings", 25.0),
            ("Vốn cổ phần của VNM cuối năm 2023 là bao nhiêu triệu đồng?", "contributed_capital", 70.0),
            ("Vốn góp của chủ sở hữu của VNM cuối năm 2023 là bao nhiêu triệu đồng?", "contributed_capital", 70.0),
        )
        for question, metric, expected in cases:
            with self.subTest(metric=metric):
                compiled = self.compiler.compile(question, self.facets)
                self.assertIsNotNone(compiled)
                assert compiled is not None
                self.assertEqual(compiled.metric, metric)
                self.assertEqual(compiled.answer, expected)
        self.assertIsNone(
            self.compiler.compile(
                "Tổng nguồn vốn đầu tư xây dựng cơ bản của VNM vào đầu năm 2023?",
                self.facets,
            )
        )
        self.assertIsNone(
            self.compiler.compile(
                "Lợi nhuận sau thuế TNDN của VNM năm 2023 là bao nhiêu triệu đồng?",
                self.facets,
            )
        )
        self.assertIsNone(
            self.compiler.compile(
                "Vốn cổ phần của VNM chênh lệch bao nhiêu so với FPT năm 2023?",
                self.facets,
            )
        )

    def test_ratio_compiles_and_uses_both_source_cells(self) -> None:
        compiled = self.compiler.compile(
            "Hệ số thanh toán hiện hành của VNM năm 2023 là bao nhiêu lần?", self.facets
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertEqual(compiled.metric, "current_ratio")
        self.assertEqual(compiled.answer, 2.0)
        self.assertEqual(len(compiled.source_cells), 2)
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
        self.assertTrue(replay["ok"], replay)
        self.assertEqual(replay["result"], 2.0)

    def test_verified_single_year_ratio_grammar_replays(self) -> None:
        cases = (
            ("Loi nhuan sau thue tren tong tai san cuoi nam cua VNM nam 2023 la bao nhieu phan tram?", "npat_to_assets_pct", 20.0),
            ("Ty trong tai san ngan han trong tong nguon von cua VNM nam 2023 la bao nhieu phan tram?", "current_assets_to_assets_pct", 40.0),
            ("Ty suat loi nhuan truoc thue cua VNM nam 2023 la bao nhieu phan tram?", "pbt_margin_pct", 20.0),
            ("Ty le chi phi quan ly doanh nghiep tren doanh thu thuan cua VNM nam 2023?", "admin_expense_to_revenue_pct", 8.0),
            ("Ty le chi phi ban hang tren doanh thu thuan cua VNM nam 2023?", "selling_expense_to_revenue_pct", 2.0),
            ("Ty le vay ngan han tren von chu so huu cua VNM nam 2023?", "short_term_borrowings_to_equity_pct", 10.0),
            ("Ty le no ngan han tren von chu so huu cua VNM nam 2023?", "current_liabilities_to_equity_pct", 40.0),
            ("Ty trong chi phi tai chinh tren doanh thu thuan cua VNM nam 2023?", "finance_expense_to_revenue_pct", 5.0),
            ("Ti trong chi phi tai chinh tren doanh thu thuan cua VNM nam 2023?", "finance_expense_to_revenue_pct", 5.0),
            ("Ty le doanh thu hoat dong tai chinh tren chi phi tai chinh cua VNM nam 2023?", "finance_revenue_to_expense_pct", 20.0),
            ("Ty le gia von hang ban tren doanh thu thuan cua VNM nam 2023?", "cogs_to_revenue_pct", 40.0),
            ("Ty suat loi nhuan rong cua VNM nam 2023?", "net_margin_pct", 10.0),
            ("Tinh ty le % chi phi ban hang va quan ly doanh nghiep tren doanh thu thuan cua VNM nam 2023?", "sga_intensity_pct", 10.0),
            ("Ty so no ngan han tren von chu so huu cua VNM nam 2023?", "current_liabilities_to_equity", 0.4),
            ("Ty le dau tu ngan han tren tien va tuong duong tien cua VNM nam 2023?", "short_term_investments_to_cash_pct", 50.0),
        )
        for question, metric, expected in cases:
            with self.subTest(metric=metric):
                compiled = self.compiler.compile(question, self.facets)
                self.assertIsNotNone(compiled)
                assert compiled is not None
                self.assertEqual(compiled.metric, metric)
                self.assertAlmostEqual(compiled.answer, expected)
                replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
                self.assertTrue(replay["ok"], replay)
                self.assertAlmostEqual(replay["result"], expected)

    def test_net_finance_result_and_current_tax_replay_from_statement_cells(self) -> None:
        cases = (
            (
                "Lợi nhuận thuần từ hoạt động tài chính của VNM năm 2023 là bao nhiêu triệu đồng?",
                "net_finance_result",
                -40.0,
            ),
            (
                "Chi phí thuế TNDN hiện hành của VNM năm 2023 là bao nhiêu triệu đồng?",
                "current_tax_expense",
                15.0,
            ),
        )
        for question, metric, expected in cases:
            with self.subTest(metric=metric):
                compiled = self.compiler.compile(question, self.facets)
                self.assertIsNotNone(compiled)
                assert compiled is not None
                self.assertEqual(compiled.metric, metric)
                self.assertAlmostEqual(compiled.answer, expected)
                self.assertNotIn(f"result = {expected:g}", compiled.pandas_query)
                replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
                self.assertTrue(replay["ok"], replay)
                self.assertAlmostEqual(replay["result"], expected)

    def test_net_other_result_is_bound_but_raw_other_income_stays_outside_grammar(self) -> None:
        compiled = self.compiler.compile(
            "Thu nhập khác thuần của VNM năm 2023 là bao nhiêu triệu đồng?",
            self.facets,
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertEqual(compiled.metric, "net_other_result")
        self.assertEqual(compiled.answer, -8.0)
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
        self.assertTrue(replay["ok"], replay)
        self.assertEqual(replay["result"], -8.0)
        self.assertIsNone(
            self.compiler.compile(
                "Thu nhập khác của VNM năm 2023 là bao nhiêu triệu đồng?",
                self.facets,
            )
        )

    def test_equity_turnover_uses_average_equity(self) -> None:
        compiled = self.compiler.compile(
            "Vong quay von chu so huu cua VNM nam 2023 la bao nhieu lan?", self.facets
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertEqual(compiled.metric, "equity_turnover_avg")
        self.assertEqual(compiled.answer, 5.0)
        self.assertEqual(len(compiled.source_cells), 3)
        self.assertEqual(
            compiled.submission_evidence(),
            [
                {"variable": "df1", "table_ref": self.table_ref},
                {
                    "variable": "df2",
                    "table_ref": "VNM_financial_statements_2022_consolidated|100",
                },
            ],
        )
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
        self.assertTrue(replay["ok"], replay)
        self.assertEqual(replay["result"], 5.0)

    def test_ambiguous_cash_equivalent_share_stays_outside_grammar(self) -> None:
        self.assertIsNone(
            self.compiler.compile(
                "Ty trong cac khoan tuong duong tien tren tong tai san cua VNM nam 2023?",
                self.facets,
            )
        )

    def test_ambiguous_or_aggregate_request_is_not_compiled(self) -> None:
        broad = {**self.facets, "tickers": ["VNM", "FPT"]}
        self.assertIsNone(
            self.compiler.compile("Tổng tài sản VNM và FPT năm 2023?", broad)
        )

    def test_extreme_panel_compiles_without_answer_literals(self) -> None:
        facets = {
            "tickers": ["VNM"],
            "years": ["2022", "2023"],
            "scope": "hợp nhất",
            "analytic": True,
        }
        compiled = self.compiler.compile(
            "Trong giai đoạn 2022-2023, tại năm VNM có doanh thu thuần cao nhất, "
            "hệ số thanh toán hiện hành là bao nhiêu lần?",
            facets,
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertEqual(compiled.metric, "extreme:revenue->current_ratio")
        self.assertEqual(compiled.answer, 2.0)
        self.assertNotIn("result = 2", compiled.pandas_query)
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
        self.assertTrue(replay["ok"], replay)
        self.assertEqual(replay["result"], 2.0)
        self.assertEqual(
            {(cell["ticker"], cell["year"]) for cell in compiled.source_cells},
            {("VNM", "2022"), ("VNM", "2023")},
        )

    def test_extreme_panel_rejects_filter_or_scenario_language(self) -> None:
        facets = {
            "tickers": ["VNM"],
            "years": ["2022", "2023"],
            "scope": "hợp nhất",
            "analytic": True,
        }
        self.assertIsNone(
            self.compiler.compile(
                "Trong các năm có doanh thu thuần dương, tại năm doanh thu thuần "
                "cao nhất, hệ số thanh toán hiện hành là bao nhiêu lần?",
                facets,
            )
        )

    def test_extreme_panel_parser_accepts_safe_metric_orderings(self) -> None:
        self.assertEqual(
            self.compiler._extreme_panel_metrics(
                "he so thanh toan nhanh cua doanh nghiep co ty le cfo tren "
                "loi nhuan sau thue cao nhat la bao nhieu lan"
            ),
            ("cfo_to_npat", "quick_ratio", True),
        )
        self.assertEqual(
            self.compiler._extreme_panel_metrics(
                "vong quay tong tai san tinh theo tong tai san binh quan cua "
                "doanh nghiep co ty trong tai san dai han tren tong tai san cao nhat"
            ),
            ("long_term_assets_share_pct", "asset_turnover_avg", True),
        )
        self.assertIsNone(
            self.compiler.compile(
                "Tổng tài sản trung bình của VNM năm 2023 là bao nhiêu?", self.facets
            )
        )

    def test_service_answers_before_unattested_model_gate(self) -> None:
        service = ProductService(
            root=self.root,
            llm_fn=lambda _system, _user: self.fail("model must not run"),
            policy=RefusalPolicy(vote_count=1),
        )
        service.enforce_model_policy = True
        service._retrieve = lambda *_args, **_kwargs: self.fail("BM25 must not run")
        with patch.dict(
            os.environ,
            {
                "KINGPRO_LLM_BASE_URL": "https://api.runpod.ai/v2/demo/openai/v1",
                "KINGPRO_LLM_MODEL": "Qwen/Qwen2.5-Coder-14B-Instruct",
                "KINGPRO_LLM_ATTESTED": "false",
            },
        ):
            response = service.ask(
                "Tổng tài sản của VNM năm 2023 là bao nhiêu triệu đồng?"
            )
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["answer"], 500.0)
        self.assertEqual(response["verification"]["mode"], "deterministic_compiler")
        self.assertTrue(response["verification"]["citation_bound"])
        self.assertTrue(response["verification"]["replay_match"])


if __name__ == "__main__":
    unittest.main()
