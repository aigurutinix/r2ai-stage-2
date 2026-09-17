import io
import json
import unittest

import pandas as pd

from vifinqa.codegen.formula_solver import (
    build_compositional_ranking_plan,
    build_filter_aggregate_plan,
    build_lease_schedule_plan,
    build_matrix_note_plan,
    build_note_axis_plan,
    build_note_detail_plan,
    build_scenario_plan,
    build_select_project_plan,
    build_temporal_event_plan,
    requires_formula_solver,
    try_formula_answer,
)
from vifinqa.retrieval.serialize import tidy_rows_from_grid


def _table(var, report_id, rows, table_pos=1,
           context="Báo cáo kết quả hoạt động kinh doanh Bảng cân đối kế toán"):
    return {"var": var, "report_id": report_id, "table_pos": table_pos,
            "report_year": int(report_id.split("_")[-2]),
            "context": context,
            "csv_text": pd.DataFrame(rows).to_csv(index=False)}


def _matrix_table(var, report_id, grid, table_pos=1, unit=1.0,
                  context="Thuyết minh báo cáo tài chính"):
    rows = tidy_rows_from_grid(grid, unit)
    return {
        "var": var, "report_id": report_id, "table_pos": table_pos,
        "report_year": int(report_id.split("_")[-2]),
        "context": context,
        "grid_json": json.dumps(grid, ensure_ascii=False),
        "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def _row(label, value, code="", col=1, col_name="2024", unit=1e9, row=1):
    return {"row": row, "label": label, "code": code, "col": col,
            "col_name": col_name, "value": value, "unit_scale": unit}


def _route(question, tickers, output_type="number", op="lookup",
           metric="doanh thu thuan", years=None):
    return {
        "question": question,
        "tickers": tickers,
        "years": years or [2024],
        "doc_type": "consolidated",
        "metric_norm": metric,
        "metric_variants": [metric],
        "output_type": output_type,
        "plan": {"op": op, "facts": []},
    }


def _eval(query, tables):
    env = {"float": float, "round": round, "min": min, "max": max, "abs": abs}
    env.update({t["var"]: pd.read_csv(io.StringIO(t["csv_text"])) for t in tables})
    return eval(query, env, {})


class FormulaSolverTests(unittest.TestCase):
    def test_exact_derived_net_financial_result_reads_both_statement_rows(self):
        tables = [_table(
            "df1", "AAA_financial_statements_2024_separate", [
                _row("Doanh thu hoạt động tài chính", 300.0,
                     code="21", row=1),
                _row("Chi phí tài chính", 120.0,
                     code="22", row=2),
            ])]
        route = _route(
            "Lãi ròng từ hoạt động tài chính của công ty mẹ AAA năm 2024 là bao nhiêu tỷ đồng?",
            ["AAA"], metric="lai rong tu hoat dong tai chinh")
        route.update(
            doc_type="separate", unit_scale=1e9,
            metric_keys=["net_financial_result"])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 180.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 180.0)
        self.assertIn("df1", ca.pandas_query)
        self.assertIn("abs", ca.pandas_query)
        self.assertEqual(len(ca.resolved), 2)
        self.assertIn("formula_exact_derived net_financial_result", ca.detail)

    def test_exact_derived_net_financial_result_fails_closed_without_expense(self):
        tables = [_table(
            "df1", "AAA_financial_statements_2024_separate", [
                _row("Doanh thu hoạt động tài chính", 300.0,
                     code="21", row=1),
            ])]
        route = _route(
            "Lãi ròng từ hoạt động tài chính của công ty mẹ AAA năm 2024 là bao nhiêu tỷ đồng?",
            ["AAA"], metric="lai rong tu hoat dong tai chinh")
        route.update(
            doc_type="separate", unit_scale=1e9,
            metric_keys=["net_financial_result"])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)

    def test_exact_derived_rejects_implausible_ocr_component(self):
        tables = [_table(
            "df1", "AAA_financial_statements_2024_separate", [
                _row("Doanh thu hoạt động tài chính", 300e9,
                     code="21", row=1, unit=1.0),
                _row("Chi phí tài chính", 4.2e21,
                     code="22", row=2, unit=1.0),
            ])]
        route = _route(
            "Lãi ròng từ hoạt động tài chính của công ty mẹ AAA năm 2024 là bao nhiêu tỷ đồng?",
            ["AAA"], metric="lai rong tu hoat dong tai chinh")
        route.update(
            doc_type="separate", unit_scale=1e9,
            metric_keys=["net_financial_result"])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)

    def test_count_direct_money_threshold(self):
        q = "Có bao nhiêu công ty có cam kết thuê hoạt động lớn hơn 40 tỷ đồng năm 2024?"
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated",
                   [_row("Cam kết thuê hoạt động", 50.0)]),
            _table("df2", "BBB_financial_statements_2024_consolidated",
                   [_row("Cam kết thuê hoạt động", 40.0)]),
            _table("df3", "CCC_financial_statements_2024_consolidated",
                   [_row("Cam kết thuê hoạt động", 100.0)]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB", "CCC"], "count", "count",
                   "cam ket thue hoat dong"),
            tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_count_negative_investing_cash_flow_years_rejects_cfo_row(self):
        q = ("AAA có số năm lưu chuyển tiền ròng từ hoạt động đầu tư âm là "
             "bao nhiêu trong các năm 2022, 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2022_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", -50.0,
                     col_name="2022", row=1),
                _row("Lưu chuyển tiền thuần từ hoạt động đầu tư", 10.0,
                     col_name="2022", row=2),
            ]),
            _table("df2", "AAA_financial_statements_2023_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động đầu tư", -20.0,
                     col_name="2023", row=1),
            ]),
            _table("df3", "AAA_financial_statements_2024_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động đầu tư", -30.0,
                     col_name="2024", row=1),
            ]),
        ]
        route = _route(q, ["AAA"], "count", "count",
                       "luu chuyen tien rong tu hoat dong dau tu",
                       years=[2022, 2023, 2024])
        route["metric_keys"] = ["cfi"]

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)
        self.assertNotIn("hoạt động kinh doanh", ca.pandas_query)

    def test_count_corporate_income_tax_rejects_personal_income_tax(self):
        q = "Có bao nhiêu công ty có thuế thu nhập doanh nghiệp lớn hơn 50 tỷ đồng?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Thuế thu nhập cá nhân", 100.0),
        ])]
        ca = try_formula_answer(
            _route(q, ["AAA"], "count", "count", "thue thu nhap doanh nghiep"),
            tables)
        self.assertFalse(ca.ok)

    def test_count_rejects_implausible_double_unit_scaling(self):
        q = "Có bao nhiêu công ty có dòng tiền hoạt động lớn hơn 1 nghìn tỷ đồng?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Dòng tiền thuần từ hoạt động kinh doanh", 932_000_000_000.0,
                 unit=1e9),
        ])]
        ca = try_formula_answer(
            _route(q, ["AAA"], "count", "count",
                   "dong tien thuan tu hoat dong kinh doanh"), tables)
        self.assertFalse(ca.ok)

    def test_count_conditions_inside_top_revenue_population(self):
        q = ("Trong 2 doanh nghiệp có doanh thu thuần lớn nhất năm 2024, "
             "có bao nhiêu doanh nghiệp vừa có hệ số thanh toán nhanh lớn hơn 1 lần, "
             "vừa có hệ số nợ phải trả trên vốn chủ sở hữu nhỏ hơn 1,5 lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 300.0)], table_pos=1),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 100.0, row=1),
                _row("Hàng tồn kho", 20.0, row=2),
                _row("Nợ ngắn hạn", 50.0, row=3),
                _row("Nợ phải trả", 120.0, row=4),
                _row("Vốn chủ sở hữu", 100.0, row=5),
            ], table_pos=2),
            _table("df3", "BBB_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 200.0)], table_pos=1),
            _table("df4", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 70.0, row=1),
                _row("Hàng tồn kho", 30.0, row=2),
                _row("Nợ ngắn hạn", 50.0, row=3),
                _row("Nợ phải trả", 80.0, row=4),
                _row("Vốn chủ sở hữu", 100.0, row=5),
            ], table_pos=2),
            _table("df5", "CCC_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 100.0)], table_pos=1),
            _table("df6", "CCC_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 100.0, row=1),
                _row("Hàng tồn kho", 10.0, row=2),
                _row("Nợ ngắn hạn", 50.0, row=3),
                _row("Nợ phải trả", 20.0, row=4),
                _row("Vốn chủ sở hữu", 100.0, row=5),
            ], table_pos=2),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB", "CCC"], "count", "count"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)
        self.assertIn("df5", ca.pandas_query)  # revenue evidence for top-N filter

    def test_ranking_by_debt_equity_ratio(self):
        q = "Hệ số nợ phải trả trên vốn chủ sở hữu lớn nhất năm 2024 là bao nhiêu lần?"
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 120.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 80.0, row=1),
                _row("Vốn chủ sở hữu", 40.0, row=2),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB"], "ratio", "ranking"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_year_ranking_returns_argmax_year_and_reads_every_period(self):
        q = ("Doanh thu thuần của AAA cao nhất trong các năm 2022, 2023 và "
             "2024 là vào năm nào?")
        tables = [
            _table("df1", "AAA_financial_statements_2022_consolidated",
                   [_row("Doanh thu thuần", 100.0, code="10", col_name="2022")]),
            _table("df2", "AAA_financial_statements_2023_consolidated",
                   [_row("Doanh thu thuần", 180.0, code="10", col_name="2023")]),
            _table("df3", "AAA_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 120.0, code="10", col_name="2024")]),
        ]
        route = _route(
            q, ["AAA"], "year", "ranking", "doanh thu thuan",
            years=[2022, 2023, 2024])
        route["plan"].update(
            dimension="year", projection="year", direction="max")

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)
        self.assertTrue(all(var in ca.pandas_query for var in ("df1", "df2", "df3")))
        self.assertIn("formula_year_ranking", ca.detail)

    def test_year_ranking_treats_nam_nao_co_as_direct(self):
        q = "Năm nào có số dư tiền cao nhất trong các năm 2023 và 2024?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 100.0,
                         code="110", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 120.0,
                         code="110", col_name="2024")]),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "tien va cac khoan tuong duong tien", [2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)

    def test_year_ranking_does_not_read_viet_nam_co_as_nested(self):
        q = ("Ngân hàng Đầu tư và Phát triển Việt Nam có các khoản phải thu "
             "bên ngoài lớn nhất vào năm nào trong các năm 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Các khoản phải thu bên ngoài", 100.0,
                         col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Các khoản phải thu bên ngoài", 120.0,
                         col_name="2024")]),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "cac khoan phai thu ben ngoai", [2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)

    def test_year_ranking_does_not_read_cong_ty_co_phan_as_nested(self):
        q = ("Tổng tiền của Công ty cổ phần Bluemarq Group cao nhất vào năm "
             "nào trong các năm 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 100.0,
                         code="110", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Tiền và các khoản tương đương tiền", 120.0,
                         code="110", col_name="2024")]),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "tien va cac khoan tuong duong tien", [2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)

    def test_year_ranking_uses_note_context_for_named_counterparty(self):
        q = ("Năm nào có số dư phải trả cho Công ty Liên doanh TNHH Crown "
             "Sài Gòn cao nhất trong các năm 2023 và 2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 900.0,
                         col_name="2023")], 1, "Đầu tư vào công ty liên kết"),
            _table("df2", "AAA_financial_statements_2023_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 100.0,
                         col_name="2023")], 2, "Phải trả người bán ngắn hạn"),
            _table("df3", "AAA_financial_statements_2024_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 800.0,
                         col_name="2024")], 1, "Đầu tư vào công ty liên kết"),
            _table("df4", "AAA_financial_statements_2024_separate",
                   [_row("Công ty Liên doanh TNHH Crown Sài Gòn", 120.0,
                         col_name="2024")], 2, "Phải trả người bán ngắn hạn"),
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       "phai tra cho cong ty lien doanh tnhh crown sai gon",
                       [2023, 2024])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)
        self.assertNotIn("df1", ca.pandas_query)
        self.assertNotIn("df3", ca.pandas_query)

    def test_year_ranking_supports_argmin(self):
        q = "Tổng tài sản của AAA thấp nhất trong giai đoạn 2022-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2022_consolidated",
                   [_row("Tổng tài sản", 100.0, code="270", col_name="2022")]),
            _table("df2", "AAA_financial_statements_2023_consolidated",
                   [_row("Tổng tài sản", 80.0, code="270", col_name="2023")]),
            _table("df3", "AAA_financial_statements_2024_consolidated",
                   [_row("Tổng tài sản", 120.0, code="270", col_name="2024")]),
        ]
        route = _route(
            q, ["AAA"], "year", "ranking", "tong tai san",
            years=[2022, 2023, 2024])
        route["plan"].update(
            dimension="year", projection="year", direction="min")

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)

    def test_year_ranking_supports_exact_formula_operands(self):
        q = "Biên lợi nhuận gộp của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated", [
                _row("Lợi nhuận gộp", 20.0, code="20", col_name="2023", row=1),
                _row("Doanh thu thuần", 100.0, code="10", col_name="2023", row=2),
            ]),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 30.0, code="20", col_name="2024", row=1),
                _row("Doanh thu thuần", 100.0, code="10", col_name="2024", row=2),
            ]),
        ]
        route = _route(
            q, ["AAA"], "year", "ranking", "bien loi nhuan gop",
            years=[2023, 2024])
        route["plan"].update(
            dimension="year", projection="year", direction="max")

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2024.0)

    def test_year_ranking_fails_closed_on_missing_period(self):
        q = "Doanh thu thuần của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [_table(
            "df1", "AAA_financial_statements_2024_consolidated",
            [_row("Doanh thu thuần", 120.0, code="10", col_name="2024")],
        )]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "doanh thu thuan",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_year_ranking_rejects_prior_value_when_current_cell_is_dash(self):
        q = "Năm nào hàng hóa có giá trị lớn nhất trong các năm 2024 và 2025?"
        tables = [
            _table(
                "df1", "AAA_financial_statements_2024_consolidated",
                [_row("Hàng hóa", 800.0, col_name="Đơn vị tính: VND")],
                context="9. Hàng tồn kho",
            ),
            _table(
                "df2", "AAA_financial_statements_2025_consolidated",
                [_row("Hàng hóa -", 1_220_617_073.0,
                      col_name="Đơn vị tính: VND")],
                context="9. Hàng tồn kho",
            ),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "hang hoa",
                   years=[2024, 2025]),
            tables,
        )

        self.assertFalse(ca.ok)

    def test_year_ranking_fails_closed_on_tie(self):
        q = "Doanh thu thuần của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Doanh thu thuần", 100.0, code="10", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Doanh thu thuần", 100.0, code="10", col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "doanh thu thuan",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)
        self.assertIn("tie", ca.detail)

    def test_year_ranking_rejects_parent_row_for_child_metric(self):
        q = ("Tài sản cố định vô hình của AAA cao nhất trong các năm 2023-2024 "
             "là năm nào?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Tài sản cố định", 100.0, code="220", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Tài sản cố định", 120.0, code="220", col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "tai san co dinh vo hinh",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_year_ranking_rejects_detail_row_for_aggregate_metric(self):
        q = "Tổng vốn chủ sở hữu của AAA cao nhất trong các năm 2023-2024 là năm nào?"
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Quỹ khác thuộc vốn chủ sở hữu", 100.0,
                         col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Quỹ khác thuộc vốn chủ sở hữu", 120.0,
                         col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "von chu so huu",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_year_ranking_does_not_drop_ratio_denominator(self):
        q = ("Năm nào AAA có tỷ trọng giá vốn cho thuê đất trên tổng giá vốn "
             "hàng bán cao nhất trong các năm 2023-2024?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated",
                   [_row("Giá vốn hàng bán", 100.0, code="11", col_name="2023")]),
            _table("df2", "AAA_financial_statements_2024_consolidated",
                   [_row("Giá vốn hàng bán", 120.0, code="11", col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "year", "ranking", "gia von hang ban",
                   years=[2023, 2024]),
            tables,
        )
        self.assertFalse(ca.ok)

    def test_direct_gross_margin(self):
        q = "Biên lợi nhuận gộp của AAA năm 2024 là bao nhiêu phần trăm?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận gộp", 30.0, row=1),
            _row("Doanh thu thuần", 120.0, row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "percent", "margin"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 25.0)

    def test_formula_rejects_segment_column_without_period_evidence(self):
        q = "Biên lợi nhuận gộp của AAA năm 2024 là bao nhiêu phần trăm?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận gộp", 30.0, code="20", col_name="Nội địa", row=1),
            _row("Doanh thu thuần", 100.0, code="10", col_name="Nội địa", row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "percent", "margin"), tables)
        self.assertFalse(ca.ok)

    def test_nested_ranking_selects_company_then_answers_other_formula(self):
        q = ("Năm 2024, hệ số thanh toán hiện hành của doanh nghiệp có "
             "hệ số nợ phải trả trên vốn chủ sở hữu cao nhất là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 200.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Nợ phải trả", 120.0, row=3),
                _row("Vốn chủ sở hữu", 100.0, row=4),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 90.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Nợ phải trả", 180.0, row=3),
                _row("Vốn chủ sở hữu", 100.0, row=4),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB"], "ratio", "ranking"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 0.9)
        self.assertEqual(_eval(ca.pandas_query, tables), 0.9)
        self.assertIn("formula_nested", ca.detail)

    def test_nested_ranking_does_not_fall_back_when_answer_formula_is_missing(self):
        q = ("Năm 2024, hệ số thanh toán hiện hành của doanh nghiệp có "
             "hệ số nợ phải trả trên vốn chủ sở hữu cao nhất là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 120.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 180.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "ratio", "ranking")

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)
        self.assertTrue(requires_formula_solver(route))

    def test_bank_nested_ranking_requires_select_then_project_solver(self):
        q = ("Thu nhập lãi thuần năm 2023 của ngân hàng có thu nhập từ "
             "hoạt động khác thấp nhất là bao nhiêu triệu đồng?")
        route = _route(q, ["AAA", "BBB"], "number", "ranking", years=[2023])

        self.assertTrue(requires_formula_solver(route))

    def test_nested_ranking_by_cfo_over_operating_profit(self):
        q = ("Năm 2024, trong hai công ty AAA và BBB, doanh nghiệp có tỷ lệ "
             "lưu chuyển tiền thuần từ hoạt động kinh doanh trên lợi nhuận thuần "
             "từ hoạt động kinh doanh thấp nhất có biên lợi nhuận ròng bao nhiêu %?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 20.0, row=1),
                _row("Lợi nhuận thuần từ hoạt động kinh doanh", 10.0, code="30", row=2),
                _row("Lợi nhuận sau thuế", 30.0, code="60", row=3),
                _row("Doanh thu thuần", 100.0, code="10", row=4),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 50.0, row=1),
                _row("Lợi nhuận thuần từ hoạt động kinh doanh", 100.0, code="30", row=2),
                _row("Lợi nhuận sau thuế", 10.0, code="60", row=3),
                _row("Doanh thu thuần", 100.0, code="10", row=4),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB"], "percent", "ranking"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_nested_top_n_group_share_refuses(self):
        q = ("Trong 3 công ty AAA, BBB và CCC, 2 doanh nghiệp có biên lợi nhuận "
             "gộp cao nhất nắm giữ bao nhiêu phần trăm tổng tiền của cả nhóm?")
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 30.0, code="20", row=1),
                _row("Doanh thu thuần", 100.0, code="10", row=2),
            ]) for i, ticker in enumerate(("AAA", "BBB", "CCC"), 1)
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB", "CCC"], "percent",
                                       "ranking"), tables)
        self.assertFalse(ca.ok)

    def test_nested_ranking_selects_year_then_answers_interest_coverage(self):
        q = ("Trong giai đoạn 2023-2024, vào năm AAA có tỷ số D/E cao nhất, "
             "hệ số khả năng thanh toán lãi vay là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated", [
                _row("Nợ phải trả", 80.0, col_name="2023", row=1),
                _row("Vốn chủ sở hữu", 100.0, col_name="2023", row=2),
                _row("Lợi nhuận trước thuế", 30.0, col_name="2023", row=3),
                _row("Chi phí lãi vay", 10.0, col_name="2023", row=4),
            ]),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", 180.0, row=1),
                _row("Vốn chủ sở hữu", 100.0, row=2),
                _row("Lợi nhuận trước thuế", 20.0, row=3),
                _row("Chi phí lãi vay", 10.0, row=4),
            ]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "ratio", "ranking", years=[2023, 2024]), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 3.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 3.0)

    def test_typed_nested_plan_separates_selector_and_projection_modes(self):
        q = ("Từ năm 2023 đến 2024, trong nhóm AAA và BBB, tại doanh nghiệp có "
             "tỷ lệ phần trăm tăng doanh thu thuần cao nhất, tỷ lệ phần trăm "
             "tăng của tổng chi phí bán hàng và chi phí quản lý doanh nghiệp "
             "là bao nhiêu %?")
        route = _route(q, ["AAA", "BBB"], "percent", "ranking",
                       years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.dimension, "entity")
        self.assertEqual(plan.selector.match.spec.name, "net_revenue")
        self.assertEqual(plan.selector.mode, "growth")
        self.assertEqual(plan.projection.match.spec.name, "sga_expense")
        self.assertEqual(plan.projection.mode, "growth")

    def test_typed_nested_plan_finds_selector_after_extreme_phrase(self):
        q = ("Trong nhóm AAA và BBB, doanh nghiệp có mức tăng lớn nhất từ năm "
             "2023 đến 2024 của tỷ lệ 365 lần hàng tồn kho bình quân đầu kỳ "
             "và cuối kỳ trên giá vốn hàng bán có tỷ lệ lợi nhuận gộp trên "
             "doanh thu thuần thay đổi bao nhiêu điểm phần trăm?")
        route = _route(q, ["AAA", "BBB"], "percentage_point", "ranking",
                       years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.selector.match.spec.name, "inventory_days")
        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual(plan.projection.match.spec.name, "gross_margin")
        self.assertEqual(plan.projection.mode, "delta")

    def test_typed_nested_plan_keeps_projection_before_year_selector(self):
        q = ("Trong các năm 2017, 2020, 2023 và 2024, tổng phải trả người "
             "bán ngắn hạn cuối năm của AAA trong năm có tổng chi phí bán "
             "hàng lớn nhất là bao nhiêu tỷ đồng?")
        route = _route(q, ["AAA"], "number", "ranking",
                       years=[2017, 2020, 2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.dimension, "year")
        self.assertEqual(plan.selector.match.spec.name, "metric:selling_expense")
        self.assertEqual(
            plan.projection.match.spec.name,
            "metric:trade_payables_short_term",
        )

    def test_typed_nested_plan_uses_distinct_note_metric_aliases(self):
        bank_q = ("Thu nhập lãi thuần năm 2023 của ngân hàng có thu nhập từ "
                  "hoạt động khác thấp nhất là bao nhiêu triệu đồng?")
        eps_q = ("Trong các năm 2023, 2024 và 2025, chi phí khác của AAA "
                 "trong năm có Lãi cơ bản và suy giảm trên mỗi cổ phiếu lớn "
                 "nhất là bao nhiêu tỷ đồng?")
        shares_q = ("Trong các công ty AAA và BBB, công ty có số lượng cổ "
                    "phiếu phổ thông cao nhất năm 2017 có tổng tài sản thuế "
                    "thu nhập hoãn lại là bao nhiêu tỷ đồng?")

        bank = build_compositional_ranking_plan(
            _route(bank_q, ["AAA", "BBB"], "number", "ranking", years=[2023]))
        eps = build_compositional_ranking_plan(
            _route(eps_q, ["AAA"], "number", "ranking",
                   years=[2023, 2024, 2025]))
        shares = build_compositional_ranking_plan(
            _route(shares_q, ["AAA", "BBB"], "number", "ranking", years=[2017]))

        self.assertIsNotNone(bank)
        self.assertEqual(bank.selector.match.spec.name,
                         "metric:bank_net_other_income")
        self.assertEqual(bank.projection.match.spec.name,
                         "metric:net_interest_income")
        self.assertIsNotNone(eps)
        self.assertEqual(eps.selector.match.spec.name, "metric:basic_eps")
        self.assertEqual(eps.projection.match.spec.name, "metric:other_expense")
        self.assertIsNotNone(shares)
        self.assertEqual(shares.selector.match.spec.name, "metric:total_shares")
        self.assertEqual(shares.projection.match.spec.name,
                         "metric:deferred_tax_asset")

    def test_typed_nested_inventory_days_selects_then_projects_margin_change(self):
        q = ("Trong nhóm AAA và BBB, doanh nghiệp có mức tăng lớn nhất từ năm "
             "2023 đến 2024 của tỷ lệ 365 lần hàng tồn kho bình quân đầu kỳ "
             "và cuối kỳ trên giá vốn hàng bán có tỷ lệ lợi nhuận gộp trên "
             "doanh thu thuần thay đổi bao nhiêu điểm phần trăm?")
        tables = []
        values = {
            "AAA": {2022: (100, 100, 20), 2023: (100, 100, 20),
                    2024: (200, 100, 30)},
            "BBB": {2022: (100, 100, 20), 2023: (100, 100, 20),
                    2024: (110, 100, 25)},
        }
        index = 1
        for ticker, by_year in values.items():
            for year, (inventory, cogs, gross_profit) in by_year.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated",
                    [
                        _row("Hàng tồn kho", inventory, code="140",
                             col_name=str(year), row=1),
                        _row("Giá vốn hàng bán", cogs, code="11",
                             col_name=str(year), row=2),
                        _row("Lợi nhuận gộp", gross_profit, code="20",
                             col_name=str(year), row=3),
                        _row("Doanh thu thuần", 100, code="10",
                             col_name=str(year), row=4),
                    ],
                ))
                index += 1
        route = _route(q, ["AAA", "BBB"], "percentage_point", "ranking",
                       years=[2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)
        self.assertIn("formula_nested_v4", ca.detail)

    def test_typed_median_filter_selects_entity_then_projects_formula(self):
        q = ("Trong nhóm AAA, BBB, CCC và DDD, trong nhóm có hệ số thanh toán "
             "nhanh năm 2023 thấp hơn trung vị, doanh nghiệp có mức tăng biên "
             "lợi nhuận gộp cao nhất từ năm 2023 đến 2024 có hệ số khả năng "
             "thanh toán lãi vay năm 2024 là bao nhiêu lần?")
        values = {
            "AAA": (30, 10, 40, 30),
            "BBB": (60, 10, 30, 30),
            "CCC": (110, 10, 50, 50),
            "DDD": (160, 10, 80, 80),
        }
        tables = []
        for index, (ticker, (current_assets, gross_2023, gross_2024,
                              pretax_2024)) in enumerate(values.items(), 1):
            tables.extend([
                _table(
                    f"df{index}_2023",
                    f"{ticker}_financial_statements_2023_consolidated",
                    [
                        _row("Tài sản ngắn hạn", current_assets, code="100",
                             col_name="2023", row=1),
                        _row("Hàng tồn kho", 10, code="140",
                             col_name="2023", row=2),
                        _row("Nợ ngắn hạn", 40, code="310",
                             col_name="2023", row=3),
                        _row("Lợi nhuận gộp", gross_2023, code="20",
                             col_name="2023", row=4),
                        _row("Doanh thu thuần", 100, code="10",
                             col_name="2023", row=5),
                    ],
                ),
                _table(
                    f"df{index}_2024",
                    f"{ticker}_financial_statements_2024_consolidated",
                    [
                        _row("Lợi nhuận gộp", gross_2024, code="20", row=1),
                        _row("Doanh thu thuần", 100, code="10", row=2),
                        _row("Lợi nhuận trước thuế", pretax_2024,
                             code="50", row=3),
                        _row("Chi phí lãi vay", 10, row=4),
                    ],
                ),
            ])
        route = _route(q, list(values), "ratio", "ranking", years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.median_filter.calculation.match.spec.name,
                         "quick_ratio")
        self.assertEqual(plan.median_filter.op, "<")
        self.assertEqual(plan.median_filter.year, 2023)
        self.assertEqual(plan.selector.match.spec.name, "gross_margin")
        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual(plan.projection.match.spec.name, "interest_coverage")
        self.assertEqual(plan.projection.mode, "level")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 4.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 4.0)
        self.assertIn("formula_nested_v4", ca.detail)

        # CCC was outside the static median subset. If its quick ratio changes
        # so it enters the subset and wins, the dynamic selection guard fails.
        mutated = [dict(table) for table in tables]
        mutated[4] = dict(mutated[4])
        frame = pd.read_csv(io.StringIO(mutated[4]["csv_text"]))
        frame.loc[frame["label"] == "Tài sản ngắn hạn", "value"] = 18
        mutated[4]["csv_text"] = frame.to_csv(index=False)
        self.assertEqual(_eval(ca.pandas_query, mutated), 0.0)

    def test_typed_median_filter_selects_year_then_projects_formula(self):
        q = ("Trong các năm 2021, 2022 và 2023, ở những năm có biên lợi nhuận "
             "gộp thấp hơn trung vị, năm có tỷ số dòng tiền hoạt động trên "
             "doanh thu thuần cao nhất có ROE là bao nhiêu phần trăm?")
        tables = []
        values = {
            2021: (10, 5, 8),
            2022: (20, 30, 12),
            2023: (30, 50, 16),
        }
        for index, (year, (gross_margin, cfo_margin, roe)) in enumerate(
                values.items(), 1):
            tables.append(_table(
                f"df{index}",
                f"AAA_financial_statements_{year}_consolidated",
                [
                    _row("Lợi nhuận gộp", gross_margin, code="20",
                         col_name=str(year), row=1),
                    _row("Doanh thu thuần", 100, code="10",
                         col_name=str(year), row=2),
                    _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh",
                         cfo_margin, col_name=str(year), row=3),
                    _row("Lợi nhuận sau thuế", roe, code="60",
                         col_name=str(year), row=4),
                    _row("Vốn chủ sở hữu", 100, code="400",
                         col_name=str(year), row=5),
                ],
            ))
        route = _route(q, ["AAA"], "percent", "ranking",
                       years=[2021, 2022, 2023])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.dimension, "year")
        self.assertEqual(plan.median_filter.calculation.match.spec.name,
                         "gross_margin")
        self.assertEqual(plan.selector.match.spec.name, "cfo_margin")
        self.assertEqual(plan.projection.match.spec.name, "roe")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 8.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 8.0)

    def test_typed_median_filter_infers_prior_year_for_growth(self):
        q = ("Năm 2025, trong nhóm ASM, DBC và VNM có tốc độ tăng trưởng doanh "
             "thu thuần cao hơn trung vị; hệ số khả năng thanh toán lãi vay "
             "của doanh nghiệp có biên lợi nhuận gộp cao nhất là bao nhiêu lần?")
        route = _route(q, ["ASM", "DBC", "VNM"], "ratio", "ranking",
                       years=[2025])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.median_filter.calculation.match.spec.name,
                         "net_revenue")
        self.assertEqual(plan.median_filter.calculation.mode, "growth")
        self.assertEqual(plan.median_filter.calculation.start_year, 2024)
        self.assertEqual(plan.median_filter.calculation.end_year, 2025)

    def test_typed_median_filter_parses_decomposed_inventory_days(self):
        q = ("Xét các doanh nghiệp HPG, HSG, MSR và NKG, trong nhóm có giá trị "
             "hàng tồn kho bình quân năm 2021 và 2022 chia cho giá vốn hàng bán "
             "năm 2022 rồi nhân 365 cao hơn trung vị, tại doanh nghiệp có mức "
             "giảm lớn nhất của giá trị này từ năm 2022 đến năm 2024, lợi nhuận "
             "gộp năm 2024 chiếm bao nhiêu phần trăm doanh thu thuần năm 2024?")
        route = _route(q, ["HPG", "HSG", "MSR", "NKG"], "percent", "ranking",
                       years=[2021, 2022, 2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.median_filter.calculation.match.spec.name,
                         "inventory_days")
        self.assertEqual(plan.median_filter.year, 2022)
        self.assertEqual(plan.selector.match.spec.name, "inventory_days")
        self.assertEqual(plan.selector.mode, "decrease")
        self.assertEqual(plan.selector.start_year, 2022)
        self.assertEqual(plan.selector.end_year, 2024)
        self.assertEqual(plan.projection.match.spec.name, "gross_margin")
        self.assertEqual(plan.projection.mode, "level")

    def test_interest_coverage_rejects_cash_flow_interest_paid(self):
        q = "Hệ số khả năng thanh toán lãi vay của AAA năm 2024 là bao nhiêu lần?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận trước thuế", 20.0, code="50", row=1),
            _row("Tiền lãi vay đã trả", -10.0, row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "ratio", "ratio"), tables)
        self.assertFalse(ca.ok)

    def test_current_ratio_rejects_other_short_term_payable(self):
        q = "Hệ số thanh toán hiện hành của AAA năm 2024 là bao nhiêu lần?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Tài sản ngắn hạn", 200.0, code="100", row=1),
            _row("Phải trả ngắn hạn khác", 100.0, code="319", row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "ratio", "ratio"), tables)
        self.assertFalse(ca.ok)

    def test_net_margin_rejects_retained_earnings(self):
        q = "Biên lợi nhuận ròng của AAA năm 2024 là bao nhiêu phần trăm?"
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Lợi nhuận sau thuế chưa phân phối", 30.0, code="421", row=1),
            _row("Doanh thu thuần", 100.0, code="10", row=2),
        ])]
        ca = try_formula_answer(_route(q, ["AAA"], "percent", "margin"), tables)
        self.assertFalse(ca.ok)

    def test_count_multiple_conditions_working_capital_and_cfo(self):
        q = ("Năm 2024, có bao nhiêu doanh nghiệp đồng thời ghi nhận vốn lưu động "
             "ròng âm và lưu chuyển tiền thuần từ hoạt động kinh doanh dương?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 80.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 10.0, row=3),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 120.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", 10.0, row=3),
            ]),
            _table("df3", "CCC_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 80.0, row=1),
                _row("Nợ ngắn hạn", 100.0, row=2),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", -10.0, row=3),
            ]),
        ]
        ca = try_formula_answer(_route(q, ["AAA", "BBB", "CCC"], "count", "count"),
                                tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)

    def test_count_direct_condition_over_years(self):
        q = ("Trong các năm 2021, 2023 và 2024, AAA có bao nhiêu năm ghi nhận "
             "quỹ khen thưởng, phúc lợi lớn hơn 40 tỷ đồng?")
        tables = [
            _table("df1", "AAA_financial_statements_2021_consolidated",
                   [_row("Quỹ khen thưởng, phúc lợi", 30.0, col_name="2021")]),
            _table("df2", "AAA_financial_statements_2023_consolidated",
                   [_row("Quỹ khen thưởng, phúc lợi", 50.0, col_name="2023")]),
            _table("df3", "AAA_financial_statements_2024_consolidated",
                   [_row("Quỹ khen thưởng, phúc lợi", 60.0, col_name="2024")]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA"], "count", "count", "quy khen thuong phuc loi",
                   years=[2021, 2023, 2024]), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_count_temporal_multi_condition(self):
        q = ("Từ 2023 sang 2024, có bao nhiêu doanh nghiệp đồng thời tăng tỷ trọng "
             "hàng tồn kho trên tổng tài sản và giảm biên lợi nhuận gộp?")
        tables = []
        values = {
            "AAA": {2023: (10.0, 100.0, 30.0, 100.0),
                    2024: (20.0, 100.0, 20.0, 100.0)},
            "BBB": {2023: (10.0, 100.0, 20.0, 100.0),
                    2024: (8.0, 100.0, 25.0, 100.0)},
        }
        idx = 1
        for ticker, periods in values.items():
            for year, (inventory, assets, gross, revenue) in periods.items():
                tables.append(_table(
                    f"df{idx}", f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Hàng tồn kho", inventory, col_name=str(year), row=1),
                        _row("Tổng tài sản", assets, col_name=str(year), row=2),
                        _row("Lợi nhuận gộp", gross, col_name=str(year), row=3),
                        _row("Doanh thu thuần", revenue, col_name=str(year), row=4),
                    ]))
                idx += 1
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB"], "count", "count", years=[2023, 2024]),
            tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)

    def test_ratio_difference_between_companies(self):
        q = ("Biên lợi nhuận gộp năm 2024 của AAA chênh lệch bao nhiêu điểm phần "
             "trăm so với BBB?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 30.0, row=1),
                _row("Doanh thu thuần", 100.0, row=2),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", 20.0, row=1),
                _row("Doanh thu thuần", 100.0, row=2),
            ]),
        ]
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB"], "percentage_point", "difference"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_average_margin_change_for_companies_with_revenue_increase(self):
        q = ("Mức thay đổi trung bình từ năm 2023 đến năm 2024 của biên lợi nhuận "
             "gộp tại các công ty có doanh thu thuần năm 2024 tăng so với năm "
             "2023 là bao nhiêu điểm phần trăm?")
        tables = []
        values = {
            "AAA": {2023: (20.0, 100.0), 2024: (36.0, 120.0)},
            "BBB": {2023: (30.0, 100.0), 2024: (20.0, 80.0)},
            "CCC": {2023: (10.0, 100.0), 2024: (30.0, 150.0)},
        }
        idx = 1
        for ticker, periods in values.items():
            for year, (gross, revenue) in periods.items():
                tables.append(_table(
                    f"df{idx}", f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Lợi nhuận gộp", gross, col_name=str(year), row=1),
                        _row("Doanh thu thuần", revenue, col_name=str(year), row=2),
                    ]))
                idx += 1
        ca = try_formula_answer(
            _route(q, ["AAA", "BBB", "CCC"], "percentage_point", "average",
                   years=[2023, 2024]), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)  # AAA +10pp, CCC +10pp
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_typed_filter_then_mean_uses_dynamic_membership(self):
        q = ("Năm 2017, trong các công ty AAA, BBB và CCC, với các công ty có "
             "tỷ lệ tài sản ngắn hạn trên nợ ngắn hạn từ 1 lần trở lên, bình "
             "quân tỷ lệ hàng tồn kho trên nợ ngắn hạn là bao nhiêu lần?")
        values = {
            "AAA": (120.0, 20.0, 100.0),
            "BBB": (80.0, 50.0, 100.0),
            "CCC": (200.0, 40.0, 100.0),
        }
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2017_consolidated", [
                _row("Tài sản ngắn hạn", current_assets, code="100",
                     col_name="2017", row=1),
                _row("Hàng tồn kho", inventory, code="140",
                     col_name="2017", row=2),
                _row("Nợ ngắn hạn", liabilities, code="310",
                     col_name="2017", row=3),
            ])
            for i, (ticker, (current_assets, inventory, liabilities))
            in enumerate(values.items(), 1)
        ]
        route = _route(q, list(values), "ratio", "average", years=[2017])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.aggregate.op, "mean")
        self.assertEqual(plan.predicates[0].calculation.match.spec.name,
                         "current_ratio")
        self.assertEqual(plan.value.primary.match.spec.name,
                         "inventory_current_liabilities")
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 0.3)
        self.assertEqual(_eval(ca.pandas_query, tables), 0.3)

        mutated = [dict(table) for table in tables]
        mutated[1] = dict(mutated[1])
        frame = pd.read_csv(io.StringIO(mutated[1]["csv_text"]))
        frame.loc[frame["label"] == "Tài sản ngắn hạn", "value"] = 150.0
        mutated[1]["csv_text"] = frame.to_csv(index=False)
        self.assertEqual(_eval(ca.pandas_query, mutated), 0.37)

    def test_filter_mean_recognizes_unquantified_revenue_increase(self):
        q = ("Từ năm 2021 đến 2022, trong bốn mã cổ phiếu AAA, BBB, CCC và "
             "DDD, với các công ty có doanh thu thuần tăng, bình quân mức "
             "thay đổi của tỷ lệ lợi nhuận gộp trên doanh thu thuần là bao "
             "nhiêu điểm phần trăm?")
        route = _route(q, ["AAA", "BBB", "CCC", "DDD"],
                       "percentage_point", "average", years=[2021, 2022])

        plan = build_filter_aggregate_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.predicates), 1)
        self.assertEqual(plan.predicates[0].calculation.match.spec.name,
                         "net_revenue")
        self.assertEqual(plan.predicates[0].calculation.mode, "delta")
        self.assertEqual(plan.predicates[0].op, ">")
        self.assertEqual(plan.predicates[0].threshold, 0.0)

    def test_filter_mean_recognizes_explicit_later_year_comparison(self):
        q = ("Trong các công ty AAA, BBB, CCC và DDD có doanh thu thuần năm "
             "2022 cao hơn năm 2021, mức thay đổi trung bình của tỷ lệ lợi "
             "nhuận gộp trên doanh thu thuần từ năm 2021 đến năm 2022 là "
             "bao nhiêu điểm phần trăm?")
        route = _route(q, ["AAA", "BBB", "CCC", "DDD"],
                       "percentage_point", "difference", years=[2022, 2021])

        plan = build_filter_aggregate_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.predicates), 1)
        self.assertEqual(plan.predicates[0].calculation.match.spec.name,
                         "net_revenue")
        self.assertEqual(plan.predicates[0].calculation.mode, "delta")
        self.assertEqual(plan.predicates[0].op, ">")

    def test_filter_sum_keeps_full_threshold_before_later_target_metric(self):
        q = ("Năm 2016, trong số các công ty AAA, BBB, CCC và DDD có tỷ lệ "
             "lợi nhuận sau thuế trên doanh thu thuần lớn hơn 10%, tổng "
             "doanh thu thuần của các công ty đó là bao nhiêu nghìn tỷ đồng?")
        route = _route(q, ["AAA", "BBB", "CCC", "DDD"], "number", "sum",
                       years=[2016])

        plan = build_filter_aggregate_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.predicates), 1)
        self.assertEqual(plan.predicates[0].op, ">")
        self.assertEqual(plan.predicates[0].threshold, 10.0)
        self.assertEqual(plan.value.primary.match.spec.name, "net_revenue")

    def test_nested_formula_accepts_opaque_label_with_exact_vas_code(self):
        q = ("Năm 2024, giá trị hàng tồn kho của doanh nghiệp có hệ số "
             "thanh toán nhanh thấp nhất trong 2 doanh nghiệp AAA và BBB "
             "là bao nhiêu tỷ đồng?")
        tables = [
            _table("df1", "AAA_financial_statements_2024_consolidated", [
                _row("A - (100=110+130+140+150)", 120.0, code="100", row=1),
                _row("Hàng tồn kho", 20.0, code="140", row=2),
                _row("Nợ ngắn hạn", 100.0, code="310", row=3),
            ]),
            _table("df2", "BBB_financial_statements_2024_consolidated", [
                _row("Tài sản ngắn hạn", 150.0, code="100", row=1),
                _row("Hàng tồn kho", 10.0, code="140", row=2),
                _row("Nợ ngắn hạn", 100.0, code="310", row=3),
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "number", "ranking", years=[2024])
        route["unit_scale"] = 1e9

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)

    def test_inventory_divided_by_current_liabilities_is_one_ratio(self):
        q = ("Năm 2024, trong các công ty AAA, BBB và CCC, nhóm có tỷ lệ "
             "hàng tồn kho chia cho nợ ngắn hạn cao hơn trung vị chiếm bao "
             "nhiêu phần trăm tổng nợ ngắn hạn của cả nhóm?")
        route = _route(q, ["AAA", "BBB", "CCC"], "percent", "ratio",
                       years=[2024])

        plan = build_filter_aggregate_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.predicates[0].calculation.match.spec.name,
                         "inventory_current_liabilities")

    def test_typed_median_filter_then_mean(self):
        q = ("Năm 2022, trong nhóm AAA, BBB, CCC và DDD, các công ty có hệ số "
             "thanh toán nhanh thấp hơn trung vị của nhóm có biên lợi nhuận "
             "ròng bình quân là bao nhiêu phần trăm?")
        values = {
            "AAA": (60.0, 10.0, 100.0, 10.0),
            "BBB": (90.0, 10.0, 100.0, 20.0),
            "CCC": (130.0, 10.0, 100.0, 30.0),
            "DDD": (160.0, 10.0, 100.0, 40.0),
        }
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2022_consolidated", [
                _row("Tài sản ngắn hạn", assets, code="100",
                     col_name="2022", row=1),
                _row("Hàng tồn kho", inventory, code="140",
                     col_name="2022", row=2),
                _row("Nợ ngắn hạn", 100.0, code="310",
                     col_name="2022", row=3),
                _row("Lợi nhuận sau thuế", margin, code="60",
                     col_name="2022", row=4),
                _row("Doanh thu thuần", 100.0, code="10",
                     col_name="2022", row=5),
            ])
            for i, (ticker, (assets, inventory, _liabilities, margin))
            in enumerate(values.items(), 1)
        ]
        route = _route(q, list(values), "percent", "average", years=[2022])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.predicates[0].reference, "median")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 15.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 15.0)
        self.assertIn("formula_filter_aggregate_v5", ca.detail)

    def test_filter_then_mean_requires_multi_period_predicates(self):
        q = ("Trong các doanh nghiệp AAA, BBB và CCC có LNST dương và CFO dương "
             "liên tục trong hai năm 2023-2024, tốc độ tăng trưởng doanh thu "
             "thuần bình quân năm 2024 là bao nhiêu phần trăm?")
        values = {
            "AAA": {2023: (10.0, 10.0, 100.0), 2024: (10.0, 10.0, 120.0)},
            "BBB": {2023: (10.0, -1.0, 100.0), 2024: (10.0, 10.0, 200.0)},
            "CCC": {2023: (10.0, 10.0, 200.0), 2024: (10.0, 10.0, 220.0)},
        }
        tables = []
        index = 1
        for ticker, periods in values.items():
            for year, (profit, cfo, revenue) in periods.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Lợi nhuận sau thuế", profit, code="60",
                             col_name=str(year), row=1),
                        _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                             col_name=str(year), row=2),
                        _row("Doanh thu thuần", revenue, code="10",
                             col_name=str(year), row=3),
                    ]))
                index += 1
        route = _route(q, list(values), "percent", "growth_pct",
                       years=[2023, 2024])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual([predicate.calculation.mode for predicate in plan.predicates],
                         ["level", "level"])
        self.assertTrue(all(predicate.years == (2023, 2024)
                            for predicate in plan.predicates))
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 15.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 15.0)

    def test_filter_then_mean_of_formula_difference(self):
        q = ("Trong nhóm AAA, BBB và CCC, các công ty duy trì dòng tiền hoạt "
             "động dương ở cả năm 2024 và 2025 nhưng doanh thu thuần năm 2025 "
             "giảm so với 2024 có chênh lệch bình quân giữa biên lợi nhuận gộp "
             "và biên lợi nhuận ròng năm 2025 là bao nhiêu điểm phần trăm?")
        values = {
            "AAA": {2024: (10, 100, 0, 0), 2025: (10, 80, 24, 8)},
            "BBB": {2024: (10, 100, 0, 0), 2025: (10, 80, 32, 12)},
            "CCC": {2024: (10, 100, 0, 0), 2025: (10, 120, 60, 12)},
        }
        tables = []
        index = 1
        for ticker, periods in values.items():
            for year, (cfo, revenue, gross, profit) in periods.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                             col_name=str(year), row=1),
                        _row("Doanh thu thuần", revenue, code="10",
                             col_name=str(year), row=2),
                        _row("Lợi nhuận gộp", gross, code="20",
                             col_name=str(year), row=3),
                        _row("Lợi nhuận sau thuế", profit, code="60",
                             col_name=str(year), row=4),
                    ]))
                index += 1
        route = _route(q, list(values), "percentage_point", "difference",
                       years=[2024, 2025])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.value.combine, "difference")
        self.assertEqual(plan.value.primary.match.spec.name, "gross_margin")
        self.assertEqual(plan.value.secondary.match.spec.name, "net_margin")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 22.5)
        self.assertEqual(_eval(ca.pandas_query, tables), 22.5)

    def test_filter_then_difference_of_group_means(self):
        q = ("Năm 2021, đối với nhóm AAA, BBB, CCC, DDD, chênh lệch về hệ số "
             "khả năng thanh toán lãi vay bình quân giữa phân nhóm có tỷ số D/E "
             "cao hơn trung vị và phân nhóm còn lại là bao nhiêu lần?")
        values = {
            "AAA": (50.0, 100.0, 2.0),
            "BBB": (100.0, 100.0, 4.0),
            "CCC": (200.0, 100.0, 8.0),
            "DDD": (300.0, 100.0, 10.0),
        }
        tables = []
        for i, (ticker, (liabilities, equity, coverage)) in enumerate(
                values.items(), 1):
            tables.append(_table(
                f"df{i}", f"{ticker}_financial_statements_2021_consolidated", [
                    _row("Nợ phải trả", liabilities, code="300",
                         col_name="2021", row=1),
                    _row("Vốn chủ sở hữu", equity, code="400",
                         col_name="2021", row=2),
                    _row("Lợi nhuận trước thuế", (coverage - 1) * 10,
                         code="50", col_name="2021", row=3),
                    _row("Chi phí lãi vay", -10.0,
                         col_name="2021", row=4),
                ]))
        route = _route(q, list(values), "ratio", "difference", years=[2021])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.aggregate.op, "difference_of_means")
        self.assertEqual(plan.aggregate.group_predicate.reference, "median")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 6.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 6.0)

    def test_filter_aggregate_resolves_values_for_excluded_members(self):
        q = ("Năm 2017, trong các công ty AAA và BBB có tỷ lệ tài sản ngắn hạn "
             "trên nợ ngắn hạn từ 1 lần trở lên, bình quân tỷ lệ hàng tồn kho "
             "trên nợ ngắn hạn là bao nhiêu lần?")
        tables = [
            _table("df1", "AAA_financial_statements_2017_consolidated", [
                _row("Tài sản ngắn hạn", 120.0, code="100", col_name="2017", row=1),
                _row("Hàng tồn kho", 20.0, code="140", col_name="2017", row=2),
                _row("Nợ ngắn hạn", 100.0, code="310", col_name="2017", row=3),
            ]),
            _table("df2", "BBB_financial_statements_2017_consolidated", [
                _row("Tài sản ngắn hạn", 80.0, code="100", col_name="2017", row=1),
                _row("Nợ ngắn hạn", 100.0, code="310", col_name="2017", row=2),
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "ratio", "average", years=[2017])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)
        self.assertIn("unresolved value BBB", ca.detail)

    def test_filter_then_share_and_sum_aggregates(self):
        share_q = ("Năm 2024, trong các công ty AAA, BBB và CCC có CFO dương, "
                   "doanh thu thuần của các công ty này chiếm bao nhiêu phần "
                   "trăm tổng doanh thu thuần của cả nhóm?")
        sum_q = ("Năm 2024, trong các công ty AAA, BBB và CCC có CFO dương, "
                 "tổng cộng doanh thu thuần của các công ty này là bao nhiêu?")
        values = {"AAA": (10.0, 100.0), "BBB": (-1.0, 200.0),
                  "CCC": (10.0, 300.0)}
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                     col_name="2024", unit=1.0, row=1),
                _row("Doanh thu thuần", revenue, code="10",
                     col_name="2024", unit=1.0, row=2),
            ])
            for i, (ticker, (cfo, revenue)) in enumerate(values.items(), 1)
        ]

        share = try_formula_answer(
            _route(share_q, list(values), "percent", "ratio", years=[2024]),
            tables)
        total = try_formula_answer(
            _route(sum_q, list(values), "number", "average", years=[2024]),
            tables)

        self.assertTrue(share.ok, share.detail)
        self.assertEqual(share.answer, 400.0 / 600.0 * 100.0)
        self.assertEqual(_eval(share.pandas_query, tables), 66.67)
        self.assertTrue(total.ok, total.detail)
        self.assertEqual(total.answer, 400.0)
        self.assertEqual(_eval(total.pandas_query, tables), 400.0)

    def test_quantified_all_period_filter_then_sum(self):
        q = ("Năm 2023, tổng doanh thu thuần của các công ty AAA, BBB và CCC "
             "có tỷ lệ lợi nhuận sau thuế trên doanh thu thuần dương trong cả "
             "ba năm 2021, 2022 và 2023 là bao nhiêu?")
        values = {
            "AAA": {2021: (10, 90), 2022: (10, 95), 2023: (10, 100)},
            "BBB": {2021: (10, 180), 2022: (-1, 190), 2023: (10, 200)},
            "CCC": {2021: (10, 280), 2022: (10, 290), 2023: (10, 300)},
        }
        tables = []
        index = 1
        for ticker, periods in values.items():
            for year, (profit, revenue) in periods.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Lợi nhuận sau thuế", profit, code="60",
                             col_name=str(year), unit=1.0, row=1),
                        _row("Doanh thu thuần", revenue, code="10",
                             col_name=str(year), unit=1.0, row=2),
                    ]))
                index += 1
        route = _route(q, list(values), "number", "ratio",
                       years=[2021, 2022, 2023])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.aggregate.op, "sum")
        self.assertEqual(plan.predicates[0].quantifier.mode, "all")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 400.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 400.0)

    def test_quantified_any_period_filter_then_mean(self):
        q = ("Trong các công ty AAA, BBB và CCC có CFO dương ít nhất một năm "
             "trong các năm 2022, 2023 và 2024, doanh thu thuần năm 2024 bình "
             "quân là bao nhiêu?")
        values = {
            "AAA": ((-1, 1, -1), 100),
            "BBB": ((-1, -1, -1), 200),
            "CCC": ((1, 1, 1), 300),
        }
        tables = []
        index = 1
        for ticker, (cfos, revenue) in values.items():
            for year, cfo in zip((2022, 2023, 2024), cfos):
                rows = [_row(
                    "Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                    col_name=str(year), unit=1.0, row=1)]
                if year == 2024:
                    rows.append(_row(
                        "Doanh thu thuần", revenue, code="10",
                        col_name="2024", unit=1.0, row=2))
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", rows))
                index += 1
        route = _route(q, list(values), "number", "average",
                       years=[2022, 2023, 2024])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.predicates[0].quantifier.mode, "any")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 200.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 200.0)

    def test_quantified_nested_plan_keeps_cfo_net_profit_as_one_formula(self):
        q = ("Trong các doanh nghiệp DBC, MPC, MSN, OGC và QNS có lợi nhuận "
             "sau thuế dương và tỷ lệ CFO trên lợi nhuận sau thuế lớn hơn "
             "0,5 trong cả hai năm 2023 và 2024, doanh nghiệp có tốc độ tăng "
             "trưởng doanh thu thuần năm 2024 so với năm 2023 cao nhất có "
             "biên lợi nhuận gộp năm 2024 là bao nhiêu phần trăm?")
        route = _route(q, ["DBC", "MPC", "MSN", "OGC", "QNS"],
                       "percent", "ranking", years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(
            [predicate.calculation.match.spec.name
             for predicate in plan.predicates],
            ["net_profit", "cfo_net_profit"],
        )
        self.assertTrue(all(predicate.quantifier.mode == "all"
                            for predicate in plan.predicates))
        self.assertEqual(plan.filters, ())

    def test_top_k_share_recomputes_membership_dynamically(self):
        q = ("Năm 2024, trong nhóm AAA, BBB và CCC, 2 doanh nghiệp có tỷ lệ "
             "lợi nhuận gộp trên doanh thu thuần cao nhất nắm giữ bao nhiêu "
             "phần trăm tổng tiền và các khoản tương đương tiền của cả nhóm?")
        values = {
            "AAA": (30.0, 100.0, 10.0),
            "BBB": (20.0, 100.0, 20.0),
            "CCC": (10.0, 100.0, 30.0),
        }
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Lợi nhuận gộp", gross, code="20", col_name="2024", row=1),
                _row("Doanh thu thuần", revenue, code="10", col_name="2024", row=2),
                _row("Tiền và các khoản tương đương tiền", cash,
                     code="110", col_name="2024", row=3),
            ])
            for i, (ticker, (gross, revenue, cash)) in enumerate(values.items(), 1)
        ]
        route = _route(q, list(values), "percent", "ranking", years=[2024])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.aggregate.rank_slice.k, 2)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 50.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 50.0)

        mutated = [dict(table) for table in tables]
        mutated[2] = dict(mutated[2])
        frame = pd.read_csv(io.StringIO(mutated[2]["csv_text"]))
        frame.loc[frame["label"] == "Lợi nhuận gộp", "value"] = 40.0
        mutated[2]["csv_text"] = frame.to_csv(index=False)
        self.assertEqual(_eval(ca.pandas_query, mutated), 66.67)

    def test_share_uses_positive_profit_denominator_scope(self):
        q = ("Năm 2024, các công ty AAA, BBB và CCC có tỷ lệ nợ phải trả trên "
             "tổng tài sản thấp hơn trung vị đóng góp bao nhiêu phần trăm vào "
             "tổng lợi nhuận sau thuế của các doanh nghiệp có lãi?")
        values = {
            "AAA": (10.0, 100.0, 10.0),
            "BBB": (20.0, 100.0, -5.0),
            "CCC": (30.0, 100.0, 30.0),
        }
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", debt, code="300", col_name="2024", row=1),
                _row("Tổng tài sản", assets, code="270", col_name="2024", row=2),
                _row("Lợi nhuận sau thuế", profit, code="60",
                     col_name="2024", row=3),
            ])
            for i, (ticker, (debt, assets, profit)) in enumerate(values.items(), 1)
        ]
        route = _route(q, list(values), "percent", "difference", years=[2024])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(len(plan.aggregate.denominator_predicates), 1)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 25.0)

    def test_partition_ratio_uses_complement_group(self):
        q = ("Năm 2024, trong các công ty AAA, BBB, CCC và DDD, tổng chi phí "
             "lãi vay của nhóm có tỷ lệ nợ phải trả trên vốn chủ sở hữu cao hơn "
             "trung vị gấp bao nhiêu lần tổng chi phí lãi vay của nhóm có tỷ lệ "
             "này bằng hoặc thấp hơn trung vị?")
        values = {
            "AAA": (50.0, 100.0, 10.0),
            "BBB": (100.0, 100.0, 20.0),
            "CCC": (200.0, 100.0, -30.0),
            "DDD": (300.0, 100.0, -40.0),
        }
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Nợ phải trả", debt, code="300", col_name="2024", row=1),
                _row("Vốn chủ sở hữu", equity, code="400", col_name="2024", row=2),
                _row("Chi phí lãi vay", interest, col_name="2024", row=3),
            ])
            for i, (ticker, (debt, equity, interest)) in enumerate(values.items(), 1)
        ]
        route = _route(q, list(values), "ratio", "difference", years=[2024])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.aggregate.op, "partition_ratio")
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 70.0 / 30.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.33)

    def test_period_aware_selects_then_projects_average_asset_turnover(self):
        q = ("Năm 2024, trong nhóm AAA và BBB, vòng quay tổng tài sản tính "
             "theo tổng tài sản bình quân của doanh nghiệp có tỷ trọng tài "
             "sản dài hạn trên tổng tài sản cao nhất là bao nhiêu vòng?")
        values = {
            "AAA": (100.0, 100.0, 80.0, 100.0),
            "BBB": (100.0, 100.0, 60.0, 300.0),
        }
        tables = []
        for i, (ticker, (opening, closing, long_term, revenue)) in enumerate(
                values.items(), 1):
            tables.extend([
                _table(f"df{i}a", f"{ticker}_financial_statements_2023_consolidated", [
                    _row("Tổng tài sản", opening, code="270",
                         col_name="2023", row=1),
                ]),
                _table(f"df{i}b", f"{ticker}_financial_statements_2024_consolidated", [
                    _row("Tổng tài sản", closing, code="270",
                         col_name="2024", row=1),
                    _row("Tài sản dài hạn", long_term, code="200",
                         col_name="2024", row=2),
                    _row("Doanh thu thuần", revenue, code="10",
                         col_name="2024", row=3),
                ]),
            ])
        route = _route(q, list(values), "ratio", "ranking", years=[2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name, "long_term_assets_ratio")
        self.assertEqual(plan.projection.match.spec.name,
                         "total_asset_turnover_average_assets")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)
        self.assertIn("formula_period_aware_v6", ca.detail)

    def test_period_aware_roe_uses_opening_and_closing_equity(self):
        q = ("Năm 2024, trong nhóm AAA, BBB và CCC, ROE cao nhất của các công "
             "ty có hệ số nợ phải trả trên vốn chủ sở hữu thấp hơn trung vị "
             "của nhóm là bao nhiêu phần trăm? ROE được tính bằng lợi nhuận "
             "sau thuế chia cho vốn chủ sở hữu bình quân đầu và cuối kỳ.")
        values = {
            "AAA": (100.0, 100.0, 50.0, 20.0),
            "BBB": (100.0, 100.0, 100.0, 40.0),
            "CCC": (100.0, 100.0, 200.0, 60.0),
        }
        tables = []
        for i, (ticker, (opening, closing, liabilities, profit)) in enumerate(
                values.items(), 1):
            tables.extend([
                _table(f"df{i}a", f"{ticker}_financial_statements_2023_consolidated", [
                    _row("Vốn chủ sở hữu", opening, code="400",
                         col_name="2023", row=1),
                ]),
                _table(f"df{i}b", f"{ticker}_financial_statements_2024_consolidated", [
                    _row("Vốn chủ sở hữu", closing, code="400",
                         col_name="2024", row=1),
                    _row("Nợ phải trả", liabilities, code="300",
                         col_name="2024", row=2),
                    _row("Lợi nhuận sau thuế", profit, code="60",
                         col_name="2024", row=3),
                ]),
            ])
        route = _route(q, list(values), "percent", "difference", years=[2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)

    def test_period_aware_year_ranking_applies_yoy_filter(self):
        q = ("Trong giai đoạn 2022-2024 của AAA, xét các năm doanh thu thuần "
             "tăng so với năm liền trước, ROE của năm có vòng quay tổng tài "
             "sản theo tài sản bình quân cao nhất là bao nhiêu phần trăm?")
        revenue = {2021: 100.0, 2022: 90.0, 2023: 120.0, 2024: 130.0}
        assets = {2021: 100.0, 2022: 100.0, 2023: 100.0, 2024: 200.0}
        tables = []
        for i, year in enumerate(revenue, 1):
            tables.append(_table(
                f"df{i}", f"AAA_financial_statements_{year}_consolidated", [
                    _row("Doanh thu thuần", revenue[year], code="10",
                         col_name=str(year), row=1),
                    _row("Tổng tài sản", assets[year], code="270",
                         col_name=str(year), row=2),
                    _row("Lợi nhuận sau thuế", 30.0, code="60",
                         col_name=str(year), row=3),
                    _row("Vốn chủ sở hữu", 150.0, code="400",
                         col_name=str(year), row=4),
                ]))
        route = _route(q, ["AAA"], "percent", "ranking",
                       years=[2022, 2023, 2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name,
                         "total_asset_turnover_average_assets")
        self.assertEqual(plan.predicates[0].calculation.mode, "yoy_delta")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)

    def test_period_aware_filter_then_mean_accrual_ratio(self):
        q = ("Trong ba mã AAA, BBB và CCC, với các công ty có lợi nhuận sau "
             "thuế dương năm 2020, bình quân của tỷ lệ chênh lệch giữa lợi "
             "nhuận sau thuế và CFO năm 2020 trên tổng tài sản bình quân năm "
             "2019 và 2020 là bao nhiêu phần trăm?")
        values = {
            "AAA": (100.0, 100.0, 20.0, 10.0),
            "BBB": (100.0, 100.0, -5.0, 0.0),
            "CCC": (200.0, 200.0, 30.0, 10.0),
        }
        tables = []
        for i, (ticker, (opening, closing, profit, cfo)) in enumerate(
                values.items(), 1):
            tables.extend([
                _table(f"df{i}a", f"{ticker}_financial_statements_2019_consolidated", [
                    _row("Tổng tài sản", opening, code="270",
                         col_name="2019", row=1),
                ]),
                _table(f"df{i}b", f"{ticker}_financial_statements_2020_consolidated", [
                    _row("Tổng tài sản", closing, code="270",
                         col_name="2020", row=1),
                    _row("Lợi nhuận sau thuế", profit, code="60",
                         col_name="2020", row=2),
                    _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                         col_name="2020", row=3),
                ]),
            ])
        route = _route(q, list(values), "percent", "difference",
                       years=[2019, 2020])

        plan = build_filter_aggregate_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.value.primary.match.spec.name,
                         "accrual_average_assets")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_period_aware_formula_fails_closed_without_opening_balance(self):
        q = ("Năm 2024, trong nhóm AAA và BBB, vòng quay tổng tài sản tính "
             "theo tổng tài sản bình quân của doanh nghiệp có tỷ trọng tài "
             "sản dài hạn trên tổng tài sản cao nhất là bao nhiêu vòng?")
        tables = [
            _table(f"df{i}", f"{ticker}_financial_statements_2024_consolidated", [
                _row("Tổng tài sản", 100.0, code="270",
                     col_name="2024", row=1),
                _row("Tài sản dài hạn", long_term, code="200",
                     col_name="2024", row=2),
                _row("Doanh thu thuần", 100.0, code="10",
                     col_name="2024", row=3),
            ])
            for i, (ticker, long_term) in enumerate(
                (("AAA", 80.0), ("BBB", 60.0)), 1)
        ]

        ca = try_formula_answer(
            _route(q, ["AAA", "BBB"], "ratio", "ranking", years=[2024]),
            tables)

        self.assertFalse(ca.ok)
        self.assertIn("unresolved", ca.detail)

    def test_period_aware_entity_ranking_checks_all_filter_years(self):
        q = ("Trong nhóm AAA, BBB và CCC, xét các công ty có CFO dương trong "
             "cả ba năm 2022, 2023 và 2024, công ty có tỷ lệ tăng doanh thu "
             "thuần năm 2024 so với năm 2023 cao nhất có tỷ lệ lợi nhuận sau "
             "thuế năm 2024 trên trung bình tổng tài sản cuối năm 2023 và "
             "cuối năm 2024 là bao nhiêu phần trăm?")
        values = {
            "AAA": ({2022: 1, 2023: 1, 2024: 1}, 100.0, 150.0, 100.0, 200.0, 15.0),
            "BBB": ({2022: 1, 2023: -1, 2024: 1}, 100.0, 200.0, 100.0, 100.0, 20.0),
            "CCC": ({2022: 1, 2023: 1, 2024: 1}, 100.0, 120.0, 100.0, 100.0, 20.0),
        }
        tables = []
        index = 1
        for ticker, (cfos, revenue_2023, revenue_2024,
                     assets_2023, assets_2024, profit) in values.items():
            for year in (2022, 2023, 2024):
                rows = [_row(
                    "Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfos[year],
                    col_name=str(year), row=1)]
                if year == 2023:
                    rows.extend([
                        _row("Doanh thu thuần", revenue_2023, code="10",
                             col_name="2023", row=2),
                        _row("Tổng tài sản", assets_2023, code="270",
                             col_name="2023", row=3),
                    ])
                if year == 2024:
                    rows.extend([
                        _row("Doanh thu thuần", revenue_2024, code="10",
                             col_name="2024", row=2),
                        _row("Tổng tài sản", assets_2024, code="270",
                             col_name="2024", row=3),
                        _row("Lợi nhuận sau thuế", profit, code="60",
                             col_name="2024", row=4),
                    ])
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", rows))
                index += 1
        route = _route(q, list(values), "percent", "growth_pct",
                       years=[2022, 2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_temporal_event_selects_first_negative_then_projects_next_year(self):
        q = ("Trong giai đoạn 2021-2023, biên lợi nhuận gộp của năm ngay sau "
             "năm đầu tiên AAA ghi nhận CFO âm là bao nhiêu phần trăm?")
        cfo = {2021: 10.0, 2022: -5.0, 2023: -2.0}
        gross = {2021: 10.0, 2022: 20.0, 2023: 30.0}
        tables = []
        for i, year in enumerate((2021, 2022, 2023), 1):
            report_id = f"AAA_financial_statements_{year}_consolidated"
            cfo_label = "Lưu chuyển tiền thuần từ hoạt động kinh doanh"
            if year == 2022:
                cfo_label = "Lưu chuyển tiền thuần từ hoạt độngkinh doanh"
            elif year == 2023:
                cfo_label = (
                    "Lưu chuyển tiền thuần sử dụng vào hoạt động kinh doanh")
            tables.append(_table(
                f"df{i * 2 - 1}", report_id, [
                    _row(cfo_label, cfo[year], code="20",
                         col_name=str(year), row=1),
                ], context="Báo cáo lưu chuyển tiền tệ"))
            tables.append(_table(
                f"df{i * 2}", report_id, [
                    _row("Lợi nhuận gộp", gross[year], code="20",
                         col_name=str(year), row=2),
                    _row("Doanh thu thuần", 100.0, code="10",
                         col_name=str(year), row=3),
                ], context="Báo cáo kết quả hoạt động kinh doanh"))
        route = _route(q, ["AAA"], "percent", "lookup",
                       years=[2021, 2022, 2023])

        plan = build_temporal_event_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.event.mode, "first")
        self.assertEqual(plan.projection.offset, 1)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 30.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 30.0)

    def test_temporal_event_argmin_projects_formula_at_next_year(self):
        q = ("AAA có hệ số dòng tiền hoạt động trên nợ ngắn hạn vào năm sau "
             "năm có hệ số thanh toán nhanh thấp nhất trong giai đoạn "
             "2021-2023 là bao nhiêu lần?")
        values = {
            2021: (100.0, 10.0, 100.0, 5.0),
            2022: (70.0, 20.0, 100.0, 10.0),
            2023: (120.0, 10.0, 100.0, 25.0),
        }
        tables = [
            _table(f"df{i}", f"AAA_financial_statements_{year}_consolidated", [
                _row("Tài sản ngắn hạn", current, code="100",
                     col_name=str(year), row=1),
                _row("Hàng tồn kho", inventory, code="140",
                     col_name=str(year), row=2),
                _row("Nợ ngắn hạn", liabilities, code="310",
                     col_name=str(year), row=3),
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                     col_name=str(year), row=4),
            ])
            for i, (year, (current, inventory, liabilities, cfo))
            in enumerate(values.items(), 1)
        ]
        route = _route(q, ["AAA"], "ratio", "ranking",
                       years=[2021, 2022, 2023])

        plan = build_temporal_event_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.axis, "year")
        self.assertEqual(plan.event.selector.match.spec.name, "quick_ratio")
        self.assertEqual(plan.projection.offset, 1)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 0.25)
        self.assertEqual(_eval(ca.pandas_query, tables), 0.25)

    def test_temporal_event_ranks_joint_entity_year_growth_grid(self):
        q = ("Trong nhóm AAA và BBB giai đoạn 2022-2023, công ty và năm có "
             "mức tăng tương đối của doanh thu thuần so với năm liền trước "
             "lớn nhất có vòng quay tổng tài sản là bao nhiêu lần?")
        values = {
            "AAA": {2021: (100.0, 50.0), 2022: (120.0, 60.0),
                    2023: (150.0, 75.0)},
            "BBB": {2021: (100.0, 50.0), 2022: (140.0, 70.0),
                    2023: (154.0, 77.0)},
        }
        tables = []
        index = 1
        for ticker, periods in values.items():
            for year, (revenue, assets) in periods.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Doanh thu thuần", revenue, code="10",
                             col_name=str(year), row=1),
                        _row("Tổng cộng tài sản", assets, code="270",
                             col_name=str(year), row=2),
                    ]))
                index += 1
        route = _route(q, ["AAA", "BBB"], "ratio", "ranking",
                       years=[2022, 2023])

        plan = build_temporal_event_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.axis, "entity_year")
        self.assertEqual(plan.event.selector.mode, "yoy_growth")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2.0)

    def test_temporal_event_does_not_leak_selector_growth_into_projection(self):
        q = ("Trong giai đoạn 2020-2024, ở năm có tốc độ tăng doanh thu "
             "thuần cao nhất trong các năm tăng trưởng dương, tỷ lệ CFO trên "
             "doanh thu thuần của năm đó là bao nhiêu phần trăm?")
        route = _route(q, ["AAA"], "percent", "ranking",
                       years=[2020, 2021, 2022, 2023, 2024])

        plan = build_temporal_event_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.event.selector.mode, "yoy_growth")
        self.assertEqual(plan.projection.calculation.mode, "level")

    def test_temporal_event_rejects_unparsed_target_before_selector(self):
        q = ("Trong các năm 2015, 2017, 2019, 2021, 2022 và 2023, giá trị "
             "mua hàng hóa và dịch vụ từ Công ty TNHH Coats Phong Phú của "
             "Tập đoàn Dệt May Việt Nam trong năm có vốn chủ sở hữu cuối năm "
             "cao nhất là bao nhiêu tỷ đồng?")
        route = _route(q, ["AAA"], "number", "ranking",
                       years=[2015, 2017, 2019, 2021, 2022, 2023])

        self.assertIsNone(build_temporal_event_plan(route))

    def test_temporal_event_defers_plain_year_ranking_to_existing_solver(self):
        q = ("Doanh thu thuần cao nhất của AAA trong giai đoạn 2021-2023 là "
             "bao nhiêu tỷ đồng?")
        route = _route(q, ["AAA"], "number", "ranking",
                       years=[2021, 2022, 2023])

        self.assertIsNone(build_temporal_event_plan(route))

    def test_temporal_event_fails_closed_without_next_year_projection(self):
        q = ("AAA có biên lợi nhuận gộp vào năm sau năm có CFO thấp nhất "
             "trong giai đoạn 2021-2022 là bao nhiêu phần trăm?")
        tables = [
            _table(f"df{i}", f"AAA_financial_statements_{year}_consolidated", [
                _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                     col_name=str(year), row=1),
                _row("Lợi nhuận gộp", 20.0, code="20",
                     col_name=str(year), row=2),
                _row("Doanh thu thuần", 100.0, code="10",
                     col_name=str(year), row=3),
            ])
            for i, (year, cfo) in enumerate(((2021, 10.0), (2022, 5.0)), 1)
        ]
        route = _route(q, ["AAA"], "percent", "ranking",
                       years=[2021, 2022])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)
        self.assertIn("projection unresolved", ca.detail)

    def test_matrix_v16_land_use_right_reads_net_closing_block(self):
        q = ("Giá trị còn lại của quyền sử dụng đất của AAA vào cuối năm "
             "2018 là bao nhiêu triệu đồng?")
        tables = [_matrix_table(
            "df1", "AAA_financial_statements_2018_consolidated", [
                ["", "Quyền sử dụng đất triệu đồng", "Tổng cộng triệu đồng"],
                ["Nguyên giá", "Nguyên giá", "Nguyên giá"],
                ["Số dư cuối năm", "1.075.116", "1.823.153"],
                ["Giá trị hao mòn luỹ kế"] * 3,
                ["Số dư cuối năm", "141.870", "709.991"],
                ["Giá trị còn lại"] * 3,
                ["Tại ngày cuối năm", "933.246", "1.113.162"],
            ], unit=1e6,
            context=("Biến động của tài sản cố định vô hình trong năm "
                     "2018 như sau"))]
        route = _route(q, ["AAA"], years=[2018])
        route["unit_scale"] = 1e6

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 933_246.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 933_246.0)

    def test_matrix_v16_named_subsidiary_ownership_uses_combined_header(self):
        q = ("Tỷ lệ sở hữu của công ty mẹ AAA tại Công ty TNHH Anh Vũ "
             "Phú Yên đến ngày 31/12/2025 là bao nhiêu phần trăm?")
        tables = [_matrix_table(
            "df1", "AAA_financial_statements_2025_separate", [
                ["Tên công ty con", "Tỷ lệ biểu quyết của Công ty",
                 "Tỷ lệ biểu quyết của Công ty", "Tỷ lệ sở hữu của Công ty",
                 "Tỷ lệ sở hữu của Công ty"],
                ["Tên công ty con", "Ngày 31 tháng 12 năm 2025",
                 "Ngày 31 tháng 12 năm 2024", "Ngày 31 tháng 12 năm 2025",
                 "Ngày 31 tháng 12 năm 2024"],
                ["Công ty TNHH Anh Vũ Phú Yên", "100,00%", "100,00%",
                 "98,50%", "95,00%"],
            ], context="Cơ cấu tổ chức các công ty con")]
        route = _route(q, ["AAA"], "percent", years=[2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 98.5)
        self.assertEqual(_eval(ca.pandas_query, tables), 98.5)

    def test_matrix_v16_closing_tax_difference_is_later_minus_earlier(self):
        q = ("Tính số chênh lệch số dư thuế TNDN phải nộp cuối kỳ của AAA "
             "giữa năm 2018 và năm 2017 là bao nhiêu tỷ đồng?")
        tables = []
        for index, (year, value) in enumerate(
                ((2017, 27_897_500_519), (2018, 56_426_836_190)), 1):
            tables.append(_matrix_table(
                f"df{index}", f"AAA_financial_statements_{year}_separate", [
                    ["", "Số đầu năm", "Số cuối năm"],
                    ["Thuế TNDN", "1", str(value)],
                ], context="Thuế và các khoản phải nộp Nhà nước"))
        route = _route(q, ["AAA"], years=[2018, 2017])
        route.update(doc_type="separate", unit_scale=1e9)

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 28.529335671)
        self.assertEqual(_eval(ca.pandas_query, tables), 28.53)

    def test_matrix_v16_borrowings_to_assets_sums_short_and_long_codes(self):
        q = "Tỷ số nợ vay trên tổng tài sản của công ty mẹ AAA năm 2022 là bao nhiêu %?"
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2022_separate", [
                ["Mã số", "TÀI SẢN", "Số cuối năm", "Số đầu năm"],
                ["270", "TỔNG CỘNG TÀI SẢN", "15000", "14000"],
            ], unit=1e3),
            _matrix_table("df2", "AAA_financial_statements_2022_separate", [
                ["Mã số", "NGUỒN VỐN", "Số cuối năm", "Số đầu năm"],
                ["320", "Vay ngắn hạn", "2000", "1000"],
                ["338", "Vay dài hạn", "3500", "4000"],
            ], table_pos=2, unit=1e3),
        ]
        route = _route(q, ["AAA"], "percent", years=[2022])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 36.6666666667)
        self.assertEqual(_eval(ca.pandas_query, tables), 36.67)

    def test_matrix_v16_nonperforming_loan_coverage_sums_groups_three_to_five(self):
        q = "Tỷ lệ bao phủ nợ xấu của công ty mẹ AAA cuối năm 2025 là bao nhiêu %?"
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2025_separate", [
                ["Chỉ tiêu", "31/12/2025", "31/12/2024"],
                ["Dự phòng rủi ro cho vay khách hàng", "1564", "1300"],
                ["Tổng tài sản Có", "100000", "90000"],
            ], unit=1e6),
            _matrix_table("df2", "AAA_financial_statements_2025_separate", [
                ["", "31/12/2025", "31/12/2024"],
                ["Nợ dưới tiêu chuẩn", "137", "100"],
                ["Nợ nghi ngờ", "137", "100"],
                ["Nợ có khả năng mất vốn", "1180", "1000"],
            ], table_pos=2, unit=1e6),
        ]
        route = _route(q, ["AAA"], "percent", years=[2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 1564 / 1454 * 100)
        self.assertEqual(_eval(ca.pandas_query, tables), 107.57)

    def test_matrix_v16_temporal_money_reductions_read_every_year(self):
        q = ("Dựa trên dữ liệu công ty mẹ AAA, khoản dự phòng phải thu khó "
             "đòi được trích lập trong năm đạt mức cao nhất trong các mốc "
             "2019, 2021 và 2025 là bao nhiêu tỷ đồng?")
        values = ((2019, 280_000_000_000), (2021, 6_000_000_000),
                  (2025, 7_000_000_000))
        tables = [
            _matrix_table(
                f"df{index}", f"AAA_financial_statements_{year}_separate", [
                    ["", "Năm nay", "Năm trước"],
                    ["Dự phòng trích lập trong năm", f"({value})", "1"],
                ], context="Dự phòng phải thu khó đòi")
            for index, (year, value) in enumerate(values, 1)
        ]
        route = _route(q, ["AAA"], years=[2019, 2021, 2025])
        route.update(doc_type="separate", unit_scale=1e9)

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 280.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 280.0)
        self.assertTrue(all(f"df{i}" in ca.pandas_query for i in range(1, 4)))

    def test_matrix_v16_tangible_fixed_asset_share_uses_only_222_and_223(self):
        q = ("Tính tỷ lệ hao mòn lũy kế trung bình của TSCĐ hữu hình năm "
             "2022 đối với AAA và BBB theo đơn vị phần trăm.")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2022_consolidated", [
                ["Mã số", "TÀI SẢN", "Tại ngày 31 tháng 12 năm",
                 "Tại ngày 31 tháng 12 năm"],
                ["Mã số", "TÀI SẢN", "2022 VND", "2021 VND"],
                ["221", "Tài sản cố định hữu hình", "60", "55"],
                ["222", "Nguyên giá", "100", "90"],
                ["223", "Giá trị khấu hao lũy kế", "(40)", "(35)"],
                ["228", "Nguyên giá tài sản vô hình", "900", "800"],
                ["229", "Giá trị hao mòn lũy kế", "(900)", "(800)"],
            ]),
            _matrix_table("df2", "BBB_financial_statements_2022_consolidated", [
                ["Mã số", "TÀI SẢN", "31/12/2022"],
                ["221", "Tài sản cố định hữu hình", "50"],
                ["222", "Nguyên giá", "100"],
                ["223", "Giá trị hao mòn lũy kế", "(50)"],
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", years=[2022])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 45.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 45.0)

    def test_matrix_v16_headerless_subsidiary_payable_uses_first_value_column(self):
        q = ("Số dư phải trả ngắn hạn khác với công ty con của công ty mẹ "
             "AAA vào cuối các năm 2021 và 2025. Giá trị lớn nhất trong các "
             "năm này là bao nhiêu tỷ đồng?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2021_separate", [
                ["Alpha", "Công ty con", "Hợp tác", "900000000000", "0"],
                ["", "", "", "1150000000000", "100000000000"],
            ], context="Phải trả ngắn hạn khác"),
            _matrix_table("df2", "AAA_financial_statements_2025_separate", [
                ["Alpha", "Công ty con", "Hợp tác", "500000000000", "0"],
                ["", "", "", "524000000000", "700000000000"],
            ], context="Phải trả ngắn hạn khác"),
        ]
        route = _route(q, ["AAA"], years=[2021, 2025])
        route.update(doc_type="separate", unit_scale=1e9)

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1150.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1150.0)

    def test_matrix_note_voting_rate_reads_only_subsidiary_block(self):
        q = ("Sự chênh lệch về tỷ lệ biểu quyết trung bình của các công ty con "
             "giữa AAA và BBB trong năm 2020 là bao nhiêu %?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2020_consolidated", [
                ["Công ty", "Tỷ lệ sở hữu và biểu quyết (%)"],
                ["Công ty con", "Công ty con"],
                ["AAA One", "100"], ["AAA Two", "80"],
                ["Công ty liên kết", "Công ty liên kết"],
                ["AAA Associate", "20"],
            ]),
            _matrix_table("df2", "BBB_financial_statements_2020_consolidated", [
                ["Công ty", "Tỷ lệ biểu quyết %"],
                ["Công ty con", "Công ty con"],
                ["BBB One", "60"], ["BBB Two", "80"],
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "difference",
                       years=[2020])

        plan = build_matrix_note_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.family, "subsidiary_voting_rate")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)
        self.assertIn("formula_matrix_note_v10", ca.detail)
        self.assertNotIn("Associate", ca.pandas_query)

    def test_matrix_v15_credit_risk_total_uses_current_year_table(self):
        q = ("Số dư tổng giá trị tài sản tài chính chịu rủi ro tín dụng của "
             "công ty mẹ Tập đoàn Bảo Việt (BVH) đến ngày 31 tháng 12 năm "
             "2018 là bao nhiêu tỷ đồng?")
        grid = [
            ["", "Chưa quá hạn", "Bị giảm giá trị", "Tổng cộng"],
            ["Tiền", "10.000.000.000", "-", "10.000.000.000"],
            ["Tổng", "90.000.000.000", "10.000.000.000", "100.000.000.000"],
        ]
        tables = [
            _matrix_table(
                "df1", "BVH_financial_statements_2018_separate", grid,
                table_pos=1,
                context=("Rủi ro tín dụng tại ngày 31 tháng 12 năm 2018 "
                         "như sau")),
            _matrix_table(
                "df2", "BVH_financial_statements_2018_separate", [
                    *grid[:2], ["Tổng", "180.000.000.000", "20.000.000.000",
                                "200.000.000.000"]],
                table_pos=2,
                context=("Rủi ro tín dụng tại ngày 31 tháng 12 năm 2017 "
                         "như sau")),
        ]
        route = _route(q, ["BVH"], years=[2018])
        route["doc_type"] = "separate"
        route["unit_scale"] = 1e9

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 100.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 100.0)
        self.assertNotIn("df2", ca.pandas_query)

    def test_matrix_v15_closing_usd_balance_keeps_currency_unit(self):
        q = "Số dư ngoại tệ USD của ACV vào cuối năm 2018 là bao nhiêu triệu USD?"
        tables = [_matrix_table(
            "df1", "ACV_financial_statements_2018_consolidated", [
                ["", "Số cuối năm", "Số đầu năm"],
                ["- Đô la Mỹ (USD)", "6.155.698,34", "6.579.341,29"],
                ["- Euro (EUR)", "-", "201,15"],
            ])]
        route = _route(q, ["ACV"], years=[2018])
        route["unit_scale"] = 1e6

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 6.15569834)
        self.assertEqual(_eval(ca.pandas_query, tables), 6.16)

    def test_matrix_v15_derivative_total_selects_contract_notional(self):
        q = ("Tổng giá trị hợp đồng công cụ phái sinh cuối năm 2022 của KLB "
             "là bao nhiêu triệu đồng?")
        tables = [_matrix_table(
            "df1", "KLB_financial_statements_2022_consolidated", [
                ["", "Tổng giá trị hợp đồng", "Tổng giá trị ghi sổ"],
                ["Số cuối năm", "Số cuối năm", "Số cuối năm"],
                ["Hoán đổi tiền tệ", "1.692.506", "21.876"],
                ["Giao dịch kỳ hạn", "1.388.270", "16.831"],
                ["Cộng", "3.080.776", "38.707"],
                ["Số đầu năm", "Số đầu năm", "Số đầu năm"],
                ["Cộng", "5.467.186", "6.036"],
            ], unit=1.0)]
        route = _route(q, ["KLB"], years=[2022])
        route["unit_scale"] = 1e6

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 3_080_776.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 3_080_776.0)

    def test_matrix_v15_fx_sensitivity_filters_net_liability_currencies(self):
        q = ("Vào năm 2016 tại Tập đoàn FPT, với các đồng tiền có công nợ "
             "tiền tệ cuối năm lớn hơn tài sản tiền tệ cuối năm, tổng mức "
             "giảm lợi nhuận trước thuế theo bảng độ nhạy nếu VND biến động "
             "bất lợi 5% là bao nhiêu tỷ đồng?")
        rows = [
            ["Đô la Mỹ (USD)", "200.000.000.000", "100", "100.000.000.000", "90"],
            ["Euro (EUR)", "50.000.000.000", "40", "80.000.000.000", "70"],
            ["Yên Nhật (JPY)", "20.000.000.000", "10", "50.000.000.000", "40"],
            ["Đô la Singapore (SGD)", "40.000.000.000", "30", "30.000.000.000", "20"],
        ]
        tables = [
            _matrix_table(
                "df1", "FPT_financial_statements_2016_consolidated", [
                    ["", "Công nợ", "Công nợ", "Tài sản", "Tài sản"],
                    ["", "Số cuối năm VND", "Số đầu năm VND",
                     "Số cuối năm VND", "Số đầu năm VND"], *rows], table_pos=1),
            _matrix_table(
                "df2", "FPT_financial_statements_2016_consolidated", [
                    ["", "Năm nay", "Năm trước"], ["", "VND", "VND"],
                    ["Đô la Mỹ (USD)", "(5.000.000.000)", "0"],
                    ["Euro (EUR)", "1.500.000.000", "0"],
                    ["Yên Nhật (JPY)", "2.000.000.000", "0"],
                    ["Đô la Singapore (SGD)", "(500.000.000)", "0"],
                ], table_pos=2, context="Phân tích độ nhạy với ngoại tệ"),
        ]
        route = _route(q, ["FPT"], years=[2015, 2016])
        route["unit_scale"] = 1e9

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 5.5)
        self.assertEqual(_eval(ca.pandas_query, tables), 5.5)
        self.assertIn("df1", ca.pandas_query)
        self.assertIn("df2", ca.pandas_query)

    def test_matrix_v15_fx_position_selects_worst_adverse_currency(self):
        q = ("Tại ngân hàng ACB, nếu VND giảm 5% so với từng đồng ngoại tệ "
             "và chỉ xét trạng thái tiền tệ nội, ngoại bảng đã công bố, đồng "
             "tiền gây bất lợi lớn nhất sẽ làm giảm LNTT theo kịch bản tương "
             "đương bao nhiêu phần trăm LNTT năm 2024?")
        tables = [
            _matrix_table(
                "df1", "ACB_financial_statements_2024_consolidated", [
                    ["", "USD", "Vàng", "EUR", "JPY", "AUD", "CAD", "Khác", "Tổng cộng"],
                    ["Tại ngày 31 tháng 12 năm 2024"] * 9,
                    ["Tổng tài sản", "100", "10", "20", "30", "40", "50", "60", "310"],
                    ["Tổng nợ phải trả", "130", "5", "10", "20", "30", "40", "50", "285"],
                    ["Trạng thái tiền tệ nội, ngoại bảng", "(1.200)", "5", "10", "20", "30", "40", "50", "(1.045)"],
                ], unit=1e6),
            _table(
                "df2", "ACB_financial_statements_2024_consolidated", [
                    _row("Lợi nhuận trước thuế", 2_000.0, code="50",
                         col_name="2024", unit=1e6),
                ], table_pos=2),
        ]
        route = _route(q, ["ACB"], "percent", "ranking",
                       years=[2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 3.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 3.0)
        self.assertIn("df1", ca.pandas_query)
        self.assertIn("df2", ca.pandas_query)

    def test_matrix_v15_var_growth_uses_total_one_day_row(self):
        q = ("Tốc độ tăng trưởng giá trị rủi ro (VaR) 1 ngày danh mục cổ "
             "phiếu niêm yết của công ty mẹ BVH trong năm 2019 là bao nhiêu "
             "phần trăm?")
        tables = []
        for index, (year, value) in enumerate(((2018, 100), (2019, 125)), 1):
            tables.append(_matrix_table(
                f"df{index}", f"BVH_financial_statements_{year}_separate", [
                    [f"Giá trị rủi ro của danh mục cổ phiếu niêm yết tại ngày 31 tháng 12 năm {year}",
                     "HOSE", "HNX", "Tổng"],
                    ["VaR (95%, 1 ngày)", "(40)", "(60)", f"({value})"],
                    ["VaR (95%, 1 tuần)", "(80)", "(120)", "(200)"],
                ], context="Rủi ro giá cổ phiếu"))
        route = _route(q, ["BVH"], "percent", "growth_pct",
                       years=[2018, 2019])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 25.0)
        self.assertIn("df1", ca.pandas_query)
        self.assertIn("df2", ca.pandas_query)

    def test_matrix_v15_fx_sensitivity_fails_when_axis_is_incomplete(self):
        q = ("Năm 2016 tại FPT, với các đồng tiền có công nợ tiền tệ cuối năm "
             "lớn hơn tài sản tiền tệ cuối năm, tổng mức giảm lợi nhuận trước "
             "thuế theo bảng độ nhạy nếu VND biến động bất lợi 5% là bao nhiêu?")
        tables = [_matrix_table(
            "df1", "FPT_financial_statements_2016_consolidated", [
                ["", "Công nợ", "Tài sản"],
                ["", "Số cuối năm VND", "Số cuối năm VND"],
                ["Đô la Mỹ (USD)", "200", "100"],
            ])]
        route = _route(q, ["FPT"], years=[2016])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)

    def test_matrix_note_real_estate_share_uses_table_total(self):
        q = ("Tính tỷ trọng trung bình dư nợ cho vay ngành bất động sản của "
             "OCB công ty mẹ tại ngày 31/12/2020 và 31/12/2025 là bao nhiêu %?")
        tables = [
            _matrix_table("df1", "OCB_financial_statements_2020_separate", [
                ["", "Số dư cuối năm VND", "Số dư đầu năm VND"],
                ["Hoạt động kinh doanh bất động sản", "2000", "1000"],
                ["Các ngành khác", "8000", "9000"],
                ["", "10000", "10000"],
            ]),
            _matrix_table("df2", "OCB_financial_statements_2025_separate", [
                ["", "31/12/2025 VND", "31/12/2024 VND"],
                ["Hoạt động kinh doanh bất động sản", "6000", "2000"],
                ["Các ngành khác", "14000", "8000"],
                ["", "20000", "10000"],
            ]),
        ]
        route = _route(q, ["OCB"], "percent", "average",
                       years=[2020, 2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 25.0)

    def test_matrix_note_fixed_asset_share_aggregates_row_codes(self):
        q = ("Giá trị trung bình của tỷ lệ hao mòn, khấu hao lũy kế trên "
             "nguyên giá tài sản cố định của AAA và BBB ở công ty mẹ vào "
             "năm 2025 là bao nhiêu phần trăm?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2025_separate", [
                ["", "TÀI SẢN", "Mã số", "Số cuối năm"],
                ["", "Tài sản cố định", "220", "60"],
                ["", "- Nguyên giá", "222", "100"],
                ["", "- Giá trị hao mòn lũy kế", "223", "(40)"],
            ]),
            _matrix_table("df2", "BBB_financial_statements_2025_separate", [
                ["", "TÀI SẢN", "Mã số", "31/12/2025"],
                ["", "Tài sản cố định", "220", "100"],
                ["", "- Nguyên giá", "222", "100"],
                ["", "- Giá trị hao mòn lũy kế", "223", "(50)"],
                ["", "- Nguyên giá", "228", "100"],
                ["", "- Giá trị hao mòn lũy kế", "229", "(50)"],
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "average",
                       years=[2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 45.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 45.0)

    def test_matrix_note_transport_segment_year_ranking(self):
        q = ("Trong các năm 2022, 2023 và 2025, năm nào PVT công ty mẹ có "
             "tỷ trọng tài sản bộ phận dịch vụ vận tải so với tổng tài sản "
             "cao nhất?")
        tables = []
        for index, (year, segment, total) in enumerate(
                ((2022, 4000, 10000), (2023, 7000, 10000),
                 (2025, 6000, 10000)), 1):
            tables.append(_matrix_table(
                f"df{index}", f"PVT_financial_statements_{year}_separate", [
                    ["Chỉ tiêu", "Dịch vụ vận tải", "Tổng"],
                    ["Tài sản bộ phận", str(segment), str(segment)],
                    ["Tài sản không phân bổ", "", str(total - segment)],
                    ["Tổng tài sản", "", str(total)],
                    ["Số đầu năm", "", ""],
                    ["Tài sản bộ phận", "99", "99"],
                ]))
        route = _route(q, ["PVT"], "year", "ranking",
                       years=[2022, 2023, 2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)
        self.assertTrue(all(f"df{i}" in ca.pandas_query for i in range(1, 4)))

    def test_matrix_note_bot_share_selects_closing_table(self):
        q = ("Giá trị trung bình của tỷ trọng tài sản bộ phận BOT trên tổng "
             "tài sản hợp nhất tại HHV qua các năm 2021 và 2022 là bao nhiêu %?")
        tables = []
        for index, (year, segment) in enumerate(
                ((2021, 8000), (2022, 9000)), 1):
            tables.extend([
                _matrix_table(
                    f"df{index * 2 - 1}",
                    f"HHV_financial_statements_{year}_consolidated", [
                        [f"01/01/{year}", "Dự án BOT", "Tổng cộng"],
                        ["Tài sản bộ phận", "1000", "1000"],
                        ["Tổng tài sản", "1000", "10000"],
                    ], table_pos=1),
                _matrix_table(
                    f"df{index * 2}",
                    f"HHV_financial_statements_{year}_consolidated", [
                        [f"31/12/{year}", "Dự án BOT", "Tổng cộng"],
                        ["Tài sản bộ phận", str(segment), str(segment)],
                        ["Tổng tài sản", str(segment), "10000"],
                    ], table_pos=2),
            ])
        route = _route(q, ["HHV"], "percent", "average",
                       years=[2021, 2022])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 85.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 85.0)
        self.assertNotIn("df1", ca.pandas_query)
        self.assertNotIn("df3", ca.pandas_query)

    def test_matrix_note_fails_closed_on_conflicting_duplicate_table(self):
        q = ("Sự chênh lệch về tỷ lệ biểu quyết trung bình của các công ty con "
             "giữa AAA và BBB trong năm 2020 là bao nhiêu %?")
        base = [["Công ty", "Tỷ lệ biểu quyết %"],
                ["Công ty con", "Công ty con"], ["One", "80"]]
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2020_consolidated",
                          base, table_pos=1),
            _matrix_table("df2", "AAA_financial_statements_2020_consolidated",
                          [*base[:2], ["One", "90"]], table_pos=2),
            _matrix_table("df3", "BBB_financial_statements_2020_consolidated",
                          base, table_pos=1),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "difference",
                       years=[2020])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)

    def test_note_detail_direct_named_child_row(self):
        q = ("Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai của công ty "
             "mẹ HNG cuối năm 2017 là bao nhiêu nghìn đồng?")
        tables = [_table(
            "df1", "HNG_financial_statements_2017_separate", [
                _row("Công ty Cổ phần Hoàng Anh Gia Lai Công ty mẹ Vay dài hạn",
                     1_957_824_733.0, col_name="2017", unit=1e3),
            ], context="Vay dài hạn với bên liên quan")]
        route = _route(q, ["HNG"], years=[2017])
        route.update(doc_type="separate", unit_scale=1e3)

        plan = build_note_detail_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.reduction, "direct")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1_957_824_733.0)
        self.assertEqual(_eval(ca.pandas_query, tables), ca.answer)
        self.assertIn("formula_note_detail_v9", ca.detail)

    def test_note_detail_growth_reads_both_exact_periods(self):
        q = ("Đến ngày 31/12/2022, giá trị còn lại tài sản cố định vô hình "
             "của OCB tăng bao nhiêu % so với ngày 31/12/2020?")
        tables = [
            _table("df1", "OCB_financial_statements_2020_consolidated", [
                _row("Tài sản cố định vô hình", 100.0, code="227",
                     col_name="2020", unit=1.0),
            ]),
            _table("df2", "OCB_financial_statements_2022_consolidated", [
                _row("Tài sản cố định vô hình", 125.0, code="227",
                     col_name="2022", unit=1.0),
            ]),
        ]
        route = _route(q, ["OCB"], "percent", "growth_pct",
                       years=[2020, 2022])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 25.0)
        self.assertTrue(all(var in ca.pandas_query for var in ("df1", "df2")))

    def test_note_detail_ratio_with_sum_denominator(self):
        q = ("Tổng nợ vay gấp tổng tiền mặt và tiền gửi ngân hàng của HNG "
             "cuối năm 2020 bao nhiêu lần?")
        tables = [_table(
            "df1", "HNG_financial_statements_2020_consolidated", [
                _row("Tổng nợ vay", 300.0, col_name="2020", unit=1.0, row=1),
                _row("Tiền mặt", 20.0, col_name="2020", unit=1.0, row=2),
                _row("Tiền gửi ngân hàng", 80.0, col_name="2020", unit=1.0,
                     row=3),
            ], context="Vay và nợ; Tiền và các khoản tương đương tiền")]
        route = _route(q, ["HNG"], "ratio", "ratio", years=[2020])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 3.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 3.0)

    def test_note_detail_difference_of_parent_child_ratios(self):
        q = ("Trong năm 2021, tỷ trọng chi phí lãi tiền gửi của AAA và BBB "
             "chênh lệch nhau bao nhiêu phần trăm?")
        tables = [
            _table("df1", "AAA_financial_statements_2021_consolidated", [
                _row("Trả lãi tiền gửi", 40.0, col_name="2021", unit=1.0,
                     row=1),
                _row("Chi phí lãi và các chi phí tương tự", 100.0,
                     col_name="2021", unit=1.0, row=2),
            ], context="Chi phí lãi"),
            _table("df2", "BBB_financial_statements_2021_consolidated", [
                _row("Trả lãi tiền gửi", 30.0, col_name="2021", unit=1.0,
                     row=1),
                _row("Chi phí lãi và các chi phí tương tự", 100.0,
                     col_name="2021", unit=1.0, row=2),
            ], context="Chi phí lãi"),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "difference",
                       years=[2021])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_note_detail_mean_ratio_across_entities(self):
        q = ("Mức trung bình của tỷ trọng thành phẩm trong tổng giá trị hàng "
             "tồn kho của AAA và BBB cuối năm 2020 là bao nhiêu phần trăm?")
        tables = [
            _table("df1", "AAA_financial_statements_2020_consolidated", [
                _row("Thành phẩm", 20.0, col_name="2020", unit=1.0, row=1),
                _row("Hàng tồn kho", 100.0, code="141", col_name="2020",
                     unit=1.0, row=2),
            ], context="Hàng tồn kho"),
            _table("df2", "BBB_financial_statements_2020_consolidated", [
                _row("Thành phẩm", 60.0, col_name="2020", unit=1.0, row=1),
                _row("Hàng tồn kho", 100.0, code="141", col_name="2020",
                     unit=1.0, row=2),
            ], context="Hàng tồn kho"),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "average", years=[2020])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 40.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 40.0)

    def test_note_detail_sum_across_years(self):
        q = ("Tổng chi phí khấu hao bất động sản đầu tư của AAA trong giai "
             "đoạn từ 2023 đến 2024 là bao nhiêu tỷ đồng?")
        tables = [
            _table("df1", "AAA_financial_statements_2023_consolidated", [
                _row("Khấu hao", 2.0, col_name="2023", row=1),
            ], context="Bất động sản đầu tư"),
            _table("df2", "AAA_financial_statements_2024_consolidated", [
                _row("Khấu hao", 3.0, col_name="2024", row=1),
            ], context="Bất động sản đầu tư"),
        ]
        route = _route(q, ["AAA"], "number", "sum", years=[2023, 2024])
        route["unit_scale"] = 1e9

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 5.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 5.0)

    def test_nested_ranking_recognizes_short_cfo_net_profit_wording(self):
        q = ("Trong các công ty AAA và BBB có lưu chuyển tiền thuần từ hoạt "
             "động kinh doanh và lợi nhuận sau thuế năm 2024 đều dương, hệ "
             "số thanh toán nhanh của doanh nghiệp có tỷ lệ dòng tiền kinh "
             "doanh trên lợi nhuận sau thuế lớn nhất là bao nhiêu lần?")
        route = _route(q, ["AAA", "BBB"], "ratio", "ranking", years=[2024])

        plan = build_compositional_ranking_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.selector.match.spec.name, "cfo_net_profit")
        self.assertEqual(plan.projection.match.spec.name, "quick_ratio")
        self.assertEqual(
            {predicate.calculation.match.spec.name for predicate in plan.predicates},
            {"net_profit"},
        )

    def test_temporal_projection_keeps_punctuated_inventory_child_metric(self):
        q = ("Trong các năm 2015, 2016 và 2017, giá gốc nguyên liệu, vật liệu "
             "cuối năm của AAA tại năm có tổng giá gốc hàng tồn kho cuối năm "
             "cao nhất là bao nhiêu tỷ đồng?")
        route = _route(q, ["AAA"], years=[2015, 2016, 2017], op="ranking")

        plan = build_temporal_event_plan(route)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.event.selector.match.spec.name,
                         "metric:inventory_gross")
        self.assertEqual(plan.projection.calculation.match.spec.name,
                         "metric:raw_material_inventory_gross")

    def test_note_detail_year_argmax_reads_every_child_row(self):
        q = ("Năm nào AAA có giá trị hàng hóa tồn kho cuối kỳ cao nhất trong "
             "các năm 2022, 2023 và 2024?")
        tables = [
            _table(f"df{index}",
                   f"AAA_financial_statements_{year}_consolidated", [
                       _row("Hàng hóa", value, col_name=str(year), unit=1.0),
                   ], context="Hàng tồn kho")
            for index, (year, value) in enumerate(
                ((2022, 10.0), (2023, 30.0), (2024, 20.0)), 1)
        ]
        route = _route(q, ["AAA"], "year", "ranking",
                       years=[2022, 2023, 2024])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)
        self.assertTrue(all(f"df{i}" in ca.pandas_query for i in range(1, 4)))

    def test_note_axis_related_party_subtotal_growth(self):
        q = ("Tổng số dư phải trả người bán ngắn hạn với bên liên quan của "
             "AAA từ cuối năm 2020 đến cuối năm 2021 thay đổi bao nhiêu phần trăm?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2020_separate", [
                ["Bên liên quan", "Số cuối năm"],
                ["Phải trả người bán ngắn hạn", "Phải trả người bán ngắn hạn"],
                ["TỔNG CỘNG", "100.000.000"],
            ]),
            _matrix_table("df2", "AAA_financial_statements_2021_separate", [
                ["Bên liên quan", "Số cuối năm"],
                ["Phải trả người bán ngắn hạn", "Phải trả người bán ngắn hạn"],
                ["TỔNG CỘNG", "75.000.000"],
            ]),
        ]
        route = _route(q, ["AAA"], "percent", "growth_pct",
                       years=[2020, 2021])
        route["doc_type"] = "separate"

        plan = build_note_axis_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.family, "related_party_trade_payables")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, -25.0)
        self.assertEqual(_eval(ca.pandas_query, tables), -25.0)

    def test_note_axis_annual_interest_share_difference_keeps_first_value(self):
        q = ("Trong năm 2021, tỷ trọng chi phí lãi tiền gửi của AAA và BBB "
             "chênh lệch nhau bao nhiêu phần trăm?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2021_consolidated", [
                ["", "Năm 2021", "Năm 2020"],
                ["Trả lãi tiền gửi", "80", "70"],
                ["Tổng cộng", "100", "100"],
            ]),
            _matrix_table("df2", "BBB_financial_statements_2021_consolidated", [
                ["", "Năm 2021", "Năm 2020"],
                ["Trả lãi tiền gửi", "60", "50"],
                ["Tổng cộng", "100", "100"],
            ]),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "difference",
                       years=[2021])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)
        self.assertTrue(all("Năm 2021" in t["csv_text"] for t in tables))

    def test_note_axis_inventory_dash_is_proven_by_subtotal(self):
        q = ("Năm nào AAA có giá trị hàng hóa tồn kho cuối kỳ cao nhất trong "
             "các năm 2024 và 2025?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2024_separate", [
                ["", "Số cuối năm", "Số đầu năm"],
                ["Hàng mua đang đi trên đường", "10.000.000", "0"],
                ["Nguyên liệu, vật liệu tồn kho", "20.000.000", "0"],
                ["Công cụ, dụng cụ", "5.000.000", "0"],
                ["Chi phí sản xuất kinh doanh dở dang", "30.000.000", "0"],
                ["Thành phẩm", "15.000.000", "0"],
                ["Hàng hóa", "20.000.000", "0"],
                ["TỔNG CỘNG", "100.000.000", "0"],
            ], context="8. Hàng tồn kho"),
            _matrix_table("df2", "AAA_financial_statements_2025_separate", [
                ["", "Số cuối năm", "Số đầu năm"],
                ["Hàng mua đang đi trên đường", "10.000.000", "0"],
                ["Nguyên liệu, vật liệu tồn kho", "20.000.000", "0"],
                ["Công cụ, dụng cụ", "5.000.000", "0"],
                ["Chi phí sản xuất kinh doanh dở dang", "30.000.000", "0"],
                ["Thành phẩm", "15.000.000", "0"],
                ["Hàng hóa", "-", "20.000.000"],
                ["TỔNG CỘNG", "80.000.000", "100.000.000"],
            ], context="9. Hàng tồn kho"),
        ]
        route = _route(q, ["AAA"], "year", "ranking", years=[2024, 2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2024.0)
        self.assertIn("df2", ca.pandas_query)

    def test_note_axis_financial_reserve_mean_uses_row_and_columns(self):
        q = ("Xác định phần trăm tỷ trọng quỹ dự phòng tài chính trong vốn chủ "
             "sở hữu cuối kỳ của AAA trung bình qua các năm 2023 và 2025.")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2023_consolidated", [
                ["", "Quỹ dự phòngtài chính", "Tổng cộng"],
                ["Tại ngày 31 tháng 12 năm 2023", "10.000.000", "100.000.000"],
            ]),
            _matrix_table("df2", "AAA_financial_statements_2025_consolidated", [
                ["", "Quỹ dự phòng tài chính", "Tổng cộng"],
                ["Tại ngày 31 tháng 12 năm 2025", "30.000.000", "100.000.000"],
            ]),
        ]
        route = _route(q, ["AAA"], "percent", "average", years=[2023, 2025])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)

    def test_note_axis_current_tax_dash_uses_total_minus_deferred(self):
        q = ("Trong số các năm 2024 và 2025, chi phí thuế TNDN hiện hành của "
             "AAA đạt mức cao nhất vào năm nào?")
        tables = [
            _matrix_table("df1", "AAA_financial_statements_2024_separate", [
                ["Mã số", "CHỈ TIÊU", "Năm nay"],
                ["51", "Chi phí thuế TNDN hiện hành", "10"],
                ["60", "Lợi nhuận sau thuế TNDN", "100"],
            ]),
            _matrix_table("df2", "AAA_financial_statements_2025_separate", [
                ["", "Năm nay", "Năm trước"],
                ["Chi phí thuế TNDN hiện hành", "-", "10"],
                ["Thu nhập thuế TNDN hoãn lại", "(5)", "(4)"],
                ["TỔNG CỘNG", "(5)", "6"],
            ]),
        ]
        route = _route(q, ["AAA"], "year", "ranking", years=[2024, 2025])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2024.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2024.0)
        self.assertIn("df2", ca.pandas_query)

    def test_lease_schedule_direct_total_selects_lessor_table(self):
        q = "Tổng cam kết cho thuê hoạt động của AAA cuối năm 2024 là bao nhiêu?"
        tables = [
            _matrix_table(
                "df1", "AAA_financial_statements_2024_separate", [
                    ["", "Số cuối năm"],
                    ["Đến 1 năm", "20.000.000"],
                    ["Trên 1 đến 5 năm", "30.000.000"],
                    ["TỔNG CỘNG", "50.000.000"],
                ], table_pos=1,
                context="Công ty thuê đất, tiền thuê phải trả trong tương lai"),
            _matrix_table(
                "df2", "AAA_financial_statements_2024_separate", [
                    ["", "Số cuối năm"],
                    ["Đến 1 năm", "40.000.000"],
                    ["Trên 1 đến 5 năm", "60.000.000"],
                    ["TỔNG CỘNG", "100.000.000"],
                ], table_pos=2,
                context="Cam kết cho thuê hoạt động, Công ty cho thuê văn phòng"),
        ]
        route = _route(q, ["AAA"], years=[2024])
        route["doc_type"] = "separate"

        plan = build_lease_schedule_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.direction, "receivable")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 100000000.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 100000000.0)
        self.assertIn("df2", ca.pandas_query)
        self.assertNotIn("df1", ca.pandas_query)

    def test_select_project_related_receivables_to_lease_total(self):
        q = ("Trong các năm 2020 và 2021, tổng tiền thuê tối thiểu phải trả "
             "là bao nhiêu tỷ đồng tại năm có số dư cuối năm của các khoản "
             "phải thu ngắn hạn khác từ các bên liên quan lớn nhất?")
        tables = [
            _matrix_table(
                "df1", "AAA_financial_statements_2020_consolidated", [
                    ["", "31/12/2020VND", "1/1/2020VND"],
                    ["Công ty TNHH Coats Phong Phú", "60.000.000.000", "-"],
                    ["Các công ty liên quan khác", "40.000.000.000", "-"],
                    ["", "100.000.000.000", "-"],
                ], table_pos=1,
                context=("Các khoản phải thu ngắn hạn khác từ các bên liên "
                         "quan như sau")),
            _matrix_table(
                "df2", "AAA_financial_statements_2021_consolidated", [
                    ["", "31/12/2021VND", "1/1/2021VND"],
                    ["Công ty TNHH Coats Phong Phú", "50.000.000.000", "-"],
                    ["Các công ty liên quan khác", "30.000.000.000", "-"],
                    ["", "80.000.000.000", "-"],
                ], table_pos=2,
                context=("Các khoản phải thu ngắn hạn khác từ các bên liên "
                         "quan như sau")),
            _matrix_table(
                "df3", "AAA_financial_statements_2020_consolidated", [
                    ["", "31/12/2020VND", "1/1/2020VND"],
                    ["Trong vòng một năm", "40.000.000.000", "-"],
                    ["Trong vòng hai đến năm năm", "60.000.000.000", "-"],
                    ["", "100.000.000.000", "-"],
                ], table_pos=3,
                context="Tiền thuê tối thiểu phải trả theo hợp đồng thuê hoạt động"),
        ]
        route = _route(q, ["AAA"], "number", "ranking", years=[2020, 2021])
        route["unit_scale"] = 1e9

        plan = build_select_project_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.family, "related_receivables_to_lease_total")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 100.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 100.0)
        self.assertIn("df1", ca.pandas_query)
        self.assertIn("df2", ca.pandas_query)
        self.assertIn("df3", ca.pandas_query)

    def test_select_project_uses_lexicographic_year_tie_breaker(self):
        q = ("Trong các năm 2020 và 2023, năm nào có tổng giá gốc nợ phải "
             "thu quá hạn lớn nhất trong số các năm mà giá gốc khoản đầu tư "
             "vào Công ty TNHH MTV Thương mại Thành Phát đạt mức cao nhất?")
        tables = []
        for index, year in enumerate((2020, 2023), start=1):
            tables.append(_matrix_table(
                f"df{index}", f"AAA_financial_statements_{year}_separate", [
                    ["TÀI SẢN", "Mã số", "Thuyết minh", f"31/12/{year} VND"],
                    ["1. Đầu tư vào công ty con", "251", "", "800.000.000.000"],
                ], table_pos=index))
            overdue = "20.000.000.000" if year == 2020 else "25.000.000.000"
            tables.append(_matrix_table(
                f"df{index + 2}", f"AAA_financial_statements_{year}_separate", [
                    ["Nợ phải thu quá hạn", f"31/12/{year}", f"31/12/{year}"],
                    ["Nợ phải thu quá hạn", "Giá gốc", "Giá trị có thể thu hồi"],
                    ["Khách hàng A", overdue, "-"],
                    ["Cộng", overdue, "-"],
                ], table_pos=index + 2))
        route = _route(q, ["AAA"], "year", "ranking", years=[2020, 2023])
        route["doc_type"] = "separate"

        plan = build_select_project_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.tie_breaker, "max")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2023.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2023.0)

    def test_select_project_scopes_projection_to_company_block(self):
        q = ("Trong các năm 2021 và 2022, giá trị mua hàng hóa và dịch vụ "
             "từ Công ty TNHH Coats Phong Phú trong năm có vốn chủ sở hữu "
             "cuối năm cao nhất là bao nhiêu tỷ đồng?")
        tables = [
            _matrix_table(
                "df1", "AAA_financial_statements_2021_consolidated", [
                    ["", "Mã số", "31/12/2021 VND"],
                    ["Vốn chủ sở hữu", "410", "9.000.000.000.000"],
                ]),
            _matrix_table(
                "df2", "AAA_financial_statements_2022_consolidated", [
                    ["", "Mã số", "31/12/2022 VND"],
                    ["Vốn chủ sở hữu", "410", "10.000.000.000.000"],
                ]),
            _matrix_table(
                "df3", "AAA_financial_statements_2022_consolidated", [
                    ["", "Giá trị giao dịch", "Giá trị giao dịch"],
                    ["", "2022VND", "2021VND"],
                    ["Công ty TNHH Coats Phong Phú", "", ""],
                    ["Mua hàng hóa và dịch vụ", "217.000.000.000", "200.000.000.000"],
                    ["Tổng Công ty May Nhà Bè - CTCP", "", ""],
                    ["Mua hàng hóa và dịch vụ", "999.000.000.000", "900.000.000.000"],
                ], table_pos=3),
        ]
        route = _route(q, ["AAA"], "number", "ranking", years=[2021, 2022])
        route["unit_scale"] = 1e9

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 217.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 217.0)

    def test_select_project_accepts_docless_lease_reports(self):
        q = ("Số tiền thuê tối thiểu phải trả trong vòng một năm theo các hợp "
             "đồng thuê hoạt động tại năm có Thuế tính theo thuế suất của "
             "Công ty lớn nhất trong các năm 2019 và 2020 là bao nhiêu tỷ đồng?")
        tables = []
        for index, (year, tax) in enumerate(((2019, 100), (2020, 150)), start=1):
            tax_table = _matrix_table(
                f"df{index}", f"AAA_financial_statements_{year}_consolidated", [
                    ["", f"{year}VND"],
                    ["Thuế tính theo thuế suất của Công ty", f"{tax}.000.000.000"],
                ], table_pos=index)
            tax_table["report_id"] = f"AAA_financial_statements_{year}"
            tables.append(tax_table)
        lease = _matrix_table(
            "df3", "AAA_financial_statements_2020_consolidated", [
                ["", "31/12/2020VND"],
                ["Trong vòng một năm", "10.000.000.000"],
                ["Từ hai đến năm năm", "40.000.000.000"],
                ["Sau năm năm", "50.000.000.000"],
                ["", "100.000.000.000"],
            ], table_pos=3,
            context="Tiền thuê tối thiểu phải trả theo hợp đồng thuê hoạt động")
        lease["report_id"] = "AAA_financial_statements_2020"
        tables.append(lease)
        route = _route(q, ["AAA"], "number", "ranking", years=[2019, 2020])
        route["unit_scale"] = 1e9

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_lease_schedule_growth_uses_short_term_payable_bucket(self):
        q = ("Tăng trưởng cam kết thuê hoạt động ngắn hạn từ 1 năm trở xuống "
             "của AAA từ cuối năm 2022 đến cuối năm 2025 là bao nhiêu %?")
        tables = [
            _matrix_table(
                "df1", "AAA_financial_statements_2022_separate", [
                    ["", "Số cuối năm"],
                    ["Từ 1 năm trở xuống", "25.000.000"],
                    ["Trên 1 năm đến 5 năm", "75.000.000"],
                    ["", "100.000.000"],
                ], context="Công ty thuê đất, tiền thuê tối thiểu phải trả"),
            _matrix_table(
                "df2", "AAA_financial_statements_2025_separate", [
                    ["", "Số cuối năm"],
                    ["Từ 1 năm trở xuống", "30.000.000"],
                    ["Trên 1 năm đến 5 năm", "70.000.000"],
                    ["", "100.000.000"],
                ], context="Cam kết thuê hoạt động, Công ty thuê đất"),
        ]
        route = _route(q, ["AAA"], "percent", "growth_pct",
                       years=[2022, 2025])
        route["doc_type"] = "separate"

        plan = build_lease_schedule_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.value_axis, "short_term")
        self.assertEqual(plan.direction, "payable")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)

    def test_lease_schedule_direct_total_for_unqualified_rented_assets(self):
        q = ("Tổng số tiền thuê tối thiểu trong tương lai của công ty mẹ AAA "
             "đến ngày 31 tháng 12 năm 2025 là bao nhiêu tỷ đồng?")
        table = _matrix_table(
            "df1", "AAA_financial_statements_2025_separate", [
                ["", "Số cuối nămVND", "Số đầu nămVND"],
                ["Trong vòng 1 năm", "17.404.125.550", "16.142.170.183"],
                ["Từ hai đến năm năm", "69.616.502.200", "64.568.680.732"],
                ["Sau năm năm", "300.635.726.790", "297.337.156.006"],
                ["", "387.656.354.540", "378.048.006.921"],
            ], context="Các khoản mục ngoài bảng cân đối kế toán - Tài sản thuê ngoài")
        table["report_id"] = "AAA_financial_statements_2025"
        route = _route(q, ["AAA"], "number", "lookup", years=[2025])
        route.update(doc_type="separate", unit_scale=1e9)

        plan = build_lease_schedule_plan(route)
        ca = try_formula_answer(route, [table])

        self.assertEqual(plan.direction, "payable")
        self.assertEqual(plan.value_axis, "total")
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 387.65635454)
        self.assertEqual(_eval(ca.pandas_query, [table]), 387.66)

    def test_lease_schedule_mean_short_term_share(self):
        q = ("Trung bình tỷ lệ phần trăm khoản tiền thuê phải thu trong tương "
             "lai đến hạn dưới 1 năm so với tổng giá trị tiền thuê trong tương "
             "lai phải thu của AAA và BBB năm 2017 là bao nhiêu phần trăm?")
        tables = [
            _matrix_table(
                "df1", "AAA_financial_statements_2017_consolidated", [
                    ["", "Năm nay"],
                    ["Dưới một năm", "40.000.000"],
                    ["Từ một đến năm năm", "60.000.000"],
                    ["TỔNG CỘNG", "100.000.000"],
                ], context="Bên cho thuê, tiền thuê phải thu trong tương lai"),
            _matrix_table(
                "df2", "BBB_financial_statements_2017_consolidated", [
                    ["", "Năm nay"],
                    ["Dưới 1 năm", "20.000.000"],
                    ["Trên 1 đến 5 năm", "80.000.000"],
                    ["Cộng", "100.000.000"],
                ], context="Công ty cho thuê tài sản"),
        ]
        route = _route(q, ["AAA", "BBB"], "percent", "average", years=[2017])

        plan = build_lease_schedule_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.value_axis, "short_term_share")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 30.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 30.0)

    def test_lease_schedule_max_total_reads_every_year(self):
        q = ("Tổng tiền thuê tối thiểu phải nhận theo hợp đồng thuê hoạt động "
             "lớn nhất của AAA trong các năm 2022 và 2025 là bao nhiêu?")
        tables = [
            _matrix_table(
                "df1", "AAA_financial_statements_2022_consolidated", [
                    ["", "Số cuối năm"],
                    ["Đến 1 năm", "40.000.000"],
                    ["Trên 1 đến 5 năm", "60.000.000"],
                    ["TỔNG CỘNG", "100.000.000"],
                ], context="Tập đoàn là bên cho thuê"),
            _matrix_table(
                "df2", "AAA_financial_statements_2025_consolidated", [
                    ["", "Số cuối năm"],
                    ["Dưới 1 năm", "50.000.000"],
                    ["Trên 1 đến 5 năm", "100.000.000"],
                    ["TỔNG CỘNG", "150.000.000"],
                ], context="Tập đoàn cho thuê văn phòng"),
        ]
        route = _route(q, ["AAA"], "number", "ranking", years=[2022, 2025])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 150000000.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 150000000.0)
        self.assertIn("df1", ca.pandas_query)
        self.assertIn("df2", ca.pandas_query)

    def test_lease_schedule_fails_closed_when_subtotal_does_not_reconcile(self):
        q = "Tổng cam kết cho thuê hoạt động của AAA cuối năm 2024 là bao nhiêu?"
        tables = [_matrix_table(
            "df1", "AAA_financial_statements_2024_consolidated", [
                ["", "Số cuối năm"],
                ["Đến 1 năm", "40.000.000"],
                ["Trên 1 đến 5 năm", "60.000.000"],
                ["TỔNG CỘNG", "120.000.000"],
            ], context="Cam kết cho thuê hoạt động")]
        route = _route(q, ["AAA"], years=[2024])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)

    def test_note_detail_ratio_fails_closed_without_parent(self):
        q = ("Tỷ trọng dự phòng chung trong tổng dự phòng rủi ro cho vay "
             "khách hàng của AAA cuối năm 2018 là bao nhiêu phần trăm?")
        tables = [_table(
            "df1", "AAA_financial_statements_2018_consolidated", [
                _row("Dự phòng chung", 40.0, col_name="2018", unit=1.0),
            ], context="Dự phòng rủi ro cho vay khách hàng")]
        route = _route(q, ["AAA"], "percent", "ratio", years=[2018])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)
        self.assertIn("unresolved", ca.detail)

    def test_matrix_v17_land_rental_cogs_projects_winning_year(self):
        q = ("Năm nào trong giai đoạn 2019 và 2022, KBC có tỷ trọng giá vốn "
             "cho thuê dài hạn đất và cơ sở hạ tầng trên tổng giá vốn hàng "
             "bán và dịch vụ cung cấp cao nhất?")
        tables = [
            _matrix_table("df1", "KBC_financial_statements_2019_consolidated", [
                ["", "Năm nay"],
                ["Giá vốn cho thuê đất và cơ sở hạ tầng cho thuê", "80.000.000"],
                ["TỔNG CỘNG", "100.000.000"],
            ]),
            _matrix_table("df2", "KBC_financial_statements_2022_consolidated", [
                ["", "Năm nay"],
                ["Giá vốn cho thuê dài hạn đất và cơ sở hạ tầng", "40.000.000"],
                ["TỔNG CỘNG", "100.000.000"],
            ]),
        ]
        route = _route(q, ["KBC"], "year", "ranking", years=[2019, 2022])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 2019.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 2019.0)
        self.assertIn("formula_matrix_note_v17", ca.detail)

    def test_matrix_v17_named_related_payable_counts_dashes_as_zero(self):
        q = ("Số năm mà CTCP Hoàng Anh Gia Lai (công ty mẹ) có số dư khoản "
             "phải trả ngắn hạn khác với Công ty TNHH MTV Kinh doanh Xuất "
             "nhập khẩu Hoàng Anh Gia Lai lớn hơn 0 trong các năm 2015, "
             "2019 và 2021 là bao nhiêu?")
        tables = []
        for index, (year, current, prior) in enumerate(
                ((2015, "-", "109.000.000"), (2019, "9.000.000", "-"),
                 (2021, "-", "9.000.000")), 1):
            tables.append(_matrix_table(
                f"df{index}",
                f"HAG_financial_statements_{year}_separate", [
                    ["Bên liên quan", "Mối quan hệ", "Giao dịch",
                     "Số cuối năm", "Số đầu năm"],
                    ["Phải trả ngắn hạn khác", "Phải trả ngắn hạn khác",
                     "Phải trả ngắn hạn khác", "Phải trả ngắn hạn khác",
                     "Phải trả ngắn hạn khác"],
                    ["Công ty TNHH MTV Kinh doanh Xuất nhập khẩu Hoàng Anh Gia Lai",
                     "Công ty con", "Mượn tạm", current, prior],
                ]))
        route = _route(q, ["HAG", "EIB"], "number", "lookup",
                       years=[2015, 2019, 2021])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 1.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 1.0)
        self.assertTrue(all(table["var"] in ca.pandas_query for table in tables))

    def test_matrix_v17_certificate_deposit_max_share(self):
        q = ("Tỷ trọng chứng chỉ tiền gửi có kỳ hạn dưới 12 tháng tại các năm "
             "cuối 2021 và 2022 của STB đạt giá trị lớn nhất là bao nhiêu %?")
        tables = [
            _matrix_table("df1", "STB_financial_statements_2021_consolidated", [
                ["", "Số cuối năm"], ["Chứng chỉ tiền gửi", "Chứng chỉ tiền gửi"],
                ["Dưới 12 tháng", "10.000.000"],
                ["Từ 12 tháng đến dưới 5 năm", "30.000.000"],
                ["Từ 5 năm trở lên", "60.000.000"],
            ], context="Phát hành giấy tờ có giá"),
            _matrix_table("df2", "STB_financial_statements_2022_consolidated", [
                ["", "Số cuối năm"], ["Chứng chỉ tiền gửi", "100.000.000"],
                ["Dưới 12 tháng", "5.000.000"],
                ["Từ 12 tháng đến dưới 5 năm", "35.000.000"],
                ["Từ 5 năm trở lên", "60.000.000"],
            ], context="Phát hành giấy tờ có giá"),
        ]
        route = _route(q, ["STB"], "percent", "ranking",
                       years=[2021, 2022])

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_matrix_v17_geographic_and_foreign_currency_means(self):
        geo_q = ("Tỷ trọng doanh thu từ khu vực Lào so với tổng doanh thu toàn "
                 "công ty của HAG trung bình trong các năm 2021 và 2022 là "
                 "bao nhiêu phần trăm?")
        geo_tables = []
        for index, (year, lao, total) in enumerate(
                ((2021, 20, 100), (2022, 40, 100)), 1):
            geo_tables.append(_matrix_table(
                f"df{index}", f"HAG_financial_statements_{year}_consolidated", [
                    ["", "Việt Nam", "Lào", "Tổng cộng"],
                    [f"Cho năm tài chính kết thúc ngày 31 tháng 12 năm {year}"] * 4,
                    ["Doanh thu từ khách hàng bên ngoài", str(total - lao),
                     str(lao), str(total)],
                ], context="Bộ phận theo khu vực địa lý"))
        geo_route = _route(geo_q, ["HAG"], "percent", "average",
                           years=[2021, 2022])
        geo = try_formula_answer(geo_route, geo_tables)
        self.assertTrue(geo.ok, geo.detail)
        self.assertEqual(geo.answer, 30.0)
        self.assertEqual(_eval(geo.pandas_query, geo_tables), 30.0)

        fx_q = ("Trung bình cộng của tỷ trọng ngoại tệ USD trong tổng dư lượng "
                "ngoại tệ ghi nhận ngoài bảng cân đối kế toán cuối năm 2023 "
                "của AAA và BBB là bao nhiêu phần trăm?")
        fx_tables = [
            _matrix_table("fx1", "AAA_financial_statements_2023_consolidated", [
                ["", "31/12/2023"], ["USD", "20.000.000"],
                ["EUR", "80.000.000"],
            ], context="Các khoản mục ngoài bảng cân đối kế toán"),
            _matrix_table("fx2", "BBB_financial_statements_2023_consolidated", [
                ["", "31/12/2023"], ["Đô la Mỹ (USD)", "60.000.000"],
                ["Euro (EUR)", "40.000.000"],
            ], context="Ngoại tệ các loại ngoài bảng cân đối kế toán"),
        ]
        fx_route = _route(fx_q, ["AAA", "BBB"], "percent", "average",
                          years=[2023])
        fx = try_formula_answer(fx_route, fx_tables)
        self.assertTrue(fx.ok, fx.detail)
        self.assertEqual(fx.answer, 40.0)
        self.assertEqual(_eval(fx.pandas_query, fx_tables), 40.0)

    def test_matrix_v17_short_term_customer_loan_mean(self):
        q = ("Trung bình tỷ trọng cho vay ngắn hạn trong tổng dư nợ cho vay "
             "khách hàng vào cuối năm 2019 tại công ty mẹ của ACB và MBB là "
             "bao nhiêu phần trăm?")
        tables = [
            _matrix_table("df1", "ACB_financial_statements_2019_separate", [
                ["", "31/12/2019"], ["Ngắn hạn", "60.000.000"],
                ["Trung hạn", "20.000.000"], ["Dài hạn", "20.000.000"],
                ["", "100.000.000"],
            ], context="Cho vay khách hàng - theo kỳ hạn"),
            _matrix_table("df2", "MBB_financial_statements_2019_separate", [
                ["", "31/12/2019"], ["Nợ ngắn hạn", "40.000.000"],
                ["Nợ trung hạn", "10.000.000"], ["Nợ dài hạn", "50.000.000"],
                ["", "100.000.000"],
            ], context="Cho vay khách hàng - phân tích dư nợ theo thời gian"),
        ]
        route = _route(q, ["ACB", "MBB"], "percent", "average", years=[2019])
        route["doc_type"] = "separate"

        ca = try_formula_answer(route, tables)

        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 50.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 50.0)

    def test_scenario_interest_shock_filters_then_projects_margin(self):
        q = ("Trong các doanh nghiệp ngành Dầu khí BSR, PLX và PVT có tỷ lệ "
             "thanh toán lãi vay thực tế trên 2 lần năm 2024, nếu chi phí lãi "
             "vay tăng 20% và các khoản khác giữ nguyên, biên lợi nhuận trước "
             "thuế thấp nhất theo kịch bản là bao nhiêu phần trăm?")
        values = {
            "BSR": (100.0, 50.0, 1000.0),
            "PLX": (180.0, 20.0, 1000.0),
            "PVT": (40.0, 40.0, 1000.0),
        }
        tables = []
        for index, (ticker, (pretax, interest, revenue)) in enumerate(
                values.items(), start=1):
            tables.append(_table(
                f"df{index}",
                f"{ticker}_financial_statements_2024_consolidated",
                [
                    _row("Doanh thu thuần", revenue, code="10", row=1),
                    _row("Chi phí lãi vay", interest, code="23", row=2),
                    _row("Lợi nhuận trước thuế", pretax, code="50", row=3),
                ],
            ))
        route = _route(q, list(values), "percent", "ranking", years=[2024])

        plan = build_scenario_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.family, "interest_shock_pretax_margin")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 9.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 9.0)
        for table in tables:
            self.assertIn(table["var"], ca.pandas_query)

    def test_scenario_fails_closed_when_population_is_incomplete(self):
        q = ("Trong các doanh nghiệp BSR, PLX và PVT có tỷ lệ thanh toán lãi "
             "vay thực tế trên 2 lần năm 2024, nếu chi phí lãi vay tăng 20%, "
             "biên lợi nhuận trước thuế thấp nhất theo kịch bản là bao nhiêu "
             "phần trăm?")
        tables = [_table(
            "df1", "BSR_financial_statements_2024_consolidated", [
                _row("Doanh thu thuần", 1000.0, code="10", row=1),
                _row("Chi phí lãi vay", 50.0, code="23", row=2),
                _row("Lợi nhuận trước thuế", 100.0, code="50", row=3),
            ])]
        route = _route(q, ["BSR", "PLX", "PVT"], "percent", "ranking",
                       years=[2024])

        ca = try_formula_answer(route, tables)

        self.assertFalse(ca.ok)
        self.assertIn("unresolved PLX", ca.detail)

    def test_v19_ratio_aliases_filter_then_select_and_project(self):
        q = ("Năm 2024, trong các doanh nghiệp AAA, BBB và CCC có tỉ số "
             "thanh toán hiện hành lớn hơn 1.5, tỉ trọng hàng tồn kho trên "
             "tổng tài sản của doanh nghiệp có tỉ số thanh toán nhanh thấp "
             "nhất là bao nhiêu phần trăm?")
        values = {
            "AAA": (100.0, 75.0, 50.0, 200.0),
            "BBB": (70.0, 65.0, 50.0, 200.0),
            "CCC": (100.0, 50.0, 50.0, 200.0),
        }
        tables = []
        for index, (ticker, (current, inventory, debt, assets)) in enumerate(
                values.items(), 1):
            tables.append(_table(
                f"df{index}",
                f"{ticker}_financial_statements_2024_consolidated", [
                    _row("Tài sản ngắn hạn", current, code="100", row=1, unit=1.0),
                    _row("Hàng tồn kho", inventory, code="140", row=2, unit=1.0),
                    _row("Nợ ngắn hạn", debt, code="310", row=3, unit=1.0),
                    _row("Tổng cộng tài sản", assets, code="270", row=4, unit=1.0),
                ]))
        route = _route(q, list(values), "percent", "ranking", years=[2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name, "quick_ratio")
        self.assertEqual(plan.projection.match.spec.name, "inventory_assets")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 37.5)
        self.assertEqual(_eval(ca.pandas_query, tables), 37.5)
        self.assertTrue(all(table["var"] in ca.pandas_query for table in tables))

    def test_v19_change_selector_projects_current_cfo_margin(self):
        q = ("Năm 2024, trong các doanh nghiệp AAA và BBB ghi nhận tăng trưởng "
             "doanh thu thuần dương so với năm 2023, tỷ số CFO trên doanh thu "
             "thuần của doanh nghiệp có mức thay đổi biên lợi nhuận gộp thấp "
             "nhất là bao nhiêu phần trăm?")
        values = {
            "AAA": {2023: (100.0, 40.0, 10.0), 2024: (120.0, 54.0, 24.0)},
            "BBB": {2023: (100.0, 40.0, 10.0), 2024: (110.0, 38.5, 33.0)},
        }
        tables = []
        index = 1
        for ticker, yearly in values.items():
            for year, (revenue, gross, cfo) in yearly.items():
                tables.append(_table(
                    f"df{index}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Doanh thu thuần", revenue, code="10", row=1,
                             col_name=str(year), unit=1.0),
                        _row("Lợi nhuận gộp", gross, code="20", row=2,
                             col_name=str(year), unit=1.0),
                        _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                             row=3, col_name=str(year), unit=1.0),
                    ]))
                index += 1
        route = _route(q, list(values), "percent", "growth_pct",
                       years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual(plan.projection.match.spec.name, "cfo_margin")
        self.assertEqual(plan.projection.mode, "level")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 30.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 30.0)

    def test_v19_binary_selector_projects_metric_of_winner(self):
        q = ("Năm 2024, hệ số D/E của công ty có hiệu số giữa CFO Margin và "
             "biên lợi nhuận ròng lớn nhất trong AAA và BBB là bao nhiêu lần?")
        values = {
            "AAA": (30.0, 20.0, 100.0, 50.0, 100.0),
            "BBB": (20.0, 30.0, 100.0, 200.0, 100.0),
        }
        tables = []
        for index, (ticker, values_) in enumerate(values.items(), 1):
            cfo, profit, revenue, liabilities, equity = values_
            tables.append(_table(
                f"df{index}",
                f"{ticker}_financial_statements_2024_consolidated", [
                    _row("Doanh thu thuần", revenue, code="10", row=1, unit=1.0),
                    _row("Lợi nhuận sau thuế", profit, code="60", row=2, unit=1.0),
                    _row("Lưu chuyển tiền thuần từ hoạt động kinh doanh", cfo,
                         row=3, unit=1.0),
                    _row("Nợ phải trả", liabilities, code="300", row=4, unit=1.0),
                    _row("Vốn chủ sở hữu", equity, code="400", row=5, unit=1.0),
                ]))
        route = _route(q, list(values), "ratio", "ranking", years=[2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name, "cfo_net_margin_gap")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 0.5)
        self.assertEqual(_eval(ca.pandas_query, tables), 0.5)

    def test_v19_projection_spread_replays_both_extremes(self):
        q = ("Năm 2024, ROA của doanh nghiệp có tỷ lệ chi phí bán hàng và "
             "quản lý doanh nghiệp trên doanh thu thuần cao nhất chênh lệch "
             "bao nhiêu điểm phần trăm so với doanh nghiệp có tỷ lệ này thấp "
             "nhất trong AAA, BBB và CCC?")
        values = {
            "AAA": (-30.0, -10.0, 100.0, 10.0, 100.0),
            "BBB": (15.0, 5.0, 100.0, 15.0, 100.0),
            "CCC": (5.0, 5.0, 100.0, 5.0, 100.0),
        }
        tables = []
        for index, (ticker, values_) in enumerate(values.items(), 1):
            selling, admin, revenue, profit, assets = values_
            tables.append(_table(
                f"df{index}",
                f"{ticker}_financial_statements_2024_consolidated", [
                    _row("Chi phí bán hàng", selling, code="25", row=1, unit=1.0),
                    _row("Chi phí quản lý doanh nghiệp", admin, code="26", row=2,
                         unit=1.0),
                    _row("Doanh thu thuần", revenue, code="10", row=3, unit=1.0),
                    _row("Lợi nhuận sau thuế", profit, code="60", row=4, unit=1.0),
                    _row("Tổng cộng tài sản", assets, code="270", row=5, unit=1.0),
                ]))
        route = _route(q, list(values), "percentage_point", "difference",
                       years=[2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.projection_reduction, "max_minus_min")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 5.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 5.0)
        self.assertTrue(all(table["var"] in ca.pandas_query for table in tables))

    def test_v20_implicit_opening_change_selects_inventory_share_winner(self):
        q = ("Trong nhóm Hoà Phát, Hoa Sen và Nam Kim, doanh nghiệp có mức "
             "tăng tỷ trọng hàng tồn kho trên tổng tài sản lớn nhất từ đầu "
             "năm đến cuối năm 2024 có biên lợi nhuận gộp năm 2024 là bao "
             "nhiêu phần trăm?")
        tickers = ["HPG", "HSG", "NKG"]
        values = {
            "HPG": {2023: (10, 100, 20), 2024: (30, 100, 40)},
            "HSG": {2023: (20, 100, 20), 2024: (25, 100, 30)},
            "NKG": {2023: (15, 100, 20), 2024: (18, 100, 25)},
        }
        tables = []
        for index, (ticker, yearly) in enumerate(values.items(), 1):
            for year, (inventory, assets, gross_profit) in yearly.items():
                tables.append(_table(
                    f"df{index}_{year}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Hàng tồn kho", inventory, code="140", row=1,
                             col_name=str(year), unit=1.0),
                        _row("Tổng cộng tài sản", assets, code="270", row=2,
                             col_name=str(year), unit=1.0),
                        _row("Lợi nhuận gộp", gross_profit, code="20", row=3,
                             col_name=str(year), unit=1.0),
                        _row("Doanh thu thuần", 100, code="10", row=4,
                             col_name=str(year), unit=1.0),
                    ]))
        route = _route(q, tickers, "percent", "ranking", years=[2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name, "inventory_assets")
        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual((plan.selector.start_year, plan.selector.end_year),
                         (2023, 2024))
        self.assertEqual(plan.projection.match.spec.name, "gross_margin")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 40.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 40.0)

    def test_v20_joint_decline_filter_selects_sga_change_winner(self):
        q = ("Trong các doanh nghiệp bất động sản (AAA,BBB,CCC) có doanh thu "
             "và biên lợi nhuận hoạt động cùng sụt giảm năm 2024, doanh "
             "nghiệp có SG&A intensity tăng mạnh nhất có biên lợi nhuận hoạt "
             "động giảm bao nhiêu điểm phần trăm?")
        values = {
            "AAA": {2023: (100, 20, 5, 5), 2024: (90, 9, 9, 9)},
            "BBB": {2023: (100, 20, 5, 5), 2024: (80, 12, 5, 5)},
            "CCC": {2023: (100, 20, 5, 5), 2024: (110, 11, 20, 20)},
        }
        tables = []
        for index, (ticker, yearly) in enumerate(values.items(), 1):
            for year, (revenue, operating, selling, admin) in yearly.items():
                tables.append(_table(
                    f"df{index}_{year}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Doanh thu thuần", revenue, code="10", row=1,
                             col_name=str(year), unit=1.0),
                        _row("Lợi nhuận thuần từ hoạt động kinh doanh", operating,
                             code="30", row=2, col_name=str(year), unit=1.0),
                        _row("Chi phí bán hàng", selling, code="25", row=3,
                             col_name=str(year), unit=1.0),
                        _row("Chi phí quản lý doanh nghiệp", admin, code="26",
                             row=4, col_name=str(year), unit=1.0),
                    ]))
        route = _route(q, list(values), "percentage_point", "ranking",
                       years=[2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name, "sga_intensity")
        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual(plan.projection.match.spec.name, "operating_margin")
        self.assertEqual(plan.projection.mode, "decrease")
        self.assertEqual(len(plan.predicates), 2)
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 10.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 10.0)

    def test_v20_decline_threshold_filters_before_turnover_ranking(self):
        q = ("Trong ngành BĐS (gồm các công ty AAA,BBB,CCC), xét các doanh "
             "nghiệp có biên lợi nhuận gộp năm 2024 giảm trên 2 điểm phần "
             "trăm so với năm 2023, tỷ số ROE năm 2024 của doanh nghiệp có "
             "vòng quay tổng tài sản tăng nhiều nhất là bao nhiêu phần trăm?")
        values = {
            "AAA": {2023: (100, 40, 100, 10, 100),
                    2024: (180, 63, 90, 20, 100)},
            "BBB": {2023: (100, 40, 100, 10, 100),
                    2024: (120, 44.4, 100, 15, 100)},
            "CCC": {2023: (100, 40, 100, 10, 100),
                    2024: (200, 78, 80, 30, 100)},
        }
        tables = []
        for index, (ticker, yearly) in enumerate(values.items(), 1):
            for year, (revenue, gross, assets, profit, equity) in yearly.items():
                tables.append(_table(
                    f"df{index}_{year}",
                    f"{ticker}_financial_statements_{year}_consolidated", [
                        _row("Doanh thu thuần", revenue, code="10", row=1,
                             col_name=str(year), unit=1.0),
                        _row("Lợi nhuận gộp", gross, code="20", row=2,
                             col_name=str(year), unit=1.0),
                        _row("Tổng cộng tài sản", assets, code="270", row=3,
                             col_name=str(year), unit=1.0),
                        _row("Lợi nhuận sau thuế", profit, code="60", row=4,
                             col_name=str(year), unit=1.0),
                        _row("Vốn chủ sở hữu", equity, code="400", row=5,
                             col_name=str(year), unit=1.0),
                    ]))
        route = _route(q, list(values), "percent", "ranking",
                       years=[2023, 2024])

        plan = build_compositional_ranking_plan(route)
        ca = try_formula_answer(route, tables)

        self.assertEqual(plan.selector.match.spec.name, "total_asset_turnover")
        self.assertEqual(plan.selector.mode, "delta")
        self.assertEqual(plan.predicates[0].calculation.match.spec.name,
                         "gross_margin")
        self.assertEqual(plan.predicates[0].calculation.mode, "decrease")
        self.assertTrue(ca.ok, ca.detail)
        self.assertEqual(ca.answer, 20.0)
        self.assertEqual(_eval(ca.pandas_query, tables), 20.0)


if __name__ == "__main__":
    unittest.main()
