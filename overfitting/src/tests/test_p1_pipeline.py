"""Tests for the P1 additions: extractive metric, shortlist, decomposition,
formula registry and answer-unit normalisation."""
import json
import unittest

import pandas as pd

from vifinqa.codegen.formulas import REGISTRY, describe_for_prompt, get
from vifinqa.codegen.fact_resolver import resolve_requirement
from vifinqa.codegen.units import (check_answer_unit, cell_is_already_percent,
                                   percent_from_cell)
from vifinqa.retrieval.shortlist import (
    _numeric_header_year,
    _period_kind,
    build_shortlist,
    render_shortlist,
)
from vifinqa.router.decompose import build_plan, detect_op, evidence_budget
from vifinqa.router.metric_phrase import extract_metric
from vifinqa.utils.viet_text import label_metric_score


class MetricPhraseTests(unittest.TestCase):
    def test_strips_quantity_prefix_and_entity_tail(self):
        mp = extract_metric(
            "Số dư trả trước cho người bán của CTCP Phát triển Bất động sản Văn "
            "Phú (VPI) cuối năm 2024 tính bằng đồng là bao nhiêu?",
            ["cong ty co phan phat trien bat dong san van phu"], ["VPI"])
        self.assertEqual(mp.core, "tra truoc cho nguoi ban")

    def test_beats_the_old_subtractive_phrase_on_the_real_regression(self):
        # measured: the old phrase scored 63 against this gold row (< 78 cut)
        gold = "Trả trước cho người bán"
        old = "so du tra truoc cho nguoi ban tinh dong"
        mp = extract_metric("Số dư trả trước cho người bán của X cuối năm 2024 "
                            "tính bằng đồng là bao nhiêu?", [], [])
        self.assertGreater(max(label_metric_score(gold, v) for v in mp.variants()),
                           label_metric_score(gold, old))

    def test_keeps_qualifier_in_wide_form(self):
        mp = extract_metric("Tổng số tiền trả trước cho người bán ngắn hạn của Y "
                            "cuối năm 2023 là bao nhiêu?", [], [])
        self.assertIn("ngan han", mp.wide)

    def test_does_not_eat_doanh_thu(self):
        mp = extract_metric("Doanh thu thuần của ABC năm 2023 là bao nhiêu tỷ đồng?",
                            [], ["ABC"])
        self.assertTrue(mp.core.startswith("doanh thu"), mp.core)


class DecomposeTests(unittest.TestCase):
    def test_detects_ops(self):
        self.assertEqual(detect_op("Lợi nhuận sau thuế năm 2023 tăng trưởng bao "
                                   "nhiêu phần trăm so với 2022?"), "growth_pct")
        self.assertEqual(detect_op("Chênh lệch doanh thu giữa A và B"), "difference")
        self.assertEqual(detect_op("Công ty nào có tổng tài sản lớn nhất"), "ranking")

    def test_detects_explicit_list_aggregates_and_count_years(self):
        self.assertEqual(
            detect_op("Tính tổng giá trị cổ tức nhận được trong các năm "
                      "2016, 2024 và 2025"),
            "sum",
        )
        self.assertEqual(
            detect_op("Tổng giá trị lãi/lỗ chênh lệch tỷ giá của các công ty A, B"),
            "sum",
        )
        self.assertEqual(
            detect_op("Tổng chi phí thuế thu nhập doanh nghiệp cao nhất trong các năm"),
            "ranking",
        )
        self.assertEqual(
            detect_op("Số năm tồn tại khoản mục X tại các mốc là bao nhiêu?"),
            "count",
        )
        self.assertEqual(
            detect_op("Chi phí lớn hơn 0 trong số năm được hỏi là bao nhiêu?"),
            "count",
        )
        self.assertEqual(
            detect_op("Có bao nhiêu đơn vị ghi nhận số dư lớn hơn 1 tỷ đồng?"),
            "count",
        )

    def test_detects_additional_difference_and_growth_wording(self):
        self.assertEqual(detect_op("A trừ đi B là bao nhiêu?"), "difference")
        self.assertEqual(
            detect_op("Hiệu số vốn chủ sở hữu của công ty A và công ty B"),
            "difference",
        )
        self.assertEqual(detect_op("A lớn hơn B bao nhiêu tỷ đồng?"), "difference")
        self.assertEqual(
            detect_op("Tổng nợ tài chính tăng bao nhiêu %?"), "growth_pct")
        self.assertEqual(
            detect_op("Tỷ lệ biến động dự phòng giữa hai kỳ là bao nhiêu %?"),
            "growth_pct",
        )
        self.assertEqual(
            detect_op("Chi phí giảm từ năm 2017 đến 2018 bao nhiêu %?"),
            "growth_pct",
        )
        self.assertEqual(
            detect_op("Biến động tổng tài sản giữa cuối năm 2019 và 2018"),
            "difference",
        )
        self.assertEqual(
            detect_op("Giá trị còn lại từ 2018 đến 2023 tăng bao nhiêu triệu đồng"),
            "difference",
        )
        self.assertEqual(
            detect_op("Thuế cuối năm 2020 bé hơn cuối năm 2018 bao nhiêu tỷ đồng"),
            "difference",
        )
        self.assertEqual(
            detect_op("Hiệu giữa trái phiếu của ngân hàng A và ngân hàng B"),
            "difference",
        )
        self.assertEqual(
            detect_op("Số dư của Eximbank hơn MBBank mấy triệu đồng"),
            "difference",
        )

    def test_multi_entity_total_is_sum_but_single_total_line_is_lookup(self):
        many = build_plan(
            "Tổng số dư phải trả ngắn hạn khác của A, B và C là bao nhiêu?",
            ["A", "B", "C"], [2024], "consolidated", "phai tra ngan han khac",
        )
        one = build_plan(
            "Tổng số dư ngoại tệ của A là bao nhiêu?",
            ["A"], [2024], "consolidated", "tong so du ngoai te",
        )
        self.assertEqual(many.op, "sum")
        self.assertEqual(one.op, "lookup")

        filtered = build_plan(
            "Tổng doanh thu của các công ty có biên lợi nhuận lớn hơn 10%",
            ["A", "B", "C"], [2024], "consolidated", "doanh thu",
        )
        self.assertEqual(filtered.op, "sum")

    def test_tong_cong_ty_is_not_an_aggregation(self):
        self.assertEqual(
            detect_op("Trả trước cho người bán của Tổng Công ty cổ phần X "
                      "cuối năm 2023 là bao nhiêu?"), "lookup")

    def test_third_party_counterparty_is_not_a_ranking(self):
        self.assertEqual(
            detect_op("Số dư phải trả người bán là bên thứ ba cuối năm 2024"),
            "lookup",
        )
        self.assertEqual(
            detect_op("Phải thu ngắn hạn của khách hàng bên thứ ba cuối năm 2021"),
            "lookup",
        )
        self.assertEqual(detect_op("Công ty đứng thứ ba về doanh thu"), "ranking")

    def test_growth_expands_to_prior_year(self):
        plan = build_plan("Doanh thu năm 2023 tăng trưởng bao nhiêu phần trăm?",
                          ["AAA"], [2023], "consolidated", "doanh thu")
        self.assertEqual(plan.op, "growth_pct")
        self.assertEqual({f.year for f in plan.facts}, {2023, 2022})

    def test_budget_grows_with_facts(self):
        one = build_plan("Doanh thu 2023", ["A"], [2023], "consolidated", "m")
        many = build_plan("Chênh lệch giữa A và B năm 2023", ["A", "B"], [2023],
                          "consolidated", "m")
        self.assertLess(evidence_budget(one), evidence_budget(many))
        self.assertLessEqual(evidence_budget(many), 12)

    def test_year_ranking_plan_declares_dimension_projection_and_direction(self):
        plan = build_plan(
            "Doanh thu thuan thap nhat trong cac nam 2022-2024 la nam nao?",
            ["AAA"], [2022, 2023, 2024], "consolidated", "doanh thu thuan",
            output_type="year",
        )
        self.assertEqual(plan.op, "ranking")
        self.assertEqual(plan.dimension, "year")
        self.assertEqual(plan.projection, "year")
        self.assertEqual(plan.direction, "min")
        self.assertEqual({fact.year for fact in plan.facts}, {2022, 2023, 2024})


class FormulaTests(unittest.TestCase):
    def test_growth_is_in_percent_units(self):
        self.assertAlmostEqual(REGISTRY["growth_pct"].fn(120.0, 100.0), 20.0)

    def test_ratio_is_in_percent_units(self):
        self.assertAlmostEqual(REGISTRY["ratio"].fn(9.0, 10.0), 90.0)

    def test_ratio_times_is_not_scaled(self):
        self.assertAlmostEqual(REGISTRY["ratio_times"].fn(9.0, 3.0), 3.0)

    def test_describe_mentions_multiple_operands(self):
        self.assertIn("BOTH", describe_for_prompt("difference", 2))
        self.assertEqual(get("nope").name, "lookup")


class UnitTests(unittest.TestCase):
    def test_ratio_cell_becomes_percent(self):
        self.assertAlmostEqual(percent_from_cell(0.9, "Tỷ lệ sở hữu", ""), 90.0)

    def test_percent_cell_is_left_alone(self):
        self.assertAlmostEqual(percent_from_cell(90.0, "Tỷ lệ sở hữu", ""), 90.0)

    def test_percent_marked_column_is_left_alone(self):
        self.assertTrue(cell_is_already_percent("Biến động", "%", 0.5))

    def test_warns_on_ratio_shaped_percent_answer(self):
        self.assertIn("RATIO", check_answer_unit(0.9, "percent"))
        self.assertIsNone(check_answer_unit(90.0, "percent"))
        self.assertIsNone(check_answer_unit(-11.0, "percent"))

    def test_year_and_count_ranges(self):
        self.assertIsNone(check_answer_unit(2023, "year"))
        self.assertIsNotNone(check_answer_unit(15.5, "year"))


class ShortlistTests(unittest.TestCase):
    def test_non_finite_header_value_is_not_a_year(self):
        self.assertIsNone(_numeric_header_year(float("inf")))

    def test_ocr_value_headers_keep_leftmost_current_period_semantics(self):
        self.assertEqual(
            _period_kind("Giá gốc/Số có khả năng trả nợ"), "current")
        self.assertEqual(_period_kind("Đơn vị tính: VND"), "current")

    def test_balance_sheet_dates_distinguish_opening_and_closing_periods(self):
        self.assertEqual(_period_kind("1/1/2024 VND"), "prior")
        self.assertEqual(_period_kind("01/01/2024 VND"), "prior")
        self.assertEqual(_period_kind("31/12/2024 VND"), "current")
        self.assertEqual(_period_kind("01.01.2024 VND"), "prior")
        self.assertEqual(_period_kind("31.12.2024 VND"), "current")

    def test_exact_resolver_recovers_split_dotted_date_header(self):
        rows = [
            {"row": 2, "label": "TÀI SẢN DÀI HẠN", "code": "200",
             "col": 3, "col_name": "Tại ngày", "value": 37.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "TÀI SẢN DÀI HẠN", "code": "200",
             "col": 4, "col_name": "Tại ngày", "value": 35.0,
             "unit_scale": 1e9},
        ]
        grid = [
            ["Mã số", "TÀI SẢN", "Thuyết minh", "Tại ngày", "Tại ngày"],
            ["Mã số", "TÀI SẢN", "Thuyết minh", "31.12.2024 VND",
             "01.01.2024 VND"],
            ["200", "TÀI SẢN DÀI HẠN", "", "37", "35"],
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 8,
            "context": "Bảng cân đối kế toán hợp nhất",
            "grid_json": json.dumps(grid, ensure_ascii=False),
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "long_term_assets", "metric_label": "tai san dai han",
            "metric_variants": ["tai san dai han"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="tai san dai han")

        self.assertIsNotNone(found)
        self.assertEqual(found.value, 37.0)
        self.assertEqual(found.col, 3)

    def test_exact_resolver_prefers_closing_header_over_ocr_unit_header(self):
        rows = [
            {"row": 21, "label": "Tài sản thuế thu nhập hoãn lại",
             "code": "262", "col": 3, "col_name": "Số cuối năm",
             "value": 13_838_474_530.0, "unit_scale": 1.0},
            {"row": 21, "label": "Tài sản thuế thu nhập hoãn lại",
             "code": "262", "col": 4, "col_name": "Đơn vị tính: VND",
             "value": 6_646_642_737.0, "unit_scale": 1.0},
        ]
        tables = [{
            "var": "df1",
            "report_id": "PVT_financial_statements_2016_consolidated",
            "report_year": 2016,
            "table_pos": 4,
            "context": "Bảng cân đối kế toán hợp nhất",
            "grid_json": "[]",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "PVT", "year": 2016, "doc_type": "consolidated",
            "metric_key": "deferred_tax_asset",
            "metric_label": "tai san thue thu nhap hoan lai",
            "metric_variants": ["tai san thue thu nhap hoan lai"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="tai san thue thu nhap hoan lai")

        self.assertIsNotNone(found)
        self.assertEqual(found.col, 3)
        self.assertEqual(found.value, 13_838_474_530.0)

    def test_start_of_report_year_serves_as_prior_year_closing_balance(self):
        rows = [
            {"row": 1, "label": "Hàng tồn kho", "code": "140", "col": 1,
             "col_name": "31/12/2024 VND", "value": 200.0, "unit_scale": 1.0},
            {"row": 1, "label": "Hàng tồn kho", "code": "140", "col": 2,
             "col_name": "1/1/2024 VND", "value": 100.0, "unit_scale": 1.0},
        ]
        tables = [{
            "var": "df1", "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024, "table_pos": 1,
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]

        candidates = build_shortlist(tables, ["hang ton kho"], [2023])

        self.assertEqual(candidates[0].col_name, "1/1/2024 VND")
        self.assertEqual(candidates[0].value, 100.0)

    def test_years_split_into_header_row_are_attached_to_value_columns(self):
        rows = [
            {"row": 1, "label": "Ma so Thuyet minh", "code": "", "col": 3,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": 3092022.0, "unit_scale": 1.0},
            {"row": 1, "label": "Ma so Thuyet minh", "code": "", "col": 4,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": 3092021.0, "unit_scale": 1.0},
            {"row": 5, "label": "Gia von hang ban va dich vu cung cap",
             "code": "11", "col": 3,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": -200.0, "unit_scale": 1.0},
            {"row": 5, "label": "Gia von hang ban va dich vu cung cap",
             "code": "11", "col": 4,
             "col_name": "Nam tai chinh ket thuc ngay 31 thang 12 nam",
             "value": -180.0, "unit_scale": 1.0},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2022_consolidated",
            "report_year": 2022,
            "table_pos": 10,
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]

        current = build_shortlist(tables, ["gia von hang ban"], [2022])
        prior = build_shortlist(tables, ["gia von hang ban"], [2021])

        self.assertEqual((current[0].col, current[0].value), (3, -200.0))
        self.assertIn("2022", current[0].col_name)
        self.assertEqual((prior[0].col, prior[0].value), (4, -180.0))
        self.assertIn("2021", prior[0].col_name)

    def setUp(self):
        rows = [
            {"row": 1, "label": "Trả trước cho người bán", "code": "132",
             "col": 1, "col_name": "31/12/2024", "value": 100.0, "unit_scale": 1e6},
            {"row": 2, "label": "Trả trước cho người bán dài hạn", "code": "212",
             "col": 1, "col_name": "31/12/2024", "value": 5.0, "unit_scale": 1e6},
            {"row": 3, "label": "Chi phí quản lý doanh nghiệp", "code": "26",
             "col": 1, "col_name": "31/12/2024", "value": 7.0, "unit_scale": 1e6},
        ]
        self.tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                        "csv_text": pd.DataFrame(rows).to_csv(index=False)}]

    def test_ranks_the_matching_row_first(self):
        cands = build_shortlist(self.tables, ["tra truoc cho nguoi ban"], [2024])
        self.assertTrue(cands)
        self.assertEqual(cands[0].label, "Trả trước cho người bán")

    def test_qualifier_mismatch_is_penalised(self):
        cands = build_shortlist(self.tables, ["tra truoc cho nguoi ban ngan han"], [2024])
        labels = [c.label for c in cands]
        self.assertIn("Trả trước cho người bán", labels)
        self.assertNotIn("Trả trước cho người bán dài hạn", labels)

    def test_render_is_compact_and_safe_when_empty(self):
        self.assertIn("no candidate row", render_shortlist([]))
        self.assertIn("df1", render_shortlist(
            build_shortlist(self.tables, ["tra truoc cho nguoi ban"], [2024])))

    def test_canonical_child_beats_lexical_parent(self):
        rows = [
            {"row": 1, "label": "Tien gui va vay cac TCTD khac", "code": "",
             "col": 1, "col_name": "So cuoi nam", "value": 100.0,
             "unit_scale": 1e6},
            {"row": 2, "label": "Vay cac TCTD khac", "code": "",
             "col": 1, "col_name": "So cuoi nam", "value": 40.0,
             "unit_scale": 1e6},
        ]
        tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                   "report_year": 2024,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["vay cac TCTD khac"], [2024],
            question="So du vay cac TCTD khac cuoi nam 2024")
        self.assertEqual(cands[0].label, "Vay cac TCTD khac")

    def test_exact_resolver_accepts_same_code_value_continuation_rows(self):
        rows = [
            {"row": 1, "label": "Lợi nhuận sau thuế TNDN (mang sang trang sau)",
             "code": "60", "col": 1, "col_name": "2024", "value": 12.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "Lợi nhuận sau thuế TNDN (mang từ trang trước)",
             "code": "60", "col": 1, "col_name": "2024", "value": 12.0,
             "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 3,
            "context": "Báo cáo kết quả hoạt động kinh doanh",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_profit", "metric_label": "loi nhuan sau thue",
            "metric_variants": ["loi nhuan sau thue"],
            "statement": "income_statement",
        }

        found = resolve_requirement(requirement, tables, question="loi nhuan sau thue")

        self.assertIsNotNone(found)
        self.assertEqual(float(found.code), 60.0)

    def test_exact_resolver_prefers_vas_code_over_unnumbered_note_row(self):
        rows = [
            {"row": 1, "label": "Lợi nhuận thuần trong năm", "code": "",
             "col": 1, "col_name": "Năm 2024", "value": 4.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "Lợi nhuận sau thuế TNDN", "code": "60",
             "col": 1, "col_name": "Năm 2024", "value": 60.0,
             "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 3,
            "context": "Báo cáo kết quả hoạt động kinh doanh",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_profit", "metric_label": "loi nhuan sau thue",
            "metric_variants": ["loi nhuan sau thue", "loi nhuan thuan trong nam"],
            "statement": "income_statement",
        }

        found = resolve_requirement(requirement, tables, question="loi nhuan sau thue")

        self.assertIsNotNone(found)
        self.assertEqual(float(found.code), 60.0)
        self.assertEqual(found.value, 60.0)

    def test_exact_resolver_reads_basic_eps_swallowed_as_row_code(self):
        grid = [
            ["", "Năm nay", "Năm trước"],
            ["Lãi cơ bản trên cổ phiếu (VND/cổ phiếu)", "1.551", "1.532"],
        ]
        rows = [{
            "row": 1,
            "label": "Lãi cơ bản trên cổ phiếu (VND/cổ phiếu)",
            "code": "1.551", "col": 2, "col_name": "Năm trước",
            "value": 1532.0, "unit_scale": 1.0,
        }]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2018_consolidated",
            "report_year": 2018,
            "table_pos": 49,
            "context": "Báo cáo kết quả hoạt động kinh doanh hợp nhất",
            "grid_json": json.dumps(grid, ensure_ascii=False),
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2018, "doc_type": "consolidated",
            "metric_key": "basic_eps", "metric_label": "lai co ban tren co phieu",
            "metric_variants": ["lai co ban tren co phieu"],
            "statement": "income_statement",
        }

        found = resolve_requirement(
            requirement, tables, question="lai co ban tren co phieu")

        self.assertIsNotNone(found)
        self.assertEqual(found.value, 1551.0)
        self.assertEqual(found.value_column, "code")
        self.assertIn("'code'", found.expr())

    def test_exact_resolver_reads_deferred_expense_swallowed_as_row_code(self):
        grid = [
            ["", "31/12/2016 Triệu VND", "31/12/2015 Triệu VND"],
            ["Chi phí chờ phân bổ", "137.314", "267.743"],
        ]
        rows = [{
            "row": 1, "label": "Chi phí chờ phân bổ", "code": "137.314",
            "col": 2, "col_name": "31/12/2015 Triệu VND",
            "value": 267743.0, "unit_scale": 1e6,
        }]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2016_separate",
            "report_year": 2016, "table_pos": 20,
            "context": "Tài sản Có khác",
            "grid_json": json.dumps(grid, ensure_ascii=False),
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2016, "doc_type": "separate",
            "metric_key": "deferred_allocation_expense",
            "metric_label": "chi phi cho phan bo",
            "metric_variants": ["chi phi cho phan bo"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="chi phi cho phan bo")

        self.assertIsNotNone(found)
        self.assertEqual(found.value, 137314.0)
        self.assertEqual(found.value_column, "code")
        self.assertIn("* 1000", found.expr())

    def test_exact_resolver_recovers_shifted_vas_code_from_raw_grid(self):
        grid = [
            ["Chỉ tiêu", "Chỉ tiêu", "Mã số", "Thuyết minh",
             "Năm 2024", "Năm 2023"],
            ["1", "Doanh thu bán hàng và cung cấp dịch vụ", "01", "",
             "120", "100"],
            ["3", "Doanh thu thuần bán hàng và cung cấp dịch vụ", "10", "",
             "110", "90"],
        ]
        rows = [
            {"row": 2, "label": "Doanh thu thuần bán hàng và cung cấp dịch vụ",
             "code": "3", "col": 2, "col_name": "Mã số", "value": 10.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "Doanh thu thuần bán hàng và cung cấp dịch vụ",
             "code": "3", "col": 4, "col_name": "Năm 2024", "value": 110.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "Doanh thu thuần bán hàng và cung cấp dịch vụ",
             "code": "3", "col": 5, "col_name": "Năm 2023", "value": 90.0,
             "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 5,
            "context": "OCR header unavailable",
            "grid_json": json.dumps(grid, ensure_ascii=False),
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
            "metric_variants": ["doanh thu thuan"],
            "statement": "income_statement",
        }

        found = resolve_requirement(
            requirement, tables, question="doanh thu thuan")

        self.assertIsNotNone(found)
        self.assertEqual(found.code, "10")
        self.assertEqual(found.col, 4)
        self.assertEqual(found.value, 110.0)

    def test_exact_resolver_rejects_explicit_prior_year_with_unit_prefix(self):
        grid = [
            ["Chỉ tiêu", "Mã số", "Thuyết minh", "Năm 2024",
             "Đơn vị tính: VND Năm 2023"],
            ["Lưu chuyển tiền thuần từ hoạt động kinh doanh", "20", "",
             "110", "90"],
        ]
        rows = [
            {"row": 1, "label": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
             "code": "20", "col": 3, "col_name": "Năm 2024", "value": 110.0,
             "unit_scale": 1e9},
            {"row": 1, "label": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
             "code": "20", "col": 4,
             "col_name": "Đơn vị tính: VND Năm 2023", "value": 90.0,
             "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 6,
            "context": "OCR header unavailable",
            "grid_json": json.dumps(grid, ensure_ascii=False),
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "cfo",
            "metric_label": "luu chuyen tien thuan tu hoat dong kinh doanh",
            "metric_variants": ["luu chuyen tien thuan tu hoat dong kinh doanh"],
            "statement": "cash_flow",
        }

        found = resolve_requirement(requirement, tables, question="cfo")

        self.assertIsNotNone(found)
        self.assertEqual(found.col, 3)
        self.assertEqual(found.value, 110.0)

    def test_exact_resolver_uses_generic_prior_column_from_following_filing(self):
        rows = [
            {"row": 1, "label": "VỐN CHỦ SỞ HỮU", "code": "400",
             "col": 3, "col_name": "Số cuối năm", "value": 200.0,
             "unit_scale": 1e9},
            {"row": 1, "label": "VỐN CHỦ SỞ HỮU", "code": "400",
             "col": 4, "col_name": "Số đầu năm", "value": 100.0,
             "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2025_consolidated",
            "report_year": 2025,
            "table_pos": 7,
            "context": "Bảng cân đối kế toán hợp nhất",
            "grid_json": "[]",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "equity", "metric_label": "von chu so huu",
            "metric_variants": ["von chu so huu"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="von chu so huu")

        self.assertIsNotNone(found)
        self.assertEqual(found.report_id,
                         "AAA_financial_statements_2025_consolidated")
        self.assertEqual(found.col, 4)
        self.assertEqual(found.value, 100.0)

    def test_exact_resolver_scans_vas_code_when_ocr_label_is_opaque(self):
        rows = [
            {"row": 1, "label": "A - (100=110+120+130+140+150)",
             "code": "100", "col": 3, "col_name": "31/12/2024VND",
             "value": 6280.0, "unit_scale": 1e9},
            {"row": 1, "label": "A - (100=110+120+130+140+150)",
             "code": "100", "col": 4, "col_name": "01/01/2024VND",
             "value": 6451.0, "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "HPX_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 4,
            "context": "Bảng cân đối kế toán hợp nhất",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "HPX", "year": 2024, "doc_type": "consolidated",
            "metric_key": "current_assets", "metric_label": "tai san ngan han",
            "metric_variants": ["tai san ngan han"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="tai san ngan han")

        self.assertIsNotNone(found)
        self.assertEqual(found.value, 6280.0)
        self.assertEqual(found.col_name, "31/12/2024VND")

    def test_exact_resolver_fails_closed_on_conflicting_vas_code_values(self):
        rows = [
            {"row": 1, "label": "A - (100=110+120+130+140+150)",
             "code": "100", "col": 3, "col_name": "31/12/2024VND",
             "value": 6280.0, "unit_scale": 1e9},
            {"row": 2, "label": "Tài sản ngắn hạn", "code": "100",
             "col": 3, "col_name": "31/12/2024VND",
             "value": 6200.0, "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "HPX_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 4,
            "context": "Bảng cân đối kế toán hợp nhất",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "HPX", "year": 2024, "doc_type": "consolidated",
            "metric_key": "current_assets", "metric_label": "tai san ngan han",
            "metric_variants": ["tai san ngan han"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="tai san ngan han")

        self.assertIsNone(found)

    def test_exact_resolver_uses_leftmost_generic_closing_header(self):
        rows = [
            {"row": 1, "label": "TÀI SẢN NGẮN HẠN", "code": "100",
             "col": 3, "col_name": "Tại ngày 31 tháng 12 năm",
             "value": 10202.0, "unit_scale": 1e9},
            {"row": 1, "label": "TÀI SẢN NGẮN HẠN", "code": "100",
             "col": 4, "col_name": "Tại ngày 31 tháng 12 năm",
             "value": 9322.0, "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "NKG_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 1,
            "context": "Bảng cân đối kế toán hợp nhất",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "NKG", "year": 2024, "doc_type": "consolidated",
            "metric_key": "current_assets", "metric_label": "tai san ngan han",
            "metric_variants": ["tai san ngan han"],
            "statement": "balance_sheet",
        }

        found = resolve_requirement(
            requirement, tables, question="tai san ngan han")

        self.assertIsNotNone(found)
        self.assertEqual(found.value, 10202.0)
        self.assertEqual(found.col, 3)

    def test_exact_code_scan_rejects_row_number_masquerading_as_vas_code(self):
        rows = [
            {"row": 1, "label": "Doanh thu thuần", "code": "10",
             "col": 3, "col_name": "Năm 2024", "value": 1615.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "10. Chi phí quản lý doanh nghiệp",
             "code": "10", "col": 3, "col_name": "Năm 2024",
             "value": 102.0, "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "HPX_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 6,
            "context": "Báo cáo kết quả hoạt động kinh doanh hợp nhất",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "HPX", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
            "metric_variants": ["doanh thu thuan"],
            "statement": "income_statement",
        }

        found = resolve_requirement(
            requirement, tables, question="doanh thu thuan")

        self.assertIsNotNone(found)
        self.assertEqual(found.value, 1615.0)
        self.assertEqual(found.label, "Doanh thu thuần")

    def test_exact_resolver_accepts_equal_inventory_totals_with_allowed_codes(self):
        rows = [
            {"row": 1, "label": "IV. Hàng tồn kho", "code": "140",
             "col": 1, "col_name": "31/12/2024", "value": 50.0,
             "unit_scale": 1e9},
            {"row": 2, "label": "1. Hàng tồn kho", "code": "141",
             "col": 1, "col_name": "31/12/2024", "value": 50.0,
             "unit_scale": 1e9},
        ]
        tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 3,
            "context": "Bảng cân đối kế toán",
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "inventory", "metric_label": "hang ton kho",
            "metric_variants": ["hang ton kho"], "statement": "balance_sheet",
        }

        found = resolve_requirement(requirement, tables, question="hang ton kho")

        self.assertIsNotNone(found)
        self.assertIn(float(found.code), {140.0, 141.0})

    def test_intangible_fixed_asset_child_beats_parent(self):
        rows = [
            {"row": 1, "label": "X Tai san co dinh", "code": "220", "col": 1,
             "col_name": "So cuoi nam", "value": 100.0, "unit_scale": 1e6},
            {"row": 2, "label": "Tai san co dinh vo hinh", "code": "227", "col": 1,
             "col_name": "So cuoi nam", "value": 40.0, "unit_scale": 1e6},
        ]
        tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                   "report_year": 2024,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["gia tri con lai cua tai san co dinh vo hinh"], [2024],
            question="Gia tri con lai cua tai san co dinh vo hinh tong cong")
        self.assertEqual(cands[0].label, "Tai san co dinh vo hinh")

    def test_named_counterparty_child_beats_generic_parent(self):
        rows = [
            {"row": 1, "label": "Vay dai han", "code": "", "col": 1,
             "col_name": "31/12/2017", "value": 100.0, "unit_scale": 1e3},
            {"row": 2,
             "label": "Cong ty Co phan HoangAnh Gia Lai Cong ty me Vay dai han",
             "code": "", "col": 1, "col_name": "31/12/2017", "value": 40.0,
             "unit_scale": 1e3},
        ]
        tables = [{"var": "df1", "report_id": "HNG_2017_separate", "table_pos": 3,
                   "report_year": 2017,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["vay dai han voi hoang anh gia lai"], [2017],
            question="Vay dai han voi Hoang Anh Gia Lai cuoi nam 2017")
        self.assertEqual(cands[0].row, 2)

    def test_opening_qualifier_selects_opening_column(self):
        rows = [
            {"row": 1, "label": "No ngan han", "code": "310", "col": 1,
             "col_name": "So cuoi nam", "value": 200.0, "unit_scale": 1e6},
            {"row": 1, "label": "No ngan han", "code": "310", "col": 2,
             "col_name": "So dau nam", "value": 120.0, "unit_scale": 1e6},
        ]
        tables = [{"var": "df1", "report_id": "R", "table_pos": 3,
                   "report_year": 2024,
                   "csv_text": pd.DataFrame(rows).to_csv(index=False)}]
        cands = build_shortlist(
            tables, ["no ngan han"], [2024],
            question="No ngan han dau nam 2024 la bao nhieu?")
        self.assertEqual(cands[0].col_name, "So dau nam")

    def test_opening_date_uses_full_header_preserved_in_grid(self):
        rows = [
            {"row": 3, "label": "Phai thu ngan han cua khach hang",
             "code": "131", "col": 3, "col_name": "Tai ngay",
             "value": 200.0, "unit_scale": 1e6},
            {"row": 3, "label": "Phai thu ngan han cua khach hang",
             "code": "131", "col": 4, "col_name": "Tai ngay",
             "value": 400.0, "unit_scale": 1e6},
        ]
        grid = [
            ["Ma so", "TAI SAN", "Thuyet minh", "Tai ngay", "Tai ngay"],
            ["Ma so", "TAI SAN", "Thuyet minh", "31.12.2022 VND",
             "01.01.2022 VND"],
        ]
        tables = [{
            "var": "df1", "report_id": "R", "table_pos": 3,
            "report_year": 2022, "grid_json": json.dumps(grid),
            "csv_text": pd.DataFrame(rows).to_csv(index=False),
        }]

        cands = build_shortlist(
            tables, ["phai thu khach hang ngan han"], [2022],
            question="So du phai thu khach hang ngan han den ngay 01/01/2022?")

        self.assertEqual(cands[0].col, 4)
        self.assertEqual(cands[0].value, 400.0)


if __name__ == "__main__":
    unittest.main()
