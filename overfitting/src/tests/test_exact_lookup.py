import io
import json
import unittest

import pandas as pd

from vifinqa.codegen.exact_lookup import try_exact_lookup_answer


def _table(rows, report_id="AAA_financial_statements_2024_consolidated"):
    return [{
        "var": "df1", "report_id": report_id, "report_year": 2024,
        "table_pos": 5, "context": "Báo cáo kết quả hoạt động kinh doanh",
        "grid_json": "[]", "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }]


def _route(requirement, *, output_type="number", unit_scale=1e9):
    return {
        "question": "Doanh thu thuần của AAA năm 2024 là bao nhiêu tỷ đồng?",
        "tickers": ["AAA"], "years": [2024], "output_type": output_type,
        "unit_scale": unit_scale, "metric_keys": [requirement["metric_key"]],
        "plan": {"op": "lookup"}, "evidence_requirements": [requirement],
    }


class ExactLookupTests(unittest.TestCase):
    def test_exact_vas_lookup_converts_unit_and_replays(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
            "metric_variants": ["doanh thu thuan"],
            "statement": "income_statement",
        }
        rows = [{
            "row": 3, "label": "Doanh thu thuần", "code": "10", "col": 3,
            "col_name": "Năm 2024", "value": 123.0, "unit_scale": 1e9,
        }]

        answer = try_exact_lookup_answer(_route(requirement), _table(rows))

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 123.0)
        self.assertEqual(answer.tier, "vas_current")
        frame = pd.read_csv(io.StringIO(_table(rows)[0]["csv_text"]))
        self.assertEqual(eval(answer.pandas_query, {"df1": frame}), 123.0)

    def test_infers_one_atomic_requirement_from_question(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
            "metric_variants": ["doanh thu thuan"],
            "statement": "income_statement",
        }
        route = _route(requirement)
        route["evidence_requirements"] = []
        rows = [{
            "row": 3, "label": "Doanh thu thuần", "code": "10", "col": 3,
            "col_name": "Năm 2024", "value": 123.0, "unit_scale": 1e9,
        }]

        answer = try_exact_lookup_answer(route, _table(rows))

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 123.0)
        self.assertIn("requirement=question", answer.detail)

    def test_coded_lookup_prefers_explicit_closing_date(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "supplier_prepayments_short_term",
            "metric_label": "tra truoc cho nguoi ban ngan han",
            "metric_variants": ["tra truoc cho nguoi ban ngan han"],
            "statement": "balance_sheet",
        }
        route = _route(requirement)
        route["question"] = "Trả trước cho người bán ngắn hạn cuối năm 2024?"
        rows = [
            {"row": 3, "label": "Trả trước cho người bán ngắn hạn",
             "code": "132", "col": 3, "col_name": "31/12/2024",
             "value": 500.0, "unit_scale": 1e9},
            {"row": 3, "label": "Trả trước cho người bán ngắn hạn",
             "code": "132", "col": 4, "col_name": "01/06/2024",
             "value": 700.0, "unit_scale": 1e9},
        ]
        tables = _table(rows)
        tables[0]["context"] = "Bảng cân đối kế toán"

        answer = try_exact_lookup_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 500.0)

    def test_note_matrix_uses_context_year_and_typed_column(self):
        route = {
            "question": "Tổng số lượng cổ phần của AAA cuối năm 2024?",
            "tickers": ["AAA"], "years": [2024],
            "doc_type": "consolidated", "output_type": "number",
            "unit_scale": 1.0, "plan": {"op": "lookup"},
            "evidence_requirements": [],
        }
        rows = [{
            "row": 3, "label": "Tổng cộng", "code": "", "col": 1,
            "col_name": "Số lượng cổ phần nắm giữ", "value": 700.0,
            "unit_scale": 1.0,
        }]
        tables = _table(rows)
        tables[0]["context"] = "Cơ cấu vốn cổ phần tại ngày 31/12/2024"
        tables[0]["grid_json"] = json.dumps([
            ["Cổ đông", "Số lượng cổ phần nắm giữ", "Tỷ lệ sở hữu"],
            ["Tổng cộng", "700", "100%"],
        ])

        answer = try_exact_lookup_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 700.0)

    def test_note_matrix_filters_to_canonical_column_qualifier(self):
        route = {
            "question": "Tổng giá gốc nợ phải thu cuối năm 2024 của AAA?",
            "tickers": ["AAA"], "years": [2024],
            "doc_type": "consolidated", "output_type": "number",
            "unit_scale": 1.0, "plan": {"op": "lookup"},
            "evidence_requirements": [],
        }
        rows = [
            {"row": 3, "label": "Tổng cộng", "code": "", "col": 1,
             "col_name": "Số cuối năm", "value": 220.0,
             "unit_scale": 1.0},
            {"row": 3, "label": "Tổng cộng", "code": "", "col": 2,
             "col_name": "Số cuối năm", "value": 3.0,
             "unit_scale": 1.0},
        ]
        tables = _table(rows)
        tables[0]["context"] = "9. Nợ xấu tại ngày 31/12/2024"
        tables[0]["grid_json"] = json.dumps([
            ["", "Số cuối năm", "Số cuối năm"],
            ["", "Giá gốc", "Giá trị có thể thu hồi"],
            ["Tổng cộng", "220", "3"],
        ])

        answer = try_exact_lookup_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 220.0)

    def test_exact_note_percent_keeps_percent_cell_scale(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "ownership_rate", "metric_label": "ty le so huu",
            "metric_variants": ["ty le so huu"], "statement": "other",
        }
        rows = [{
            "row": 2, "label": "Tỷ lệ sở hữu", "code": "", "col": 1,
            "col_name": "31/12/2024 %", "value": 35.0, "unit_scale": 1.0,
        }]
        tables = _table(rows)
        tables[0]["context"] = "Tỷ lệ sở hữu tại công ty con"

        answer = try_exact_lookup_answer(
            _route(requirement, output_type="percent", unit_scale=1.0), tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 35.0)
        self.assertEqual(answer.tier, "note_exact")

    def test_refuses_multiple_canonical_requirements(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
            "metric_variants": ["doanh thu thuan"],
            "statement": "income_statement",
        }
        route = _route(requirement)
        route["evidence_requirements"].append(dict(
            requirement, metric_key="gross_profit", metric_label="loi nhuan gop"))

        answer = try_exact_lookup_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("requirements=2", answer.detail)

    def test_refuses_opening_date_without_prior_period_mapping(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "net_revenue", "metric_label": "doanh thu thuan",
            "metric_variants": ["doanh thu thuan"],
            "statement": "income_statement",
        }
        route = _route(requirement)
        route["question"] = "Doanh thu thuần của AAA tại ngày 01/01/2024?"

        answer = try_exact_lookup_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("opening-period", answer.detail)

    def test_refuses_child_detail_missing_from_canonical_metric(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "short_term_borrowings",
            "metric_label": "vay va no thue tai chinh ngan han",
            "metric_variants": ["vay va no thue tai chinh ngan han"],
            "statement": "balance_sheet",
        }
        route = _route(requirement)
        route["question"] = "Khoản vay ngắn hạn từ ngân hàng của AAA là bao nhiêu?"

        answer = try_exact_lookup_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("detail=tu ngan hang", answer.detail)

    def test_refuses_lexical_child_lost_by_router(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "bank_cash", "metric_label": "tien mat",
            "metric_variants": ["tien mat"], "statement": "balance_sheet",
        }
        route = _route(requirement)
        route["question"] = "Tiền mặt ngoại tệ cuối năm của AAA là bao nhiêu?"

        answer = try_exact_lookup_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("detail=ngoai te", answer.detail)

    def test_refuses_cost_qualifier_missing_from_parent_metric(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "separate",
            "metric_key": "investments_in_subsidiaries",
            "metric_label": "dau tu vao cong ty con",
            "metric_variants": ["dau tu vao cong ty con"],
            "statement": "balance_sheet",
        }
        route = _route(requirement)
        route["question"] = "Nguyên giá đầu tư vào công ty con của AAA?"

        answer = try_exact_lookup_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("detail=nguyen gia", answer.detail)

    def test_refuses_named_company_counterparty_missing_from_metric(self):
        requirement = {
            "ticker": "AAA", "year": 2024, "doc_type": "consolidated",
            "metric_key": "bonds_issued", "metric_label": "trai phieu phat hanh",
            "metric_variants": ["trai phieu phat hanh"],
            "statement": "balance_sheet",
        }
        route = _route(requirement)
        route["question"] = (
            "Số dư trái phiếu phát hành đến Công ty Thành Công của AAA?")

        answer = try_exact_lookup_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("detail=den cong ty", answer.detail)


if __name__ == "__main__":
    unittest.main()
