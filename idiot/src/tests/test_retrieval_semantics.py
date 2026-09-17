from __future__ import annotations

import unittest

from road2ai_vifinqa.corpus import TableAsset
from road2ai_vifinqa.retrieval import _table_semantic_prior


class RetrievalSemanticsTest(unittest.TestCase):
    def test_point_in_time_question_prefers_balance_sheet(self) -> None:
        rows = [["Chỉ tiêu", "2024"], ["Tổng tài sản", "1.000"]]
        balance = TableAsset(
            "AAA_financial_statements_2024_consolidated",
            1,
            1,
            "BÁO CÁO TÌNH HÌNH TÀI CHÍNH",
            rows,
        )
        income = TableAsset(
            "AAA_financial_statements_2024_consolidated",
            2,
            2,
            "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            rows,
        )
        question = "Tổng tài sản của AAA cuối năm 2024 là bao nhiêu?"
        self.assertGreater(
            _table_semantic_prior(balance, question),
            _table_semantic_prior(income, question),
        )

    def test_flow_question_prefers_income_statement(self) -> None:
        rows = [["Chỉ tiêu", "2024"], ["Doanh thu thuần", "1.000"]]
        balance = TableAsset("doc", 1, 1, "BẢNG CÂN ĐỐI KẾ TOÁN", rows)
        income = TableAsset("doc", 2, 2, "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH", rows)
        question = "Doanh thu thuần năm 2024 là bao nhiêu?"
        self.assertGreater(
            _table_semantic_prior(income, question),
            _table_semantic_prior(balance, question),
        )


if __name__ == "__main__":
    unittest.main()
