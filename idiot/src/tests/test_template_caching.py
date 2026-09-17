from __future__ import annotations

import unittest

from road2ai_vifinqa.corpus import DocumentRef, RowAsset, TableAsset
from road2ai_vifinqa.template_solver import TemplateSolver
from road2ai_vifinqa.text import fold_text


class _FakeCorpus:
    def __init__(self) -> None:
        self.tickers = frozenset({"AAA"})
        self.company_names = {"AAA": "Công ty Cổ phần AAA"}
        self.document_calls = 0
        self.document = DocumentRef(
            "AAA_financial_statements_2024_consolidated",
            "AAA",
            2024,
            "consolidated",
            "unused.txt",
            1,
        )
        rows = [
            ["Chỉ tiêu", "Mã số", "31/12/2024 Triệu VND"],
            ["Doanh thu thuần", "10", "1.000"],
        ]
        self.asset = TableAsset(
            self.document.doc_id,
            1,
            1,
            "Báo cáo kết quả hoạt động kinh doanh",
            rows,
        )
        self.row = RowAsset(
            self.document.doc_id,
            1,
            1,
            rows[1],
            fold_text(" ".join(rows[1])),
        )

    def documents_for_question(self, _question: str):
        self.document_calls += 1
        return [self.document]

    def rows_for_documents(self, _documents):
        return [self.row]

    def table(self, _doc_id: str, _table_id: int):
        return self.asset


class TemplateCacheTest(unittest.TestCase):
    def test_reuses_issuer_hits_and_table_analyzer_across_operands(self) -> None:
        corpus = _FakeCorpus()
        solver = TemplateSolver(corpus)  # type: ignore[arg-type]
        first = solver._best_direct_hit(
            "doanh thu thuần của AAA năm 2024",
            "doanh thu thuần",
            2024,
            ticker="AAA",
        )
        second = solver._best_direct_hit(
            "doanh thu thuần của AAA năm 2024",
            "doanh thu thuần",
            2024,
            ticker="AAA",
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(corpus.document_calls, 1)
        self.assertEqual(len(solver._issuer_hits_cache), 1)
        self.assertEqual(len(solver._table_analyzer_cache), 0)

        semantic_corpus = _FakeCorpus()
        semantic_solver = TemplateSolver(semantic_corpus, semantic_columns=True)  # type: ignore[arg-type]
        semantic_solver._best_direct_hit(
            "doanh thu thuần của AAA năm 2024",
            "doanh thu thuần",
            2024,
            ticker="AAA",
        )
        semantic_solver._best_direct_hit(
            "doanh thu thuần của AAA năm 2024",
            "doanh thu thuần",
            2024,
            ticker="AAA",
        )
        self.assertEqual(semantic_corpus.document_calls, 1)
        self.assertEqual(len(semantic_solver._table_analyzer_cache), 1)


if __name__ == "__main__":
    unittest.main()
