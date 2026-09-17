from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from road2ai_vifinqa.build_index import build_index
from road2ai_vifinqa.corpus import Corpus


class SemanticIndexTest(unittest.TestCase):
    def test_builds_and_reads_semantic_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_root = root / "reports"
            doc_dir = report_root / "AAA" / "2024" / "AAA_financial_statements_2024_consolidated"
            doc_dir.mkdir(parents=True)
            source = doc_dir / "report_extracted.txt"
            source.write_text(
                """===== PAGE 1 =====
BÁO CÁO TÌNH HÌNH TÀI CHÍNH
Đơn vị: triệu VND
<table>
<tr><th>Mã</th><th>Chỉ tiêu</th><th>31/12/2024</th></tr>
<tr><td>270</td><td>TỔNG CỘNG TÀI SẢN</td><td>1.000</td></tr>
</table>
""",
                encoding="utf-8",
            )
            index = root / "tables.sqlite3"
            manifest = root / "manifest.json"
            result = build_index(
                force=True,
                output_path=index,
                manifest_path=manifest,
                expected_tables=1,
                report_root=report_root,
            )
            self.assertEqual(result["format_version"], 2)
            self.assertTrue(json.loads(manifest.read_text(encoding="utf-8"))["semantic_columns"])
            with Corpus(index) as corpus:
                table = corpus.table("AAA_financial_statements_2024_consolidated", 1)
                self.assertEqual(table.statement_kind, "balance_sheet")
                self.assertEqual(table.unit_scale, 1_000_000.0)
                self.assertIn(0, table.header_rows)
                rows = corpus.rows_for_documents([corpus.document(table.doc_id)])
                self.assertEqual(rows[1].row_label, "TỔNG CỘNG TÀI SẢN")


if __name__ == "__main__":
    unittest.main()
