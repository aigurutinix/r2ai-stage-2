from unittest import TestCase
from unittest.mock import Mock

from road2ai_vifinqa.corpus import TableAsset
from road2ai_vifinqa.source_units import continuation_scale, declared_scale
from road2ai_vifinqa.text import english_vnd_scale, source_scale
from road2ai_vifinqa.table_semantics import TableAnalyzer


class SourceUnitsTest(TestCase):
    def setUp(self):
        self.headers = [["Chỉ tiêu", "Năm 2018", "Năm 2019"], ["Chỉ tiêu", "Năm 2018", "Năm 2019"]]
        self.previous = TableAsset("ABC", 2, 57, "Đơn vị: Triệu đồng, %", self.headers)
        self.current = TableAsset("ABC", 3, 58, "===== PAGE 58 =====", self.headers)
        self.corpus = Mock()
        self.corpus.table.return_value = self.previous

    def test_repeated_header_inherits_explicit_previous_unit(self):
        self.assertEqual(continuation_scale(self.corpus, self.current), 1e6)

    def test_narrative_currency_is_not_a_declaration(self):
        self.assertIsNone(declared_scale("Lợi nhuận tăng 10 triệu đồng"))

    def test_different_headers_prevent_unit_leak(self):
        current = TableAsset("ABC", 3, 58, "===== PAGE 58 =====", [["Chỉ tiêu", "Năm 2019", "Năm 2020"]])
        self.assertIsNone(continuation_scale(self.corpus, current))

    def test_new_section_prevents_unit_leak(self):
        current = TableAsset("ABC", 3, 58, "===== PAGE 58 ===== 4. Thu nhập khác", self.headers)
        self.assertIsNone(continuation_scale(self.corpus, current))

    def test_local_base_currency_prevents_inheritance(self):
        current = TableAsset("ABC", 3, 58, "Đơn vị tính: VND", self.headers)
        self.assertIsNone(continuation_scale(self.corpus, current))

    def test_explicit_units(self):
        self.assertEqual(declared_scale("Đơn vị tính: nghìn đồng"), 1000)
        self.assertEqual(declared_scale("ĐVT: tỷ VND"), 1e9)
        self.assertEqual(declared_scale("Đơn vị: VND"), 1)

    def test_english_vnd_units_in_both_orders(self):
        for label, expected in (("31/12/2018VND million", 1e6),
                                ("Unit: millions of shares; VND", 1),
                                ("VND (thousands)", 1e3),
                                ("Millions VND", 1e6),
                                ("Unit: VND billion", 1e9),
                                ("Trillion VND", 1e12),
                                ("Tỷ VND", 1e9)):
            with self.subTest(label=label):
                self.assertEqual(source_scale(label), expected)

    def test_currency_and_multiplier_must_be_adjacent(self):
        self.assertIsNone(english_vnd_scale("VND; total of 30 million shares"))
        self.assertIsNone(english_vnd_scale("Millions USD"))

    def test_english_unit_attached_to_date(self):
        rows = [["Items", "31/12/2018VND million", "31/12/2017VND million"],
                ["Loans", "877.765", "723.517"]]
        analyzer = TableAnalyzer(rows)
        self.assertEqual(analyzer.cell(1, 1).unit_scale, 1e6)
