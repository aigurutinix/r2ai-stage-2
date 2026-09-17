import unittest
import pandas as pd

from tools.benchmark_reasoning_arithmetic import first_present


class ArithmeticInputTest(unittest.TestCase):
    def test_mixed_csv_row_uses_first_non_missing_field(self):
        row = pd.Series({"value": float("nan"), "vnd_value": 1200.0, "raw_number": 1.2})
        self.assertEqual(first_present(row, "value", "vnd_value", "raw_number"), 1200.0)

    def test_actual_zero_is_not_treated_as_missing(self):
        row = pd.Series({"value": 0.0, "vnd_value": 1200.0})
        self.assertEqual(first_present(row, "value", "vnd_value"), 0.0)

    def test_mixed_labels_and_absent_values(self):
        row = pd.Series({"label": float("nan"), "row_label": "Tài sản cố định"})
        self.assertEqual(first_present(row, "label", "row_label"), "Tài sản cố định")
        self.assertIsNone(first_present(row, "value", "vnd_value"))


if __name__ == "__main__":
    unittest.main()
