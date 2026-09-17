import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from vifinqa.submission.build import (
    _read_unique_by_id,
    _validate_codegen_evidence,
    _validate_replay,
    _validate_zip_layout,
)


class SubmissionGuardTests(unittest.TestCase):
    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text(
                json.dumps({"id": 1}) + "\n" + json.dumps({"id": 1}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate ids"):
                _read_unique_by_id(path, "test")

    def test_dataframe_without_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without evidence"):
            _validate_codegen_evidence(
                7, "float(df2['value'].iloc[0])", [{"var": "df1"}], "ok"
            )

    def test_successful_constant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not reference"):
            _validate_codegen_evidence(7, "12.34", [], "ok")

    def test_comments_and_strings_do_not_create_fake_evidence_refs(self):
        code = "note = 'df3 is text'\n# df2 is a comment\nanswer = float(df1.iloc[0, 0])"
        _validate_codegen_evidence(7, code, [{"var": "df1"}], "ok")

    def test_failed_placeholder_constant_is_allowed(self):
        _validate_codegen_evidence(7, "0.0", [], "failed")

    def test_zip_layout_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "submission.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("results.json", "[]")
                z.writestr("data/a.csv", "x\n")
            _validate_zip_layout(path, "results.json", {"a.csv"})
            with zipfile.ZipFile(path, "a") as z:
                z.writestr("unexpected.txt", "x")
            with self.assertRaisesRegex(ValueError, "invalid submission zip layout"):
                _validate_zip_layout(path, "results.json", {"a.csv"})

    def test_replay_detects_answer_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "data/a.csv").write_text("value\n2\n", encoding="utf-8")
            entry = {
                "id": 1,
                "answer": 3.0,
                "pandas_query": "float(df1['value'].iloc[0])",
                "evidence": [{"variable": "df1", "csv_path": "data/a.csv"}],
            }
            with self.assertRaisesRegex(ValueError, "replay failed"):
                _validate_replay([entry], root)


if __name__ == "__main__":
    unittest.main()
