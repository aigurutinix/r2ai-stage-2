from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.submission.export_batch import export_batch_results  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BatchSubmissionExportTests(unittest.TestCase):
    def test_answered_batch_exports_replayable_deterministic_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "build" / "tables" / "source.csv"
            source.parent.mkdir(parents=True)
            source.write_text("metric,2024\nrevenue,123.5\n", encoding="utf-8-sig")
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "index": 0,
                                "id": 7,
                                "question": "Revenue?",
                                "status": "answered",
                                "response": {
                                    "status": "answered",
                                    "submission_export": {
                                        "source_artifact": ".",
                                        "answer": 123.5,
                                        "relevant_docs": ["DOC"],
                                        "relevant_tables": ["DOC|10"],
                                        "evidence": [
                                            {"variable": "df1", "source_path": "build/tables/source.csv"}
                                        ],
                                        "pandas_query": "_dfvals = list(dfs.values())\ndf1 = _dfvals[0]\nresult = float(df1.iloc[0, 1])",
                                    },
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            first_dir = root / "candidate_a"
            second_dir = root / "candidate_b"
            first_zip = root / "a.zip"
            second_zip = root / "b.zip"
            first = export_batch_results(
                batch,
                root=root,
                output_dir=first_dir,
                archive_path=first_zip,
                expected_count=1,
            )
            second = export_batch_results(
                batch,
                root=root,
                output_dir=second_dir,
                archive_path=second_zip,
                expected_count=1,
            )
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(second["status"], "PASS")
            self.assertEqual(sha256(first_zip), sha256(second_zip))
            with zipfile.ZipFile(first_zip) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertEqual(bundle.namelist(), ["submission.json", "data/source.csv"])
                row = json.loads(bundle.read("submission.json"))[0]
            self.assertEqual(row["answer"], 123.5)
            self.assertEqual(row["evidence"], [{"variable": "df1", "csv_path": "data/source.csv"}])

    def test_raw_fallback_record_resolves_against_fallback_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "baseline" / "data" / "fallback.csv"
            source.parent.mkdir(parents=True)
            source.write_text("metric,2024\nvalue,9\n", encoding="utf-8-sig")
            raw = {
                "id": 1,
                "question": "Same question",
                "answer": 9.0,
                "relevant_docs": ["DOC"],
                "relevant_tables": ["DOC|1"],
                "evidence": [{"variable": "df1", "csv_path": "data/fallback.csv"}],
                "pandas_query": "_dfvals = list(dfs.values())\ndf1 = _dfvals[0]\nresult = float(df1.iloc[0, 1])",
            }
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": 1,
                                "question": "Same question",
                                "status": "fallback",
                                "response": raw,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = export_batch_results(
                batch,
                root=root,
                output_dir=root / "out",
                archive_path=root / "out.zip",
                fallback_artifact="baseline",
            )
            self.assertEqual(report["validation"]["execution_checks"][0]["match"], True)

    def test_refusal_fails_closed_without_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": 1,
                                "question": "Unknown",
                                "status": "refused",
                                "response": {"status": "refused"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            archive = root / "out.zip"
            with self.assertRaisesRegex(ValueError, "cannot export"):
                export_batch_results(
                    batch,
                    root=root,
                    output_dir=out,
                    archive_path=archive,
                )
            self.assertFalse(out.exists())
            self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
