"""Tests for the rollbackable v272 q749/q780 lineage cleanup."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v272_q749_q780_cleanup as builder


class V272CleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = builder.SOURCE
        cls.candidate = builder.OUTPUT

    def test_candidate_exists_and_submission_is_byte_identical(self) -> None:
        self.assertTrue(self.candidate.is_dir())
        self.assertEqual(
            hashlib.sha256((self.baseline / "submission.json").read_bytes()).digest(),
            hashlib.sha256((self.candidate / "submission.json").read_bytes()).digest(),
        )

    def test_non_lineage_payload_is_byte_identical(self) -> None:
        allowed = {
            Path("data/q749_source_cells.csv"),
            Path("data/q780_source_cells.csv"),
            Path("source_audit.json"),
            Path("v272_q749_q780_cleanup_audit.json"),
        }
        baseline_files = {p.relative_to(self.baseline) for p in self.baseline.rglob("*") if p.is_file() and p.relative_to(self.baseline) not in allowed}
        candidate_files = {p.relative_to(self.candidate) for p in self.candidate.rglob("*") if p.is_file() and p.relative_to(self.candidate) not in allowed}
        self.assertEqual(baseline_files, candidate_files)
        for relative in baseline_files:
            self.assertEqual(hashlib.sha256((self.baseline / relative).read_bytes()).digest(), hashlib.sha256((self.candidate / relative).read_bytes()).digest(), relative)

    def test_only_two_lineage_changes_are_present(self) -> None:
        q749 = self.candidate / "data" / "q749_source_cells.csv"
        self.assertTrue(q749.is_file())
        with q749.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["ticker"], row["raw"], row["row_idx"]) for row in rows], [("EIB", "380", "2"), ("ACB", "450.276", "1")])
        self.assertEqual(rows[1]["typed_factor"], "1000.0")
        with (self.candidate / "data" / "q780_source_cells.csv").open(encoding="utf-8-sig", newline="") as handle:
            q780 = list(csv.DictReader(handle))
        self.assertEqual(q780[1]["typed_factor"], "1.0")
        self.assertEqual(q780[1]["raw"], "(210.684)")

    def test_q749_manifest_survives_pandas_numeric_inference(self) -> None:
        frame = pd.read_csv(self.candidate / "data" / "q749_source_cells.csv")
        values = [
            float(row.raw) * float(row.typed_factor) * float(row.scale)
            for row in frame.itertuples(index=False)
        ]
        self.assertEqual(values, [380.0, 450276.0])
        self.assertEqual(values[0] - values[1], -449896.0)

    def test_source_audit_coordinates_and_answers(self) -> None:
        audit = json.loads((self.candidate / "source_audit.json").read_text(encoding="utf-8-sig"))
        by_id = {int(item["id"]): item for item in audit}
        self.assertEqual(by_id[749]["answer"], -449896.0)
        self.assertEqual(by_id[749]["sources"][1]["row"], 1)
        self.assertEqual(by_id[749]["sources"][1]["typed_factor"], 1000.0)
        self.assertEqual(by_id[780]["answer"], 1113749.0)
        self.assertEqual(by_id[780]["sources"][1]["typed_factor"], 1.0)

    def test_builder_refuses_overwrite_for_rollback_safety(self) -> None:
        with self.assertRaises(FileExistsError):
            builder.build()


if __name__ == "__main__":
    unittest.main()
