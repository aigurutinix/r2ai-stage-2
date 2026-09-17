"""Exact semantic-diff and physical-masthead tests for rollbackable V283."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v283_hut_scope as builder


class V283Tests(unittest.TestCase):
    def test_refuses_overwrite(self) -> None:
        with self.assertRaises(FileExistsError):
            builder.build()

    def test_submission_semantic_diff_is_exact(self) -> None:
        base = {
            int(row["id"]): row
            for row in json.loads((builder.SOURCE / "submission.json").read_text(encoding="utf-8"))
        }
        candidate = {
            int(row["id"]): row
            for row in json.loads((builder.OUTPUT / "submission.json").read_text(encoding="utf-8"))
        }
        diffs = []
        for question_id in base:
            fields = sorted(
                key
                for key in set(base[question_id]) | set(candidate[question_id])
                if base[question_id].get(key) != candidate[question_id].get(key)
            )
            if fields:
                diffs.append((question_id, fields))
        self.assertEqual(
            diffs,
            [
                (98, ["answer", "relevant_docs", "relevant_tables"]),
                (714, ["answer", "relevant_docs", "relevant_tables"]),
            ],
        )
        self.assertEqual(candidate[98]["answer"], 146.47)
        self.assertEqual(candidate[714]["answer"], 168.74)
        for question_id in (98, 714):
            self.assertEqual(
                candidate[question_id]["pandas_query"], base[question_id]["pandas_query"]
            )
            self.assertEqual(candidate[question_id]["evidence"], base[question_id]["evidence"])

    def test_compact_manifests_are_exact(self) -> None:
        expected = {
            98: [("cdkt:140", "146.469.679.444", "HUT_financial_statements_2024_consolidated|325", "12", "4")],
            714: [
                ("kqkd:21", "874.739.630.652", "HUT_financial_statements_2024_separate|399", "6", "4"),
                ("kqkd:22", "706.004.285.205", "HUT_financial_statements_2024_separate|399", "7", "4"),
            ],
        }
        for question_id, wanted in expected.items():
            with (builder.OUTPUT / "data" / f"q{question_id}_source_cells.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            actual = [
                (row["metric_key"], row["raw"], row["source_table"], row["row_idx"], row["col_idx"])
                for row in rows
            ]
            self.assertEqual(actual, wanted)

    def test_source_audit_and_masthead_proof(self) -> None:
        audits = {
            int(row["id"]): row
            for row in json.loads((builder.OUTPUT / "source_audit.json").read_text(encoding="utf-8"))
        }
        self.assertEqual(audits[98]["answer"], 146.47)
        self.assertEqual(audits[714]["answer"], 168.74)
        proof = json.loads(
            (builder.OUTPUT / "v283_hut_scope_audit.json").read_text(encoding="utf-8")
        )["physical_masthead_proof"]
        self.assertEqual(len(proof["checks"]), 3)
        self.assertIn("RIENG", proof["container_swap"]["HUT_financial_statements_2024_consolidated"])
        self.assertIn("HOP NHAT", proof["container_swap"]["HUT_financial_statements_2024_separate"])

    def test_non_target_files_are_byte_identical(self) -> None:
        allowed = {
            Path("submission.json"),
            Path("source_audit.json"),
            Path("data/q98_source_cells.csv"),
            Path("data/q714_source_cells.csv"),
            Path("v283_hut_scope_audit.json"),
        }
        base = {
            path.relative_to(builder.SOURCE)
            for path in builder.SOURCE.rglob("*")
            if path.is_file() and path.relative_to(builder.SOURCE) not in allowed
        }
        candidate = {
            path.relative_to(builder.OUTPUT)
            for path in builder.OUTPUT.rglob("*")
            if path.is_file() and path.relative_to(builder.OUTPUT) not in allowed
        }
        self.assertEqual(base, candidate)
        for relative in base:
            self.assertEqual(
                hashlib.sha256((builder.SOURCE / relative).read_bytes()).digest(),
                hashlib.sha256((builder.OUTPUT / relative).read_bytes()).digest(),
                str(relative),
            )


if __name__ == "__main__":
    unittest.main()
