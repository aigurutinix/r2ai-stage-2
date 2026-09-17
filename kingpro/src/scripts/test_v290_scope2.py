"""Exact diff and trusted-payload tests for rollbackable V290."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v290_scope2 as builder


def keyed(path: Path) -> dict[int, dict]:
    return {int(row["id"]): row for row in json.loads(path.read_text(encoding="utf-8-sig"))}


class V290Tests(unittest.TestCase):
    def test_overwrite_guard(self) -> None:
        with self.assertRaises(FileExistsError):
            builder.build()

    def test_exact_submission_diff_and_q714_preserved(self) -> None:
        base = keyed(builder.SOURCE / "submission.json")
        candidate = keyed(builder.OUTPUT / "submission.json")
        diffs = []
        for question_id in base:
            fields = sorted(
                field
                for field in set(base[question_id]) | set(candidate[question_id])
                if base[question_id].get(field) != candidate[question_id].get(field)
            )
            if fields:
                diffs.append((question_id, fields))
        self.assertEqual(
            diffs,
            [
                (98, ["answer", "relevant_docs", "relevant_tables"]),
                (764, ["answer", "relevant_tables"]),
            ],
        )
        self.assertEqual(candidate[714], base[714])
        self.assertEqual(
            (builder.OUTPUT / "data" / "q714_source_cells.csv").read_bytes(),
            (builder.SOURCE / "data" / "q714_source_cells.csv").read_bytes(),
        )

    def test_q98_exact_v225_union_payload(self) -> None:
        candidate = keyed(builder.OUTPUT / "submission.json")[98]
        trusted = keyed(builder.Q98_PAYLOAD / "submission.json")[98]
        for field in ("answer", "relevant_docs", "relevant_tables", "pandas_query", "evidence"):
            self.assertEqual(candidate[field], trusted[field])
        self.assertEqual(
            candidate["relevant_docs"],
            ["HUT_financial_statements_2024_separate", "HUT_financial_statements_2024_consolidated"],
        )
        self.assertEqual(
            candidate["relevant_tables"],
            ["HUT_financial_statements_2024_separate|328", "HUT_financial_statements_2024_consolidated|325"],
        )
        self.assertEqual(
            (builder.OUTPUT / "data" / "q98_source_cells.csv").read_bytes(),
            (builder.Q98_PAYLOAD / "data" / "q98_source_cells.csv").read_bytes(),
        )
        self.assertEqual(
            keyed(builder.OUTPUT / "source_audit.json")[98],
            keyed(builder.Q98_PAYLOAD / "source_audit.json")[98],
        )

    def test_q764_exact_v287_payload_with_docs_preserved(self) -> None:
        candidate = keyed(builder.OUTPUT / "submission.json")[764]
        trusted = keyed(builder.Q764_PAYLOAD / "submission.json")[764]
        baseline = keyed(builder.SOURCE / "submission.json")[764]
        for field in ("answer", "relevant_tables", "pandas_query", "evidence"):
            self.assertEqual(candidate[field], trusted[field])
        self.assertEqual(candidate["relevant_docs"], baseline["relevant_docs"])
        self.assertEqual(
            (builder.OUTPUT / "data" / "q764_source_cells.csv").read_bytes(),
            (builder.Q764_PAYLOAD / "data" / "q764_source_cells.csv").read_bytes(),
        )
        self.assertEqual(
            keyed(builder.OUTPUT / "source_audit.json")[764],
            keyed(builder.Q764_PAYLOAD / "source_audit.json")[764],
        )

    def test_non_target_payload_byte_identical(self) -> None:
        allowed = {
            Path("submission.json"),
            Path("source_audit.json"),
            Path("data/q98_source_cells.csv"),
            Path("data/q764_source_cells.csv"),
            Path("data/DPM_financial_statements_2015_consolidated_1909.csv"),
            Path("v290_scope2_audit.json"),
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
