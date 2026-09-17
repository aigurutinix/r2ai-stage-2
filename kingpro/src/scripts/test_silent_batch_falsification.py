"""Unit tests for the deterministic v261 silent-batch falsification audit."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_silent_batch_falsification as audit


class SilentBatchFalsificationTests(unittest.TestCase):
    def test_displayed_parenthesized_million_value(self) -> None:
        self.assertEqual(audit.number("(210.684)"), -210684.0)

    def test_run_reproduces_all_current_answers(self) -> None:
        report = audit.run()
        self.assertEqual(report["summary"]["answer_mismatches"], 0)
        self.assertEqual(report["coverage"]["questions"], 10)

    def test_fail_closed_findings_are_specific(self) -> None:
        report = audit.run()
        by_id = {row["id"]: row for row in report["records"]}
        self.assertIn("lineage_manifest_missing", by_id[749]["flags"])
        self.assertIn("typed_factor_conflicts_with_parenthesized_physical_unit", by_id[780]["flags"])
        self.assertIn("near_tie_selector_gap_pp<=0.1", by_id[401]["flags"])
        for qid in (448, 385, 554, 563, 572, 533, 783):
            self.assertEqual(by_id[qid]["flags"], [])

    def test_report_is_json_serializable_and_has_top_alternatives(self) -> None:
        report = audit.run()
        json.dumps(report, ensure_ascii=False)
        self.assertTrue(all("top_alternatives" in row for row in report["records"]))


if __name__ == "__main__":
    unittest.main()
