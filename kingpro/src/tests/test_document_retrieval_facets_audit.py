from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_document_retrieval_facets import audit  # noqa: E402


class DocumentRetrievalFacetAuditTests(unittest.TestCase):
    def test_checked_in_full_reports_pass(self) -> None:
        report = audit()
        self.assertTrue(report["passed"])
        self.assertGreater(report["metrics"]["macro_f2_delta"], 0.04)
        self.assertGreater(report["metrics"]["macro_recall_delta"], 0.04)

    def test_f2_regression_fails_closed(self) -> None:
        chosen_path = (
            ROOT
            / "build"
            / "demo_compliance"
            / "document_retrieval_coverage_semantic_v20_final.json"
        )
        payload = json.loads(chosen_path.read_text(encoding="utf-8"))
        payload["modes"]["1"]["macro_f2"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "chosen.json"
            modified.write_text(json.dumps(payload), encoding="utf-8")
            report = audit(chosen_path=modified)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["macro_f2_improved"])


if __name__ == "__main__":
    unittest.main()
