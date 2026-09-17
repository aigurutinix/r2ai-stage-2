from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_retrieval_reranker import audit  # noqa: E402


class RetrievalRerankerAuditTests(unittest.TestCase):
    def test_checked_in_full_reports_pass(self) -> None:
        report = audit()
        self.assertTrue(report["passed"])
        self.assertEqual(report["regressions"], [])
        self.assertGreater(
            report["bucket_deltas"]["all"]["pipeline_recall_at_8_delta"], 0.07
        )

    def test_bucket_regression_fails_closed(self) -> None:
        reranked_path = (
            ROOT / "build" / "demo_compliance" /
            "retrieval_provenance_semantic_v6_reranked.json"
        )
        payload = json.loads(reranked_path.read_text(encoding="utf-8"))
        payload["summary"]["ratio_language"]["pipeline_mrr40"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "reranked.json"
            modified.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            report = audit(reranked_path=modified)
        self.assertFalse(report["passed"])
        self.assertIn("ratio_language", report["regressions"])
        self.assertFalse(report["checks"]["no_bucket_regressed_on_recall_or_mrr"])


if __name__ == "__main__":
    unittest.main()
