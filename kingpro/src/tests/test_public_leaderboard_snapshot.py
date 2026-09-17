from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from snapshot_public_leaderboard import SCORE_ORDER, normalize_row, snapshot  # noqa: E402


class PublicLeaderboardSnapshotTests(unittest.TestCase):
    def test_normalizes_dashboard_order_and_finds_rank(self) -> None:
        scores = [
            {"column_key": key, "score": str(index / 10)}
            for index, key in enumerate(SCORE_ORDER)
        ]
        payload = {
            "id": 40,
            "primary_index": 0,
            "count": 2,
            "submissions": [
                {"id": 3622, "owner": "kingpro", "scores": scores},
                {"id": 1, "owner": "other", "scores": []},
            ],
        }
        report = snapshot(payload, "KINGPRO")
        self.assertEqual(report["rank"], 1)
        self.assertEqual(report["selected_submission"]["id"], 3622)
        self.assertEqual(
            list(report["selected_submission"]["scores"]), list(SCORE_ORDER)
        )

    def test_missing_metric_remains_unknown(self) -> None:
        row = normalize_row(
            {
                "id": 9,
                "owner": "kingpro",
                "scores": [{"column_key": "EXECUTION_ACCURACY", "score": "0.7"}],
            }
        )
        self.assertEqual(row["scores"]["EXECUTION_ACCURACY"], 0.7)
        self.assertIsNone(row["scores"]["ANSWER_ACCURACY"])


if __name__ == "__main__":
    unittest.main()
