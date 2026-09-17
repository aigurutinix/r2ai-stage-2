from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "leaderboard_scores.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("leaderboard_scores", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_preserves_dashboard_metric_order() -> None:
    row = {
        "id": 2231,
        "filename": "BM25 Baseline",
        "created_when": "2026-07-31T00:00:00Z",
        "status": "Finished",
        "on_leaderboard": True,
        "scores": [
            {"column_key": key, "score": index / 10}
            for index, key in enumerate(reversed(MODULE.SCORE_ORDER), 1)
        ],
    }

    normalized = MODULE.normalize(row)

    assert tuple(normalized["scores"]) == (
        "EXECUTION_ACCURACY",
        "TABLES_F2MACRO",
        "DOCS_F2MACRO",
        "TABLES_PRECISION",
        "TABLES_RECALL",
        "TABLES_MRR5",
        "DOCS_PRECISION",
        "DOCS_RECALL",
        "DOCS_MRR5",
        "ANSWER_ACCURACY",
    )
