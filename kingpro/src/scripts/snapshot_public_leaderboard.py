"""Capture the public R2AI leaderboard without browser credentials.

This is a read-only fallback for the common Windows failure mode where Chrome
keeps the participant profile's cookie database locked.  It intentionally
reads only the public phase endpoint, so it can prove the selected row and its
complete score vector but cannot expose unselected submissions.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_URL = (
    "https://leaderboard.aiguru.com.vn/api/phases/40/"
    "get_leaderboard/?page=1&page_size=100"
)
SCORE_ORDER = (
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


def fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "KINGPRO-readonly-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_scores = {
        str(item["column_key"]): float(item["score"])
        for item in row.get("scores", [])
        if isinstance(item, dict) and "column_key" in item and "score" in item
    }
    return {
        "id": int(row["id"]),
        "owner": str(row.get("owner", "")),
        "created_when": row.get("created_when"),
        "scores": {key: raw_scores.get(key) for key in SCORE_ORDER},
    }


def snapshot(payload: dict[str, Any], owner: str) -> dict[str, Any]:
    rows = [normalize_row(row) for row in payload.get("submissions", [])]
    owner_rows = [row for row in rows if row["owner"].casefold() == owner.casefold()]
    return {
        "source": "public_phase_leaderboard",
        "phase_id": int(payload.get("id", 0) or 0),
        "primary_index": int(payload.get("primary_index", 0) or 0),
        "owner": owner,
        "selected_submission": owner_rows[0] if owner_rows else None,
        "rank": next(
            (index for index, row in enumerate(rows, start=1) if row in owner_rows),
            None,
        ),
        "leaderboard_count": int(payload.get("count", len(rows)) or len(rows)),
        "rows_returned": len(rows),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--owner", default="kingpro")
    parser.add_argument("--expect-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    report = snapshot(fetch_json(args.url, args.timeout), args.owner)
    report["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["endpoint"] = args.url
    selected = report["selected_submission"]
    if selected is None:
        raise SystemExit("owner not present on public leaderboard: {0}".format(args.owner))
    if args.expect_id is not None and selected["id"] != args.expect_id:
        raise SystemExit(
            "selected submission drift: expected {0}, got {1}".format(
                args.expect_id, selected["id"]
            )
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
