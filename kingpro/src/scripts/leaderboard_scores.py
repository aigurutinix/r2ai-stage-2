"""Read Codabench submission scores through an already logged-in Chrome tab.

This is deliberately read-only: it calls the same GET endpoint used by the
results page through Chrome DevTools and never reads, prints, or stores browser
cookies. Chrome must already expose DevTools on port 9222.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from submit_via_cdp import Cdp


DEFAULT_TARGET = "leaderboard.aiguru.com.vn/competitions/14/"
DEFAULT_PHASE = 40
# Match the left-to-right order rendered by the competition dashboard.  This
# is deliberately separate from any intuitive grouping of metrics: Answer is
# the final column on the current R2AI leaderboard, not the second column.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read all R2AI leaderboard scores")
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--phase", type=int, default=DEFAULT_PHASE)
    parser.add_argument("--target-fragment", default=DEFAULT_TARGET)
    parser.add_argument("--limit", type=int, default=0, help="0 prints every submission")
    parser.add_argument("--json", action="store_true", help="emit normalized JSON")
    parser.add_argument("--out", type=Path, help="optional normalized JSON output")
    return parser.parse_args()


def page_targets(port: int, fragment: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/json/list", timeout=5
    ) as response:
        items = json.load(response)
    return [
        item
        for item in items
        if item.get("type") == "page" and fragment in item.get("url", "")
    ]


def choose_target(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise SystemExit("No logged-in competition tab found on Chrome DevTools")
    results = [item for item in items if "#/results-tab" in item.get("url", "")]
    return results[0] if results else items[0]


def browser_get(cdp: Cdp, path: str) -> dict[str, Any]:
    expression = (
        "(async()=>{const r=await fetch(" + json.dumps(path) + ");"
        "const body=await r.json();return {status:r.status,body};})()"
    )
    result = cdp.call(
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    value = result["result"].get("value")
    if not isinstance(value, dict) or value.get("status") != 200:
        raise RuntimeError(f"leaderboard GET failed: {value!r}")
    return value["body"]


def read_submissions(cdp: Cdp, phase: int) -> list[dict[str, Any]]:
    path: str | None = f"/api/submissions/?phase={phase}&page=1&page_size=50"
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    while path:
        payload = browser_get(cdp, path)
        for row in payload.get("results", []):
            submission_id = int(row["id"])
            if submission_id not in seen:
                rows.append(row)
                seen.add(submission_id)
        next_url = payload.get("next")
        if next_url and "/api/" in next_url:
            path = "/api/" + next_url.split("/api/", 1)[1]
        else:
            path = None
    return sorted(rows, key=lambda row: int(row["id"]), reverse=True)


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    scores = {item["column_key"]: float(item["score"]) for item in row.get("scores", [])}
    return {
        "id": int(row["id"]),
        "filename": row.get("filename", ""),
        "created_when": row.get("created_when"),
        "status": row.get("status"),
        "on_leaderboard": bool(row.get("on_leaderboard")),
        "scores": {key: scores.get(key) for key in SCORE_ORDER},
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print("ID    LB  Status     Exec    Answer  TblF2   DocF2   Filename")
    for row in rows:
        scores = row["scores"]
        fmt = lambda key: "—" if scores[key] is None else f"{scores[key]:.4f}"
        print(
            f"{row['id']:<5} {'*' if row['on_leaderboard'] else '-':<3} "
            f"{row['status']:<10} {fmt('EXECUTION_ACCURACY'):<7} "
            f"{fmt('ANSWER_ACCURACY'):<7} {fmt('TABLES_F2MACRO'):<7} "
            f"{fmt('DOCS_F2MACRO'):<7} {row['filename']}"
        )


def main() -> int:
    args = parse_args()
    target = choose_target(page_targets(args.port, args.target_fragment))
    cdp = Cdp(target["webSocketDebuggerUrl"])
    try:
        rows = [normalize(row) for row in read_submissions(cdp, args.phase)]
    finally:
        cdp.close()
    if args.limit > 0:
        rows = rows[: args.limit]
    payload = {"phase": args.phase, "count": len(rows), "submissions": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
