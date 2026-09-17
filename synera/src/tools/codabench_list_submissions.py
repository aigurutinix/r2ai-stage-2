"""List the authenticated participant's Codabench submissions safely.

Credentials are requested interactively and are never written or printed.
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

import requests


def checked(response: requests.Response, step: str) -> requests.Response:
    if response.ok:
        return response
    raise RuntimeError(
        f"{step} failed: HTTP {response.status_code}: {response.text[:2_000]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--base-url", default="https://leaderboard.aiguru.com.vn")
    parser.add_argument("--phase", type=int, default=40)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        help="Only print these submission IDs after fetching the participant list.",
    )
    parser.add_argument("--download-prediction", type=int, metavar="SUBMISSION_ID")
    parser.add_argument("--download-scoring", type=int, metavar="SUBMISSION_ID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    password = getpass.getpass("Password: ")
    session = requests.Session()
    token_response = checked(
        session.post(
            f"{base}/api/api-token-auth/",
            data={"username": args.username, "password": password},
            timeout=30,
        ),
        "authentication",
    )
    session.headers.update({"Authorization": f"Token {token_response.json()['token']}"})

    download_id = args.download_prediction or args.download_scoring
    if download_id is not None:
        if args.output is None:
            parser.error("--output is required with a download option")
        details = checked(
            session.get(
                f"{base}/api/submissions/{download_id}/get_details/",
                timeout=30,
            ),
            "submission details",
        ).json()
        result_key = "prediction_result" if args.download_prediction is not None else "scoring_result"
        result_url = details.get(result_key)
        if not result_url:
            raise RuntimeError(f"submission has no downloadable {result_key}")
        content = checked(session.get(result_url, timeout=180), f"{result_key} download").content
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        print(json.dumps({"output": str(output), "bytes": len(content)}))
        return 0

    url: str | None = f"{base}/api/submissions/"
    params: dict[str, object] | None = {
        "phase": args.phase,
        "page_size": min(args.limit, 500),
    }
    rows: list[dict[str, object]] = []
    while url and len(rows) < args.limit:
        payload = checked(session.get(url, params=params, timeout=30), "submission list").json()
        params = None
        if isinstance(payload, list):
            page = payload
            url = None
        else:
            page = payload.get("results", [])
            url = payload.get("next")
        for item in page:
            scores = item.get("scores", [])
            if args.primary_only:
                scores = [score for score in scores if score.get("is_primary")]
            rows.append(
                {
                    "id": item.get("id"),
                    "filename": item.get("filename"),
                    "created_when": item.get("created_when"),
                    "status": item.get("status"),
                    "scores": scores,
                }
            )
            if len(rows) >= args.limit:
                break

    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row.get("id") in wanted]
    rendered = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "rows": len(rows)}))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
