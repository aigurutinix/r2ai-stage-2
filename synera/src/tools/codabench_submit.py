"""Submit one audited Road2AI archive through Codabench's documented REST flow.

Credentials are read interactively and are never written to disk or printed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

import requests


TERMINAL_STATES = {"Finished", "Failed", "Cancelled", "Unknown"}


def checked(response: requests.Response, step: str) -> requests.Response:
    if response.ok:
        return response
    body = response.text[:2_000]
    raise RuntimeError(f"{step} failed: HTTP {response.status_code}: {body}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--username", required=True)
    parser.add_argument("--base-url", default="https://leaderboard.aiguru.com.vn")
    parser.add_argument("--competition", type=int, default=14)
    parser.add_argument("--phase", type=int, default=40)
    parser.add_argument("--task", type=int, default=46)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    archive = args.archive.resolve(strict=True)
    if archive.suffix.lower() != ".zip":
        raise ValueError("archive must be a ZIP file")

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
    token = token_response.json()["token"]
    session.headers.update({"Authorization": f"Token {token}"})

    eligibility = checked(
        session.get(f"{base}/api/can_make_submission/{args.phase}/", timeout=30),
        "eligibility check",
    ).json()
    if not eligibility.get("can"):
        raise RuntimeError(f"submission rejected before upload: {eligibility.get('reason')}")

    size = archive.stat().st_size
    dataset_payload = {
        "type": "submission",
        "competition": args.competition,
        "request_sassy_file_name": archive.name,
        "file_name": archive.name,
        "file_size": size,
    }
    dataset = checked(
        session.post(f"{base}/api/datasets/", json=dataset_payload, timeout=30),
        "dataset creation",
    ).json()

    with archive.open("rb") as handle:
        upload = requests.put(
            dataset["sassy_url"],
            data=handle,
            headers={"Content-Type": "application/zip"},
            timeout=180,
        )
    checked(upload, "archive upload")
    checked(
        session.put(f"{base}/api/datasets/completed/{dataset['key']}/", json={}, timeout=30),
        "upload completion",
    )

    submission_payload = {
        "data": dataset["key"],
        "phase": args.phase,
        "fact_sheet_answers": {},
        "tasks": [args.task],
        "organization": None,
        "queue": None,
    }
    submission = checked(
        session.post(f"{base}/api/submissions/", json=submission_payload, timeout=30),
        "submission creation",
    ).json()
    submission_id = int(submission["id"])
    print(json.dumps({"submission_id": submission_id, "status": submission.get("status")}), flush=True)

    deadline = time.monotonic() + args.timeout_seconds
    last_status = None
    result = submission
    while time.monotonic() < deadline:
        result = checked(
            session.get(f"{base}/api/submissions/{submission_id}/", timeout=30),
            "submission polling",
        ).json()
        status = result.get("status")
        if status != last_status:
            print(json.dumps({"submission_id": submission_id, "status": status}), flush=True)
            last_status = status
        if status in TERMINAL_STATES:
            break
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError(f"submission {submission_id} did not finish before timeout")

    safe_result = {
        "submission_id": submission_id,
        "filename": result.get("filename"),
        "status": result.get("status"),
        "status_details": result.get("status_details"),
        "scores": result.get("scores", []),
    }
    print(json.dumps(safe_result, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "Finished" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
