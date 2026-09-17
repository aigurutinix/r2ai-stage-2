"""Read authenticated Codabench phase metadata without persisting credentials.

The password is requested through ``getpass``.  Authentication tokens, signed
URLs, storage keys, and other secrets are removed from the printed payload.
"""

from __future__ import annotations

import argparse
import getpass
import json
from typing import Any

import requests


SENSITIVE_PARTS = ("token", "password", "secret", "sassy", "url", "key")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in SENSITIVE_PARTS)
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def checked(response: requests.Response, label: str) -> requests.Response:
    if response.ok:
        return response
    raise RuntimeError(
        f"{label} failed: HTTP {response.status_code}: {response.text[:1000]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--base-url", default="https://leaderboard.aiguru.com.vn")
    parser.add_argument("--phase", type=int, default=40)
    parser.add_argument("--competition", type=int, default=14)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    password = getpass.getpass("Password: ")
    session = requests.Session()
    response = checked(
        session.post(
            f"{base}/api/api-token-auth/",
            data={"username": args.username, "password": password},
            timeout=30,
        ),
        "authentication",
    )
    session.headers.update({"Authorization": f"Token {response.json()['token']}"})

    payload: dict[str, Any] = {}
    for label, path in (
        ("phase", f"/api/phases/{args.phase}/"),
        ("competition", f"/api/competitions/{args.competition}/"),
    ):
        item = checked(session.get(base + path, timeout=30), label).json()
        payload[label] = sanitize(item)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
