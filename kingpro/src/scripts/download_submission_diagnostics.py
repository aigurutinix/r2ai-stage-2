"""Download diagnostics for one owned Codabench submission via logged-in Chrome.

Only GET requests are issued.  Browser cookies and signed storage URLs are never
printed or written to disk; the manifest contains only file metadata and hashes.
Chrome must already expose DevTools on port 9222 and have the competition open.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from leaderboard_scores import choose_target, page_targets
from submit_via_cdp import Cdp


DEFAULT_TARGET = "leaderboard.aiguru.com.vn/competitions/14/"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "build" / "submission_diagnostics"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download read-only diagnostics for an owned Codabench submission"
    )
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--target-fragment", default=DEFAULT_TARGET)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=50 * 1024 * 1024,
        help="refuse any individual response larger than this many bytes",
    )
    return parser.parse_args()


def evaluate(cdp: Cdp, expression: str) -> Any:
    result = cdp.call(
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    remote = result.get("result", {})
    if "exceptionDetails" in result or remote.get("subtype") == "error":
        raise RuntimeError(f"browser evaluation failed: {result!r}")
    return remote.get("value")


def get_json(cdp: Cdp, path: str) -> Any:
    expression = f"""
    (async () => {{
      const response = await fetch({json.dumps(path)});
      const text = await response.text();
      return {{status: response.status, text}};
    }})()
    """
    value = evaluate(cdp, expression)
    if not isinstance(value, dict) or value.get("status") != 200:
        raise RuntimeError(f"GET {path} failed: status={value.get('status') if isinstance(value, dict) else None}")
    return json.loads(value["text"])


def get_bytes(cdp: Cdp, url: str, max_bytes: int) -> tuple[bytes, str]:
    expression = f"""
    (async () => {{
      const response = await fetch({json.dumps(url)});
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.length > {max_bytes}) return {{status: 413, bytes: bytes.length}};
      let binary = '';
      const step = 0x8000;
      for (let i = 0; i < bytes.length; i += step) {{
        binary += String.fromCharCode(...bytes.subarray(i, i + step));
      }}
      return {{
        status: response.status,
        contentType: response.headers.get('content-type') || '',
        body: btoa(binary),
        bytes: bytes.length
      }};
    }})()
    """
    value = evaluate(cdp, expression)
    if not isinstance(value, dict) or value.get("status") != 200:
        raise RuntimeError(
            f"diagnostic download failed: status={value.get('status') if isinstance(value, dict) else None}, "
            f"bytes={value.get('bytes') if isinstance(value, dict) else None}"
        )
    data = base64.b64decode(value["body"], validate=True)
    if len(data) != value.get("bytes"):
        raise RuntimeError("download length mismatch")
    return data, str(value.get("contentType", ""))


def extension(data: bytes, content_type: str, default: str) -> str:
    if data.startswith(b"PK\x03\x04"):
        return ".zip"
    if "html" in content_type:
        return ".html"
    if "json" in content_type:
        return ".json"
    if content_type.startswith("text/"):
        return ".txt"
    return default


def safe_name(value: str) -> str:
    return SAFE_NAME.sub("_", value).strip("._") or "diagnostic"


def main() -> int:
    args = parse_args()
    target = choose_target(page_targets(args.port, args.target_fragment))
    cdp = Cdp(target["webSocketDebuggerUrl"])
    try:
        details = get_json(cdp, f"/api/submissions/{args.submission_id}/get_details/")
        detailed_url = get_json(
            cdp, f"/api/submissions/{args.submission_id}/get_detail_result/"
        )
        sources: list[tuple[str, str]] = []
        for item in details.get("logs", []):
            if isinstance(item, dict) and item.get("name") and item.get("data_file"):
                sources.append((f"log_{safe_name(str(item['name']))}", item["data_file"]))
        for name in ("scoring_result", "prediction_result"):
            if details.get(name):
                sources.append((name, details[name]))
        if isinstance(detailed_url, str) and detailed_url:
            sources.append(("detailed_result", detailed_url))

        out_dir = args.out_root.resolve() / str(args.submission_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        seen: set[tuple[str, str]] = set()
        for logical_name, url in sources:
            key = (logical_name, url)
            if key in seen:
                continue
            seen.add(key)
            data, content_type = get_bytes(cdp, url, args.max_bytes)
            suffix = extension(data, content_type, ".bin")
            path = out_dir / f"{logical_name}{suffix}"
            path.write_bytes(data)
            records.append(
                {
                    "name": logical_name,
                    "file": path.name,
                    "bytes": len(data),
                    "content_type": content_type,
                    "sha256": hashlib.sha256(data).hexdigest().upper(),
                }
            )
    finally:
        cdp.close()

    manifest = {
        "submission_id": args.submission_id,
        "request_mode": "GET-only via logged-in browser",
        "files": records,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded {len(records)} diagnostic files to {out_dir}")
    for record in records:
        print(f"  {record['file']:<42} {record['bytes']:>9} bytes")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
