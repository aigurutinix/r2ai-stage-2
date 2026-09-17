"""Fetch one authenticated Codabench detailed-result HTML artifact safely.

The signed result URL is requested through an already logged-in competition
tab.  Browser cookies are never read or printed.  The returned URL is accepted
only when it points back to the expected leaderboard host and private result
prefix, then the HTML is saved locally for reproducible error analysis.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from leaderboard_scores import Cdp, choose_target, page_targets


EXPECTED_HOST = "leaderboard.aiguru.com.vn"
EXPECTED_PREFIX = "/private/detailed_result/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument(
        "--target-fragment",
        default="leaderboard.aiguru.com.vn/competitions/14/",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def signed_url(cdp: Cdp, submission_id: int) -> str:
    path = f"/api/submissions/{submission_id}/get_detail_result/"
    expression = (
        "(async()=>{const r=await fetch(" + json.dumps(path) + ");"
        "const body=await r.json();return {status:r.status,body};})()"
    )
    evaluated = cdp.call(
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        },
    )
    value = evaluated["result"].get("value")
    if not isinstance(value, dict) or value.get("status") != 200:
        raise RuntimeError(f"detail-result API failed: {value!r}")
    url = value.get("body")
    if not isinstance(url, str):
        raise RuntimeError("detail-result API did not return a URL")
    return url


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise RuntimeError("refusing non-HTTPS detailed-result URL")
    if parsed.hostname != EXPECTED_HOST:
        raise RuntimeError(f"refusing unexpected detailed-result host: {parsed.hostname}")
    if not parsed.path.startswith(EXPECTED_PREFIX):
        raise RuntimeError(f"refusing unexpected detailed-result path: {parsed.path}")


def browser_download(cdp: Cdp, url: str) -> tuple[bytes, str]:
    expression = (
        "(async()=>{const r=await fetch(" + json.dumps(url) + ");"
        "const text=await r.text();return {status:r.status,"
        "contentType:r.headers.get('content-type')||'',text};})()"
    )
    evaluated = cdp.call(
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        },
    )
    value = evaluated["result"].get("value")
    if not isinstance(value, dict) or value.get("status") != 200:
        raise RuntimeError(f"browser detailed-result fetch failed: {value!r}")
    text = value.get("text")
    if not isinstance(text, str):
        raise RuntimeError("browser detailed-result fetch returned no text")
    return text.encode("utf-8"), str(value.get("contentType", ""))


def cdp_stream_download(cdp: Cdp, url: str) -> tuple[bytes, str]:
    frame_id = cdp.call("Page.getFrameTree")["frameTree"]["frame"]["id"]
    loaded = cdp.call(
        "Network.loadNetworkResource",
        {
            "frameId": frame_id,
            "url": url,
            "options": {"disableCache": True, "includeCredentials": True},
        },
    ).get("resource", {})
    if not loaded.get("success"):
        raise RuntimeError(f"CDP stream fetch failed: {loaded!r}")
    stream = loaded.get("stream")
    if not stream:
        raise RuntimeError(f"CDP stream fetch returned no body: {loaded!r}")
    chunks: list[bytes] = []
    try:
        while True:
            part = cdp.call("IO.read", {"handle": stream, "size": 1024 * 1024})
            data = part.get("data", "")
            if part.get("base64Encoded"):
                chunks.append(base64.b64decode(data))
            else:
                chunks.append(str(data).encode("utf-8"))
            if part.get("eof"):
                break
    finally:
        cdp.call("IO.close", {"handle": stream})
    headers = loaded.get("headers") or {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "")
    return b"".join(chunks), content_type


def rendered_download(port: int, url: str) -> tuple[bytes, str]:
    encoded = urllib.parse.quote(url, safe="")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/json/new?{encoded}", method="PUT"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        target = json.load(response)
    target_id = target.get("id")
    websocket_url = target.get("webSocketDebuggerUrl")
    if not target_id or not websocket_url:
        raise RuntimeError("Chrome did not create a detailed-result target")

    cdp = Cdp(websocket_url)
    try:
        html = ""
        for _ in range(40):
            state = cdp.evaluate("document.readyState")
            html = cdp.evaluate("document.documentElement?.outerHTML || ''")
            if state == "complete" and isinstance(html, str) and html:
                break
            time.sleep(0.25)
        if not isinstance(html, str):
            raise RuntimeError("rendered detailed-result target returned no HTML")
        return html.encode("utf-8"), "text/html; rendered-by=chrome"
    finally:
        cdp.close()
        close_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/json/close/{target_id}", method="PUT"
        )
        try:
            urllib.request.urlopen(close_request, timeout=5).close()
        except OSError:
            pass


def main() -> int:
    args = parse_args()
    target = choose_target(page_targets(args.port, args.target_fragment))
    cdp = Cdp(target["webSocketDebuggerUrl"])
    try:
        url = signed_url(cdp, args.submission_id)
        validate_url(url)
        payload, content_type = browser_download(cdp, url)
        if not payload:
            payload, content_type = cdp_stream_download(cdp, url)
    finally:
        cdp.close()
    if not (
        content_type.startswith("text/html")
        or content_type.startswith("application/octet-stream")
    ):
        raise RuntimeError(f"unexpected content type: {content_type}")
    if not payload:
        payload, content_type = rendered_download(args.port, url)
    if b"<html" not in payload[:4096].lower():
        preview = payload[:200].decode("utf-8", errors="replace")
        raise RuntimeError(
            "downloaded artifact does not look like HTML; "
            f"bytes={len(payload)} preview={preview!r}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(
        json.dumps(
            {
                "submission_id": args.submission_id,
                "output": str(args.out.resolve()),
                "bytes": len(payload),
                "content_type": content_type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
