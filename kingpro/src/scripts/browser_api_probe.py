"""Inspect authenticated Codabench browser resources without exposing cookies.

The script connects to Chrome DevTools, selects already-open pages by URL
fragment, and prints resource URLs or performs same-origin GET probes.  It is
strictly read-only and useful for discovering score/detail endpoints used by
the competition UI.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any

from submit_via_cdp import Cdp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--page-fragment", required=True)
    parser.add_argument("--resource-pattern", default="api|detail|private")
    parser.add_argument("--get", action="append", default=[])
    parser.add_argument(
        "--expression",
        action="append",
        default=[],
        help="read-only JavaScript expression to evaluate in the matched page",
    )
    return parser.parse_args()


def pages(port: int, fragment: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/json/list", timeout=5
    ) as response:
        items = json.load(response)
    return [
        item
        for item in items
        if item.get("type") == "page" and fragment in item.get("url", "")
    ]


def main() -> int:
    args = parse_args()
    matched = pages(args.port, args.page_fragment)
    if not matched:
        raise SystemExit("No matching Chrome page")

    output: list[dict[str, Any]] = []
    for item in matched:
        cdp = Cdp(item["webSocketDebuggerUrl"])
        try:
            pattern = json.dumps(args.resource_pattern)
            resources = cdp.evaluate(
                "(() => {"
                f"const re = new RegExp({pattern}, 'i');"
                "return performance.getEntriesByType('resource')"
                ".map(entry => entry.name).filter(name => re.test(name));"
                "})()"
            )
            probes = []
            for path in args.get:
                expression = (
                    "(async()=>{const r=await fetch(" + json.dumps(path) + ");"
                    "const text=await r.text();return {status:r.status,"
                    "contentType:r.headers.get('content-type'),"
                    "text:text.slice(0,4000)};})()"
                )
                evaluated = cdp.call(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                )
                response = evaluated["result"].get("value")
                probes.append({"path": path, "response": response})
            expressions = []
            for expression in args.expression:
                expressions.append(
                    {"expression": expression, "value": cdp.evaluate(expression)}
                )
            output.append(
                {
                    "page": item.get("url"),
                    "resources": resources,
                    "probes": probes,
                    "expressions": expressions,
                }
            )
        finally:
            cdp.close()

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
