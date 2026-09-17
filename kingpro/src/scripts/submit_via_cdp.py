"""Attach one approved submission archive through Chrome DevTools port 9222.

The competition page starts its own upload when CDP sets the file input. This
script deliberately does not dispatch an additional ``change`` event and does
not click any submit control. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import websocket


DEFAULT_TARGET = "leaderboard.aiguru.com.vn/competitions/14/#/participate-tab"
DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / ".submission-automation"


class Cdp:
    def __init__(self, websocket_url: str) -> None:
        self.ws = websocket.create_connection(
            websocket_url, suppress_origin=True, timeout=20
        )
        self.counter = 0

    def close(self) -> None:
        self.ws.close()

    def call(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        message_id = self.counter
        self.ws.send(
            json.dumps(
                {"id": message_id, "method": method, "params": params or {}}
            )
        )
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result", {})

    def evaluate(self, expression: str) -> object:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
        )
        return result["result"].get("value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or attach exactly one ZIP to the AI Guru file input via "
            "Chrome DevTools. Chrome must already be running with port 9222."
        )
    )
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--target-fragment", default=DEFAULT_TARGET)
    parser.add_argument("--target-id", help="Required only when no unique focused tab exists")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually set the file input once; omitted means read-only dry-run",
    )
    parser.add_argument(
        "--confirm-sha256",
        help="Full expected SHA-256; mandatory with --execute",
    )
    parser.add_argument(
        "--resume-change",
        action="store_true",
        help=(
            "Resume a locked file_set_once receipt by dispatching exactly one "
            "change event on the already-populated file input"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_archive(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"REFUSED: file does not exist: {path}")
    if path.suffix.lower() != ".zip":
        raise SystemExit("REFUSED: submission must be a .zip archive")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            bad_member = archive.testzip()
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"REFUSED: invalid ZIP: {exc}") from exc
    if bad_member:
        raise SystemExit(f"REFUSED: ZIP CRC failed at {bad_member}")
    duplicate_count = len(names) - len(set(names))
    if duplicate_count:
        raise SystemExit(f"REFUSED: ZIP has {duplicate_count} duplicate entries")
    return {"bytes": path.stat().st_size, "entries": len(names)}


def targets(port: int, fragment: str) -> list[dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
        items = json.load(response)
    return [
        item
        for item in items
        if item.get("type") == "page" and fragment in item.get("url", "")
    ]


def focus_state(item: dict) -> dict:
    cdp = Cdp(item["webSocketDebuggerUrl"])
    try:
        value = cdp.evaluate(
            "(() => ({focused: document.hasFocus(), visibility: document.visibilityState}))()"
        )
        return value if isinstance(value, dict) else {}
    finally:
        cdp.close()


def choose_target(items: list[dict], requested_id: str | None) -> dict:
    if requested_id:
        selected = [item for item in items if item.get("id") == requested_id]
        if len(selected) != 1:
            raise SystemExit(f"REFUSED: target id {requested_id!r} is not a matching page")
        return selected[0]
    if len(items) == 1:
        return items[0]
    focused = []
    for item in items:
        state = focus_state(item)
        if state.get("focused") and state.get("visibility") == "visible":
            focused.append(item)
    if len(focused) == 1:
        return focused[0]
    choices = [{"id": item.get("id"), "url": item.get("url")} for item in items]
    raise SystemExit(
        "REFUSED: no unique focused competition tab; use --target-id from: "
        + json.dumps(choices, ensure_ascii=False)
    )


def inspect_page(cdp: Cdp, filename: str) -> dict:
    quoted_name = json.dumps(filename)
    expression = f"""
    (() => {{
      const wanted = {quoted_name};
      const inputs = [...document.querySelectorAll('input[type=file][name=data_file]')];
      const body = document.body?.innerText || '';
      const inFlight = /(^|\\n)\\s*(Preparing|Scoring|Đang chuẩn bị|Đang chấm)\\s*($|\\n)/i.test(body);
      return {{
        url: location.href,
        title: document.title,
        focused: document.hasFocus(),
        visibility: document.visibilityState,
        inputCount: inputs.length,
        selectedFiles: inputs.flatMap(input => [...input.files].map(file => file.name)),
        filenameAlreadyVisible: body.includes(wanted),
        inFlight
      }};
    }})()
    """
    value = cdp.evaluate(expression)
    if not isinstance(value, dict):
        raise SystemExit("REFUSED: could not inspect competition page")
    return value


def refusal_reasons(page: dict, lock_path: Path) -> list[str]:
    reasons = []
    if page.get("inputCount") != 1:
        reasons.append(f"expected one input[name=data_file], found {page.get('inputCount')}")
    if page.get("selectedFiles"):
        reasons.append(f"file input is not empty: {page['selectedFiles']}")
    if page.get("filenameAlreadyVisible"):
        reasons.append("the same filename is already visible in submission history")
    if page.get("inFlight"):
        reasons.append("a Preparing/Scoring submission is visible")
    if lock_path.exists():
        reasons.append(f"this SHA already has a permanent one-shot lock: {lock_path}")
    return reasons


def write_lock(lock_path: Path, payload: dict) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"REFUSED: one-shot lock already exists: {lock_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def update_lock(lock_path: Path, payload: dict) -> None:
    lock_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    archive = args.file.expanduser().resolve()
    archive_info = validate_archive(archive)
    archive_sha = sha256(archive)
    lock_path = args.state_dir.resolve() / f"{archive_sha}.json"

    matching = targets(args.port, args.target_fragment)
    if not matching:
        raise SystemExit(
            f"REFUSED: no Chrome page on port {args.port} matches {args.target_fragment!r}"
        )
    target = choose_target(matching, args.target_id)
    cdp = Cdp(target["webSocketDebuggerUrl"])
    try:
        before = inspect_page(cdp, archive.name)
        if args.resume_change:
            if not args.execute:
                raise SystemExit("REFUSED: --resume-change requires --execute")
            if not args.confirm_sha256:
                raise SystemExit("REFUSED: --confirm-sha256 is mandatory with --resume-change")
            if args.confirm_sha256.strip().upper() != archive_sha:
                raise SystemExit("REFUSED: --confirm-sha256 does not match the file")
            if not lock_path.is_file():
                raise SystemExit("REFUSED: no one-shot receipt exists for --resume-change")
            receipt = json.loads(lock_path.read_text(encoding="utf-8"))
            if receipt.get("state") != "file_set_once":
                raise SystemExit(
                    "REFUSED: receipt state is not file_set_once: "
                    + repr(receipt.get("state"))
                )
            if before.get("selectedFiles") != [archive.name]:
                raise SystemExit(
                    "REFUSED: selected file does not exactly match the locked archive: "
                    + repr(before.get("selectedFiles"))
                )
            changed = cdp.evaluate(
                "(() => {"
                "const input=document.querySelector('input[type=file][name=data_file]');"
                "if(!input)return {ok:false,reason:'missing-input'};"
                "input.dispatchEvent(new Event('change',{bubbles:true}));"
                "return {ok:true,files:[...input.files].map(file=>file.name)};"
                "})()"
            )
            if not isinstance(changed, dict) or not changed.get("ok"):
                raise SystemExit("REFUSED: change event was not dispatched")
            receipt.update(
                state="change_dispatched_once",
                changeDispatchedAt=datetime.now(timezone.utc).isoformat(),
            )
            update_lock(lock_path, receipt)
            time.sleep(3)
            after = inspect_page(cdp, archive.name)
            print(
                json.dumps(
                    {
                        "result": "CHANGE_DISPATCHED_ONCE_DO_NOT_RETRY",
                        "lock": str(lock_path),
                        "event": changed,
                        "pageAfter": after,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        reasons = refusal_reasons(before, lock_path)
        report = {
            "mode": "execute" if args.execute else "dry-run",
            "file": str(archive),
            "sha256": archive_sha,
            **archive_info,
            "targetId": target["id"],
            "page": before,
            "eligible": not reasons,
            "refusalReasons": reasons,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))

        if not args.execute:
            return 0
        if reasons:
            raise SystemExit("REFUSED: execute preconditions failed")
        if not args.confirm_sha256:
            raise SystemExit("REFUSED: --confirm-sha256 is mandatory with --execute")
        if args.confirm_sha256.strip().upper() != archive_sha:
            raise SystemExit("REFUSED: --confirm-sha256 does not match the file")

        receipt = {
            "state": "armed",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "file": str(archive),
            "filename": archive.name,
            "sha256": archive_sha,
            "targetId": target["id"],
            "pageUrl": before.get("url"),
        }
        write_lock(lock_path, receipt)

        root = cdp.call("DOM.getDocument", {"depth": -1, "pierce": True})["root"]["nodeId"]
        node_id = cdp.call(
            "DOM.querySelector",
            {"nodeId": root, "selector": "input[type=file][name=data_file]"},
        ).get("nodeId")
        if not node_id:
            receipt.update(state="refused_after_lock", error="file input disappeared")
            update_lock(lock_path, receipt)
            raise SystemExit("REFUSED: file input disappeared after arming")

        # Critical invariant: exactly one CDP attachment call, no synthetic events,
        # no click. The React page owns the resulting upload lifecycle.
        cdp.call("DOM.setFileInputFiles", {"nodeId": node_id, "files": [str(archive)]})
        receipt.update(
            state="file_set_once",
            fileSetAt=datetime.now(timezone.utc).isoformat(),
        )
        update_lock(lock_path, receipt)

        time.sleep(2)
        after = inspect_page(cdp, archive.name)
        print(
            json.dumps(
                {
                    "result": "FILE_SET_ONCE_DO_NOT_RETRY",
                    "lock": str(lock_path),
                    "pageAfter": after,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        cdp.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, urllib.error.URLError, websocket.WebSocketException) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
