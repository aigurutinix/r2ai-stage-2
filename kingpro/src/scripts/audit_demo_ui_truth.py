"""Read-only browser audit for the canonical low-clutter KINGPRO UI.

The audit uses an isolated headless Chrome profile. It proves the local UI can
run one grounded question, expose measured backend trace and fixed evidence,
and animate the Trace, Evidence and Chat/Batch controls. It does not attest a
remote model, private score or BTC eligibility.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import websocket


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

REQUIRED_DEMO_TEXT = (
    "KINGPRO",
    "FINANCIAL QA",
    "Chat",
    "Batch",
    "Stage-safe",
    "1012 verified programs",
    "Nguồn kiểm chứng",
)

FORBIDDEN_DEMO_TEXT = (
    "System operational",
    "Runtime Operational",
    "Ready · 58/58 audit",
    "115/115 tests",
    "PUBLIC SCORE · V184 · ID 3492",
    "PUBLIC SCORE · V276 · ID 3742",
    "PUBLIC SCORE · V290 · ID 3745",
    "PUBLIC SCORE · V206 · ID 3605",
    "PUBLIC SCORE · V207 · ID 3622",
    "Compliance candidate v188",
    "artifact này chưa được đo",
)


class Cdp:
    def __init__(self, websocket_url: str) -> None:
        self.ws = websocket.create_connection(websocket_url, suppress_origin=True, timeout=10)
        self.counter = 0
        self.events: list[dict[str, Any]] = []

    def close(self) -> None:
        self.ws.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        message_id = self.counter
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                result = message.get("result", {})
                return result if isinstance(result, dict) else {}
            self.events.append(message)

    def evaluate(self, expression: str) -> Any:
        response = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        result = response.get("result", {})
        if result.get("subtype") == "error":
            raise RuntimeError(result.get("description") or "browser evaluation failed")
        return result.get("value")

    def drain(self, seconds: float = 0.35) -> None:
        deadline = time.monotonic() + seconds
        self.ws.settimeout(0.05)
        try:
            while time.monotonic() < deadline:
                try:
                    self.events.append(json.loads(self.ws.recv()))
                except websocket.WebSocketTimeoutException:
                    continue
        finally:
            self.ws.settimeout(10)


def _wait_for(predicate: Callable[[], bool], label: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for {0}".format(label))


def _target(port: int) -> dict[str, Any]:
    with urllib.request.urlopen("http://127.0.0.1:{0}/json/list".format(port), timeout=2) as response:
        items = json.load(response)
    pages = [item for item in items if item.get("type") == "page"]
    if len(pages) != 1:
        raise RuntimeError("expected one isolated Chrome page, found {0}".format(len(pages)))
    return pages[0]


def _click_text(cdp: Cdp, selector: str, text: str) -> bool:
    expression = """
    (() => {
      const wanted = %s;
      const node = [...document.querySelectorAll(%s)]
        .find((item) => (item.textContent || '').includes(wanted));
      if (!node) return false;
      node.click();
      return true;
    })()
    """ % (json.dumps(text), json.dumps(selector))
    return cdp.evaluate(expression) is True


def _console_errors(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for event in events:
        method = event.get("method")
        params = event.get("params") if isinstance(event.get("params"), dict) else {}
        if method == "Runtime.exceptionThrown":
            detail = params.get("exceptionDetails", {})
            errors.append(str(detail.get("text") or detail.get("exception") or "runtime exception"))
        if method == "Log.entryAdded":
            entry = params.get("entry", {})
            if isinstance(entry, dict) and entry.get("level") == "error":
                errors.append(str(entry.get("text") or "browser log error"))
    return errors


def audit(url: str, chrome: Path, port: int, screenshot: Path) -> dict[str, Any]:
    if not chrome.is_file():
        raise FileNotFoundError("Chrome not found: {0}".format(chrome))
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kingpro-ui-audit-") as profile:
        process = subprocess.Popen(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--remote-debugging-port={0}".format(port),
                "--user-data-dir={0}".format(profile),
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp: Cdp | None = None
        try:
            _wait_for(lambda: _target(port).get("webSocketDebuggerUrl") is not None, "Chrome DevTools")
            cdp = Cdp(str(_target(port)["webSocketDebuggerUrl"]))
            for domain in ("Page.enable", "Runtime.enable", "Log.enable"):
                cdp.call(domain)
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
            )
            cdp.call("Page.navigate", {"url": url})
            _wait_for(
                lambda: cdp.evaluate("document.readyState === 'complete'") is True
                and "Anh muốn phân tích báo cáo nào?" in str(cdp.evaluate("document.body.innerText")),
                "canonical KINGPRO workspace",
            )
            _wait_for(
                lambda: "1012 verified programs" in str(cdp.evaluate("document.body.innerText")).casefold(),
                "live health payload",
            )
            initial_text = str(cdp.evaluate("document.body.innerText"))
            folded_initial = initial_text.casefold()
            required = {text: text.casefold() in folded_initial for text in REQUIRED_DEMO_TEXT}
            forbidden_absent = {
                text: text.casefold() not in folded_initial for text in FORBIDDEN_DEMO_TEXT
            }

            scenario_clicked = _click_text(cdp, ".suggestions button", "Lãi tiền gửi năm 2018")
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.assistantMessage') !== null") is True
                and "208.253,2" in str(cdp.evaluate("document.body.innerText")),
                "grounded registry answer",
                timeout=30.0,
            )
            answer_text = str(cdp.evaluate("document.body.innerText"))
            grounded_answer_visible = all(
                marker in answer_text
                for marker in ("Verified registry replay", "208.253,2", "triệu đồng")
            )
            measured_trace_visible = (
                "MEASURED BACKEND TRACE" in answer_text
                and cdp.evaluate("document.querySelector('.traceMotion.open') !== null") is True
            )
            evidence_visible = all(
                marker in answer_text
                for marker in ("Bảng dữ liệu đã dùng", "VJC_financial_statements_2018_separate")
            )
            _wait_for(
                lambda: cdp.evaluate(
                    "document.querySelector('.tableLaunch:not(:disabled)') !== null"
                ) is True,
                "source table preview readiness",
            )
            table_modal_clicked = _click_text(cdp, ".tableLaunch", "Mở bảng lớn")
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.tableModal') !== null") is True,
                "large source table modal",
            )
            modal_text = str(cdp.evaluate("document.body.innerText")).casefold()
            table_modal_visible = all(
                marker.casefold() in modal_text
                for marker in ("Bảng nguồn đã dùng", "Lãi tiền gửi", "208.253.201.298")
            )
            table_headers = cdp.evaluate(
                "[...document.querySelectorAll('.tableModal thead th')].map((node) => node.textContent)"
            )
            excel_headers_promoted = table_headers == ["Chỉ tiêu", "2018 · VND", "2017 · VND"]
            source_cell = cdp.evaluate(
                """
                (() => {
                  const cell = document.querySelector('.selectedSourceCell');
                  const row = cell?.closest('tr');
                  return {
                    count: document.querySelectorAll('.selectedSourceCell').length,
                    text: cell?.textContent,
                    rowLabel: row?.querySelector('td')?.textContent,
                    column: cell ? [...row.children].indexOf(cell) : -1,
                  };
                })()
                """
            )
            causal_lineage_visible = source_cell == {
                "count": 1,
                "text": "208.253.201.298",
                "rowLabel": "Lãi tiền gửi",
                "column": 1,
            }
            table_modal_closed = cdp.evaluate(
                "document.querySelector('.tableModal header button')?.click(); true"
            ) is True
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.tableModal') === null") is True,
                "close large source table modal",
            )

            trace_hide = _click_text(cdp, ".traceToggle", "Ẩn luồng")
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.traceMotion.closed') !== null") is True,
                "trace collapse motion",
            )
            trace_show = _click_text(cdp, ".traceToggle", "Hiện luồng")
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.traceMotion.open') !== null") is True,
                "trace expand motion",
            )
            evidence_hide = cdp.evaluate("document.querySelector('.evidenceToggle')?.click(); true") is True
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.evidencePanel.closed') !== null") is True,
                "evidence collapse motion",
            )
            evidence_show = cdp.evaluate("document.querySelector('.evidenceToggle')?.click(); true") is True
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.evidencePanel.open') !== null") is True,
                "evidence expand motion",
            )
            batch_clicked = _click_text(cdp, ".modeSwitch button", "Batch")
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.batchPage') !== null") is True,
                "batch workspace",
            )
            chat_clicked = _click_text(cdp, ".modeSwitch button", "Chat")
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.conversation') !== null") is True,
                "return to chat",
            )

            image = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            screenshot.write_bytes(base64.b64decode(str(image.get("data", ""))))
            cdp.drain()
            console_errors = _console_errors(cdp.events)
            checks = {
                "required_truthful_claims_visible": all(required.values()),
                "stale_or_misleading_claims_absent": all(forbidden_absent.values()),
                "grounded_scenario_runs_on_real_api": scenario_clicked and grounded_answer_visible,
                "measured_backend_trace_visible": measured_trace_visible,
                "fixed_evidence_panel_bound_to_source": evidence_visible,
                "large_table_modal_interactive": (
                    table_modal_clicked and table_modal_visible and table_modal_closed
                ),
                "generic_columns_promoted_to_excel_headers": excel_headers_promoted,
                "counterfactual_cell_lineage_visible": causal_lineage_visible,
                "trace_motion_interactive": trace_hide and trace_show,
                "evidence_motion_interactive": evidence_hide and evidence_show,
                "chat_batch_motion_interactive": batch_clicked and chat_clicked,
                "browser_console_clean": not console_errors,
                "screenshot_written": screenshot.is_file() and screenshot.stat().st_size > 0,
            }
            return {
                "passed": all(checks.values()),
                "url": url,
                "checks": checks,
                "required_text": required,
                "forbidden_text_absent": forbidden_absent,
                "console_errors": console_errors,
                "screenshot": str(screenshot.resolve()),
                "claim_limit": (
                    "This proves the canonical local UI rendered a grounded answer, measured "
                    "trace and source-bound evidence with working motion controls; it does not "
                    "attest the remote model, private score or BTC eligibility."
                ),
            }
        finally:
            if cdp is not None:
                cdp.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3010/")
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--port", type=int, default=9333)
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=ROOT / "build" / "demo_compliance" / "demo_ui_truth.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "demo_compliance" / "demo_ui_truth.json",
    )
    args = parser.parse_args()
    report = audit(args.url, args.chrome.resolve(), args.port, args.screenshot.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
