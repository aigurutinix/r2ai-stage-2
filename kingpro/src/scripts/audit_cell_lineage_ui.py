"""Browser proof that audited cell lineage reaches the visible evidence table."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

from audit_demo_ui_truth import Cdp, DEFAULT_CHROME, _console_errors, _target, _wait_for


ROOT = Path(__file__).resolve().parents[1]
QUESTION = (
    "Tiền gửi của khách hàng cuối năm 2019 của Ngân hàng TMCP Sài Gòn - "
    "Hà Nội là bao nhiêu triệu đồng?"
)
EXPECTED_HEADERS = [
    "Chỉ tiêu",
    "Quá hạn · Trên 3 tháng",
    "Quá hạn · Dưới 3 tháng",
    "Trong hạn · Đến 1 tháng",
    "Trong hạn · Từ 1 đến 3 tháng",
    "Trong hạn · Từ 3 đến 12 tháng",
    "Trong hạn · Từ 1 đến 5 năm",
    "Trong hạn · Trên 5 năm",
    "Trong hạn · Tổng cộng",
]


def audit(url: str, chrome: Path, port: int, screenshot: Path) -> dict:
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kingpro-lineage-ui-") as profile:
        process = subprocess.Popen(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp: Cdp | None = None
        try:
            _wait_for(lambda: _target(port).get("webSocketDebuggerUrl") is not None, "Chrome")
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
                and cdp.evaluate("document.querySelector('textarea') !== null") is True,
                "KINGPRO composer",
            )
            question_json = json.dumps(QUESTION, ensure_ascii=False)
            filled = cdp.evaluate(
                """
                (() => {
                  const input = document.querySelector('textarea');
                  if (!input) return false;
                  const setter = Object.getOwnPropertyDescriptor(
                    HTMLTextAreaElement.prototype, 'value'
                  ).set;
                  setter.call(input, %s);
                  input.dispatchEvent(new Event('input', { bubbles: true }));
                  return true;
                })()
                """ % question_json
            ) is True
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.composer button[type=submit]:not(:disabled)') !== null") is True,
                "enabled submit",
            )
            submitted = cdp.evaluate(
                "document.querySelector('.composer button[type=submit]')?.click(); true"
            ) is True
            _wait_for(
                lambda: "259.236.746" in str(cdp.evaluate("document.body.innerText"))
                and cdp.evaluate("document.querySelector('.tableLaunch:not(:disabled)') !== null") is True,
                "audited q69 answer",
                timeout=30,
            )
            opened = cdp.evaluate("document.querySelector('.tableLaunch')?.click(); true") is True
            _wait_for(
                lambda: cdp.evaluate("document.querySelector('.selectedSourceCell') !== null") is True,
                "verified source cell",
            )
            _wait_for(
                lambda: cdp.evaluate(
                    """
                    (() => {
                      const wrap = document.querySelector('.tableModalWrap');
                      const cell = document.querySelector('.selectedSourceCell');
                      const table = document.querySelector('.tableModalWrap table');
                      if (!wrap || !cell || !table) return false;
                      const outer = wrap.getBoundingClientRect();
                      const grid = table.getBoundingClientRect();
                      const centered = Math.abs(
                        (grid.left + grid.width / 2) - (outer.left + outer.width / 2)
                      ) <= 2;
                      return wrap.scrollTop > 100
                        && wrap.scrollWidth <= wrap.clientWidth + 2
                        && centered;
                    })()
                    """
                ) is True,
                "auto-scroll to audited cell",
            )
            state = cdp.evaluate(
                """
                (() => {
                  const wrap = document.querySelector('.tableModalWrap');
                  const cell = document.querySelector('.selectedSourceCell');
                  const row = cell?.closest('tr');
                  const outer = wrap?.getBoundingClientRect();
                  const inner = cell?.getBoundingClientRect();
                  const grid = document.querySelector('.tableModalWrap table')?.getBoundingClientRect();
                  return {
                    headers: [...document.querySelectorAll('.tableModalWrap th')]
                      .map((node) => node.textContent),
                    cellCount: document.querySelectorAll('.selectedSourceCell').length,
                    rowCount: document.querySelectorAll('.selectedSourceRow').length,
                    cellText: cell?.textContent,
                    rowLabel: row?.querySelector('td')?.textContent,
                    cellIndex: cell ? [...row.children].indexOf(cell) : -1,
                    background: cell ? getComputedStyle(cell).backgroundColor : '',
                    boxShadow: cell ? getComputedStyle(cell).boxShadow : '',
                    scrollTop: wrap?.scrollTop || 0,
                    scrollLeft: wrap?.scrollLeft || 0,
                    horizontalCenterDelta: (outer && inner)
                      ? Math.abs((inner.left + inner.width / 2) - (outer.left + outer.width / 2))
                      : null,
                    wrapWidth: outer?.width || 0,
                    tableCenterDelta: (outer && grid)
                      ? Math.abs((grid.left + grid.width / 2) - (outer.left + outer.width / 2))
                      : null,
                    tableFitsViewport: Boolean(wrap && wrap.scrollWidth <= wrap.clientWidth + 2),
                    cellVisible: Boolean(outer && inner
                      && inner.bottom > outer.top && inner.top < outer.bottom
                      && inner.right > outer.left && inner.left < outer.right),
                  };
                })()
                """
            )
            image = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            screenshot.write_bytes(base64.b64decode(str(image.get("data", ""))))
            cdp.drain()
            console_errors = _console_errors(cdp.events)
            checks = {
                "question_submitted": filled and submitted,
                "q69_answer_replayed": "259.236.746" in str(cdp.evaluate("document.body.innerText")),
                "multi_level_headers_promoted": state["headers"] == EXPECTED_HEADERS,
                "exactly_one_source_cell_highlighted": state["cellCount"] == 1 and state["rowCount"] == 1,
                "correct_total_cell_highlighted": (
                    state["cellText"] == "259.236.746"
                    and state["rowLabel"] == "Tiền gửi của khách hàng"
                    and state["cellIndex"] == 8
                ),
                "verified_cell_visually_distinct": (
                    state["background"] == "rgb(204, 239, 220)"
                    and "rgb(32, 166, 106)" in state["boxShadow"]
                ),
                "modal_auto_scrolled_to_cell": (
                    state["scrollTop"] > 100
                    and state["cellVisible"] is True
                ),
                "table_horizontally_centered": (
                    state["tableFitsViewport"] is True
                    and state["tableCenterDelta"] is not None
                    and state["tableCenterDelta"] <= 2
                ),
                "browser_console_clean": not console_errors,
                "screenshot_written": screenshot.is_file() and screenshot.stat().st_size > 0,
            }
            return {
                "passed": opened and all(checks.values()),
                "url": url,
                "checks": checks,
                "state": state,
                "console_errors": console_errors,
                "screenshot": str(screenshot.resolve()),
                "claim_limit": (
                    "This proves one manifest-audited cell is re-exposed and visibly "
                    "highlighted at its exact physical row/column in the local UI."
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
    if hasattr(__import__("sys").stdout, "reconfigure"):
        __import__("sys").stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000/")
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    parser.add_argument("--port", type=int, default=9341)
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=ROOT / "build" / "demo_compliance" / "cell_lineage_q69.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "demo_compliance" / "cell_lineage_q69.json",
    )
    args = parser.parse_args()
    report = audit(args.url, args.chrome.resolve(), args.port, args.screenshot.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
