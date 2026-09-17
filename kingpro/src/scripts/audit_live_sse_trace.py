"""Audit end-to-end SSE ordering through the Next.js proxy."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = (
    "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) "
    "là bao nhiêu triệu đồng?"
)


def audit(url: str, question: str, timeout: float) -> dict:
    body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict] = []
    current_event = "message"
    data_lines: list[str] = []
    content_type = ""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = str(response.headers.get("Content-Type") or "")
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    payload = json.loads("\n".join(data_lines))
                    events.append(
                        {
                            "event": current_event,
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                            "data": payload,
                        }
                    )
                current_event = "message"
                data_lines = []
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

    names = [event["event"] for event in events]
    statuses = [event for event in events if event["event"] == "status"]
    result_events = [event for event in events if event["event"] == "result"]
    result = result_events[0]["data"] if result_events else {}
    first_status_ms = statuses[0]["elapsed_ms"] if statuses else None
    result_ms = result_events[0]["elapsed_ms"] if result_events else None
    heartbeat_events = [
        event
        for event in statuses
        if isinstance(event.get("data"), dict)
        and event["data"].get("metadata", {}).get("heartbeat") is True
    ]
    requires_heartbeat = bool(result_ms is not None and result_ms >= 1000)
    checks = {
        "content_type_event_stream": content_type.startswith("text/event-stream"),
        "status_precedes_result": bool(
            statuses
            and result_events
            and events.index(statuses[0]) < events.index(result_events[0])
        ),
        "started_event_present": any(event["data"].get("status") == "started" for event in statuses),
        "completed_event_present": any(event["data"].get("status") == "completed" for event in statuses),
        "heartbeat_when_request_exceeds_one_second": (not requires_heartbeat) or bool(heartbeat_events),
        "first_event_arrives_before_result": bool(
            first_status_ms is not None and result_ms is not None and first_status_ms < result_ms
        ),
        "result_answer_matches": result.get("status") == "answered"
        and abs(float(result.get("answer", 0)) - 208253.2) < 1e-9,
        "result_contains_measured_trace": bool(result.get("trace"))
        and result["trace"][0].get("status") == "completed",
        "done_is_terminal": bool(names) and names[-1] == "done",
    }
    return {
        "passed": all(checks.values()),
        "url": url,
        "content_type": content_type,
        "event_sequence": names,
        "first_status_ms": first_status_ms,
        "result_ms": result_ms,
        "heartbeat_count": len(heartbeat_events),
        "checks": checks,
        "claim_limit": (
            "This proves local end-to-end stream ordering and heartbeat behavior "
            "for one grounded scenario; it is not a latency SLA or private-score claim."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000/api/ask/stream")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "live_sse_trace_audit.json")
    args = parser.parse_args()
    report = audit(args.url, args.question, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
