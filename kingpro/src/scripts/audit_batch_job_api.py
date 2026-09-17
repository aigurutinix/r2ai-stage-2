"""Audit durable batch create/events/status/download through the Next proxy."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    {
        "id": 1,
        "question": "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?",
    },
    {"id": 2, "question": "Tổng tài sản của FPT năm 2024 là bao nhiêu tỷ đồng?"},
    {
        "id": 3,
        "question": (
            "Năm 2022, trong nhóm HPG, HSG, MSR và NKG, các công ty có hệ số "
            "thanh toán nhanh thấp hơn trung vị của nhóm có biên lợi nhuận ròng "
            "bình quân là bao nhiêu phần trăm?"
        ),
    },
]


def _json_request(url: str, *, payload: dict | None = None, timeout: float = 60) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _stream_events(url: str, timeout: float) -> list[dict]:
    events: list[dict] = []
    name = "message"
    data_lines: list[str] = []
    with urllib.request.urlopen(url, timeout=timeout) as response:
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    events.append({"event": name, "data": json.loads("\n".join(data_lines))})
                name = "message"
                data_lines = []
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
    return events


def audit(base_url: str, timeout: float) -> dict:
    created = _json_request(
        f"{base_url.rstrip('/')}/api/batch/jobs",
        payload={"items": QUESTIONS, "workers": 3, "fallback_on_refusal": False},
        timeout=timeout,
    )
    job_id = str(created.get("job_id") or "")
    events = _stream_events(
        f"{base_url.rstrip('/')}/api/batch/jobs/{job_id}/events?cursor=0",
        timeout,
    )
    snapshot = _json_request(f"{base_url.rstrip('/')}/api/batch/jobs/{job_id}", timeout=timeout)
    download = _json_request(
        f"{base_url.rstrip('/')}/api/batch/jobs/{job_id}/download",
        timeout=timeout,
    )
    submission_export = _json_request(
        f"{base_url.rstrip('/')}/api/batch/jobs/{job_id}/submission",
        payload={},
        timeout=timeout,
    )
    submission_snapshot = _json_request(
        f"{base_url.rstrip('/')}/api/batch/jobs/{job_id}",
        timeout=timeout,
    )
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/api/batch/jobs/{job_id}/submission/download",
        timeout=timeout,
    ) as response:
        submission_zip = response.read()
    with zipfile.ZipFile(io.BytesIO(submission_zip), "r") as bundle:
        archive_corrupt = bundle.testzip()
        archive_names = bundle.namelist()
        packed_submission = json.loads(bundle.read("submission.json"))
    names = [event["event"] for event in events]
    items = [event for event in events if event["event"] == "item"]
    results = download.get("results") if isinstance(download.get("results"), list) else []
    checks = {
        "job_created": bool(job_id) and created.get("total") == 3,
        "queued_and_running_events_present": names.count("status") >= 2,
        "all_item_events_present": len(items) == 3,
        "single_result_event_present": names.count("result") == 1,
        "done_is_terminal": bool(names) and names[-1] == "done",
        "snapshot_success": snapshot.get("status") == "success",
        "download_ready": snapshot.get("download_ready") is True,
        "download_summary_complete": download.get("summary", {}).get("completed") == 3,
        "download_order_preserved": [record.get("id") for record in results] == [1, 2, 3],
        "no_fallback_or_error": download.get("summary", {}).get("fallback") == 0
        and download.get("summary", {}).get("error") == 0,
        "durable_paths_present": all(
            Path(str(snapshot.get(key))).is_file()
            for key in ("output_path", "manifest_path", "report_path")
        ),
        "submission_export_pass": submission_export.get("status") == "PASS",
        "submission_snapshot_ready": submission_snapshot.get("submission_ready") is True,
        "submission_download_hash_matches": (
            hashlib.sha256(submission_zip).hexdigest().upper()
            == submission_export.get("archive", {}).get("sha256")
        ),
        "submission_archive_crc_pass": archive_corrupt is None,
        "submission_archive_layout_valid": (
            archive_names[0] == "submission.json"
            and all(name.startswith("data/") for name in archive_names[1:])
        ),
        "submission_record_count_matches": len(packed_submission) == len(QUESTIONS),
    }
    return {
        "passed": all(checks.values()),
        "job_id": job_id,
        "event_sequence": names,
        "checks": checks,
        "snapshot": submission_snapshot,
        "claim_limit": (
            "This proves one local three-question durable job through the Next proxy; "
            "it is not a full 1,012-question private run or score claim."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "batch_job_api_audit.json")
    args = parser.parse_args()
    report = audit(args.base_url, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
