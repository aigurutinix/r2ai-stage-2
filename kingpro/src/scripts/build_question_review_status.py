#!/usr/bin/env python3
"""Sinh bảng trạng thái audit từng câu từ ledger nguồn chuẩn.

Mục đích của file này là ngăn việc dò lại câu đã khóa sau khi mất context.
Nguồn sự thật duy nhất cho trạng thái "đã khóa" là
knowledge/vothuong/question_source_verdicts.json. Các review lịch sử không có
verdict trong ledger này không được tự động xem là đã khóa nguồn 5 lớp.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"
DEFAULT_STATE = ROOT / "knowledge/vothuong/current_state.json"
DEFAULT_SUBMISSION = (
    ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3/submission.json"
)
DEFAULT_PUBLIC_QUEUE = ROOT / "build/v227_residual_public100_fused_v217.json"
DEFAULT_BATCH_QUEUE = ROOT / "build/v227_residual_batch150_fused_v217.json"
DEFAULT_JSON_OUT = ROOT / "knowledge/vothuong/question_review_status.json"
DEFAULT_MD_OUT = ROOT / "knowledge/vothuong/QUESTION_REVIEW_STATUS.md"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compact(value, limit: int = 84) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def queue_items(path: Path) -> list[dict]:
    data = load_json(path)
    queue = data.get("queue")
    if not isinstance(queue, list) or not all(isinstance(item, dict) for item in queue):
        raise ValueError(f"Queue không hợp lệ: {path}")
    return queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--public-queue", type=Path, default=DEFAULT_PUBLIC_QUEUE)
    parser.add_argument("--batch-queue", type=Path, default=DEFAULT_BATCH_QUEUE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger_data = load_json(args.ledger)
    state = load_json(args.state)
    submission = load_json(args.submission)
    public_queue = queue_items(args.public_queue)
    batch_queue = queue_items(args.batch_queue)

    if not isinstance(submission, list):
        raise ValueError("submission.json phải là một danh sách predictions")
    verdicts = ledger_data.get("verdicts")
    if not isinstance(verdicts, dict):
        raise ValueError("Ledger thiếu object verdicts")

    question_by_id = {int(item["id"]): item for item in submission}
    all_ids = list(question_by_id)
    closed_ids = [int(question_id) for question_id in verdicts]
    closed_set = set(closed_ids)

    unknown = sorted(closed_set - set(all_ids))
    if unknown:
        raise ValueError(f"Ledger chứa question ID ngoài submission: {unknown}")

    no_change_ids = [
        question_id
        for question_id in closed_ids
        if verdicts[str(question_id)].get("mutation") == "none"
    ]
    cleanup_ids = [
        question_id
        for question_id in closed_ids
        if str(verdicts[str(question_id)].get("mutation", "")).startswith("pending_batch")
    ]
    other_closed_ids = [
        question_id
        for question_id in closed_ids
        if question_id not in set(no_change_ids) | set(cleanup_ids)
    ]

    public_pending = [item for item in public_queue if int(item["id"]) not in closed_set]
    public_ids = {int(item["id"]) for item in public_queue}
    batch_pending = [item for item in batch_queue if int(item["id"]) not in closed_set]
    batch_only_pending = [item for item in batch_pending if int(item["id"]) not in public_ids]
    all_not_source_closed_ids = [question_id for question_id in all_ids if question_id not in closed_set]

    raw_next = state["individual_source_audit"].get("next_question")
    configured_next = int(raw_next) if raw_next is not None else None
    if configured_next in closed_set:
        configured_next = int(public_pending[0]["id"]) if public_pending else None

    pending_fixes = {
        int(item["question_id"]): item
        for item in state["individual_source_audit"].get("pending_batch_fixes", [])
    }

    def closed_record(question_id: int) -> dict:
        verdict = verdicts[str(question_id)]
        source = question_by_id[question_id]
        result = {
            "id": question_id,
            "question": source.get("question"),
            "answer": verdict.get("answer", source.get("answer")),
            "status": verdict.get("status"),
            "mutation": verdict.get("mutation"),
            "artifacts": verdict.get("artifacts", []),
        }
        if question_id in pending_fixes:
            result["pending_fix"] = pending_fixes[question_id]
        return result

    def pending_record(item: dict) -> dict:
        question_id = int(item["id"])
        return {
            "id": question_id,
            "question": item.get("question", question_by_id[question_id].get("question")),
            "current_answer": item.get(
                "current_answer", question_by_id[question_id].get("answer")
            ),
            "risk_score": item.get("risk_score"),
            "zone": item.get("zone"),
            "reason": item.get("reason"),
        }

    generated_at = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds")
    output = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "scope_note": (
            "'Chưa khóa' nghĩa là chưa có verdict audit nguồn 5 lớp trong ledger mới; "
            "không có nghĩa câu đó chưa từng xuất hiện trong review/thử nghiệm lịch sử."
        ),
        "sources": {
            "ledger": str(args.ledger.relative_to(ROOT)).replace("\\", "/"),
            "state": str(args.state.relative_to(ROOT)).replace("\\", "/"),
            "submission": str(args.submission.relative_to(ROOT)).replace("\\", "/"),
            "public_queue": str(args.public_queue.relative_to(ROOT)).replace("\\", "/"),
            "batch_queue": str(args.batch_queue.relative_to(ROOT)).replace("\\", "/"),
        },
        "next_question": configured_next,
        "totals": {
            "corpus": len(all_ids),
            "source_closed": len(closed_ids),
            "closed_no_change": len(no_change_ids),
            "closed_pending_batch_cleanup": len(cleanup_ids),
            "closed_other": len(other_closed_ids),
            "not_source_closed": len(all_not_source_closed_ids),
            "public100_total": len(public_queue),
            "public100_pending": len(public_pending),
            "batch150_total": len(batch_queue),
            "batch150_pending": len(batch_pending),
            "batch150_only_pending": len(batch_only_pending),
        },
        "closed_no_change": [closed_record(question_id) for question_id in no_change_ids],
        "closed_pending_batch_cleanup": [
            closed_record(question_id) for question_id in cleanup_ids
        ],
        "closed_other": [closed_record(question_id) for question_id in other_closed_ids],
        "priority_pending_public100": [pending_record(item) for item in public_pending],
        "priority_pending_batch150_only": [
            pending_record(item) for item in batch_only_pending
        ],
        "all_not_source_closed_ids": all_not_source_closed_ids,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Trạng thái audit từng câu",
        "",
        f"> Cập nhật: `{generated_at}`. Câu tiếp theo: **q{configured_next}**.",
        "",
        "`Chưa khóa` ở đây nghĩa là chưa có verdict audit nguồn 5 lớp trong ledger mới;",
        "không có nghĩa câu đó chưa từng được xem trong review hoặc phép thử lịch sử.",
        "JSON đầy đủ còn lưu toàn bộ ID chưa khóa để tool có thể tiếp tục không trùng.",
        "",
        "## Tổng quan",
        "",
        "| Nhóm | Số câu | Hành động |",
        "|---|---:|---|",
        f"| Toàn bộ submission | {len(all_ids)} | — |",
        f"| Đã khóa, không đổi | {len(no_change_ids)} | Không dò lại nếu không có bằng chứng mới |",
        f"| Đã khóa answer, chờ cleanup batch | {len(cleanup_ids)} | Không dò lại answer; chỉ áp dụng cleanup khi đủ batch |",
        f"| Đã khóa loại khác | {len(other_closed_ids)} | Xem verdict riêng |",
        f"| Chưa khóa nguồn 5 lớp | {len(all_not_source_closed_ids)} | Tiếp tục theo queue ưu tiên |",
        f"| Còn lại trong public100 | {len(public_pending)}/{len(public_queue)} | Soi theo thứ tự bên dưới |",
        f"| Còn lại trong batch150 | {len(batch_pending)}/{len(batch_queue)} | Queue mở rộng |",
        "",
        "## Đã khóa — không dò lại",
        "",
        "| Câu | Answer | Trạng thái | Câu hỏi |",
        "|---:|---:|---|---|",
    ]
    for question_id in no_change_ids:
        row = closed_record(question_id)
        lines.append(
            f"| q{question_id} | {compact(row['answer'], 24)} | "
            f"`{compact(row['status'], 45)}` | {compact(row['question'])} |"
        )

    lines.extend(
        [
            "",
            "## Đã khóa answer — chờ cleanup theo batch",
            "",
            "| Câu | Answer | Cleanup còn chờ |",
            "|---:|---:|---|",
        ]
    )
    for question_id in cleanup_ids:
        row = closed_record(question_id)
        change = row.get("pending_fix", {}).get("change", "Xem verdict JSON")
        lines.append(
            f"| q{question_id} | {compact(row['answer'], 24)} | {compact(change, 120)} |"
        )

    if other_closed_ids:
        lines.extend(["", "## Đã khóa — trạng thái khác", ""])
        lines.append(", ".join(f"q{question_id}" for question_id in other_closed_ids))

    lines.extend(
        [
            "",
            "## Chưa khóa — queue public100",
            "",
            "Đây là thứ tự làm việc chính. Sau mỗi verdict phải chạy lại script này.",
            "",
            "| STT | Câu | Risk | Answer hiện tại | Câu hỏi |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for index, item in enumerate(public_pending, 1):
        row = pending_record(item)
        lines.append(
            f"| {index} | q{row['id']} | {compact(row['risk_score'], 12)} | "
            f"{compact(row['current_answer'], 24)} | {compact(row['question'])} |"
        )

    lines.extend(
        [
            "",
            "## Chưa khóa — phần mở rộng batch150 không trùng public100",
            "",
            "| STT | Câu | Risk | Answer hiện tại | Câu hỏi |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for index, item in enumerate(batch_only_pending, 1):
        row = pending_record(item)
        lines.append(
            f"| {index} | q{row['id']} | {compact(row['risk_score'], 12)} | "
            f"{compact(row['current_answer'], 24)} | {compact(row['question'])} |"
        )

    lines.extend(
        [
            "",
            "## Quy tắc cập nhật",
            "",
            "1. Ghi verdict mới vào `question_source_verdicts.json` ngay khi khóa câu.",
            "2. Cập nhật `current_state.json` nếu câu tiếp theo hoặc cleanup batch thay đổi.",
            "3. Chạy `python scripts/build_question_review_status.py`.",
            "4. Không dò lại nhóm đã khóa trừ khi xuất hiện bằng chứng nguồn mới hoặc audit tool có lỗi hệ thống.",
            "",
        ]
    )
    args.md_out.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"Đã ghi {args.md_out.relative_to(ROOT)} và {args.json_out.relative_to(ROOT)}: "
        f"khóa={len(closed_ids)}, chờ_cleanup={len(cleanup_ids)}, "
        f"public100_còn={len(public_pending)}, next=q{configured_next}"
    )


if __name__ == "__main__":
    main()
