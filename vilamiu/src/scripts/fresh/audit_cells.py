"""Show the model the cell it chose and ask whether that row answers the question.

Classifying every wrong answer against the questions whose value is known put 22% on the
wrong cell — the largest remaining class, and larger than unit errors and arithmetic
combined. The failure has a shape, visible by hand: question 135 asks for receivables from
CUSTOMERS and the answer came from the note on OTHER receivables. Both notes have a row
called "Bên thứ ba", so whichever note the reader landed in, the row label looked right.

That is not a question a wider shortlist or a better retriever fixes, because the table was
offered and the row was read. It is a question about whether this row is the indicator the
question named — and asking that directly is a different task from finding the figure,
which is why a second pass over the same model can catch it.

The prompt is deliberately narrow: one table, the heading above it, the chosen row and its
neighbours, and the column headers. Nothing else competes for attention, and the auditor
is asked to name a replacement row rather than to re-answer, so a correction stays
verifiable — the new value is read from the cell by code, not reported by the model.

Usage:
  python scripts/fresh/audit_cells.py --results artifacts/fresh/cot_unitfixed.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from fix_units import contexts_of, declared_scale  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
NEIGHBOURS = 8
LABEL_CHARS = 76

SYSTEM = """Bạn kiểm tra xem MỘT ô dữ liệu có đúng là ô mà câu hỏi cần hay không.

Bạn nhận: câu hỏi, đoạn văn phía trên bảng (thường là tiêu đề thuyết minh), tiêu đề các
cột, và một số dòng lân cận. Một dòng được đánh dấu `>>>` là dòng đã được chọn.

Việc của bạn là trả lời: dòng và cột được chọn có đúng là chỉ tiêu câu hỏi yêu cầu không.

Hãy đặc biệt cảnh giác với các cặp dễ lẫn có cùng tên dòng nhưng khác thuyết minh:
"phải thu khách hàng" khác "phải thu khác"; "trước thuế" khác "sau thuế"; "ngắn hạn" khác
"dài hạn"; "nguyên giá" khác "giá trị còn lại"; hợp nhất khác công ty mẹ. Hãy đọc đoạn văn
phía trên bảng để biết bảng này nói về chỉ tiêu nào.

Trả lời bằng ĐÚNG một đối tượng JSON:
{"dung": true}
hoặc
{"dung": false, "dong": <số dòng đúng trong bảng này>, "cot": <số cột đúng>}

Nếu không có dòng nào trong bảng này đúng, trả {"dung": false, "dong": -1, "cot": -1}."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

(văn bản phía trên bảng: …{context})

{header}
{rows}

Dòng `>>>` và cột c{col} có đúng là chỉ tiêu câu hỏi cần?"""


def locate(refs: list[dict], value: float, unit: float,
           grids: dict, contexts: dict) -> tuple | None:
    """The single cell that explains this answer, or None if it is ambiguous."""

    found = []
    for ref in refs:
        path = (ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"] /
                ref["doc"] / f"{ref['doc']}_extracted_tables"
                / f"table_{ref['table_id']}.csv")
        if path not in grids:
            if len(grids) > 300:
                grids.clear()
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    grids[path] = list(csv_mod.reader(handle))
            except OSError:
                grids[path] = []
        grid = grids[path]
        if not grid:
            continue
        if ref["doc"] not in contexts:
            contexts[ref["doc"]] = contexts_of(
                ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"] /
                ref["doc"] / f"{ref['doc']}_extracted.txt")
        for r_index, row in enumerate(grid[1:]):
            for c_index, cell in enumerate(row):
                raw = str(cell).strip()
                if not raw:
                    continue
                parsed = ps.parse_vn_number(raw)
                if parsed is None or not parsed:
                    continue
                if any(abs(abs(parsed) * s / unit - abs(value)) <= 0.01
                       for s in SCALES):
                    found.append((ref, grid, r_index, c_index))
                    break
    return found[0] if len(found) == 1 else None


def render_window(grid: list[list[str]], row: int, col: int) -> tuple[str, str]:
    width = min(6, max((len(r) for r in grid), default=0))
    header = grid[0] if grid else []
    head = "  ".join(
        f"c{i}={str(header[i]).strip()[:LABEL_CHARS if i == 0 else 24]}"
        if i < len(header) else f"c{i}=" for i in range(width))
    lines = []
    low = max(0, row - NEIGHBOURS)
    high = min(len(grid) - 1, row + NEIGHBOURS + 1)
    for index in range(low, high):
        cells = grid[index + 1]
        text = " | ".join(
            str(cells[i]).strip()[:LABEL_CHARS if i == 0 else 24]
            if i < len(cells) else "" for i in range(width))
        mark = ">>>" if index == row else "   "
        lines.append(f"{mark} r{index} | {text}")
    return head, "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/fresh/cot_unitfixed.jsonl")
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/prompts_audit.jsonl")
    parser.add_argument("--plan", default="artifacts/fresh/audit_plan.json")
    parser.add_argument("--ids-file", default="")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    refs_of = {}
    for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            refs_of[record["id"]] = record["meta"]["refs"]

    only = None
    if args.ids_file:
        only = set(json.loads((ROOT / args.ids_file).read_text(encoding="utf-8")))

    counters: Counter[str] = Counter()
    grids: dict[Path, list[list[str]]] = {}
    contexts: dict[str, dict[int, str]] = {}
    plan = {}

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for line in (ROOT / args.results).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            qid = record["id"]
            if only is not None and qid not in only:
                continue
            value = record.get("answer")
            if value is None:
                counters["khong co dap an"] += 1
                continue
            _name, unit = unit_of(questions[qid])
            if not unit:
                counters["khong ro don vi"] += 1
                continue
            spot = locate(refs_of.get(qid, []), float(value), unit, grids, contexts)
            if spot is None:
                counters["khong dinh vi duoc o (bo qua)"] += 1
                continue
            ref, grid, row, col = spot
            head, window = render_window(grid, row, col)
            context = contexts[ref["doc"]].get(ref["table_id"], "")[-260:]
            handle.write(json.dumps({
                "id": qid, "system": SYSTEM,
                "user": USER.format(question=questions[qid], context=context,
                                    header=head, rows=window, col=col),
            }, ensure_ascii=False) + "\n")
            plan[str(qid)] = {"ref": ref["ref"], "doc": ref["doc"],
                             "ticker": ref["ticker"], "year": ref["year"],
                             "table_id": ref["table_id"], "row": row, "col": col,
                             "answer": float(value)}
            counters["co the soat"] += 1

    (ROOT / args.plan).write_text(json.dumps(plan, ensure_ascii=False),
                                  encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
