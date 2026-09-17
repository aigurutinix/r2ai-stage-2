"""Can the model name the right row when the table is handed to it?

Every intervention today moved the phrasing, the voting, the table count or the
branch order, and every one came back flat. None of them touched the step that the
organisers' own error analysis puts at 54.7% of failures and that hand-reading
found in three of four wrong answers: picking the row.

This isolates it. The gold table — the one that actually holds the answer — is the
only table shown, so retrieval cannot be the cause and neither can dilution. The
model is asked for a row index and nothing else. What comes back is compared with
the row the gold value sits on.

The number decides where the remaining effort goes:

* near 50% -- row selection is the ceiling of the whole system and everything else
  is rounding. Work here or nowhere.
* near 85% -- the model reads a table well when it is given the right one, and the
  loss is upstream in choosing the table, or downstream in the arithmetic.

Usage (against a served model):
  PYTHONPATH=src python scripts/_probe_row_pick.py \
      --model Qwen/Qwen3-14B-AWQ --local-url http://127.0.0.1:18000/v1 --limit 150
"""

from __future__ import annotations

import argparse
import collections
import os
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import close, parse_number  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.tables import headers as header_mod  # noqa: E402

# Spell the period columns out as absolute dates instead of asking the model to
# convert them. 26.4% of probe answers pick the right row and the wrong column,
# and the system prompt already states the conversion rule.
ABS_HEADERS = os.environ.get("VIFIN_ABS_HEADERS", "0") != "0"
# Bind each column index to its header text instead of a free-standing ruler.
COL_TAGS = os.environ.get("VIFIN_COL_TAGS", "0") != "0"

SYSTEM = """Bạn đọc bảng trong báo cáo tài chính Việt Nam và chỉ ra DÒNG chứa số
mà câu hỏi cần.

Bảng được in kèm chỉ số dòng: r0 là dòng tiêu đề, r1, r2, … là các dòng dữ liệu.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"row": <chỉ số dòng>, "col": <chỉ số cột>}

Quy tắc:
- Chọn dòng có nhãn khớp chỉ tiêu câu hỏi hỏi tới, không phải dòng có số lớn nhất.
- Nhãn thật thường có tiền tố đánh số ("1. ", "V. ") mà câu hỏi không có.
- Một dòng cộng có thể để trống ô nhãn; nếu câu hỏi hỏi tổng thì đó là dòng cần.
- Cột: bảng tài chính luôn có nhiều cột kỳ. Bỏ qua cột "Mã số" và
  "Thuyết minh".
- BẢNG NÀY THUỘC BÁO CÁO NĂM {report_year}. Vì vậy:
    "Số cuối năm" / "Số cuối kỳ" / "31/12/{report_year}"  =  cuối năm {report_year}
    "Số đầu năm"  / "Số đầu kỳ"  / "01/01/{report_year}"  =  cuối năm {prev_year}
    "Năm nay" = năm {report_year};  "Năm trước" = năm {prev_year}
  Đối chiếu kỳ câu hỏi hỏi với quy đổi trên rồi mới chọn cột.
- Nếu không dòng nào chứa chỉ tiêu đó, trả {"row": -1, "col": -1}."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

BẢNG — {caption}
{table}

Dòng nào và cột nào chứa số câu hỏi cần?"""

JSON_RE = re.compile(r"\{[^{}]*\}")
MAX_ROWS = 60
MAX_CELL = 200
MAX_HEADER_CELL = 120


def render(grid) -> str:
    """Rows prefixed r0.., and a column-number line above them.

    Without that line the model answered with the index of the first *data*
    column, not counting the label column: 48 of 52 column errors were exactly
    -1. That is a convention mismatch, not a misreading of the table.
    """

    width = max(len(row) for row in grid[:MAX_ROWS])
    if COL_TAGS:
        # A separate ruler line above the table cannot line up with the columns,
        # because the cells have different widths and nothing pads them. The
        # measurement says it never worked: 49 of 58 column errors are exactly
        # -1, and the model names c0 — the label column — where the gold cell is
        # c1. It is counting data columns and skipping the label. Binding the
        # index to the header text removes the alignment problem entirely.
        header = " | ".join(f"c{c}={str(cell)[:MAX_HEADER_CELL]}"
                            for c, cell in enumerate(grid[0]))
        body = [f"r{i}: " + " | ".join(str(cell)[:MAX_CELL] for cell in row)
                for i, row in enumerate(grid[:MAX_ROWS])
                if i > 0]
        return "\n".join([f"r0: {header}"] + body)
    lines = [" | ".join(f"[c{c}]" for c in range(width))]
    for index, row in enumerate(grid[:MAX_ROWS]):
        # The header row gets a wider budget than the body. Once the resolver
        # appends "[31/12/2021]" a header can reach 34 characters, exactly the
        # body limit, and truncating it would cut off the thing being tested.
        budget = MAX_HEADER_CELL if index == 0 else MAX_CELL
        cells = " | ".join(str(cell)[:budget] for cell in row)
        lines.append(f"r{index}: {cells}")
    return "\n".join(lines)


def gold_row(grid, answer, context: str):
    """The (row, column) whose cell equals the gold answer, raw or scaled."""

    for index, row in enumerate(grid[1:], start=1):
        for column, cell in enumerate(row):
            value = parse_number(cell)
            if value is None or value == 0:
                continue
            if close(value, answer, 5e-4):
                return index, column
            scaled = value * lookup_mod.column_scale(grid, column, context)
            if close(scaled, answer, 5e-4):
                return index, column
            for unit in (1e12, 1e9, 1e6, 1e3):
                if close(value / unit, answer, 5e-4) or close(scaled / unit, answer, 5e-4):
                    return index, column
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--out", default="artifacts/_row_pick.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    client = (ChatClient.local(args.model, args.local_url) if args.local_url
              else ChatClient.from_env(ROOT, model=args.model))

    work = []
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(work) >= args.limit:
            continue
        record = json.loads(line)
        answer = parse_number(record.get("answer"))
        refs = record.get("relevant_tables") or []
        if answer is None or not refs:
            continue
        doc, table_id = refs[0].rsplit("|table_", 1)
        key = TableKey(doc, int(table_id))
        grid = store.rows(key)
        if not grid or len(grid) < 3:
            continue
        meta = store.meta(key)
        context = (f"{getattr(meta, 'unit_page', '')} {getattr(meta, 'unit_doc', '')} "
                   f"{getattr(meta, 'caption', '')}")
        located = gold_row(grid, answer, context)
        # Without a locatable gold cell there is nothing to be right or wrong about.
        if located is None:
            continue
        target, target_col = located
        year_match = re.search(r"_(19|20)(\d{2})_", doc)
        report_year = int(year_match.group(1) + year_match.group(2)) if year_match else 0
        shown = header_mod.resolve_grid(grid, report_year) if ABS_HEADERS else grid
        work.append((record, shown, target, target_col,
                     str(getattr(meta, "caption", ""))[:110], report_year))

    print(f"{len(work)} câu có dòng gold xác định được, model={args.model}", flush=True)

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        record, grid, target, target_col, caption, report_year = item
        user = USER.format(question=record["question"], caption=caption,
                           table=render(grid))
        # The column is meaningless without knowing which report this is: "Số đầu
        # năm" of the 2021 report is the close of 2020. 34.7% of answers picked
        # the right row and the wrong column, and the prompt never said the year.
        # `.format` cannot be used: the prompt shows a JSON object and its
        # braces would be read as fields.
        system = (SYSTEM.replace("{report_year}", str(report_year))
                        .replace("{prev_year}", str(report_year - 1)))
        try:
            reply = client.complete(system, user)
        except RuntimeError:
            return
        match = JSON_RE.search(reply or "")
        picked = picked_col = None
        if match:
            try:
                data = json.loads(match.group(0))
                picked = int(data.get("row", -1))
                picked_col = int(data.get("col", -1))
            except (ValueError, TypeError):
                picked = picked_col = None

        with lock:
            tally["n"] += 1
            if picked is None:
                tally["không đọc được trả lời"] += 1
            elif picked < 0:
                tally["model nói không có"] += 1
            elif picked == target and picked_col == target_col:
                tally["ĐÚNG CẢ Ô"] += 1
            elif picked == target:
                tally["đúng dòng, SAI CỘT"] += 1
            elif picked in (target - 1, target + 1):
                tally["lệch một dòng"] += 1
            else:
                tally["sai dòng"] += 1
            rows_out.append({"id": record.get("id"), "gold_row": target,
                             "gold_col": target_col, "picked": picked,
                             "picked_col": picked_col})
            if tally["n"] % 25 == 0:
                n = tally["n"]
                print(f"  {n}/{len(work)}  đúng ô={tally['ĐÚNG CẢ Ô'] / n:.1%}"
                      f"  {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} câu, chỉ đưa MỘT bảng — bảng chứa đáp án")
    for name, count in tally.most_common():
        if name == "n":
            continue
        print(f"  {name:26s} {count:4d}  {count / n:6.1%}")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")
    print(f"  chi tiết -> {args.out}")


if __name__ == "__main__":
    main()
