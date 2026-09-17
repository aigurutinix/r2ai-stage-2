"""Hand the model a handful of candidate ROWS and ask which one, and which column.

Lexical scoring generates candidates well and selects them badly. Read by hand on a fresh
sample of eight, it picked `phải trả phí quản lý` for a question about `phải thu phí quản
lý`, `Tổng lợi nhuận trước thuế` for `Tổng cam kết cho thuê`, and `số trích lập dự phòng
trái phiếu` for `số dư trái phiếu`. Every one of those shares most of its tokens with the
right answer and means something else. No reweighting fixes that; the distinction is
semantic.

So the split is: code finds the candidates, the model chooses among them, code reads the
cell. That is the organisers' own 87% setting rather than their 64% one — the model is
choosing between five labelled rows, not between nine tables.

Three things travel with each candidate because each was a wrong answer traced by hand:

  the section heading   a row called `Tổ chức kinh tế` means deposits or loans depending
                        on the heading above it, and the heading is printed
  the column headers    a bank's tax note prints `1/1/2020` beside `31/12/2020`, and
                        picking by position answered the opening balance
  the raw cell text     `(4.354.219)` in brackets is a deduction, and whether the question
                        wants the magnitude or the signed figure is a question about the
                        question

Usage:
  python scripts/fresh/build_pick_prompts.py --limit 120 --out artifacts/fresh/prompts_pick.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from find_statements import locate_columns  # noqa: E402
from fix_units import contexts_of  # noqa: E402
from label_match import score  # noqa: E402
from measure_pointers import POINTER_RE, heading_numbers  # noqa: E402
from measure_unique import variants_from  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402
from table_reading import row_label, scale_from_headers, section_label  # noqa: E402

CODE_RE = re.compile(r"^\d{1,3}$")
CANDIDATES = 6
CELL_CHARS = 30

SYSTEM = """Bạn chọn ĐÚNG MỘT ô dữ liệu trả lời câu hỏi, trong số các ứng viên được đưa.

Mỗi ứng viên là một dòng đã được tìm thấy trong báo cáo, kèm:
- `[k]` số thứ tự ứng viên
- tiêu đề mục phía trên dòng đó (nếu có) — thứ quyết định nghĩa của nhãn dòng
- nhãn dòng, và Mã số nếu có
- các ô giá trị của dòng, mỗi ô kèm tiêu đề cột của nó

Trả lời bằng ĐÚNG một đối tượng JSON:
{"ung_vien": <k>, "cot": <chỉ số cột>, "do_lon": true|false}

Quy tắc:
- `ung_vien` là số `[k]` của dòng đúng. Nếu KHÔNG ứng viên nào đúng, trả `-1`.
- `cot` là chỉ số cột (`c3`, `c4`...) của ô đúng — chọn theo kỳ mà câu hỏi hỏi. Đọc tiêu
  đề cột: `31/12/2020` khác `1/1/2020`; `Số cuối năm` khác `Số đầu năm`; `Năm nay` khác
  `Năm trước`. Cột đầu tiên KHÔNG mặc định là kỳ hiện tại.
- `do_lon`: đặt `true` nếu câu hỏi muốn ĐỘ LỚN của một khoản giảm trừ in trong ngoặc
  (ví dụ "số dư dự phòng" mà báo cáo in `(4.354.219)`); đặt `false` nếu dấu có nghĩa
  (ví dụ lưu chuyển tiền thuần âm, lỗ).
- Cảnh giác cặp đối nghĩa mà giống chữ: "phải thu" khác "phải trả"; "tiền gửi" khác
  "cho vay"; "số dư" khác "số trích lập dự phòng"; "cam kết cho thuê" khác "lợi nhuận
  trước thuế"; "trước thuế" khác "sau thuế"; "nguyên giá" khác "giá trị còn lại".
- Nếu câu hỏi hỏi một chỉ tiêu TỔNG, ưu tiên dòng tổng chứ không phải một dòng con.
- Đừng chọn dòng GỘP hai chỉ tiêu khi câu hỏi chỉ hỏi một trong hai. Ví dụ câu hỏi
  "tiền gửi tại các TCTD khác" thì dòng "Tiền gửi tại các TCTD khác VÀ cho vay các TCTD
  khác" là SAI — phải tìm dòng chỉ nói về tiền gửi.
- Dòng mở đầu bằng "Trừ:" hoặc nằm trong một bảng đối chiếu là khoản KHẤU TRỪ, không phải
  số dư. Câu hỏi hỏi "số dư" của một quỹ thì phải lấy dòng số dư, không lấy dòng trừ.
- Khi có nhiều ứng viên cùng nói về chỉ tiêu đó, ưu tiên dòng có Mã số — đó là dòng của
  báo cáo chính, còn dòng không có Mã số thường là chi tiết hoặc đối chiếu."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{candidates}

Ứng viên nào, cột nào?"""


def render(index: int, item: dict) -> str:
    lines = [f"[{index}] bảng {item['table_id']}"]
    if item["heading"]:
        lines.append(f"    (mục: {item['heading'][:150]})")
    code = f"  Mã số {item['code']}" if item["code"] else ""
    lines.append(f"    nhãn: {item['label'][:110]}{code}")
    for column, header, raw in item["cells"]:
        lines.append(f"      c{column}  [{header[:34]}]  {raw[:CELL_CHARS]}")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="artifacts/fresh/spec_replies.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/prompts_pick.jsonl")
    parser.add_argument("--plan", default="artifacts/fresh/pick_plan.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    variants = variants_from(ROOT / args.specs)
    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    written = 0
    plan: dict[str, list[dict]] = {}
    skipped: dict[str, int] = {}

    def note(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for question in questions:
            qid = question["id"]
            names = variants.get(qid)
            if not names:
                note("khong co spec")
                continue
            text = question["question"]
            _name, unit = unit_of(text)
            if not unit:
                note("khong ro don vi cau hoi")
                continue
            tickers = sorted(resolver.resolve(text))
            years = sorted({y for y in YEAR_RE.findall(text)})
            if not tickers or not years:
                note("khong xac dinh duoc ma/nam")
                continue
            base = ROOT / "data" / "official_corpus" / tickers[0] / years[-1]
            if not base.is_dir():
                note("khong co tai lieu")
                continue
            want_separate = bool(PARENT_RE.search(text))
            docs = sorted(p for p in base.iterdir() if p.is_dir())
            docs = ([d for d in docs if ("separate" in d.name) == want_separate]
                    + [d for d in docs if ("separate" in d.name) != want_separate])

            found: list[tuple[float, dict]] = []
            for doc_dir in docs:
                tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
                if not tables_dir.is_dir():
                    continue
                grids, paths = {}, {}
                for csv_path in sorted(tables_dir.glob("table_*.csv")):
                    table_id = int(csv_path.stem.split("_")[-1])
                    try:
                        with csv_path.open(encoding="utf-8-sig", newline="") as file:
                            grids[table_id] = list(csv_mod.reader(file))
                        paths[table_id] = csv_path
                    except OSError:
                        continue
                anchors = contexts_of(doc_dir / f"{doc_dir.name}_extracted.txt")
                by_number: dict[str, list[int]] = {}
                for table_id in grids:
                    for number in heading_numbers(anchors.get(table_id, "")):
                        by_number.setdefault(number, []).append(table_id)

                pointed: set[int] = set()
                for table_id, grid in grids.items():
                    if not grid:
                        continue
                    code_col, pointer_col = locate_columns(grid)
                    skip = set()
                    if code_col is not None:
                        skip.add(code_col)
                    if pointer_col is not None:
                        skip.add(pointer_col)
                    for index, row in enumerate(grid[1:]):
                        if not row:
                            continue
                        code = ""
                        if code_col is not None and len(row) > code_col:
                            maybe = str(row[code_col]).strip()
                            if CODE_RE.match(maybe):
                                code = maybe
                        if pointer_col is not None and len(row) > pointer_col:
                            candidate = str(row[pointer_col]).strip()
                            if POINTER_RE.match(candidate):
                                key = re.sub(r"[\s.]", "", candidate).upper()
                                pointed.update(by_number.get(key, []))
                        label = section_label(grid, index, skip)
                        value = score(names, label)
                        if not value:
                            continue
                        cells = []
                        for column, cell in enumerate(row):
                            if column in skip:
                                continue
                            raw = str(cell).strip()
                            if not raw or ps.parse_vn_number(raw) is None:
                                continue
                            header = ""
                            for head_row in grid[:2]:
                                if column < len(head_row):
                                    piece = str(head_row[column]).strip()
                                    if piece and piece not in header:
                                        header = f"{header} {piece}".strip()
                            cells.append((column, header, raw))
                        if not cells:
                            continue
                        found.append((value, {
                            "table_id": table_id,
                            "row": index,
                            "code": code,
                            "label": row_label(row, skip),
                            "heading": label[:len(label) - len(row_label(row, skip))
                                             ].strip(),
                            "cells": cells,
                            "csv": str(paths[table_id].relative_to(ROOT)
                                       ).replace("\\", "/"),
                            "scale": scale_from_headers(grid),
                        }))
                if found:
                    break

            if not found:
                note("khong co ung vien")
                continue
            found.sort(key=lambda item: -item[0])
            chosen = []
            seen = set()
            for _value, item in found:
                key = (item["table_id"], item["row"])
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(item)
                if len(chosen) >= CANDIDATES:
                    break

            blocks = [render(index, item) for index, item in enumerate(chosen)]
            handle.write(json.dumps({
                "id": qid, "system": SYSTEM,
                "user": USER.format(question=text, candidates="\n\n".join(blocks)),
            }, ensure_ascii=False) + "\n")
            plan[str(qid)] = chosen
            written += 1

    (ROOT / args.plan).write_text(json.dumps(plan, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"{written} prompt")
    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  bo qua — {reason}: {count}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
