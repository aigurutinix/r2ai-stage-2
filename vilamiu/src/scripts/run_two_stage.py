"""Understand the question first, then read one table — two model calls, not one.

Every mechanism this project has tried makes a single call carry the whole task: pick
among eight tables, find the row, find the column, and often write pandas too. Measured
through the real path, those land at 19.5% (write a program), 21.7% (return
coordinates) and 27.3% (copy the value). Handed a single table, the same model finds
the right cell 69.1% of the time. The difference is the choosing.

So the work is split at that seam:

  stage 1  the question alone, no tables. Return the note heading the figure lives
           under and the row it sits on. This is translation, which is what the model
           is good at: reports say "9. CHO VAY KHÁCH HÀNG", questions say "dư nợ cho
           vay các tổ chức kinh tế, cá nhân trong nước".
  stage 2  the predicted heading pins one table out of the eight-table shortlist, and
           only that table is shown. Return {row, col}. A deterministic
           `num(df, r, c)` is emitted from the coordinates, so the program always runs
           and always reads a real cell.

The heading is matched against `artifacts/table_notes.jsonl`, which recovers the real
note heading for 95.5% of tables. The store's own `caption` field is a heading on only
22.5% of them — form codes, company names and addresses otherwise — and an earlier
version of this experiment scored badly for exactly that reason: the model's headings
were right and had nothing to match.

Note this is not the change that lost on the leaderboard. That one put note headings
into the corpus-wide bag-of-words index, where headings are near-identical across
hundreds of companies and blunt it. Here the candidates are already the eight tables of
one company in one year, so cross-company similarity is irrelevant.

Output is the same record shape as `run_generate.py` writes, so `run_submit.py` can
consume it as the program cache with no new branch:
  PLANNED_FILE=planned_v3.jsonl python scripts/run_submit.py 10 5 artifacts/two_stage.jsonl

Usage:
  PYTHONPATH=src python scripts/run_two_stage.py --model Qwen/Qwen3-14B \
      --local-url http://127.0.0.1:18000/v1 --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

STAGE1_SYSTEM = """Bạn đọc câu hỏi về báo cáo tài chính Việt Nam và cho biết con số đó
nằm ở đâu. KHÔNG có bảng nào được đưa cho bạn — chỉ suy từ câu hỏi.

Báo cáo tài chính Việt Nam chia thành các thuyết minh có tiêu đề, ví dụ:
  "Tiền và các khoản tương đương tiền"
  "Cho vay khách hàng"
  "Phát hành giấy tờ có giá"
  "Tài sản cố định hữu hình"
  "Phải thu ngắn hạn khác"
  "Chi phí sản xuất kinh doanh theo yếu tố"
  "Thông tin về các bên liên quan"
  "Báo cáo bộ phận"

Câu hỏi thường gọi tên một DÒNG bên trong thuyết minh, không phải tên thuyết minh.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"note": "<tiêu đề thuyết minh>", "row": "<tên dòng>"}

Quy tắc:
- `note` là cụm danh từ, không số thứ tự, không tên công ty, không năm.
- `row` là khoản mục câu hỏi hỏi tới, viết theo lời báo cáo.
- Giữ đúng các cặp phân biệt: hữu hình/vô hình, ngắn hạn/dài hạn, trước thuế/sau thuế,
  phải thu/phải trả.
- Nếu con số nằm ngay trên báo cáo chính, dùng "Bảng cân đối kế toán",
  "Báo cáo kết quả hoạt động kinh doanh" hoặc "Báo cáo lưu chuyển tiền tệ"."""

STAGE1_USER = """Câu hỏi: {question}

Con số này nằm trong thuyết minh nào, và ở dòng nào?"""

STAGE2_SYSTEM = """Bạn định vị MỘT Ô trong một bảng của báo cáo tài chính Việt Nam.

Dòng đánh r1, r2, … (r0 là dòng tiêu đề). Cột đánh c0, c1, … ghi ngay trên dòng tiêu đề.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"row": <chỉ số dòng>, "col": <chỉ số cột>}

Quy tắc:
- Chọn dòng có nhãn khớp chỉ tiêu, không phải dòng có số lớn nhất. Nhãn thật thường có
  tiền tố đánh số ("1. ", "V. ") mà câu hỏi không có.
- Dòng cộng có thể để trống ô nhãn; nếu câu hỏi hỏi tổng thì lấy dòng đó.
- Bỏ qua cột "Mã số" và "Thuyết minh" — chúng không chứa giá trị.
- BẢNG NÀY THUỘC BÁO CÁO NĂM {year}. "Số cuối năm" = cuối năm {year};
  "Số đầu năm" = cuối năm {prev}; "Năm nay" = năm {year}; "Năm trước" = năm {prev}.
- Nếu không dòng nào chứa chỉ tiêu đó, trả {{"row": -1, "col": -1}}."""

STAGE2_USER = """<câu_hỏi>
{question}
</câu_hỏi>

<chỉ_tiêu_cần_tìm>
thuyết minh: {note}
dòng: {row}
</chỉ_tiêu_cần_tìm>

BẢNG — {title}
{table}

Ô nào chứa số câu hỏi cần?"""

JSON_RE = re.compile(r"\{[^{}]*\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
YEAR_RE = re.compile(r"_((?:19|20)\d{2})_")
MAX_ROWS = 60


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def overlap(a: str, b: str) -> float:
    left = {t for t in fold(a).split() if len(t) > 2}
    right = {t for t in fold(b).split() if len(t) > 2}
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if not shared:
        return 0.0
    coverage = shared / len(left)
    focus = shared / len(right)
    return 2 * coverage * focus / (coverage + focus)


def render(grid) -> str:
    """The column index bound to the header text, and full cell contents.

    Truncating cells at 30 characters deletes the tail of 18.7% of the labels that
    hold answers, and Vietnamese line items carry their distinguishing word there.
    """

    lines = [",".join(f"c{c}={cell}" for c, cell in enumerate(grid[0]))]
    for index, row in enumerate(grid[1:MAX_ROWS], start=1):
        lines.append(f"r{index}: " + ",".join(str(cell) for cell in row))
    return "\n".join(lines)


def emit(row: int, column: int, factor: float) -> str:
    """A self-contained program reading grid cell (row, column).

    `frame_from_rows` consumes grid row 0 as the header, so grid row r is DataFrame
    row r-1.
    """

    return f'''def num(frame, r, c):
    text = str(frame.iloc[r, c]).strip()
    if text in ("-", "", "--", "\\u2013", "\\u2014", "nan", "None"):
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace("%", "").replace(" ", "")
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(".", "")
    value = float(text)
    return -value if negative else value

result = round(abs(num(df, {row - 1}, {column})) * {factor!r}, 2)'''


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--cache", default="artifacts/two_stage.jsonl")
    parser.add_argument("--spec", default="artifacts/spec.jsonl")
    # The leaderboard says the ranking is already near-optimal where it works:
    # TABLES_MRR5 0.6374 against TABLES_RECALL 0.7086 means that when a gold table
    # is in our list, its reciprocal rank averages 0.90 — it is at rank 1 almost
    # every time. So pinning by note title can only re-order a list whose head is
    # usually already right, and may move away from it. `--pin rank1` skips stage 1
    # and reads the top-ranked table, which isolates "read one table" from "choose
    # the table" and costs half the model calls.
    parser.add_argument("--pin", choices=("note", "rank1"), default="note")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    client = (ChatClient.local(args.model, args.local_url) if args.local_url
              else ChatClient.from_env(ROOT, model=args.model))

    notes: dict[tuple[str, int], str] = {}
    notes_path = ROOT / "artifacts" / "table_notes.jsonl"
    for line in notes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            notes[(row["doc"], int(row["table_id"]))] = row["note"]
    print(f"{len(notes)} bảng có tiêu đề thuyết minh", flush=True)

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    cache_path = ROOT / args.cache
    spec_path = ROOT / args.spec
    done = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in questions if q.id not in done]
    print(f"{len(questions)} câu, {len(done)} đã có, {len(todo)} cần chạy", flush=True)

    handle = cache_path.open("a", encoding="utf-8")
    spec_handle = spec_path.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"n": 0, "ok": 0, "no_spec": 0, "no_cell": 0}
    started = time.time()

    def ask(system: str, user: str) -> dict | None:
        try:
            reply = client.complete(system, user)
        except RuntimeError:
            return None
        match = JSON_RE.search(THINK_RE.sub("", reply or ""))
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            return None

    def work(question) -> None:
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        if not keys:
            return

        note = row_hint = ""
        best_key, best_score = keys[0], -1.0
        if args.pin == "note":
            spec = ask(STAGE1_SYSTEM, STAGE1_USER.format(question=question.question))
            note = str((spec or {}).get("note", "")).strip()
            row_hint = str((spec or {}).get("row", "")).strip()
            if not note and not row_hint:
                with lock:
                    counters["no_spec"] += 1
                return
            # The heading pins one table among the candidates of this company and year.
            for key in keys:
                title = notes.get((key.doc_name, key.table_id), "")
                if not title:
                    continue
                score = max(overlap(note, title), 0.7 * overlap(row_hint, title))
                if score > best_score:
                    best_key, best_score = key, score

        grid = store.rows(best_key)
        if not grid or len(grid) < 2:
            return
        meta = store.meta(best_key)
        year_match = YEAR_RE.search(best_key.doc_name)
        year = int(year_match.group(1)) if year_match else 0
        title = notes.get((best_key.doc_name, best_key.table_id),
                          str(getattr(meta, "caption", "")))

        placed = ask(
            STAGE2_SYSTEM.replace("{year}", str(year)).replace("{prev}", str(year - 1)),
            STAGE2_USER.format(question=question.question, note=note, row=row_hint,
                               title=title[:110], table=render(grid)))
        try:
            row_index = int((placed or {})["row"])
            column = int((placed or {})["col"])
        except (KeyError, TypeError, ValueError):
            with lock:
                counters["no_cell"] += 1
            return
        if row_index < 1 or column < 1 or row_index >= len(grid) \
                or column >= len(grid[row_index]):
            with lock:
                counters["no_cell"] += 1
            return

        context = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        scale = lookup_mod.column_scale(grid, column, context)
        factor = scale / (question.unit_scale or 1.0)
        code = emit(row_index, column, factor)
        outcome = run_query(code, {"df": grid})

        record = {
            "id": question.id,
            "ok": outcome.ok,
            "value": outcome.value,
            "error": (outcome.error or "")[:300],
            "attempts": 1,
            "code": code,
            "variables": ["df"],
            "refs": {"df": f"{best_key.doc_name}|{int(meta.start_line)}"},
            "keys": [[best_key.doc_name, best_key.table_id]],
        }
        with lock:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            spec_handle.write(json.dumps(
                {"id": question.id, "note": note, "row": row_hint,
                 "picked": [best_key.doc_name, best_key.table_id],
                 "title": title[:110], "score": round(best_score, 3)},
                ensure_ascii=False) + "\n")
            spec_handle.flush()
            counters["n"] += 1
            counters["ok"] += int(outcome.ok)
            if counters["n"] % 50 == 0:
                print(f"  {counters['n']}/{len(todo)}  chạy được="
                      f"{counters['ok'] / counters['n']:.1%}  "
                      f"{time.time() - started:.0f}s", flush=True)

    def guarded(question) -> None:
        try:
            work(question)
        except Exception as exc:  # noqa: BLE001 - one question must not end the run
            print(f"  id={question.id} bỏ qua: {type(exc).__name__}: {exc}"[:150],
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(guarded, todo))
    handle.close()
    spec_handle.close()
    print(f"xong: {counters['n']} câu, chạy được {counters['ok']}, "
          f"không có spec {counters['no_spec']}, không định vị được "
          f"{counters['no_cell']}", flush=True)


if __name__ == "__main__":
    main()
