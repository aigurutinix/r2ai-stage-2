"""Ask the model for TWO cells and an operation, for the 251 ratio questions.

The single-cell reader cannot answer a ratio by construction, and the audit shows
it: on the 172 questions outside its shape, 88% of its answers contradict their own
question. Ratio/share/growth questions are 251 of the 1012 — the largest block this
project has never addressed with generation — and the deterministic ratio branches
answer 29 of them.

The judgement here needs no gold, which is the point. A ratio proposal is checked
three ways that cost nothing:

  labels    the numerator row label must share words with the question's numerator
            phrase, and the denominator's with its denominator phrase
  range     a share is in [0, 100]; a growth rate is rarely outside [-100, 500]
  same-row  numerator and denominator drawn from the same row and column is a
            tell-tale of the model giving up (it produces exactly 100.0)

Run against OpenRouter first: a GPU rental is 20x the cost of probing 60 questions
through an API, and the question is whether the shape works at all.

Usage:
  PYTHONPATH=src python scripts/_probe_ratio_cells.py --n 60 --model qwen/qwen3-14b
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SYSTEM = """Bạn tính MỘT TỶ LỆ từ một bảng báo cáo tài chính Việt Nam.

Dòng đánh r1, r2, … (r0 là dòng tiêu đề). Cột đánh c0, c1, … ghi ngay trên dòng tiêu đề.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"tu_dong": <dòng tử số>, "tu_cot": <cột tử số>,
 "mau_dong": <dòng mẫu số>, "mau_cot": <cột mẫu số>,
 "phep": "chia" | "tang_truong"}

Quy tắc:
- "chia": kết quả = tử/mẫu × 100. Dùng cho tỷ trọng, tỷ lệ, cơ cấu, "chiếm bao nhiêu %".
- "tang_truong": kết quả = (tử-mẫu)/mẫu × 100. Tử là kỳ SAU, mẫu là kỳ TRƯỚC.
- Tử số và mẫu số PHẢI khác ô nhau. Nếu chúng trùng thì đề bài không hỏi tỷ lệ.
- Mẫu số của tỷ trọng thường là dòng tổng ("TỔNG CỘNG", "Cộng"), có thể để trống ô nhãn.
- BẢNG NÀY THUỘC BÁO CÁO NĂM {year}. "Số cuối năm" = cuối năm {year};
  "Số đầu năm" = cuối năm {prev}; "Năm nay" = năm {year}; "Năm trước" = năm {prev}.
- Nếu bảng không đủ hai chỉ tiêu đó, trả {{"tu_dong": -1, "tu_cot": -1, "mau_dong": -1, "mau_cot": -1, "phep": "chia"}}."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

BẢNG — {title}
{table}

Tử số ở ô nào, mẫu số ở ô nào?"""

JSON_RE = re.compile(r"\{[^{}]*\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
YEAR_RE = re.compile(r"_((?:19|20)\d{2})_")
MAX_ROWS = 60
STOP = {"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "trong", "cho",
        "tai", "theo", "phan", "tram", "ctcp", "chiem", "bang", "moi"}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def tokens(text: str) -> set[str]:
    return {t for t in fold(text).split() if len(t) > 2 and t not in STOP}


def covered(question: str, label: str) -> float:
    right = tokens(label)
    return len(tokens(question) & right) / len(right) if right else 0.0


def render(grid) -> str:
    lines = [",".join(f"c{c}={cell}" for c, cell in enumerate(grid[0]))]
    for index, row in enumerate(grid[1:MAX_ROWS], start=1):
        lines.append(f"r{index}: " + ",".join(str(cell) for cell in row))
    return "\n".join(lines)


def parse_json(text: str) -> dict | None:
    text = THINK_RE.sub("", text or "")
    match = None
    for match in JSON_RE.finditer(text):
        pass
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def number(text: str) -> float | None:
    text = str(text).strip()
    if text in ("-", "", "--", "–", "—", "nan", "None"):
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace("%", "").replace(" ", "")
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(".", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="artifacts/probe_ratio.jsonl")
    args = parser.parse_args()

    import importlib.util

    shape_spec = importlib.util.spec_from_file_location(
        "qs", ROOT / "scripts" / "_question_shape.py")
    shape_mod = importlib.util.module_from_spec(shape_spec)
    shape_spec.loader.exec_module(shape_mod)

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    ratio = [q for q in questions if shape_mod.shape(q.question) == "ty le"]
    ratio = ratio[:args.n]
    print(f"{len(ratio)} cau ty le (tren {len(questions)} cau)")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    client = ChatClient.from_env(ROOT, model=args.model)
    notes = {}
    notes_path = ROOT / "artifacts" / "table_notes.jsonl"
    if notes_path.exists():
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                notes[(record["doc"], int(record["table_id"]))] = record.get("note", "")

    lock = threading.Lock()
    counters: Counter[str] = Counter()
    results = []

    def work(question) -> None:
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        if not keys:
            counters["no_table"] += 1
            return
        key = keys[0]
        grid = store.rows(key)
        if not grid or len(grid) < 3:
            counters["tiny"] += 1
            return
        match = YEAR_RE.search(key.doc_name)
        year = int(match.group(1)) if match else 2024
        title = notes.get((key.doc_name, key.table_id), "")
        try:
            reply = client.complete(
                SYSTEM.replace("{year}", str(year)).replace("{prev}", str(year - 1)),
                USER.format(question=question.question, title=title, table=render(grid)),
            )
        except Exception as error:  # noqa: BLE001
            with lock:
                counters["transport"] += 1
                if counters["transport"] <= 3:
                    print(f"  loi goi: {type(error).__name__}: {error}")
            return

        spec = parse_json(reply)
        if not spec:
            counters["no_json"] += 1
            return
        try:
            tr, tc = int(spec["tu_dong"]), int(spec["tu_cot"])
            dr, dc = int(spec["mau_dong"]), int(spec["mau_cot"])
        except (KeyError, TypeError, ValueError):
            counters["bad_json"] += 1
            return
        if tr < 1 or dr < 1:
            counters["gave_up"] += 1
            return
        if (tr, tc) == (dr, dc):
            counters["same_cell"] += 1
            return
        if tr >= len(grid) or dr >= len(grid):
            counters["out_of_range"] += 1
            return

        top = number(grid[tr][tc]) if tc < len(grid[tr]) else None
        bottom = number(grid[dr][dc]) if dc < len(grid[dr]) else None
        if top is None or not bottom:
            counters["nil_cell"] += 1
            return
        op = str(spec.get("phep", "chia"))
        value = ((top - bottom) / bottom if op == "tang_truong" else top / bottom) * 100

        record = {
            "id": question.id,
            "question": question.question,
            "table": f"{key.doc_name}|{key.table_id}",
            "op": op,
            "value": round(value, 2),
            "tu": [tr, tc, str(grid[tr][0])[:60], str(grid[tr][tc])[:24]],
            "mau": [dr, dc, str(grid[dr][0])[:60], str(grid[dr][dc])[:24]],
            "tu_khop": round(covered(question.question, grid[tr][0]), 2),
            "mau_khop": round(covered(question.question, grid[dr][0]), 2),
            "trong_khoang": -100.0 <= value <= 500.0,
        }
        with lock:
            counters["ok"] += 1
            results.append(record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, ratio))

    out = ROOT / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
                   encoding="utf-8")
    print("ket qua:", dict(counters))
    if results:
        good = sum(1 for r in results if r["trong_khoang"])
        both = sum(1 for r in results if r["tu_khop"] >= 0.3 and r["mau_khop"] >= 0.3)
        print(f"  trong khoang hop ly: {good}/{len(results)} "
              f"({100 * good / len(results):.0f}%)")
        print(f"  ca hai nhan khop >=0.3: {both}/{len(results)} "
              f"({100 * both / len(results):.0f}%)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
