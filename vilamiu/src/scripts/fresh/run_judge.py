"""An adjudicator that reads the evidence, not the votes.

The mentor's reference architecture ends in a Critic that talks back to the Programmer.
Ours has never talked back: when two passes disagree, or a gate rejects a reading, the
question silently falls to the old answer (~40% right). That silent pool is ~340 questions.

Voting cannot resolve them — the samples share habits, and the published result is that
majority voting entrenches the shared error. What can resolve them is what this sends: for
each candidate, the program is executed and the ROWS IT ACTUALLY READ are printed — row
label, cell value, column header — and the model adjudicates on that printed text. The
judge sees `A đọc dòng 'Tiền mặt' = 1.535.353.919` beside `B đọc dòng 'Tiền và các khoản
tương đương tiền' = 948.303.528.748` and the question's own words, which is exactly the
comparison every hand-read in this project used to overturn a wrong answer.

The winner's program already exists, so shipping needs nothing new. A verdict of `khong`
(neither) keeps the incumbent — the judge is allowed to refuse.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

sys.path.insert(0, str(ROOT / "scripts" / "fresh"))
from bind_label import own_label  # noqa: E402
from find_statements import locate_columns  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "sandbox", ROOT / "vifinqa-official/src/vifinqa/answering/sandbox.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)

CELL_RE = re.compile(
    r'dfs\[\s*["\']([^"\']+)["\']\s*\]\s*\.iloc\[\s*(\d+)\s*,\s*(\d+)\s*\]')
JSON_RE = re.compile(r'\{[^{}]*"chon"[^{}]*\}')
THINK_RE = re.compile(r"^.*</think>\s*", re.S)

SYSTEM = """Bạn là trọng tài kiểm chứng số liệu báo cáo tài chính Việt Nam.

Hai lời giải A và B cho cùng một câu hỏi ra hai đáp án khác nhau. Với mỗi lời giải, bên
dưới in ra NHỮNG Ô MÀ NÓ THỰC SỰ ĐỌC: nhãn dòng, giá trị ô, và tiêu đề cột — lấy thẳng từ
dữ liệu, không phải model nào tự thuật lại.

Cách phân xử, theo thứ tự:
1. NHÃN DÒNG phải khớp đúng chỉ tiêu câu hỏi nêu. Nhãn thừa chữ (câu hỏi nói "Tiền" mà
   nhãn là "Tiền mặt") hay thiếu chữ đều là sai dòng.
2. TIÊU ĐỀ CỘT phải khớp kỳ câu hỏi nêu (năm nào, đầu/cuối năm, năm nay/năm trước).
3. Đơn vị bảng và phép quy đổi về đơn vị câu hỏi phải đúng.
4. Phép tính (cộng/trừ/chia) phải đúng nghĩa câu hỏi (tổng thì không lấy thành phần,
   chênh lệch không nêu chiều thì không âm).

Nếu cả hai cùng sai, hoặc không đủ bằng chứng, chọn "khong".

Trả lời đúng MỘT dòng JSON: {"chon": "A" | "B" | "khong", "ly_do": "<một câu>"}"""


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPEN_ROUTER_KEY") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("khong tim thay OPEN_ROUTER_KEY")


_grid_cache: dict[str, list[list[str]]] = {}


def grid_for(ref: str, catalogue: dict[str, str]) -> list[list[str]]:
    if ref in _grid_cache:
        return _grid_cache[ref]
    import csv as csv_mod
    grid: list[list[str]] = []
    relative = catalogue.get(ref)
    path = ROOT / relative if relative else None
    if path is None or not path.exists():
        doc, _, raw = ref.rpartition("|table_")
        found = list((ROOT / "data/official_corpus").glob(f"*/*/{doc}"))
        if found:
            path = found[0] / f"{doc}_extracted_tables" / f"table_{raw}.csv"
    if path is not None and path.exists():
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                grid = list(csv_mod.reader(handle))
        except OSError:
            grid = []
    while len(_grid_cache) >= 300:
        _grid_cache.pop(next(iter(_grid_cache)))
    _grid_cache[ref] = grid
    return grid


def evidence_of(record: dict) -> str:
    """The rows a program reads, printed from the data itself."""

    query = record.get("pandas_query") or ""
    catalogue = record.get("csvs") or {}
    lines = []
    for ref, row_text, col_text in CELL_RE.findall(query)[:8]:
        row, col = int(row_text), int(col_text)
        grid = grid_for(ref, catalogue)
        # `iloc` counts data rows; the CSV's first line is the header.
        if not grid or row + 1 >= len(grid):
            lines.append(f"- {ref.split('|')[-1]}.iloc[{row},{col}]: NGOAI BANG")
            continue
        code_col, pointer_col = locate_columns(grid)
        skip = {c for c in (code_col, pointer_col) if c is not None}
        label = own_label(grid[row + 1], skip) or "(khong nhan)"
        value = (grid[row + 1][col] if col < len(grid[row + 1]) else "")
        header = (grid[0][col] if col < len(grid[0]) else "")
        table = ref.split("|")[-1]
        lines.append(f"- {table} dòng {row} cột {col}: nhãn='{label[:70]}' "
                     f"giá trị='{value}' tiêu đề cột='{str(header)[:44]}'")
    if not lines:
        lines.append("- (chương trình không đọc ô cụ thể nào — tính gộp)")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", default="artifacts/fresh/progA_results.jsonl")
    parser.add_argument("--second", default="artifacts/fresh/progB_results.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/judge_results.jsonl")
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    def load(relative: str) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("answer") in (None, ""):
                continue
            out[int(row["id"])] = row
        return out

    first, second = load(args.first), load(args.second)
    questions = {json.loads(l)["id"]: json.loads(l)["question"]
                 for l in (ROOT / "data/questions/questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if l.strip()}

    disputes = []
    for qid in sorted(set(first) & set(second)):
        a, b = float(first[qid]["answer"]), float(second[qid]["answer"])
        if abs(a - b) > max(0.01, abs(a) * 1e-9):
            disputes.append(qid)

    target = ROOT / args.out
    done = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    disputes = [q for q in disputes if q not in done]
    if args.limit:
        disputes = disputes[: args.limit]
    print(f"{len(disputes)} cau tranh chap, {len(done)} da xu", flush=True)

    key = api_key()
    lock = threading.Lock()
    handle = target.open("a", encoding="utf-8")
    counters = {"A": 0, "B": 0, "khong": 0, "khong parse duoc": 0, "loi goi": 0}
    started = time.time()

    def work(qid: int) -> None:
        a, b = first[qid], second[qid]
        user = (f"<câu_hỏi>\n{questions[qid]}\n</câu_hỏi>\n\n"
                f"LỜI GIẢI A — đáp án {float(a['answer']):,.2f}\n"
                f"Ô mà A đọc:\n{evidence_of(a)}\n\n"
                f"LỜI GIẢI B — đáp án {float(b['answer']):,.2f}\n"
                f"Ô mà B đọc:\n{evidence_of(b)}\n\n"
                "Phân xử theo 4 bước và trả về một dòng JSON.")
        body = json.dumps({
            "model": args.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": 0.0, "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }).encode("utf-8")
        reply = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    ENDPOINT, data=body,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(request, timeout=240) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                reply = ((payload.get("choices") or [{}])[0].get("message")
                         or {}).get("content") or ""
                break
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (
                        400, 401, 402, 403):
                    break
                if attempt == 2:
                    break
                time.sleep(3 * (attempt + 1))
        if reply is None:
            with lock:
                counters["loi goi"] += 1
            return
        match = JSON_RE.search(THINK_RE.sub("", reply))
        verdict_ = None
        if match:
            try:
                verdict_ = json.loads(match.group(0))
            except ValueError:
                verdict_ = None
        with lock:
            if not verdict_ or verdict_.get("chon") not in ("A", "B", "khong"):
                counters["khong parse duoc"] += 1
                return
            choice = verdict_["chon"]
            counters[choice] += 1
            handle.write(json.dumps(
                {"id": qid, "chon": choice,
                 "ly_do": str(verdict_.get("ly_do") or "")[:200]},
                ensure_ascii=False) + "\n")
            handle.flush()
            total = sum(counters.values())
            if total % 25 == 0:
                print(f"  {total}/{len(disputes)}  {time.time() - started:.0f}s",
                      flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, disputes))
    handle.close()
    print()
    for name, count in counters.items():
        print(f"  {name}: {count}")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
