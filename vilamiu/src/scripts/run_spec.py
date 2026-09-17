"""Turn each question into a structure, using the model for the wording only.

The deterministic branches each carry their own family of regular expressions to
read the question: `ratio.RATIO_RE` for "tỷ trọng X trên Y", `compose` for the
operations, `lookup.extract_metric` for the plain metric, plus a dozen patterns
that strip owner clauses, leading years and trailing periods. That is the part
that cannot be finished — Vietnamese has more ways to phrase a share than the
alternation has branches, and 94 of the 157 in-scope ratio questions fall outside
it.

This replaces the reading, not the answering. The prompt contains THE QUESTION AND
NOTHING ELSE — no table, no rows, no cells. What comes back is the operation and
the metric phrases, and those phrases are then fed to the existing label matcher,
column picker, unit scaler and sanity gate. The division is deliberate: asked to
pick a cell out of a table it was handed, this model cost 0.38 questions per row
changed on the board (20/08); asked to name what the question is asking for, it is
doing the job the regexes are bad at.

Because the prompt has no table in it, a full sweep of 1012 questions is a few
hundred tokens each instead of 16k — cheap enough for an API and not worth a GPU
rental on its own.

Usage:
  PYTHONPATH=src python scripts/run_spec.py --model qwen/qwen3-14b
  PYTHONPATH=src python scripts/run_spec.py --local-url http://127.0.0.1:18000/v1 \
      --model Qwen/Qwen3-14B
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402

SYSTEM = """Bạn đọc MỘT CÂU HỎI về báo cáo tài chính Việt Nam và nói nó hỏi phép tính gì.
Bạn KHÔNG được thấy bảng, và KHÔNG cần biết con số. Chỉ mô tả cấu trúc câu hỏi.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"phep": "<tên phép>", "tu": "<chỉ tiêu tử số>", "mau": "<chỉ tiêu mẫu số>",
 "bien_the": ["<cách báo cáo in tên chỉ tiêu tử số>", "..."]}

Tên phép, chọn đúng một:
- "doc"          : chỉ đọc một chỉ tiêu, không tính gì. mau = ""
- "chia"         : tử/mẫu, gồm tỷ trọng, tỷ lệ, cơ cấu, "chiếm bao nhiêu %",
                   "bằng bao nhiêu % của", hệ số, tỷ suất, vòng quay
- "tang_truong"  : (sau-trước)/trước, gồm "tăng bao nhiêu %", "tốc độ tăng trưởng"
- "hieu"         : sau trừ trước, gồm chênh lệch, biến động, "tăng bao nhiêu đồng"
- "tong"         : cộng nhiều chỉ tiêu hoặc nhiều công ty
- "trung_binh"   : bình quân qua các kỳ

Quy tắc viết chỉ tiêu:
- Viết ĐÚNG tên dòng như báo cáo in ra, bỏ hết tên công ty, mã cổ phiếu, năm,
  ngày, đơn vị tiền, và bỏ các chữ "của", "tại", "cuối năm", "hợp nhất", "công ty mẹ".
  "Tỷ trọng tài sản ngắn hạn trong tổng nguồn vốn của công ty mẹ ABC cuối năm 2024"
  -> {"phep":"chia","tu":"tài sản ngắn hạn","mau":"tổng nguồn vốn"}
- Với "tang_truong" và "hieu", tử và mẫu là CÙNG một chỉ tiêu; ghi nó vào cả hai.
- Với "doc" và "tong", mau = "".
- Đừng thêm chữ nào câu hỏi không có. Đừng suy ra công thức kế toán, trừ các tên
  chuẩn: ROA = "lợi nhuận sau thuế" / "tổng cộng tài sản"; ROE = "lợi nhuận sau
  thuế" / "vốn chủ sở hữu"; biên lợi nhuận gộp = "lợi nhuận gộp" / "doanh thu thuần".

"bien_the": 3 đến 5 cách mà BÁO CÁO TÀI CHÍNH thật sẽ in tên chỉ tiêu tử số, kể cả
khi khác hẳn chữ trong câu hỏi. Đây là phần quan trọng nhất: bộ khớp nhãn của hệ
thống chỉ đếm từ trùng, nên nó không nối được "vốn cổ phần" với dòng thật là "Vốn
góp của chủ sở hữu", và 18% câu hỏi hiện không khớp được dòng nào.
  "vốn cổ phần"      -> ["Vốn góp của chủ sở hữu", "Vốn cổ phần", "Vốn điều lệ"]
  "tiền mặt"         -> ["Tiền", "Tiền mặt", "Tiền và các khoản tương đương tiền"]
  "nợ ngắn hạn"      -> ["Nợ ngắn hạn", "NỢ NGẮN HẠN", "Tổng nợ ngắn hạn"]
Viết như dòng in trong bảng: không kèm tên công ty, không kèm năm, không kèm đơn vị.
Nếu không nghĩ ra cách nói khác thì để đúng một phần tử là chính chỉ tiêu đó."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

Câu hỏi này hỏi phép tính gì?"""

JSON_RE = re.compile(r"\{[^{}]*\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
OPS = ("doc", "chia", "tang_truong", "hieu", "tong", "trung_binh")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--local-url", default="")
    parser.add_argument("--out", default="artifacts/spec_q.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N in-scope questions, for a probe")
    parser.add_argument("--all", action="store_true",
                        help="every question, not only the rate-unit ones")
    args = parser.parse_args()

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    scope = questions if args.all else [
        q for q in questions if q.target_unit in ("phan_tram", "lan", "vong")]
    out_path = ROOT / args.out
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in scope if q.id not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(scope)} cau trong pham vi, {len(done)} da co, {len(todo)} can chay",
          flush=True)
    if not todo:
        return

    client = (ChatClient.local(args.model, args.local_url) if args.local_url
              else ChatClient.from_env(ROOT, model=args.model))
    handle = out_path.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters: Counter[str] = Counter()
    started = time.time()

    def work(question) -> None:
        try:
            reply = client.complete(SYSTEM, USER.format(question=question.question))
        except Exception as error:  # noqa: BLE001
            with lock:
                counters["transport"] += 1
                if counters["transport"] <= 3:
                    print(f"  loi goi: {type(error).__name__}: {error}", flush=True)
            return
        match = None
        for match in JSON_RE.finditer(THINK_RE.sub("", reply or "")):
            pass
        if match is None:
            counters["no_json"] += 1
            return
        try:
            spec = json.loads(match.group(0))
        except json.JSONDecodeError:
            counters["bad_json"] += 1
            return
        op = str(spec.get("phep", "")).strip()
        if op not in OPS:
            counters["op_la"] += 1
            return
        variants = spec.get("bien_the") or []
        if isinstance(variants, str):
            variants = [variants]
        record = {
            "id": question.id,
            "op": op,
            "tu": str(spec.get("tu", "")).strip(),
            "mau": str(spec.get("mau", "")).strip(),
            # Deduplicated but not filtered: a phrasing that matches no row costs
            # one more `match_row` call and changes nothing, while the one that
            # matches is the whole point.
            "bien_the": list(dict.fromkeys(
                str(v).strip() for v in variants if str(v).strip()))[:5],
        }
        if not record["tu"]:
            counters["khong co tu"] += 1
            return
        with lock:
            counters[op] += 1
            counters["ok"] += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if counters["ok"] % 50 == 0:
                print(f"  {counters['ok']}/{len(todo)}  {time.time() - started:.0f}s",
                      flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()
    print("ket qua:", dict(counters))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
