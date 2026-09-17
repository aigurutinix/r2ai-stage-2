"""Write the prompts locally so the rented box only has to call the model.

Rendering happens here, not there: the block needs the statement index, the note index
and the corpus, and shipping all of that to a rented machine is how a run ends up
measuring stale code. The box receives a jsonl of finished prompts, calls the local
server, and writes replies. Nothing else.

The probe set is deliberately not random. It is drawn so the answer can be judged
without gold:

  half from the questions the rule-based matcher answers CONFIDENTLY — that set is
  52% correct, so agreement is a floor and disagreement is where the model wins or
  loses
  half from the questions it refuses — where the alternative today is a blank, so
  anything correct is new

The model is asked for a source, a code and a period, all of which must appear in the
block it was shown. A code outside the block is rejected without being trusted, which
is the property the earlier cell-picking attempt could not offer.

Usage:
  python scripts/fresh/build_prompts.py --n 100 --out artifacts/fresh/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from render_block import Corpus  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

SYSTEM = """Bạn đọc một câu hỏi về báo cáo tài chính Việt Nam và chỉ ra Ô chứa đáp án.

Bạn được cho DANH SÁCH ĐẦY ĐỦ các dòng của ba báo cáo chính, mỗi dòng có Mã số, tên
chỉ tiêu, giá trị kỳ này và kỳ trước. Kèm theo là mục lục các thuyết minh, mỗi thuyết
minh đã được xác định là chi tiết hóa dòng nào.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"nguon": "cdkt" | "kqkd" | "lctt" | "TM", "ma": "<Mã số hoặc số TM>", "ky": "current" | "prior"}

Quy tắc:
- `ma` PHẢI là một Mã số có trong danh sách, hoặc số của một thuyết minh trong mục lục.
  Không được bịa mã.
- Chọn thuyết minh (`"nguon": "TM"`) khi câu hỏi hỏi một mục CHI TIẾT không có trong ba
  báo cáo chính — ví dụ "lãi tiền gửi", "doanh thu cho thuê tàu bay", "dư nợ ngành
  thương mại", tên một công ty con, tên một người.
- Chọn báo cáo chính khi câu hỏi hỏi đúng một dòng của nó.
- `ky`: "cuối năm N" / "trong năm N" / "đến ngày 31/12/N" của báo cáo năm N là
  `current`; "đầu năm N" là `prior`.
- Đừng chọn theo giá trị lớn hay nhỏ. Chọn theo tên chỉ tiêu khớp câu hỏi.
- Chú ý các cặp dễ lẫn: "trước thuế" khác "sau thuế"; "phải nộp" khác "đã nộp";
  "ngắn hạn" khác "dài hạn"; "nguyên giá" khác "giá trị còn lại"; "vay" khác "nợ"."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{block}

Ô nào chứa đáp án?"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--plan", default="artifacts/fresh/answer_plan.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/prompts.jsonl")
    parser.add_argument("--n", type=int, default=100, help="0 = every question")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)
    resolver = TickerResolver()

    confident = set()
    plan_path = ROOT / args.plan
    if plan_path.exists():
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                confident.add(json.loads(line)["id"])

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    usable, skipped = [], 0
    for question in questions:
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        if len(found) != 1 or not years:
            skipped += 1
            continue
        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        block = corpus.block(ticker, year, scope)
        if not block:
            other = "consolidated" if scope == "separate" else "separate"
            block = corpus.block(ticker, year, other)
            if block:
                scope = other
        if not block:
            skipped += 1
            continue
        usable.append({
            "id": question["id"], "question": text, "ticker": ticker,
            "year": year, "scope": scope, "block": block,
            "in_plan": question["id"] in confident,
        })

    print(f"{len(usable)} cau dung duoc, {skipped} cau khong dung khoi nao")
    if args.n:
        rng = random.Random(args.seed)
        with_plan = [q for q in usable if q["in_plan"]]
        without = [q for q in usable if not q["in_plan"]]
        rng.shuffle(with_plan)
        rng.shuffle(without)
        half = args.n // 2
        usable = with_plan[:half] + without[:args.n - half]
        rng.shuffle(usable)
        print(f"probe: {sum(1 for q in usable if q['in_plan'])} cau bo luat tu tin, "
              f"{sum(1 for q in usable if not q['in_plan'])} cau bo luat tu choi")

    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as handle:
        for item in usable:
            handle.write(json.dumps({
                "id": item["id"],
                "system": SYSTEM,
                "user": USER.format(question=item["question"], block=item["block"]),
                "meta": {k: item[k] for k in
                         ("question", "ticker", "year", "scope", "in_plan")},
            }, ensure_ascii=False) + "\n")

    sizes = sorted(len(item["block"]) for item in usable)
    print(f"kich thuoc khoi: p50={sizes[len(sizes) // 2]} "
          f"p90={sizes[9 * len(sizes) // 10]} max={sizes[-1]} ky tu")
    print(f"  uoc token moi prompt: p50={sizes[len(sizes) // 2] // 3.5:.0f} "
          f"max={sizes[-1] // 3.5:.0f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
