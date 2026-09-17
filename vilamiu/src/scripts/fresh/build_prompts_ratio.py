"""Prompts for the rate questions: name TWO codes, not one.

248 questions ask for a %, a multiple or a turnover and are dropped by the builder,
because a single cell cannot be converted into a rate. The model plan addresses many of
them, but a single address is the wrong shape for the answer.

The fix is the same division as everywhere else, applied twice: the model names the
numerator line and the denominator line, and code reads both cells at full precision
with each table's own scale and divides. A rate is where per-read accuracy compounds,
so the reads have to stay on the verified path — the identities check them at 97–99.6%
and the cross-year comparison at 90.9%.

Two named ratios are given in the instructions because their definition is not visible
in the question: ROA and ROE divide by equity or assets, and the corpus prints both.
Everything else the question states explicitly.

Usage:
  python scripts/fresh/build_prompts_ratio.py --out artifacts/fresh/prompts_ratio.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from render_block import Corpus  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

PERCENT_RE = re.compile(r"%|phần trăm", re.I)
TIMES_RE = re.compile(r"bao nhiêu lần|vòng quay|hệ số", re.I)

SYSTEM = """Bạn đọc một câu hỏi về một TỶ LỆ trong báo cáo tài chính Việt Nam và chỉ ra
HAI Ô cần chia cho nhau.

Bạn được cho danh sách đầy đủ các dòng của ba báo cáo chính, mỗi dòng có Mã số, tên chỉ
tiêu, giá trị kỳ này và kỳ trước. Kèm mục lục thuyết minh.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"tu": {"nguon": "cdkt|kqkd|lctt|TM", "ma": "..."},
 "mau": {"nguon": "cdkt|kqkd|lctt|TM", "ma": "..."},
 "ky": "current" | "prior",
 "phep": "phan_tram" | "lan"}

Quy tắc:
- Cả hai `ma` PHẢI có trong danh sách. Không bịa mã.
- `phan_tram` nghĩa là kết quả = tử/mẫu × 100. `lan` nghĩa là tử/mẫu.
- Tử và mẫu phải là HAI dòng khác nhau.
- Mẫu số của "tỷ trọng A trong B" là B. Nếu câu hỏi không nêu mẫu số, dùng dòng tổng
  hợp lý nhất: tổng tài sản (270), tổng nguồn vốn (440), hoặc doanh thu thuần (10).
- Các tỷ lệ có tên riêng: ROA = lợi nhuận sau thuế / tổng cộng tài sản;
  ROE = lợi nhuận sau thuế / vốn chủ sở hữu; biên lợi nhuận gộp = lợi nhuận gộp /
  doanh thu thuần; hệ số nợ = nợ phải trả / tổng cộng tài sản.
- Nếu câu hỏi dùng tỷ lệ chỉ để LỌC rồi hỏi một chỉ tiêu khác, thì tỷ lệ KHÔNG phải
  đáp án — khi đó trả {"tu": {"nguon": "", "ma": ""}, "mau": {"nguon": "", "ma": ""},
  "ky": "current", "phep": "phan_tram"} để bỏ qua.
- ĐỪNG tự tính ra con số. Chỉ nói đọc ô nào."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{block}

Chia ô nào cho ô nào?"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/prompts_ratio.jsonl")
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)
    resolver = TickerResolver()

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    written = 0
    skipped = {"co don vi tien": 0, "khong phai cau ty le": 0,
               "nhieu ma hoac khong nam": 0, "khong co khoi": 0}
    sizes = []
    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for question in questions:
            text = question["question"]
            _name, unit = unit_of(text)
            if unit:
                skipped["co don vi tien"] += 1
                continue
            if not (PERCENT_RE.search(text) or TIMES_RE.search(text)):
                skipped["khong phai cau ty le"] += 1
                continue
            found = resolver.resolve(text)
            years = YEAR_RE.findall(text)
            if len(found) != 1 or not years:
                skipped["nhieu ma hoac khong nam"] += 1
                continue
            ticker = next(iter(found))
            year = max(years)
            scope = "separate" if PARENT_RE.search(text) else "consolidated"
            block = corpus.block(ticker, year, scope) or corpus.block(
                ticker, year, "consolidated" if scope == "separate" else "separate")
            if not block:
                skipped["khong co khoi"] += 1
                continue
            if not corpus.block(ticker, year, scope):
                scope = "consolidated" if scope == "separate" else "separate"

            handle.write(json.dumps({
                "id": question["id"],
                "system": SYSTEM,
                "user": USER.format(question=text, block=block),
                "meta": {"question": text, "ticker": ticker, "year": year,
                         "scope": scope},
            }, ensure_ascii=False) + "\n")
            written += 1
            sizes.append(len(block))

    print(f"{written} prompt cau ty le")
    for name, count in skipped.items():
        if count:
            print(f"  bo qua — {name}: {count}")
    if sizes:
        sizes.sort()
        print(f"  uoc token: p50={sizes[len(sizes) // 2] // 2.5:.0f} "
              f"max={sizes[-1] // 2.5:.0f}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
