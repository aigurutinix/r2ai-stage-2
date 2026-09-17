"""Prompts for the questions naming several companies.

247 questions name more than one company and get no prompt at all, because a block is
one company's statements and four full blocks overflow the window. The compact
rendering fixes that: headline lines only, no note index, about 2,000 tokens per
company, so four companies and the question fit inside 16k.

The division of labour is the one that worked on the single-company run, applied to a
harder shape. A cohort question usually filters — "trong nhóm …, doanh nghiệp có X cao
nhất, thì Y là bao nhiêu" — and the filtering is a comparison the model can perform,
because every figure is in front of it. What the model must NOT do is state the answer:
it names the metric, the operation and the companies, and code reads the cells and does
the arithmetic at full precision with the right unit scale.

So the output stays a closed choice:

  ma, nguon, ky   the line to read, identical for every company
  phep            one of: mot_ma, tong, hieu, trung_binh, lon_nhat, nho_nhat
  ma_ck           which companies to read it for, in order

`mot_ma` with a single ticker covers every filter question: the model applies the
filter using the numbers it can see, and reports which company won rather than what
its figure was.

Usage:
  python scripts/fresh/build_prompts_cohort.py --out artifacts/fresh/prompts_cohort.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from render_block import Corpus  # noqa: E402
from render_compact import compact  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

SYSTEM = """Bạn đọc một câu hỏi so sánh nhiều doanh nghiệp và chỉ ra CÁCH TÍNH đáp án.

Bạn được cho các dòng đầu mục của ba báo cáo chính cho từng doanh nghiệp, mỗi dòng có
Mã số, tên chỉ tiêu, giá trị kỳ này và kỳ trước.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"nguon": "cdkt" | "kqkd" | "lctt", "ma": "<Mã số>", "ky": "current" | "prior",
 "phep": "mot_ma" | "tong" | "hieu" | "trung_binh" | "lon_nhat" | "nho_nhat",
 "ma_ck": ["<mã CK>", ...]}

Quy tắc:
- `ma` PHẢI là một Mã số có trong danh sách. Không bịa mã.
- `ma_ck` chỉ gồm các mã có trong dữ liệu được cho, theo đúng thứ tự cần dùng.
- Nếu câu hỏi LỌC rồi hỏi chỉ tiêu của doanh nghiệp thắng — ví dụ "trong nhóm …,
  doanh nghiệp có vòng quay cao nhất, thì lợi nhuận là bao nhiêu" — thì bạn hãy TỰ so
  sánh bằng các con số đã cho, rồi trả `"phep": "mot_ma"` với `ma_ck` chỉ chứa doanh
  nghiệp thắng, và `ma` là chỉ tiêu CÂU HỎI CẦN BÁO, không phải chỉ tiêu dùng để lọc.
- `"hieu"` cần đúng hai mã, theo thứ tự câu hỏi nêu.
- ĐỪNG tự tính ra con số cuối. Hệ thống sẽ đọc ô và tính; bạn chỉ nói đọc ô nào.
- `ky`: "cuối năm N" / "trong năm N" của báo cáo năm N là `current`; "đầu năm N" là
  `prior`."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{blocks}

Đọc ô nào, của những doanh nghiệp nào, và ghép bằng phép gì?"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/prompts_cohort.jsonl")
    parser.add_argument("--max-companies", type=int, default=6)
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)
    resolver = TickerResolver()

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    written = 0
    skipped = {"khong phai cau nhom": 0, "khong co nam": 0,
               "khong cong ty nao co khoi": 0, "qua nhieu cong ty": 0}
    sizes = []
    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for question in questions:
            text = question["question"]
            found = sorted(resolver.resolve(text))
            if len(found) < 2:
                skipped["khong phai cau nhom"] += 1
                continue
            years = YEAR_RE.findall(text)
            if not years:
                skipped["khong co nam"] += 1
                continue
            if len(found) > args.max_companies:
                skipped["qua nhieu cong ty"] += 1
                continue
            year = max(years)
            scope = "separate" if PARENT_RE.search(text) else "consolidated"

            blocks, have = [], []
            for ticker in found:
                text_block = compact(corpus, ticker, year, scope)
                if not text_block:
                    other = "consolidated" if scope == "separate" else "separate"
                    text_block = compact(corpus, ticker, year, other)
                if text_block:
                    blocks.append(text_block)
                    have.append(ticker)
            if not blocks:
                skipped["khong cong ty nao co khoi"] += 1
                continue

            body = "\n\n".join(blocks)
            handle.write(json.dumps({
                "id": question["id"],
                "system": SYSTEM,
                "user": USER.format(question=text, blocks=body),
                "meta": {"question": text, "year": year, "scope": scope,
                         "tickers": have, "named": found},
            }, ensure_ascii=False) + "\n")
            written += 1
            sizes.append(len(body))

    print(f"{written} prompt cau nhom")
    for name, count in skipped.items():
        if count:
            print(f"  bo qua — {name}: {count}")
    if sizes:
        sizes.sort()
        print(f"kich thuoc: p50={sizes[len(sizes) // 2]} p90="
              f"{sizes[9 * len(sizes) // 10]} max={sizes[-1]} ky tu")
        print(f"  uoc token: p50={sizes[len(sizes) // 2] // 2.5:.0f} "
              f"max={sizes[-1] // 2.5:.0f}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
