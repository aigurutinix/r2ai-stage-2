"""Ask the model for the indicator and the ways it might be printed. Nothing else.

The reach measurement matches question wording against row labels with one folded token
set, and 18% of questions find nothing that way. That is the failure the design hands to a
model on purpose: a question says "chi phí quản lý doanh nghiệp" and the statement prints
"Chi phí quản lý DN", or the question says "lãi tiền gửi" and the note prints "Lãi tiền gửi,
tiền cho vay". Generating the variants is a language job; matching them is not.

The model sees the question and no tables. That keeps this stage cheap, keeps it checkable
against a closed vocabulary, and keeps it out of the business of choosing cells — which is
where every model pass in this project has gone wrong.

Usage:
  python scripts/fresh/build_spec_prompts.py --limit 120 --out artifacts/fresh/prompts_spec.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = """Bạn đọc một câu hỏi về báo cáo tài chính Việt Nam và trích ra chỉ tiêu mà câu \
hỏi cần, cùng những cách mà chỉ tiêu đó có thể được IN RA trong báo cáo.

Bạn KHÔNG trả lời câu hỏi. Bạn không thấy bảng nào. Chỉ đọc câu hỏi.

Trả lời bằng ĐÚNG một đối tượng JSON:
{"chi_tieu": [{"ten": "<tên chỉ tiêu chuẩn>",
               "bien_the": ["<cách in 1>", "<cách in 2>", "<cách in 3>"]}],
 "phep_tinh": "<một trong: single, difference, ratio, growth, share, sum, average, minimum, maximum, argmin, argmax, count>",
 "ky": "<một trong: cuoi_nam, dau_nam, nam_nay, nam_truoc, khong_ro>"}

Quy tắc:
- `bien_the` là cách nhãn dòng có thể được in trong bảng, KHÔNG phải cách diễn đạt lại câu
  hỏi. Ví dụ câu hỏi "chi phí quản lý doanh nghiệp" thì biến thể gồm
  "Chi phí quản lý doanh nghiệp", "Chi phí quản lý DN", "9. Chi phí quản lý doanh nghiệp".
- Giữ nguyên chính tả và dấu tiếng Việt như báo cáo thường in.
- Nếu câu hỏi cần nhiều chỉ tiêu (một tỷ số, một hiệu), liệt kê đủ từng chỉ tiêu.
- Đừng thêm tên công ty, năm, hay đơn vị vào `bien_the`.
- `phep_tinh` phải nằm trong danh sách trên, không được tự đặt tên khác."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

Chỉ tiêu nào, và nhãn dòng có thể in như thế nào?"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--out", default="artifacts/fresh/prompts_spec.jsonl")
    args = parser.parse_args()

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps({
                "id": question["id"], "system": SYSTEM,
                "user": USER.format(question=question["question"]),
            }, ensure_ascii=False) + "\n")
    print(f"{len(questions)} prompt -> {ROOT / args.out}")


if __name__ == "__main__":
    main()
