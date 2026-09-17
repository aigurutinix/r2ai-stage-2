"""Give the program prompts a parser and a column rule, both from measured failures.

Four programs were run end to end before spending a pass on 976. None produced a number,
and the three execution errors named their causes exactly:

  '(45061179652)'  the model wrote its own Vietnamese number parser and left out the
  '(5)'            bracketed negative, so every cost line killed its program.
  'c3'             the model addressed a column by the label `c3` that the rendering
                   prints above it, as though it were a column name in the frame.

Neither is a reasoning failure, so neither is fixed by a better model. The parser is
supplied for the model to copy verbatim — it still chooses the rows, the tables and the
arithmetic, which is the part that needs a model — and the column rule is stated as a
prohibition rather than a description, because describing the mapping did not stop it.

Only the `system` field changes, so retrieval is untouched and no prompt is rebuilt.

Usage:
  python scripts/fresh/patch_prog_prompts.py artifacts/fresh/prompts_progall.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ADDITION = '''
<parse_helper>
Hàm `_num(x)` ĐÃ CÓ SẴN trong môi trường chạy. Nó nhận nội dung thô của một ô và trả về
float: dấu chấm là phân cách nghìn, dấu phẩy là thập phân, ngoặc là số âm, gạch ngang là 0.
- Dùng `_num(...)` cho MỌI ô bạn đọc.
- ĐỪNG định nghĩa lại `_num`, và đừng tự viết hàm parse khác.
- Đừng dùng `float(...)` trực tiếp trên nội dung ô.
</parse_helper>

<column_rule>
- `r12` và `c3` chỉ là NHÃN in ra để bạn đọc, KHÔNG phải tên dòng/cột trong DataFrame.
- Chỉ được truy cập ô bằng số nguyên: `_num(dfs["<table_ref>"].iloc[12, 3])` cho `r12`, `c3`.
- TUYỆT ĐỐI không viết `df["c3"]`, `df.loc[:, "c3"]`, `df["r12"]` — sẽ lỗi ngay.
- Chỉ dùng `table_ref` xuất hiện trong prompt, copy y nguyên.
</column_rule>
'''


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("can duong dan file prompt")
        raise SystemExit(2)

    path = ROOT / sys.argv[1]
    rows = [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for record in rows:
        if "<parse_helper>" not in record["system"]:
            record["system"] = record["system"].rstrip() + "\n" + ADDITION
            changed += 1
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    print(f"{len(rows)} prompt, da them huong dan cho {changed}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
