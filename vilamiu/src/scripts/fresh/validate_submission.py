"""Kiểm tra một bài nộp trước khi tải lên, theo đúng luật của Ban Tổ chức.

Bốn điều kiện được kiểm, và điều kiện thứ ba là điều kiện mới nhất — BTC nêu rõ rằng một
`pandas_query` gán hằng số, nhân hai hằng số, hay đọc một cột `answer` đã lưu sẵn thì **câu
đó không được tính điểm**. Ba ví dụ BTC đưa ra đều nằm trong bộ mẫu bên dưới.

Việc thi hành dùng **chính sandbox của BTC** (`vifinqa-official/src/vifinqa/answering/
sandbox.py`), không phải một bản mô phỏng. Điểm này từng làm sai một lần: sandbox nạp biến
`dfs` theo khoá `table_ref` (và `df` khi chỉ có một bảng), nên một bộ kiểm tra tự viết nạp
`df1..dfN` sẽ báo lỗi trên toàn bộ 1.012 dòng dù bài nộp hoàn toàn hợp lệ.

Usage:
  python scripts/fresh/validate_submission.py --zip submissions/legal.zip
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Ba dạng BTC nêu đích danh, cộng dạng "không đọc dữ liệu nào".
LITERAL = re.compile(
    r"^\s*result\s*=\s*(?:round\(\s*)?-?[\d.]+\s*(?:[*/+-]\s*[\d.eE+-]+\s*)*"
    r"(?:,\s*\d+\s*\))?\s*$", re.M)
ANSWER_COLUMN = re.compile(r'\[\s*["\']answer["\']\s*\]')
READS_DATA = re.compile(r"dfs\s*\[|\bdf\d*\s*[\.\[]")

_spec = importlib.util.spec_from_file_location(
    "sandbox", ROOT / "vifinqa-official/src/vifinqa/answering/sandbox.py")
sandbox = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sandbox)


def offence(query: str) -> str | None:
    if not READS_DATA.search(query):
        return "khong doc du lieu tu DataFrame"
    if LITERAL.search(query):
        return "gan hang so cho result"
    if ANSWER_COLUMN.search(query):
        return "doc cot 'answer' co san"
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--expect", type=int, default=1012)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.zip) as archive:
        names = set(archive.namelist())
        if "submission.json" not in names:
            raise SystemExit("THIEU submission.json o cap ngoai cung cua ZIP")
        rows = json.loads(archive.read("submission.json"))
        blobs = {n: archive.read(n) for n in names if n != "submission.json"}

    problems: dict[str, list[str]] = {
        "thieu truong": [], "csv_path sai": [], "csv thieu trong zip": [],
        "vi pham luat hang so": [], "chuong trinh loi": [], "chay nhung lech answer": [],
    }
    required = {"id", "question", "answer", "relevant_docs", "relevant_tables",
                "evidence", "pandas_query"}
    good = 0

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for name, blob in blobs.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)

        for record in rows:
            tag = f"id={record.get('id')}"
            if not required <= set(record):
                problems["thieu truong"].append(tag)
                continue
            query = record.get("pandas_query") or ""
            paths: dict[str, Path] = {}
            broken = False
            for item in record.get("evidence") or []:
                csv_path = str(item.get("csv_path") or "")
                if not csv_path.startswith("data/"):
                    problems["csv_path sai"].append(f"{tag} {csv_path}")
                    broken = True
                    continue
                if csv_path not in blobs:
                    problems["csv thieu trong zip"].append(f"{tag} {csv_path}")
                    broken = True
                    continue
                stem = Path(csv_path).stem
                doc, _, number = stem.rpartition("_table_")
                paths[f"{doc}|table_{number}"] = root / csv_path
            if record.get("answer") in (None, ""):
                continue
            why = offence(query)
            if why:
                problems["vi pham luat hang so"].append(f"{tag} {why}")
                continue
            if broken or not paths:
                continue
            try:
                produced = float(sandbox.run_pandas_code(query, paths))
            except Exception as error:  # noqa: BLE001
                problems["chuong trinh loi"].append(
                    f"{tag} {type(error).__name__}: {str(error)[:60]}")
                continue
            if abs(produced - float(record["answer"])) > 0.011:
                problems["chay nhung lech answer"].append(
                    f"{tag} chay ra {produced:.4f}, khai {record['answer']}")
                continue
            good += 1

    print(f"{args.zip}: {len(rows)} dong (yeu cau {args.expect})")
    if len(rows) != args.expect:
        print(f"  !! SO DONG SAI — thieu cau se bi tinh la du doan khong hop le")
    answered = sum(1 for r in rows if r.get("answer") not in (None, ""))
    print(f"  co dap an                 : {answered}")
    print(f"  chay lai va KHOP answer   : {good}")
    for key, items in problems.items():
        if items:
            print(f"  {key:26s}: {len(items)}")
            for line in items[:args.show]:
                print(f"      {line}")
    fatal = (len(rows) != args.expect
             or problems["thieu truong"] or problems["csv_path sai"]
             or problems["csv thieu trong zip"])
    if fatal:
        print("\nKET LUAN: CHUA NOP DUOC — con loi cau truc o tren.")
        raise SystemExit(1)
    if problems["vi pham luat hang so"]:
        print(f"\nCANH BAO: {len(problems['vi pham luat hang so'])} cau vi pham luat hang so "
              f"-> BTC se KHONG tinh diem cho cac cau nay.")
    print("\nKET LUAN: cau truc hop le, nop duoc.")


if __name__ == "__main__":
    main()
