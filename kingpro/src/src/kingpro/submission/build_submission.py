"""Sinh submission.json + đóng gói submission.zip đúng đặc tả R2AI Stage 2.

Đặc tả (Submission Instructions): ZIP gồm submission.json (ngoài cùng) + data/ chứa CSV.
Mỗi câu: id, question, answer(float), relevant_docs, relevant_tables(=id|dòng),
evidence[{variable, csv_path}], pandas_query.

Chế độ:
  --mode retrieval : chỉ điền relevant_docs/relevant_tables (dò ĐỊNH DẠNG, answer=0.0). Việc 1.
  (answering đầy đủ sẽ thêm sau khi có model mở.)

Dò định dạng dòng (A/B): --line-offset 0 (mặc định, dòng tại thẻ <table>) hoặc thử -1/khác.

Chạy:
  python src/kingpro/submission/build_submission.py --mode retrieval -k 5 --out sub_probe
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kingpro.retrieval.bm25_index import retrieve  # noqa: E402
from kingpro.evaluation.metrics import doc_of  # noqa: E402

QUESTIONS = Path("data/questions/questions.jsonl")


def shift_line(table_ref: str, offset: int) -> str:
    if offset == 0:
        return table_ref
    rid, _, line = table_ref.rpartition("|")
    if line.isdigit():
        return f"{rid}|{int(line) + offset}"
    return table_ref


def build(mode: str, k: int, out_dir: Path, line_offset: int = 0, limit: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)  # zip cần thư mục data/ (rỗng ở chế độ retrieval)

    questions = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8")]
    if limit:
        questions = questions[:limit]

    rows = []
    for q in questions:
        hits = retrieve(q["question"], k=k)
        refs = [shift_line(h["table_ref"], line_offset) for h in hits]
        docs = list(dict.fromkeys(doc_of(r) for r in refs))
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": 0.0,                 # placeholder ở chế độ retrieval
                "relevant_docs": docs,
                "relevant_tables": refs,
                "evidence": [],
                "pandas_query": "",
            }
        )

    sub_json = out_dir / "submission.json"
    with open(sub_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_json, "submission.json")
        z.writestr("data/.keep", "")
    print(f"submission.json: {len(rows)} câu | zip -> {zip_path}")
    # kiểm nhanh cấu trúc
    n_empty = sum(1 for r in rows if not r["relevant_tables"])
    print(f"  câu không có bảng: {n_empty} | ví dụ refs[0]: {rows[0]['relevant_tables'][:2]}")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="retrieval", choices=["retrieval"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("sub_probe"))
    ap.add_argument("--line-offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.mode, a.k, a.out, a.line_offset, a.limit)
