"""Lấy N câu MEDIUM (2-giá-trị-1-phép: tăng trưởng/chênh lệch, ĐƠN thực thể) + bảng ứng viên
+ code base hiện tại, để agent xác thực đáp án THẬT -> Medium clean gold (đo formula-auditor).
Chạy: PYTHONUTF8=1 python scripts/build_verify_medium.py 33
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.evaluation.metrics import doc_of

N = int(sys.argv[1]) if len(sys.argv) > 1 else 33
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}

# MEDIUM pattern: có phép tính 2 giá trị, ĐƠN thực thể, loại 'chênh lệch' nằm trong nhãn (lookup giả)
MED = re.compile(r"tăng trưởng|tốc độ tăng|tăng.{0,10}(phần trăm|%)|(phần trăm|%).{0,6}so với|"
                 r"chênh lệch.{0,25}(giữa|và|so với)|so với.{0,10}(năm|cùng kỳ).{0,20}(tăng|giảm|bao nhiêu)", re.I)


def csv_text(tref, maxc=2600):
    r = CAT.get(tref)
    if not r:
        return None
    try:
        t = open("build/tables/" + r["csv_path"], encoding="utf-8-sig").read()
    except Exception:
        return None
    return t[:maxc] + ("\n...(cắt)" if len(t) > maxc else "")


qs = [json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8")]
retrieve_decomposed("khoi dong")
import os
os.makedirs("build/verify_med", exist_ok=True)
out = []
for q in qs:
    if len(out) >= N:
        break
    if not MED.search(q["question"]):
        continue
    try:
        fac = extract_all_facets(q["question"])
    except Exception:
        continue
    if len(fac.get("tickers", [])) != 1:
        continue
    try:
        hits = retrieve_decomposed(q["question"])
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q["question"], rel, n=4)
        tables = [{"ref": h["table_ref"], "csv": csv_text(h["table_ref"])} for h in atabs if csv_text(h["table_ref"])]
        if not tables:
            continue
        rec = {"id": q["id"], "question": q["question"], "base_answer": base.get(q["id"], {}).get("answer"),
               "base_query": base.get(q["id"], {}).get("pandas_query", ""), "tables": tables}
        out.append(rec)
    except Exception:
        continue

for i, o in enumerate(out):
    json.dump(o, open(f"build/verify_med/m{i}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Medium verify: {len(out)} câu -> build/verify_med/m0..m{len(out)-1}.json")
