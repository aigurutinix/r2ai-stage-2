"""Dựng record xác thực cho các câu bị analytic_cross ĐÈ (kiểm FIX/BREAK trước khi nộp).
Mỗi record: id, question, base_answer (sub_audit), cross_answer + 2 ô cross đã lấy, bảng ứng viên.
Chạy: PYTHONUTF8=1 python scripts/build_verify_cross.py
"""
import json
import os
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.evaluation.metrics import doc_of

IDS = [449, 588, 592, 593, 609, 613, 617, 621, 629, 636, 638, 643, 650, 655]
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
audit = {r["id"]: r for r in json.load(open("sub_audit/submission.json", encoding="utf-8"))}
cross = {r["id"]: r for r in json.load(open("sub_cross/submission.json", encoding="utf-8"))}


def csv_text(tref, maxc=2600):
    r = CAT.get(tref)
    if not r:
        return None
    try:
        t = open("build/tables/" + r["csv_path"], encoding="utf-8-sig").read()
    except Exception:
        return None
    return t[:maxc] + ("\n...(cắt)" if len(t) > maxc else "")


retrieve_decomposed("warm")
os.makedirs("build/verify_cross", exist_ok=True)
out = []
for qid in IDS:
    q = QMAP[qid]
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    atabs = tables_in_reports(q, rel, n=6)
    tables = [{"ref": h["table_ref"], "csv": csv_text(h["table_ref"])} for h in atabs if csv_text(h["table_ref"])]
    rec = {"id": qid, "question": q,
           "base_answer": audit.get(qid, {}).get("answer"),
           "cross_answer": cross.get(qid, {}).get("answer"),
           "cross_query": cross.get(qid, {}).get("pandas_query", ""),
           "tables": tables}
    out.append(rec)

for i, o in enumerate(out):
    json.dump(o, open(f"build/verify_cross/c{i}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"verify_cross: {len(out)} câu -> build/verify_cross/c0..c{len(out)-1}.json")
print("IDS:", [o["id"] for o in out])
