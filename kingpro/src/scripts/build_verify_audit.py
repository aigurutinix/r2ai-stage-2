"""Dựng record kiểm 18 câu unit-audit ứng viên (mở khoá từ 320): base vs det ai đúng?
Chạy: PYTHONUTF8=1 python scripts/build_verify_audit.py
"""
import json
import os
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.evaluation.metrics import doc_of

ids = json.load(open("build/audit_safe_ids.json"))
fixes = {f["id"]: f for f in json.load(open("sub_auditall_fixes.json", encoding="utf-8"))}
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}


def csv_text(tref, maxc=2400):
    r = CAT.get(tref)
    if not r:
        return None
    try:
        t = open("build/tables/" + r["csv_path"], encoding="utf-8-sig").read()
    except Exception:
        return None
    return t[:maxc] + ("\n...(cắt)" if len(t) > maxc else "")


retrieve_decomposed("warm")
os.makedirs("build/verify_audit", exist_ok=True)
out = []
for qid in ids:
    q = QMAP[qid]
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    atabs = tables_in_reports(q, rel, n=5)
    tables = [{"ref": h["table_ref"], "csv": csv_text(h["table_ref"])} for h in atabs if csv_text(h["table_ref"])]
    out.append({"id": qid, "question": q, "base_answer": fixes[qid]["base"],
                "det_answer": fixes[qid]["det"], "tables": tables})

for i, o in enumerate(out):
    json.dump(o, open(f"build/verify_audit/a{i}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"verify_audit: {len(out)} câu -> a0..a{len(out)-1}.json ; ids={[o['id'] for o in out]}")
