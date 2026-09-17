"""Pre-compute đầu vào cho workflow dựng GOLD SẠCH: lấy N câu lookup đơn (1 công ty, 1 năm),
retrieve bảng ứng viên, dump kèm CSV text để agent đọc + xác thực đúng ô.
Chạy: PYTHONUTF8=1 python scripts/build_verify_input.py 40
"""
import json
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.evaluation.metrics import doc_of

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


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
out = []
for q in qs:
    if len(out) >= N:
        break
    fac = extract_all_facets(q["question"])
    if fac.get("analytic") or len(fac.get("tickers", [])) != 1 or len(fac.get("years", [])) != 1:
        continue
    try:
        hits = retrieve_decomposed(q["question"])
        rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q["question"], rel_docs, n=4)
        tables = []
        for h in atabs:
            t = csv_text(h["table_ref"])
            if t:
                tables.append({"ref": h["table_ref"], "csv": t})
        if tables:
            out.append({"id": q["id"], "question": q["question"], "tables": tables})
    except Exception:
        continue

json.dump(out, open("build/verify_input.json", "w", encoding="utf-8"), ensure_ascii=False)
os.makedirs("build/verify", exist_ok=True) if (os := __import__("os")) else None
for i, o in enumerate(out):
    json.dump(o, open(f"build/verify/q{i}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"verify_input.json + build/verify/q0..q{len(out)-1}.json: {len(out)} câu, tổng bảng {sum(len(o['tables']) for o in out)}")
