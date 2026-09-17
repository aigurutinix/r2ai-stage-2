"""Canary UPSIDE (confidence routing của GPT): trên nền sub_det, ĐÈ base ở câu ĐÃ trả nhưng
deterministic có MÃ SỐ khớp CHÍNH XÁC (conf=3) và LỆCH base >1%. Test deterministic có hơn base
trên lookup chuẩn không. GPU-FREE. Điểm xấu thì KHÔNG đưa lên leaderboard (giữ 0.1858).
Chạy: PYTHONUTF8=1 python scripts/postprocess_override.py sub_det sub_ov
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.answering.operand_pipeline import deterministic_answer, build_idf
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_det"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_ov"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def csv_full(tref):
    r = CAT.get(tref)
    return "build/tables/" + r["csv_path"] if r else None


def safe(tref):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tref) + ".csv"


if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
existing = set(os.listdir(f"{OUT}/data"))
retrieve_decomposed("khoi dong")

n_ans = n_ov = 0
for e in rows:
    if not (e.get("pandas_query") or "").strip():
        continue                                        # rỗng đã lo ở sub_det
    # chỉ xét câu 1 công ty + 1 năm (lookup chuẩn)
    try:
        fac = extract_all_facets(e["question"])
    except Exception:
        continue
    if len(fac.get("tickers", [])) != 1 or len(fac.get("years", [])) != 1 or fac.get("analytic"):
        continue
    n_ans += 1
    q = e["question"]
    try:
        hits = retrieve_decomposed(q)
        rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel_docs, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])}
                  for h in atabs if csv_full(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
                         for t in tables])
        res = deterministic_answer(q, tables, idf)
    except Exception:
        res = None
    if not (res and res.get("conf") == 3 and res.get("answer") is not None):
        continue                                        # CHỈ đè khi mã số khớp chính xác
    old = coerce_number(e.get("answer"))
    new = float(res["answer"])
    if old is not None and abs(new - old) <= 0.01 * max(abs(old), 1.0):
        continue                                        # trùng base -> không cần đè
    ev = res["evidence"][0]
    nm = safe(ev["table_ref"])
    if nm not in existing:
        shutil.copyfile(csv_full(ev["table_ref"]), f"{OUT}/data/{nm}")
        existing.add(nm)
    e["answer"] = new
    e["pandas_query"] = res["pandas_query"]
    e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
    n_ov += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"answered_1tk1yr={n_ans} OVERRIDE_maso_exact={n_ov} -> {zp} {os.path.getsize(zp)} bytes")
