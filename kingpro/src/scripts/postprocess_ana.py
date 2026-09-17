"""Canary ANALYTICAL: trên nền sub_det, điền/đè đáp án cho câu ANALYTICAL đơn-thực-thể
cùng-chỉ-tiêu/2-năm (chênh lệch, tăng trưởng) bằng analytic_answer. Base ~0 điểm ở nhóm này -> NET GAIN.
GPU-FREE. Chạy: PYTHONUTF8=1 python scripts/postprocess_ana.py sub_det sub_ana
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.answering.operand_pipeline import analytic_answer, _classify_op, build_idf
from kingpro.evaluation.metrics import doc_of
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_det"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_ana"
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

n_cand = n_set = n_fill = n_over = 0
for e in rows:
    q = e["question"]
    if _classify_op(q) is None:
        continue
    try:
        fac = extract_all_facets(q)
    except Exception:
        continue
    if len(fac.get("tickers", [])) != 1:        # chỉ đơn-thực-thể (cùng chỉ tiêu)
        continue
    n_cand += 1
    was_empty = not (e.get("pandas_query") or "").strip()
    try:
        hits = retrieve_decomposed(q)
        rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel_docs, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])}
                  for h in atabs if csv_full(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
                         for t in tables])
        res = analytic_answer(q, tables, idf)
    except Exception:
        res = None
    if not (res and res.get("answer") is not None and res.get("conf", 0) >= 2):
        continue
    ev = res["evidence"][0]
    nm = safe(ev["table_ref"])
    if nm not in existing:
        shutil.copyfile(csv_full(ev["table_ref"]), f"{OUT}/data/{nm}")
        existing.add(nm)
    e["answer"] = float(res["answer"])
    e["pandas_query"] = res["pandas_query"]
    e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
    n_set += 1
    n_fill += was_empty
    n_over += (not was_empty)

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"analytical_candidates={n_cand} SET={n_set} (fill_empty={n_fill}, override={n_over}) -> {zp} {os.path.getsize(zp)}")
