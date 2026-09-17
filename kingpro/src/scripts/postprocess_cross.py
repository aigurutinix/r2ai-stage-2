"""CHÉO-BẢNG: trên nền sub_audit, câu tăng trưởng/chênh lệch đơn-thực-thể cách >=2 năm ->
analytic_cross (2 báo cáo năm riêng, cùng chỉ tiêu). Đo trên 23 gold: FIX 6 BREAK 0 @conf>=2.
Chỉ đè khi conf>=2. GPU-FREE. Chạy: PYTHONUTF8=1 python scripts/postprocess_cross.py sub_audit sub_cross
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.answering.operand_pipeline import analytic_cross, _classify_op, build_idf
from kingpro.evaluation.metrics import doc_of
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_audit"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_cross"
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

n_cand = n_set = 0
touched = []
for e in rows:
    q = e["question"]
    if _classify_op(q) is None:
        continue
    try:
        fac = extract_all_facets(q)
    except Exception:
        continue
    if len(fac.get("tickers", [])) != 1:
        continue
    n_cand += 1
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])}
                  for h in atabs if csv_full(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
        res = analytic_cross(q, tables, idf)
    except Exception:
        res = None
    if not (res and res.get("answer") is not None and res.get("conf", 0) >= 2):
        continue
    # copy CSVs + remap evidence -> data/<safe>.csv theo THỨ TỰ (df1, df2)
    new_ev = []
    for ev in res["evidence"]:
        tref = ev["table_ref"]
        nm = safe(tref)
        if nm not in existing:
            shutil.copyfile(csv_full(tref), f"{OUT}/data/{nm}")
            existing.add(nm)
        new_ev.append({"variable": ev["variable"], "csv_path": f"data/{nm}"})
    e["answer"] = float(res["answer"])
    e["pandas_query"] = res["pandas_query"]
    e["evidence"] = new_ev
    n_set += 1
    touched.append((e["id"], res["answer"], res["conf"], res["cross"]))

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"cross_candidates={n_cand} OVERRIDE={n_set} -> {zp} {os.path.getsize(zp)}")
for qid, a, c, cr in touched:
    print(f"  Q{qid} ans={a} conf={c} cross={cr}")
