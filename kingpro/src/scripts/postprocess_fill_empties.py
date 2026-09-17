"""RISK-FREE: lấp câu RỖNG còn lại (base=0 điểm) bằng best-of {cross>=2, ana>=1, det>=1}.
Rỗng đã là 0 nên điền BẤT KỲ chỉ có thể >=, không thể giảm. Trên nền sub_cross.
Chạy: PYTHONUTF8=1 python scripts/postprocess_fill_empties.py sub_cross sub_cross2
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import analytic_cross, analytic_answer, deterministic_answer, build_idf
from kingpro.evaluation.metrics import doc_of
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_cross"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_cross2"
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

n_fill = 0
touched = []
for e in rows:
    if (e.get("pandas_query") or "").strip():
        continue                    # chỉ đụng câu RỖNG
    q = e["question"]
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])}
                  for h in atabs if csv_full(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
        cx = analytic_cross(q, tables, idf)
        an = analytic_answer(q, tables, idf)
        de = deterministic_answer(q, tables, idf)
    except Exception:
        cx = an = de = None
    pick = None
    if cx and cx.get("conf", 0) >= 2:
        # cross: evidence 2 bảng (df1,df2), có sẵn pandas_query/evidence
        pick = ("cross", cx["answer"], cx["pandas_query"],
                [(ev["variable"], ev["table_ref"]) for ev in cx["evidence"]])
    elif an and an.get("conf", 0) >= 1:
        pick = ("ana", an["answer"], an["pandas_query"], [("df1", an["evidence"][0]["table_ref"])])
    elif de and de.get("conf", 0) >= 1:
        pick = ("det", de["answer"], de["pandas_query"], [("df1", de["evidence"][0]["table_ref"])])
    if not pick:
        continue
    kind, ans, code, evs = pick
    new_ev = []
    for var, tref in evs:
        nm = safe(tref)
        if nm not in existing:
            shutil.copyfile(csv_full(tref), f"{OUT}/data/{nm}")
            existing.add(nm)
        new_ev.append({"variable": var, "csv_path": f"data/{nm}"})
    e["answer"] = float(ans)
    e["pandas_query"] = code
    e["evidence"] = new_ev
    n_fill += 1
    touched.append((e["id"], kind, ans))

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"FILL_EMPTIES={n_fill} -> {zp} {os.path.getsize(zp)}")
for qid, k, a in touched:
    print(f"  Q{qid} [{k}] ans={a}")
