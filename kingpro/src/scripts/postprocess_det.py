"""Canary AN TOÀN + GPU-FREE: lấp các câu base RỖNG bằng đáp án TẤT ĐỊNH (operand_pipeline).
Rỗng=0 điểm nên chỉ có thể GIỮ (0) hoặc TĂNG (+1). Nộp -> leaderboard làm trọng tài (không có gold sạch).
Chạy: PYTHONUTF8=1 python scripts/postprocess_det.py sub_fill sub_det
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import deterministic_answer, build_idf
from kingpro.evaluation.metrics import doc_of
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_fill"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_det"
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

n_empty = n_fill = 0
for e in rows:
    if (e.get("pandas_query") or "").strip():
        continue
    n_empty += 1
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
    if res and res.get("answer") is not None:
        ev = res["evidence"][0]
        tref = ev["table_ref"]
        nm = safe(tref)
        if nm not in existing:
            shutil.copyfile(csv_full(tref), f"{OUT}/data/{nm}")
            existing.add(nm)
        e["answer"] = float(res["answer"])
        e["pandas_query"] = res["pandas_query"]
        e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
        n_fill += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"empty={n_empty} filled_deterministic={n_fill} -> {zp} {os.path.getsize(zp)} bytes")
