"""Áp các override ĐÃ KIỂM lên sub_final: Q70 (deterministic FIX, verify workflow), Q710 (ratio,
base=2.9e13 vô lý nên risk-free). Rồi re-zip. Chạy: PYTHONUTF8=1 python scripts/apply_verified.py sub_final
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import deterministic_answer, ratio_answer, build_idf
from kingpro.evaluation.metrics import doc_of
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "sub_final"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}


def csv_full(t):
    r = CAT.get(t)
    return "build/tables/" + r["csv_path"] if r else None


def safe(t):
    return re.sub(r"[^0-9A-Za-z_]+", "_", t) + ".csv"


rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
byid = {e["id"]: e for e in rows}
existing = set(os.listdir(f"{OUT}/data"))
retrieve_decomposed("warm")

TARGETS = {70: "det", 710: "ratio"}
done = []
for qid, kind in TARGETS.items():
    q = QMAP[qid]
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    atabs = tables_in_reports(q, rel, n=6)
    tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])} for h in atabs if csv_full(h["table_ref"])]
    idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
    res = deterministic_answer(q, tables, idf) if kind == "det" else ratio_answer(q, tables, idf)
    if not (res and res.get("conf", 0) >= 2):
        done.append((qid, "SKIP no-res"))
        continue
    ev = res["evidence"][0]
    nm = safe(ev["table_ref"])
    if nm not in existing:
        shutil.copyfile(csv_full(ev["table_ref"]), f"{OUT}/data/{nm}")
        existing.add(nm)
    e = byid[qid]
    e["answer"] = float(res["answer"])
    e["pandas_query"] = res["pandas_query"]
    e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
    done.append((qid, f"applied {res['answer']}"))

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"applied -> {zp} {os.path.getsize(zp)}")
for qid, st in done:
    print(f"  Q{qid}: {st}")
