"""BUCKET tomorrow-bandit: OVERRIDE lookup có chọn lọc. Trên nền sub_mai, với các câu trong
build/override_bucket_ids.json (det conf>=2, base!=det, lookup sạch), thay base bằng deterministic.
1 phép biến đổi duy nhất, bucket ĐỦ LỚN để đọc tín hiệu trên 506 câu chấm (GPT). GPU-free.
Chạy: PYTHONUTF8=1 python scripts/build_bucket_override.py
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
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

SRC = "sub_mai"
OUT = "sub_bkt_override"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
ids = set(json.load(open("build/override_bucket_ids.json")))


def cf(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def safe(tr):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tr) + ".csv"


if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
existing = set(os.listdir(f"{OUT}/data"))
retrieve_decomposed("warm")

n = 0
for e in rows:
    if e["id"] not in ids:
        continue
    q = QMAP[e["id"]]
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": cf(h["table_ref"])} for h in atabs if cf(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
        r = deterministic_answer(q, tables, idf)
    except Exception:
        r = None
    if not (r and r.get("conf", 0) >= 2):
        continue
    if abs((coerce_number(r["answer"]) or 0) - (coerce_number(e.get("answer")) or 1e30)) <= 0.01:
        continue          # base==det roi
    ev = r["evidence"][0]
    nm = safe(ev["table_ref"])
    if nm not in existing:
        shutil.copyfile(cf(ev["table_ref"]), f"{OUT}/data/{nm}")
        existing.add(nm)
    e["answer"] = float(r["answer"])
    e["pandas_query"] = r["pandas_query"]
    e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
    n += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"BUCKET_OVERRIDE: {n} cau override lookup -> {zp} {os.path.getsize(zp)}")
