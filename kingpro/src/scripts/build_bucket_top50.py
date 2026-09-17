"""SUB1 mai (GPT chốt): best + ~50 câu override MẠNH NHẤT (không nộp 172).
Gate: det conf>=2, KHÔNG note/ownership%/aggregate/superlative, chỉ 1-value lookup, ưu tiên
conf=3 (mã-số exact) rồi conf=2 margin lớn. Xếp hạng, lấy top 50. Nền sub_mai.
Chạy: PYTHONUTF8=1 python scripts/build_bucket_top50.py
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import deterministic_answer, score_rows, resolve_columns, label_column, first_data_row, build_idf
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

SRC = "sub_mai"
OUT = "sub_bkt_top50"
N_TOP = 50
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
ids = json.load(open("build/override_bucket_ids.json"))
BAD = re.compile(r"trong nhóm|trong số|cao nhất|thấp nhất|trung bình|trung vị|tỷ lệ|tỷ trọng|chiếm|"
                 r"sở hữu|lợi ích|biểu quyết|các công ty|các doanh nghiệp|đầu năm|đầu kỳ", re.I)


def cf(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def safe(tr):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tr) + ".csv"


retrieve_decomposed("warm")
scored = []
for qid in ids:
    q = QMAP[qid]
    if BAD.search(q):
        continue
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": cf(h["table_ref"])} for h in atabs if cf(h["table_ref"])]
        dfs = [pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables]
        idf = build_idf(dfs)
        r = deterministic_answer(q, tables, idf)
    except Exception:
        r = None
    if not (r and r.get("conf", 0) >= 2):
        continue
    # margin: điểm score_rows top1/top2 trên bảng đã chọn (đo độ tách bạch)
    margin = 0.0
    try:
        ev = r["evidence"][0]
        df = pd.read_csv(cf(ev["table_ref"]), dtype=str, keep_default_na=False, encoding="utf-8-sig")
        cand = score_rows(df, q, label_column(df, resolve_columns(df)), idf, first_data_row(df), topk=2)
        if cand:
            t1 = cand[0][0]
            t2 = cand[1][0] if len(cand) > 1 else 0.01
            margin = t1 / max(t2, 0.01)
    except Exception:
        pass
    # ưu tiên: conf=3 (mã-số) trước, rồi margin lớn
    rank = (r["conf"], margin)
    scored.append((rank, qid, r))

scored.sort(key=lambda x: (-x[0][0], -x[0][1]))
top = scored[:N_TOP]
print(f"cand sạch = {len(scored)}, lấy top {len(top)} (conf3={sum(1 for x in top if x[0][0]>=3)})")

if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
byid = {e["id"]: e for e in rows}
existing = set(os.listdir(f"{OUT}/data"))
n = 0
for rank, qid, r in top:
    ev = r["evidence"][0]
    nm = safe(ev["table_ref"])
    if nm not in existing:
        shutil.copyfile(cf(ev["table_ref"]), f"{OUT}/data/{nm}")
        existing.add(nm)
    e = byid[qid]
    e["answer"] = float(r["answer"])
    e["pandas_query"] = r["pandas_query"]
    e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
    n += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump([qid for _, qid, _ in top], open("build/top50_ids.json", "w"))
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"BUCKET_TOP50: override {n} cau manh nhat -> {zp} {os.path.getsize(zp)}")
