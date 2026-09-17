"""AUDITOR đơn vị MỞ KHOÁ TOÀN BỘ (bỏ lọc ticker==1): 320 câu giả-đa-thực-thể + lỗi đơn vị
RATIO/AGG chưa từng được audit. Nền sub_cross. Chỉ sửa khi base lệch deterministic ĐÚNG bội 10^k
(=cùng ô, sai đơn vị) + det conf>=2. Log từng fix (id/base/det/q) để verify trước khi nộp.
Chạy: PYTHONUTF8=1 python scripts/postprocess_audit_all.py sub_cross sub_auditall
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports, extract_all_facets
from kingpro.answering.operand_pipeline import deterministic_answer, analytic_answer, analytic_cross, _classify_op, build_idf
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_cross"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_auditall"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}


def csv_full(t):
    r = CAT.get(t)
    return "build/tables/" + r["csv_path"] if r else None


def safe(t):
    return re.sub(r"[^0-9A-Za-z_]+", "_", t) + ".csv"


def is_pow10(bv, dv):
    if bv is None or dv is None or dv == 0 or bv == 0:
        return False
    r = abs(bv / dv)
    return any(k != 0 and abs(r - 10.0 ** k) <= 0.02 * (10.0 ** k) for k in range(-13, 14))


if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
existing = set(os.listdir(f"{OUT}/data"))
retrieve_decomposed("warm")

# câu đã bị auditor/cross đụng ở các pass trước (so với sub_det gốc) -> để không đếm trùng
n_ans = n_fix = 0
fixes = []
for e in rows:
    if not (e.get("pandas_query") or "").strip():
        continue
    try:
        if len(extract_all_facets(e["question"]).get("tickers", [])) == 1:
            continue                     # câu 1-ticker đã được audit ở sub_audit -> chỉ chạy phần MỚI mở khoá
    except Exception:
        continue
    n_ans += 1
    q = e["question"]
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])} for h in atabs if csv_full(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
        res = (analytic_answer(q, tables, idf) if _classify_op(q) else None) or deterministic_answer(q, tables, idf)
    except Exception:
        res = None
    if not (res and res.get("conf", 0) >= 2 and res.get("answer") is not None):
        continue
    ba = coerce_number(e.get("answer"))
    if not is_pow10(ba, res["answer"]):
        continue
    ev = res["evidence"][0]
    nm = safe(ev["table_ref"])
    if nm not in existing:
        shutil.copyfile(csv_full(ev["table_ref"]), f"{OUT}/data/{nm}")
        existing.add(nm)
    fixes.append({"id": e["id"], "base": ba, "det": res["answer"], "conf": res["conf"], "q": q[:90]})
    e["answer"] = float(res["answer"])
    e["pandas_query"] = res["pandas_query"]
    e["evidence"] = [{"variable": "df1", "csv_path": f"data/{nm}"}]
    n_fix += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(fixes, open(f"{OUT}_fixes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"answered_ALL={n_ans} AUDIT_FIX_donvi={n_fix} -> {zp} {os.path.getsize(zp)}")
for f in fixes:
    print(f"  Q{f['id']} base={f['base']} -> det={f['det']} (c{f['conf']}) | {f['q']}")
