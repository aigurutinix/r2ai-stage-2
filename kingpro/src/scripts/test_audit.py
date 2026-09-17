"""Test AUDITOR đơn vị trên GOLD SẠCH (đo TRƯỚC khi nộp). Luật: base sai deterministic đúng bội 10^k
(= cùng ô, lệch đơn vị) + det tự tin -> sửa base theo det (đơn vị tất định). Đếm FIX vs BREAK.
Chạy: PYTHONUTF8=1 python scripts/test_audit.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import deterministic_answer, analytic_answer, _classify_op, build_idf
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
gold = [json.loads(l) for l in open("build/clean_gold.jsonl", encoding="utf-8")]
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}


def csv_full(t):
    r = CAT.get(t)
    return "build/tables/" + r["csv_path"] if r else None


def eq(a, b):
    return a is not None and b is not None and abs(a - b) <= 0.01 + 0.005 * max(abs(b), 1.0)


def is_pow10(base_v, det_v):
    """base = det * 10^k, k != 0 (cùng ô, lệch đơn vị)."""
    if base_v is None or det_v is None or det_v == 0 or base_v == 0:
        return False
    r = abs(base_v / det_v)
    for k in range(-13, 14):
        if k != 0 and abs(r - 10.0 ** k) <= 0.02 * (10.0 ** k):
            return True
    return False


retrieve_decomposed("warm")


def det_of(q):
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    atabs = tables_in_reports(q, rel, n=6)
    tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])} for h in atabs if csv_full(h["table_ref"])]
    if not tables:
        return None, 0
    idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
    r = (analytic_answer(q, tables, idf) if _classify_op(q) else None) or deterministic_answer(q, tables, idf)
    return (r["answer"], r.get("conf", 0)) if r else (None, 0)


n = base_ok = aud_ok = fix = brk = 0
for g in gold:
    n += 1
    gv = g["gold"]
    ba = coerce_number(base.get(g["id"], {}).get("answer")) if (base.get(g["id"], {}).get("pandas_query") or "").strip() else None
    dv, conf = det_of(g["question"])
    # AUDIT: chỉ sửa khi base lệch det đúng bội 10^k và det tự tin >=2
    audited = ba
    if ba is not None and dv is not None and conf >= 2 and is_pow10(ba, dv):
        audited = dv
    if eq(ba, gv):
        base_ok += 1
    if eq(audited, gv):
        aud_ok += 1
    if not eq(ba, gv) and eq(audited, gv):
        fix += 1
    if eq(ba, gv) and not eq(audited, gv):
        brk += 1

print(f"GOLD SẠCH N={n}")
print(f"BASE        : {base_ok}/{n} = {base_ok/n:.3f}")
print(f"BASE+AUDIT  : {aud_ok}/{n} = {aud_ok/n:.3f}   (FIX {fix} câu sai->đúng, BREAK {brk} câu đúng->sai)")
