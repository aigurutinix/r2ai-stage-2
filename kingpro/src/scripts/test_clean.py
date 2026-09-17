"""Đo grounding THẬT trên GOLD SẠCH (32 câu agent xác thực). So deterministic vs base.
Chạy: PYTHONUTF8=1 python scripts/test_clean.py
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
    return a is not None and abs(a - b) <= 0.01 + 0.005 * max(abs(b), 1.0)


retrieve_decomposed("warm")


def my_answer(q):
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    atabs = tables_in_reports(q, rel, n=6)
    tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])} for h in atabs if csv_full(h["table_ref"])]
    if not tables:
        return None
    idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
    if _classify_op(q):
        r = analytic_answer(q, tables, idf)
        if r:
            return r["answer"]
    r = deterministic_answer(q, tables, idf)
    return r["answer"] if r else None


n = det_cov = det_ok = base_ok = 0
fails = []
for g in gold:
    n += 1
    gv = g["gold"]
    a = my_answer(g["question"])
    if a is not None:
        det_cov += 1
        if eq(a, gv):
            det_ok += 1
        else:
            fails.append((g["id"], a, gv, g["question"][:55]))
    ba = coerce_number(base.get(g["id"], {}).get("answer")) if (base.get(g["id"], {}).get("pandas_query") or "").strip() else None
    if eq(ba, gv):
        base_ok += 1

print(f"GOLD SẠCH N={n}")
print(f"DETERMINISTIC: trả lời {det_cov}, ĐÚNG {det_ok} = {det_ok/n:.3f} (toàn bộ) | {det_ok/max(det_cov,1):.3f} (trên câu trả lời)")
print(f"BASE (LLM):    ĐÚNG {base_ok} = {base_ok/n:.3f}")
print("--- deterministic SAI (để soi) ---")
for qid, a, gv, q in fails[:10]:
    print(f"Q{qid} em={a} gold={gv} | {q}")
