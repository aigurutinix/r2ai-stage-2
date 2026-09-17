"""Đo analytic_answer trên 23 Medium clean gold (build/clean_gold_med.jsonl).
So: (1) sub_audit hiện tại đúng mấy câu, (2) analytic_answer đúng mấy câu, ở conf nào.
abs_tol 0.01 tuyệt đối (như grader). Chạy: PYTHONUTF8=1 python scripts/test_med.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import analytic_answer, analytic_cross, _classify_op, build_idf
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_audit/submission.json", encoding="utf-8"))}
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
gold = [json.loads(l) for l in open("build/clean_gold_med.jsonl", encoding="utf-8")]


def csv_full(tref):
    r = CAT.get(tref)
    return "build/tables/" + r["csv_path"] if r else None


def match(a, b, tol=0.01):
    a, b = coerce_number(a), coerce_number(b)
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


retrieve_decomposed("warm")
base_ok = 0
rows = []
for g in gold:
    qid, truth = g["id"], g["true_answer"]
    q = QMAP.get(qid, "")
    op = _classify_op(q)
    b_ans = base.get(qid, {}).get("answer")
    b_ok = match(b_ans, truth)
    base_ok += b_ok
    try:
        hits = retrieve_decomposed(q)
        rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(q, rel, n=6)
        tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])}
                  for h in atabs if csv_full(h["table_ref"])]
        idf = build_idf([pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig") for t in tables])
        cx = analytic_cross(q, tables, idf)
    except Exception:
        cx = None
    cx_ans = cx["answer"] if cx else None
    cx_conf = cx["conf"] if cx else 0
    cx_ok = bool(cx) and match(cx_ans, truth)
    tag = ("BREAK" if (b_ok and cx and not cx_ok) else
           "FIX" if (not b_ok and cx_ok) else
           "both_ok" if (b_ok and cx_ok) else
           "both_wrong" if cx else "NO_FIRE")
    rows.append((qid, op, tag, cx_conf, cx["cross"] if cx else None, truth, b_ans, cx_ans))

print(f"N={len(gold)} MEDIUM gold | sub_audit ĐÚNG = {base_ok}/{len(gold)}")
cx_fire = sum(1 for r in rows if r[2] != 'NO_FIRE')
print(f"analytic_CROSS: FIRE={cx_fire} đúng={sum(1 for r in rows if r[2] in('FIX','both_ok'))}")
fix = [r for r in rows if r[2] == 'FIX']
brk = [r for r in rows if r[2] == 'BREAK']
print(f">>> conf>=2: FIX={len([r for r in fix if r[3]>=2])} BREAK={len([r for r in brk if r[3]>=2])}")
print(f">>> ALL conf: FIX={len(fix)} BREAK={len(brk)}")
print("--- chi tiết ---")
for qid, op, tag, c, cross, tr, ba, ca in rows:
    print(f"Q{qid} {op or '-':6} {tag:11} c={c} cross={cross} truth={tr} base={ba} cross_ans={ca}")
