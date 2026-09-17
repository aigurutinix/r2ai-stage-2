"""GPU #3b context-sweep (GPT): rerun single-entity với 2 context khác nhau, override khi HỘI TỤ.
B1 = top-3 bảng retrieval. B2 = top-1 bảng + 2 bảng lân cận (cùng report, line gần). Mỗi context 3 gen.
Gate: B1 3/3 + B2 3/3 + B1==B2 (abs<=0.01) + != base. Override (risk-free với base-vô-lý).
Chạy: PYTHONUTF8=1 python scripts/run_context_sweep.py
"""
import json
import os
import re
import sys

sys.path.insert(0, "src")
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))
os.environ["KINGPRO_LLM_BASE_URL"] = os.environ.get("BASE_CODER", "")
os.environ["KINGPRO_LLM_MODEL"] = os.environ.get("MODEL_CODER", "Qwen/Qwen2.5-Coder-14B-Instruct")

from kingpro.answering.llm_client import chat
from kingpro.answering.program_engine import run_program
from kingpro.evaluation.metrics import coerce_number
from kingpro.retrieval.bm25_index import extract_all_facets

GRADER_PY = ".venv-grader/Scripts/python.exe"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
CATBYREPORT = {}
for r in CAT.values():
    CATBYREPORT.setdefault(r["report_id"], []).append(r)
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}

# pool = ctx candidates FAIL + single-entity absurd (chưa fix)
done_pass = {644, 680, 710}
pool = []
for c in json.load(open("build/ctx_candidates.json", encoding="utf-8")):
    if c["id"] not in done_pass:
        pool.append(c)
for c in json.load(open("build/absurd_candidates.json", encoding="utf-8")):
    q = QMAP[c["id"]]
    if re.search(r"trong nhóm|trong số|các (công ty|doanh nghiệp)|mã cổ phiếu|cao nhất|thấp nhất", q, re.I):
        continue
    try:
        if len(extract_all_facets(q).get("tickers", [])) != 1:
            continue
    except Exception:
        continue
    if c["id"] not in {x["id"] for x in pool} and c["id"] not in done_pass:
        pool.append(c)


def cf(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def neighbors(top_tref):
    """Bảng lân cận: cùng report, |line| gần nhất."""
    r = CAT.get(top_tref)
    if not r:
        return []
    same = [x for x in CATBYREPORT.get(r["report_id"], []) if x["table_ref"] != top_tref]
    same.sort(key=lambda x: abs(x.get("line", 0) - r.get("line", 0)))
    return [x["table_ref"] for x in same[:2]]


def llm(system, user):
    return chat(system, user, temperature=0.5, max_tokens=900, timeout=150)


def eq(a, b):
    a, b = coerce_number(a), coerce_number(b)
    return a is not None and b is not None and abs(a - b) <= 0.01


out = []
for c in pool:
    qid = c["id"]
    q = QMAP[qid]
    old = coerce_number(base[qid].get("answer"))
    tabs = c["tables"]
    B1 = [{"table_ref": tr, "csv_path": cf(tr)} for tr in tabs[:3] if cf(tr)]
    nb = neighbors(tabs[0])
    B2refs = [tabs[0]] + nb
    B2 = [{"table_ref": tr, "csv_path": cf(tr)} for tr in B2refs if cf(tr)]
    try:
        r1 = run_program(q, B1, llm, max_fix=2, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
        r2 = run_program(q, B2, llm, max_fix=2, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        out.append({"id": qid, "err": str(ex)[:80]})
        print(f"Q{qid} ERR", flush=True)
        continue
    a1 = r1.get("answer") if r1.get("ok") else None
    a2 = r2.get("answer") if r2.get("ok") else None
    c1 = r1.get("agree", 0) == 3 and r1.get("attempts", 0) == 3
    c2 = r2.get("agree", 0) == 3 and r2.get("attempts", 0) == 3
    converge = a1 is not None and a2 is not None and eq(a1, a2)
    changed = a1 is not None and old is not None and not eq(a1, old)
    plausible = a1 is not None and abs(a1) < 1e6
    passed = bool(c1 and c2 and converge and changed and plausible)
    rec = {"id": qid, "old": old, "B1": a1, "B2": a2, "c1": c1, "c2": c2, "PASS": passed,
           "query": r1.get("pandas_query", ""), "B1_tables": [t["table_ref"] for t in B1]}
    out.append(rec)
    print(f"Q{qid} old={old} B1={a1}({c1}) B2={a2}({c2}) conv={converge} -> {'PASS' if passed else 'skip'}", flush=True)

json.dump(out, open("build/context_sweep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
npass = sum(1 for r in out if r.get("PASS"))
print(f"\n=== SWEEP PASS = {npass}/{len(pool)} ===")
