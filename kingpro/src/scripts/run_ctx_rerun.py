"""GPU #3 (GPT): retrieval-conditioned rerun. Với 14 câu retrieval mơ hồ:
Context A = evidence base cũ; Context B = top-3 bảng retrieval (base table + alts).
Sinh 2 gen trên A + 3 gen trên B (Qwen-Coder-14B, temp 0.5). Gate CHẶT:
B 3/3 consensus + B_ans != base cũ + A KHÔNG đồng thuận với base cũ -> thay.
Chạy: PYTHONUTF8=1 python scripts/run_ctx_rerun.py
"""
import json
import os
import sys

sys.path.insert(0, "src")
# env cho llm_client
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))
os.environ["KINGPRO_LLM_BASE_URL"] = os.environ.get("BASE_CODER", os.environ.get("KINGPRO_LLM_BASE_URL", ""))
os.environ["KINGPRO_LLM_MODEL"] = os.environ.get("MODEL_CODER", os.environ.get("KINGPRO_LLM_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct"))
os.environ["KINGPRO_LLM_API_KEY"] = os.environ.get("KINGPRO_LLM_API_KEY", os.environ.get("RUNPOD_API_KEY", "EMPTY"))

from kingpro.answering.llm_client import chat
from kingpro.answering.program_engine import run_program
from kingpro.evaluation.metrics import coerce_number

GRADER_PY = ".venv-grader/Scripts/python.exe"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
cands = json.load(open("build/ctx_candidates.json", encoding="utf-8"))
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}


def cf(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def llm(system, user):
    return chat(system, user, temperature=0.5, max_tokens=900, timeout=150)


def eq(a, b):
    a, b = coerce_number(a), coerce_number(b)
    return a is not None and b is not None and abs(a - b) <= 0.01


out = []
for c in cands:
    qid = c["id"]
    q = base[qid]["question"] if "question" in base[qid] else None
    from_q = {x["id"]: x for x in json.load(open("data/questions/questions.jsonl", encoding="utf-8"))} if q is None else None
    if q is None:
        q = json.loads([l for l in open("data/questions/questions.jsonl", encoding="utf-8") if f'"id": {qid},' in l or f'"id":{qid},' in l][0])["question"]
    old = coerce_number(base[qid].get("answer"))
    # Context A = evidence base cũ (dùng chính csv trong sub_base_fix/data)
    A_tables = [{"table_ref": f"A{i}", "csv_path": "sub_base_fix/" + ev["csv_path"]}
                for i, ev in enumerate(base[qid].get("evidence", []))]
    # Context B = top-3 bảng retrieval
    B_tables = [{"table_ref": tr, "csv_path": cf(tr)} for tr in c["tables"][:3] if cf(tr)]
    try:
        rA = run_program(q, A_tables, llm, max_fix=2, n_vote=2, timeout=6.0, python_exe=GRADER_PY)
        rB = run_program(q, B_tables, llm, max_fix=2, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        out.append({"id": qid, "err": str(ex)[:120]})
        print(f"Q{qid} ERR {str(ex)[:80]}", flush=True)
        continue
    A_ans = rA.get("answer") if rA.get("ok") else None
    B_ans = rB.get("answer") if rB.get("ok") else None
    B_cons = rB.get("agree", 0) == 3 and rB.get("attempts", 0) == 3
    A_supports_old = (A_ans is not None and eq(A_ans, old))
    B_changed = (B_ans is not None and old is not None and not eq(B_ans, old))
    passed = bool(B_cons and B_changed and not A_supports_old)
    rec = {"id": qid, "old": old, "A_ans": A_ans, "B_ans": B_ans, "B_cons": B_cons,
           "A_supports_old": A_supports_old, "PASS": passed,
           "B_query": rB.get("pandas_query", ""), "B_tables": [t["table_ref"] for t in B_tables]}
    out.append(rec)
    print(f"Q{qid} old={old} A={A_ans} B={B_ans} Bcons={B_cons} Asupp={A_supports_old} -> {'PASS' if passed else 'skip'}", flush=True)

json.dump(out, open("build/ctx_rerun.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
npass = sum(1 for r in out if r.get("PASS"))
print(f"\n=== GATE PASS = {npass}/{len(cands)} (GPT: 0-3 kèo chết, 4-15 nộp) ===")
