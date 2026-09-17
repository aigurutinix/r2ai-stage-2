"""MỞ RỘNG đòn thắng: rerun Context B cho câu base VÔ LÝ (risk-free vì base=0 điểm).
Chỉ Context B = top-3 bảng retrieval, 3 gen (temp 0.5). Gate: B 3/3 consensus + |B|<1e6 (hợp lý %)
+ B != base. Override risk-free. Chạy: PYTHONUTF8=1 python scripts/run_absurd_rerun.py
"""
import json
import os
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

GRADER_PY = ".venv-grader/Scripts/python.exe"
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
cands = json.load(open("build/absurd_candidates.json", encoding="utf-8"))
QMAP = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}


def cf(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def llm(system, user):
    return chat(system, user, temperature=0.5, max_tokens=900, timeout=150)


out = []
for c in cands:
    qid = c["id"]
    q = QMAP[qid]
    old = coerce_number(c["base_ans"])
    B_tables = [{"table_ref": tr, "csv_path": cf(tr)} for tr in c["tables"][:3] if cf(tr)]
    try:
        rB = run_program(q, B_tables, llm, max_fix=2, n_vote=3, timeout=6.0, python_exe=GRADER_PY)
    except Exception as ex:
        out.append({"id": qid, "err": str(ex)[:100]})
        print(f"Q{qid} ERR {str(ex)[:60]}", flush=True)
        continue
    B = rB.get("answer") if rB.get("ok") else None
    B_cons = rB.get("agree", 0) == 3 and rB.get("attempts", 0) == 3
    plausible = B is not None and abs(B) < 1e6
    changed = B is not None and old is not None and abs(B - old) > 0.01
    passed = bool(B_cons and plausible and changed)
    out.append({"id": qid, "old": old, "B": B, "B_cons": B_cons, "PASS": passed,
                "B_query": rB.get("pandas_query", ""), "B_tables": [t["table_ref"] for t in B_tables]})
    print(f"Q{qid} old={old:.3g} B={B} cons={B_cons} plaus={plausible} -> {'PASS' if passed else 'skip'}", flush=True)

json.dump(out, open("build/absurd_rerun.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
npass = sum(1 for r in out if r.get("PASS"))
print(f"\n=== ABSURD GATE PASS = {npass}/{len(cands)} (risk-free override) ===")
