"""GOP ket qua vong Auditor vao agent_full_v2.json — CO CHON LOC.

Chi nhan dap an thoa DU CA BA dieu kien:
  1. answer khac None (Auditor da chap nhan o vong chay)
  2. co `pandas` VA co `refs` — refs rong nghia la agent xuat mot HANG SO khong co dau vet bang
     chung; build_submission tu choi dung (dung), va bia refs de lach la dieu khong duoc lam
  3. kiem lai mien gia tri lan nua (phong khi vong chay dung ban domain() cu)

Moi id duoc gop deu thuoc tap DA CHUNG MINH SAI -> can duoi cua thay doi nay bang 0.

Chay: python merge_audit.py            (in bao cao, KHONG ghi)
      python merge_audit.py --apply    (ghi that, sao luu .bak)
"""
import os, sys, json, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P
from agent_audit import domain

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "agent_full_v2.json")
AUD = os.environ.get("AUDIT_IN", os.path.join(HERE, "agent_audit.json"))

full = json.load(open(FULL, encoding="utf-8"))
byid = {e["id"]: e for e in full}
QT = {q["id"]: q["question"] for q in P.Q}
D = json.load(open(AUD, encoding="utf-8"))

stat = {"nhan": 0, "bo_answer_None": 0, "bo_refs_rong": 0, "bo_sai_mien": 0}
for x in D:
    qid = x["id"]
    if x.get("answer") is None:
        stat["bo_answer_None"] += 1; continue
    if not x.get("pandas") or not x.get("refs"):
        stat["bo_refs_rong"] += 1; continue
    _, ok = domain(QT[qid])
    if ok is None or not ok(float(x["answer"])):
        stat["bo_sai_mien"] += 1; continue
    rec = {"id": qid, "question": QT[qid], "answer": x["answer"], "pandas": x["pandas"],
           "refs": x["refs"], "votes": x.get("votes", 0), "n": x.get("n", 0)}
    byid[qid] = rec
    stat["nhan"] += 1

print(f"ket qua vong Auditor: {len(D)} cau")
for k, v in stat.items():
    print(f"  {k:<16}: {v}")
print(f"\nagent_full_v2.json: {len(full)} -> {len(byid)} entry"
      f" | co dap an: {sum(1 for e in full if e.get('answer') is not None)}"
      f" -> {sum(1 for e in byid.values() if e.get('answer') is not None)}")

if "--apply" in sys.argv:
    shutil.copy(FULL, FULL + ".bak")
    json.dump(sorted(byid.values(), key=lambda e: e["id"]), open(FULL, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\nDA GHI {FULL}  (sao luu: agent_full_v2.json.bak)")
else:
    print("\n(chay lai voi --apply de ghi that)")
