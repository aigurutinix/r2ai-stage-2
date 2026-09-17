"""CHAN DOAN vi sao tie-break bang reranker di ngang (eval_rr.py: nua A -1, nua B +4 → trai dau).

Do truc tiep DO CHINH XAC CUA TUNG TIN HIEU trong vung hoa, thay vi chi nhin diem cuoi:
  - cach hien tai (diem tu vung, hoa thi lay nhan ngan hon)
  - thu tu reranker (rerank_cache.json)
  - tran oracle: co it nhat MOT ung vien trong vung hoa mang gia tri dung hay khong
Neu hai tin hieu chinh xac ngang nhau → hoan doi chi xao lai tap cau dung, dung ky vong bang 0.
Day la ket luan da rut ra voi BM25 (58/85 vs 58/85); kiem xem reranker co khac khong.

Chay: python diag_rr.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = [d for d in json.load(open(os.path.join(HERE, "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None]
RR = json.load(open(os.path.join(HERE, "rerank_cache.json"), encoding="utf-8"))

_C = {}
def ing(p):
    if p not in _C:
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    return _C[p]

def table_query(qt):
    t = P.target_label(qt)
    return t if len(t) >= 6 else qt

def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)

GAP = 0.05
tie = cur_ok = rr_ok = oracle = no_cand = same_pick = 0
rr_missing = 0
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    tgt = P.target_label(d["question"]); qdir = P.qdir_of(d["question"]); gold = d["llm_val"]
    cands = P.score_cands(rows, tgt, qdir)
    if len(cands) < 2 or cands[0][0] - cands[1][0] >= GAP:
        continue                                   # khong hoa → khong can thiep
    tied = [(ov, r) for ov, r in cands if cands[0][0] - ov < GAP]
    if len(tied) < 2:
        continue
    tie += 1
    rr = RR.get(f"{fr[1]}|{table_query(d['question'])}")
    if not rr:
        rr_missing += 1

    def val_of(r):
        return P.pick_value({"maso": "", "label": r["label"], "tid": r["tid"], "cur": r["cur"],
                             "prev": r["prev"], "ov": 0.0,
                             "hdrs": r.get("hdrs") or [], "vals": r.get("vals") or []}, qdir)

    good = [r for _, r in tied if close(val_of(r), gold)]
    if not good:
        no_cand += 1
        continue
    oracle += 1
    pick_cur = tied[0][1]
    if rr:
        pos = {ln: i for i, ln in enumerate(rr)}; far = len(rr) + 1
        pick_rr = max(tied, key=lambda x: (-pos.get(x[1]["line"], far), x[0]))[1]
    else:
        pick_rr = pick_cur
    cur_ok += close(val_of(pick_cur), gold)
    rr_ok += close(val_of(pick_rr), gold)
    same_pick += (pick_cur is pick_rr)

print(f"GAP = {GAP}")
print(f"cau roi vao VUNG HOA               : {tie}")
print(f"  trong do khong ung vien nao dung : {no_cand}   (tran cua MOI tin hieu bang)")
print(f"  co it nhat 1 ung vien dung        : {oracle}")
print(f"    -> cach HIEN TAI chon dung      : {cur_ok}/{oracle} = {100*cur_ok/max(1,oracle):.0f}%")
print(f"    -> RERANKER chon dung           : {rr_ok}/{oracle} = {100*rr_ok/max(1,oracle):.0f}%")
print(f"    -> hai cach chon Y HET          : {same_pick}/{oracle}")
print(f"  thieu khoa rerank                 : {rr_missing}")
