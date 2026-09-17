"""CHAN DOAN KHAU DUNG UNG VIEN: voi moi cau dang SAI, gia tri dung nam o dau?

Vi sao: diag_rr.py cho thay trong vung hoa, 34/119 ca (29%) KHONG ung vien nao mang gia tri dung
=> nghen khong nam o khau PHAN XU (chon bang / tie-break) ma o khau DUNG UNG VIEN. Cau hoi quyet
dinh: dong dung co nam trong `cands` khong? Ba kha nang dan toi ba ket luan trai nguoc han:

  (A) CHAM DIEM  — dong dung CO trong cands nhung bi xep thap  => con du dia, sua cong thuc cham.
  (B) CONG CUNG  — dong dung bi `inter==0: continue` loai thang => can loi thoat cho zero-overlap.
  (C) CHON COT   — dung dong roi nhung gia tri dung o COT khac  => viec cua pick_col, khong phai locate.
  (D) HET DUONG  — khong dong nao trong ca bao cao mang gia tri dung => loi parse/report, locate bo tay.

Neu (D) chiem da so thi khau chon dong da het du dia va nen dung.

Chay: python diag_pool.py
"""
import os, sys, json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = [d for d in json.load(open(os.path.join(HERE, "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None]

_C = {}
def ing(p):
    if p not in _C:
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    return _C[p]

def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)

def as_loc(r):
    return {"maso": "", "label": r["label"], "tid": r["tid"], "cur": r["cur"], "prev": r["prev"],
            "ov": 0.0, "hdrs": r.get("hdrs") or [], "vals": r.get("vals") or []}

GAP = 0.05
kind = Counter(); ranks = []; examples = {"A": [], "B": [], "C": [], "D": []}
n_wrong = n_ok = 0
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    qt = d["question"]; tgt = P.target_label(qt); qdir = P.qdir_of(qt); gold = d["llm_val"]
    loc = P.locate(rows, tgt, qdir)
    if loc and close(P.pick_value(loc, qdir), gold):
        n_ok += 1; continue
    n_wrong += 1

    # dong "dung" theo COT MAC DINH (dung cach pick_value doc) va theo BAT KY cot nao
    hit_def = [r for r in rows if r["cur"] is not None and r["label"]
               and close(P.pick_value(as_loc(r), qdir), gold)]
    hit_any = [r for r in rows if r["cur"] is not None and r["label"]
               and any(close(v, gold) for v in [r["cur"], r["prev"], *(r.get("vals") or [])])]

    if not hit_any:
        kind["D"] += 1
        if len(examples["D"]) < 3: examples["D"].append((d["id"], qt[:80]))
        continue
    if not hit_def:                      # dung dong ton tai nhung chi khop o COT khac
        kind["C"] += 1
        if len(examples["C"]) < 3: examples["C"].append((d["id"], qt[:80]))
        continue

    cands = P.score_cands(rows, tgt, qdir)
    order = {id(r): i for i, (_, r) in enumerate(cands)}
    rk = [order[id(r)] for r in hit_def if id(r) in order]
    if not rk:
        kind["B"] += 1
        if len(examples["B"]) < 3: examples["B"].append((d["id"], qt[:80]))
    else:
        kind["A"] += 1; ranks.append(min(rk) + 1)
        if len(examples["A"]) < 3: examples["A"].append((d["id"], qt[:80]))

tot = sum(kind.values())
print(f"dev set: {n_ok} dung / {n_wrong} sai\n")
lab = {"A": "CHAM DIEM  - dong dung CO trong cands, bi xep thap",
       "B": "CONG CUNG  - bi `inter==0` loai thang khoi cands",
       "C": "CHON COT   - dung dong, gia tri o COT khac",
       "D": "HET DUONG  - khong dong nao mang gia tri dung"}
for k in "ABCD":
    print(f"  ({k}) {lab[k]:<48} {kind[k]:>3}  {100*kind[k]/max(1,tot):>5.1f}%")

if ranks:
    ranks.sort()
    b = Counter("1" if r == 1 else "2-3" if r <= 3 else "4-10" if r <= 10 else "11-30" if r <= 30 else ">30"
                for r in ranks)
    print(f"\nNhom (A) — thu hang cua dong dung trong cands (n={len(ranks)}, trung vi {ranks[len(ranks)//2]}):")
    for k in ("1", "2-3", "4-10", "11-30", ">30"):
        if b[k]: print(f"    hang {k:>6}: {b[k]:>3}")
print()
for k in "ABCD":
    for i, q in examples[k]:
        print(f"  {k} id{i}: {q}")
