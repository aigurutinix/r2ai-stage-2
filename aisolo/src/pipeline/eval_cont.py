"""DO HAI HUONG CHUA NHAU (CONT_IN / CONT_OF) — chinh nua A, NGHIEM THU nua B.

Xuat phat tu ca that trong nhom hang 2-3 (diag_rank23.py):
  id12  target 'Von co phan da phat hanh'
        CHON  'Von co phan'                                  ov=0.818   <- khuc cat, RUNG dinh ngu
        DUNG  'Von co phan da phat hanhCo phieu pho thong'   ov=0.817   <- chua TRON target
Hai nhan cach nhau 0.001 vi bonus "chua nguyen cum" +0.15 ap NHU NHAU cho ca hai huong chua.
Nhung hai huong khong tuong duong ve ngu nghia: nhan CHUA TRON target la dung thu duoc hoi roi
mo ta them; nhan la KHUC CAT da danh roi dinh ngu cua cau hoi.

CONT_IN = CONT_OF = 0.15 => hanh vi y het truoc. Quet CONT_IN len va/hoac CONT_OF xuong.

Chay: python eval_cont.py
"""
import os, sys, json, importlib

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

CASES = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    CASES.append((d["id"], rows, P.target_label(d["question"]), P.qdir_of(d["question"]), d["llm_val"]))
print(f"dev set dung duoc: {len(CASES)} cau")

ids = sorted(c[0] for c in CASES)
A = {i for k, i in enumerate(ids) if k % 2 == 0}
B = {i for k, i in enumerate(ids) if k % 2 == 1}


def run(cin, cof):
    P.CONT_IN, P.CONT_OF = cin, cof          # score_cands doc bien module-level
    a = b = 0
    for qid, rows, tgt, qdir, gold in CASES:
        loc = P.locate(rows, tgt, qdir)
        ok = close(P.pick_value(loc, qdir), gold) if loc else False
        if qid in A:
            a += ok
        else:
            b += ok
    return a, b


GRID = [(0.15, 0.15), (0.20, 0.15), (0.25, 0.15), (0.30, 0.15),
        (0.15, 0.10), (0.15, 0.05), (0.15, 0.00), (0.25, 0.05), (0.30, 0.00)]
print(f"\nnua A (chinh) = {len(A)} cau | nua B (NGHIEM THU) = {len(B)} cau\n")
print(f"{'CONT_IN':>8} {'CONT_OF':>8} | {'A':>18} | {'B':>18}")
print("-" * 62)
res = {}; base = None
for cin, cof in GRID:
    a, b = run(cin, cof)
    res[(cin, cof)] = (a, b)
    if base is None:
        base = (a, b)
    print(f"{cin:>8} {cof:>8} | {a:>4} ({a-base[0]:+d}) {100*a/len(A):>5.1f}% | {b:>4} ({b-base[1]:+d}) {100*b/len(B):>5.1f}%")

best = max((k for k in res if k != (0.15, 0.15)), key=lambda k: res[k][0])
print(f"\n=> Chon tu nua A: CONT_IN={best[0]} CONT_OF={best[1]} (A {res[best][0]-base[0]:+d} cau)")
print(f"*** NGHIEM THU nua B: {res[best][1]-base[1]:+d} cau "
      f"({100*base[1]/len(B):.1f}% -> {100*res[best][1]/len(B):.1f}%) ***")
print("\nTIEU CHI DAT TRUOC: nua B cung dau nua A VA >= +3 cau (da thu 4 gia thuyet tren cung dev set,")
print("nguong +1 khong con phan biet duoc voi may rui do thu nhieu lan).")
