"""ĐO TIE-BREAK CẤP BẢNG (GAP_THR) trên dev set — chỉnh nửa A, NGHIỆM THU nửa B.

Vì sao hướng này: phân rã 99 ca sai cho thấy 71% có nhãn đúng nằm ở BẢNG KHÁC, trong đó 20% là
CÙNG MỘT CHUỖI NHÃN ở nhiều bảng — điểm từ vựng bằng nhau tuyệt đối nên chỉnh trọng số vô ích.
Chỉ tín hiệu cấp bảng mới phân biệt được.

Vì sao chỉ can thiệp khi HOÀ: đo trên chính dev set, cách biệt top1-top2 <0.05 → đúng 48.1%
(104 câu), >=0.3 → đúng 87.9% (66 câu). Can thiệp vùng chắc chắn là tự bắn vào chân.

Chia đôi BẮT BUỘC: router `ov` từng cho +8 trên toàn bộ rồi chỉ còn +1 khi nghiệm thu.

Chạy: python eval_gap.py
"""
import os, re, sys, json

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
        txt = open(p, encoding="utf-8", errors="replace").read()
        _C[p] = (P.ingest(txt), P.table_tokens(txt))
    return _C[p]


def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)


CASES = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    (rows, uf, src), tt = ing(str(fr[0]))
    CASES.append((d["id"], rows, tt, P.target_label(d["question"]), P.qdir_of(d["question"]), d["llm_val"]))
print(f"dev set dung duoc: {len(CASES)} cau")

ids = sorted(c[0] for c in CASES)
A = {i for k, i in enumerate(ids) if k % 2 == 0}
B = {i for k, i in enumerate(ids) if k % 2 == 1}


def run(gap):
    a = b = 0
    for qid, rows, tt, tgt, qdir, gold in CASES:
        loc = P.locate(rows, tgt, qdir, tabtoks=(tt if gap else None), gap=gap)
        ok = close(P.pick_value(loc, qdir), gold) if loc else False
        if qid in A:
            a += ok
        else:
            b += ok
    return a, b


print(f"\nnua A (chinh) = {len(A)} cau | nua B (NGHIEM THU) = {len(B)} cau\n")
print(f"{'GAP':>6} | {'A':>16} | {'B':>16}")
print("-" * 46)
res = {}
base = None
for g in (0.0, 0.03, 0.05, 0.08, 0.12, 0.20):
    a, b = run(g)
    res[g] = (a, b)
    if base is None:
        base = (a, b)
    print(f"{g:>6} | {a:>4} ({a-base[0]:+d}) {100*a/len(A):>5.1f}% | {b:>4} ({b-base[1]:+d}) {100*b/len(B):>5.1f}%")

best = max((g for g in res if g > 0), key=lambda g: res[g][0])
print(f"\n=> Chon tu nua A: GAP_THR={best} (A {res[best][0]-base[0]:+d} cau)")
print(f"*** NGHIEM THU nua B: {res[best][1]-base[1]:+d} cau "
      f"({100*base[1]/len(B):.1f}% -> {100*res[best][1]/len(B):.1f}%) ***")
print("\nTIEU CHI DAT TRUOC: nua B phai CUNG DAU voi nua A. Trai dau hoac B<=0 -> LOAI.")
