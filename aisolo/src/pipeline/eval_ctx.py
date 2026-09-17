"""ĐO trọng số NGỮ CẢNH CHA (CTX_W) trên dev set — chỉnh nửa A, NGHIỆM THU nửa B.

Dev set 312 câu có đáp án xác minh độc lập (gemma4:31b-cloud đọc bảng thô). Thước đo này HỢP LỆ
cho thay đổi TỪ VỰNG/deterministic như CTX_W; nó chỉ VÔ HIỆU với hướng LLM-đọc-bảng
(xem memory qwen-router-selfconsistency-deadend).

Chia đôi là bắt buộc: router `ov` từng cho +8 trên toàn bộ rồi chỉ còn +1 trên nửa nghiệm thu.

Chạy: python eval_ctx.py
"""
import os, re, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

D = [d for d in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None]

_C = {}
def ing(p):
    if p not in _C:
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    return _C[p]


def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)


# nạp trước report của mọi câu để vòng quét trọng số không phải parse lại
CASES = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    CASES.append((d["id"], rows, P.target_label(d["question"]), P.qdir_of(d["question"]), d["llm_val"]))
print(f"dev set dung duoc: {len(CASES)} cau")

ids = [c[0] for c in CASES]
A = {i for k, i in enumerate(sorted(ids)) if k % 2 == 0}
B = {i for k, i in enumerate(sorted(ids)) if k % 2 == 1}


def run(w):
    P.CTX_W = w
    a = b = 0
    for qid, rows, tgt, qdir, gold in CASES:
        loc = P.locate(rows, tgt, qdir)
        ok = close(P.pick_value(loc, qdir), gold) if loc else False
        if qid in A:
            a += ok
        else:
            b += ok
    return a, b


print(f"\nnua A (chinh) = {len(A)} cau | nua B (NGHIEM THU) = {len(B)} cau\n")
print(f"{'CTX_W':>6} | {'A':>12} | {'B':>12}")
print("-" * 38)
base = None
res = {}
for w in (0.0, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0):
    a, b = run(w)
    res[w] = (a, b)
    if base is None:
        base = (a, b)
    print(f"{w:>6} | {a:>4} ({a-base[0]:+d}) {100*a/len(A):>5.1f}% | {b:>4} ({b-base[1]:+d}) {100*b/len(B):>5.1f}%")

best_w = max((w for w in res if w > 0), key=lambda w: res[w][0])
print(f"\n=> Chon tu nua A: CTX_W={best_w} (A {res[best_w][0]-base[0]:+d} cau)")
print(f"*** NGHIEM THU nua B: {res[best_w][1]-base[1]:+d} cau "
      f"({100*base[1]/len(B):.1f}% -> {100*res[best_w][1]/len(B):.1f}%) ***")
