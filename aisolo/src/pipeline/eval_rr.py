"""DO TIE-BREAK CAP BANG BANG RERANKER (rerank_cache.json) — chinh nua A, NGHIEM THU nua B.

Vi sao lam lai huong da bi loai: eval_gap.py do BM25 lam tin hieu bang → di ngang (68% vs 68%),
ket luan luc do ghi la "tin hieu bang vo dung". Ket luan DUNG phai la "BM25 khong du manh":
theo bang do cua BTC (slide 26), BM25 Recall@10 = 47.4%, con Qwen3-Embedding + reranker = 80.8%.
Ta DA CO san thu tu reranker cho 995/1012 cau trong rerank_cache.json, nhung no chi duoc dung de
dien truong relevant_tables (xem docstring primary_lines: "Chi dung cho relevant_tables") —
duong sinh dap an goi locate() tren TOAN BO dong cua ca bao cao, khong he biet gi ve bang.

Chi can thiep khi HOA diem tu vung: cach biet top1-top2 <0.05 → dang dung 48.1%; >=0.3 → 87.9%.

Chia doi BAT BUOC: router `ov` tung cho +8 tren toan bo roi chi con +1 khi nghiem thu.

Chay: python eval_rr.py
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
print(f"rerank_cache: {len(RR)} khoa")

_C = {}
def ing(p):
    if p not in _C:
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    return _C[p]


def table_query(qt):                      # ban sao cua build_submission.table_query (tranh side-effect khi import)
    t = P.target_label(qt)
    return t if len(t) >= 6 else qt


def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)


CASES = []
hit = 0
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    rr = RR.get(f"{fr[1]}|{table_query(d['question'])}")
    hit += bool(rr)
    CASES.append((d["id"], rows, rr, P.target_label(d["question"]), P.qdir_of(d["question"]), d["llm_val"]))
print(f"dev set dung duoc: {len(CASES)} cau | co khoa rerank: {hit} ({100*hit/max(1,len(CASES)):.1f}%)")

ids = sorted(c[0] for c in CASES)
A = {i for k, i in enumerate(ids) if k % 2 == 0}
B = {i for k, i in enumerate(ids) if k % 2 == 1}


def run(gap):
    a = b = 0
    for qid, rows, rr, tgt, qdir, gold in CASES:
        loc = P.locate(rows, tgt, qdir, rrank=(rr if gap else None), gap=gap)
        ok = close(P.pick_value(loc, qdir), gold) if loc else False
        if qid in A:
            a += ok
        else:
            b += ok
    return a, b


print(f"\nnua A (chinh) = {len(A)} cau | nua B (NGHIEM THU) = {len(B)} cau\n")
print(f"{'GAP':>6} | {'A':>18} | {'B':>18}")
print("-" * 50)
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
