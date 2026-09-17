"""DO TIE-BREAK BANG TOKEN THUA CUA CAU HOI (residual) — chinh nua A, NGHIEM THU nua B.

Vi sao huong nay khac han hai lan thu truoc:
  - eval_gap.py dung BM25 voi query = target_label. Nhung target_label chinh la chuoi ma CA HAI
    bang deu khop ngang nhau (do la ly do chung hoa diem) => query khong mang thong tin phan biet.
    Ket qua dung nhu du doan: 68% vs 68%, di ngang.
  - eval_rr.py dung thu tu reranker, cung xep theo do lien quan voi CAU HOI/target => 71% vs 68%.
  - O day query = TOKEN THUA: token cua cau hoi TRU token da nam trong nhan. Do chinh la phan
    ngu nghia chua duoc dung den, va la phan duy nhat co the phan biet hai bang cung chua nhan do.
    Vi du id3 target "Chi phi du phong": hai bang deu co nhan nay, nhung tu "rui ro tin dung"
    trong cau hoi chi xuat hien o bang thuyet minh.

Chi can thiep khi HOA. Chia doi BAT BUOC (router `ov` tung +8 roi con +1 khi nghiem thu).

Chay: python eval_resid.py          (MODE=bm25 | ov, mac dinh bm25)
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = [d for d in json.load(open(os.path.join(HERE, "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None]
MODE = os.environ.get("MODE", "bm25")

_C = {}
def ing(p):
    if p not in _C:
        txt = open(p, encoding="utf-8", errors="replace").read()
        tt = P.table_tokens(txt)
        _C[p] = (P.ingest(txt), tt, {t["tid"]: set(t["toks"]) for t in tt})
    return _C[p]

def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)

def as_loc(r):
    return {"maso": "", "label": r["label"], "tid": r["tid"], "cur": r["cur"], "prev": r["prev"],
            "ov": 0.0, "hdrs": r.get("hdrs") or [], "vals": r.get("vals") or []}

CASES = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    (rows, uf, src), tt, tset = ing(str(fr[0]))
    qt = d["question"]; tgt = P.target_label(qt)
    resid = P.toks(qt) - P.toks(tgt)                    # token cau hoi CHUA duoc nhan dung den
    CASES.append((d["id"], rows, tt, tset, tgt, resid, P.qdir_of(qt), d["llm_val"]))
print(f"dev set dung duoc: {len(CASES)} cau")

ids = sorted(c[0] for c in CASES)
A = {i for k, i in enumerate(ids) if k % 2 == 0}
B = {i for k, i in enumerate(ids) if k % 2 == 1}


def pick(rows, tt, tset, tgt, resid, qdir, gap):
    # PHAI lap lai short-circuit MAIN_CODE cua locate(), neu khong baseline bi thap gia (199 vs 208)
    # va can thiep se de len ca nhung ca locate() von giai bang Ma so TT200 — so sanh sai ban chat.
    tn = P.strip_vn(tgt)
    code = next((c for k, c in P.MAIN_CODE.items() if k in tn), None)
    if code and not any(x in tn for x in ["chua phan phoi", "thang du"]):
        cand = [r for r in rows if r["ma"] == code and r["cur"] is not None]
        if cand:
            return cand[0]
    cands = P.score_cands(rows, tgt, qdir)
    if not cands:
        return None
    if not gap or len(cands) < 2 or cands[0][0] - cands[1][0] >= gap:
        return cands[0][1]
    tied = [(ov, r) for ov, r in cands if cands[0][0] - ov < gap]
    if len(tied) < 2 or not resid:
        return cands[0][1]
    if MODE == "bm25":
        bm = P.bm25_scores(list(resid), [t["toks"] for t in tt])
        sc = {t["tid"]: s for t, s in zip(tt, bm)}
    else:                                                # do phu don gian: |resid & token bang| / |resid|
        sc = {tid: len(resid & tk) / max(1, len(resid)) for tid, tk in tset.items()}
    return max(tied, key=lambda x: (sc.get(x[1]["tid"], 0.0), x[0]))[1]


def run(gap):
    a = b = 0
    for qid, rows, tt, tset, tgt, resid, qdir, gold in CASES:
        r = pick(rows, tt, tset, tgt, resid, qdir, gap)
        ok = close(P.pick_value(as_loc(r), qdir), gold) if r is not None else False
        if qid in A:
            a += ok
        else:
            b += ok
    return a, b


print(f"MODE={MODE}\nnua A (chinh) = {len(A)} cau | nua B (NGHIEM THU) = {len(B)} cau\n")
print(f"{'GAP':>6} | {'A':>18} | {'B':>18}")
print("-" * 50)
res = {}; base = None
for g in (0.0, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35):
    a, b = run(g)
    res[g] = (a, b)
    if base is None:
        base = (a, b)
    print(f"{g:>6} | {a:>4} ({a-base[0]:+d}) {100*a/len(A):>5.1f}% | {b:>4} ({b-base[1]:+d}) {100*b/len(B):>5.1f}%")

best = max((g for g in res if g > 0), key=lambda g: res[g][0])
print(f"\n=> Chon tu nua A: GAP={best} (A {res[best][0]-base[0]:+d} cau)")
print(f"*** NGHIEM THU nua B: {res[best][1]-base[1]:+d} cau "
      f"({100*base[1]/len(B):.1f}% -> {100*res[best][1]/len(B):.1f}%) ***")
print("\nTIEU CHI DAT TRUOC: nua B phai CUNG DAU voi nua A. Trai dau hoac B<=0 -> LOAI.")
