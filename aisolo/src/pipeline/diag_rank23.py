"""PHAN LOAI CAC CA DONG DUNG DUNG HANG 2-3 (41 ca theo diag_pool.py).

Day la nhom dac nhat va gan dich nhat: dong dung DA nam trong cands, chi thua dong hang 1 mot
chut diem tu vung. Cau hoi: dong hang 1 (sai) va dong dung khac nhau o CHO NAO? Neu co mot dac
trung lap lai thi do la don bay; neu moi ca mot kieu thi khong co gi de sua.

Phan nhom:
  ID-TABLE  nhan GIONG HET, khac BANG          -> chi tin hieu cap bang moi phan biet (da do: reranker 71%)
  ID-SAME   nhan GIONG HET, CUNG bang          -> khac dong trong 1 bang, can tin hieu vi tri/ctx
  CTX       nhan khac nhau nhung NGU CANH CHA phan biet duoc -> "Add group context" cua BTC
  SUB       nhan dung la con/cha cua nhan chon (chuoi long nhau)
  OTHER     con lai

Chay: python diag_rank23.py
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

MAXR = int(os.environ.get("MAXR", "3"))
kind = Counter(); rowsout = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    qt = d["question"]; tgt = P.target_label(qt); qdir = P.qdir_of(qt); gold = d["llm_val"]
    loc = P.locate(rows, tgt, qdir)
    if loc and close(P.pick_value(loc, qdir), gold):
        continue
    cands = P.score_cands(rows, tgt, qdir)
    if not cands:
        continue
    good = [(i, ov, r) for i, (ov, r) in enumerate(cands)
            if close(P.pick_value(as_loc(r), qdir), gold)]
    if not good or good[0][0] + 1 > MAXR or good[0][0] == 0:
        continue
    gi, gov, gr = good[0]
    cov, cr = cands[0]

    gctx = (gr.get("ctx") or "").strip(); cctx = (cr.get("ctx") or "").strip()
    if gr["label"] == cr["label"]:
        k = "ID-TABLE" if gr["tid"] != cr["tid"] else "ID-SAME"
    elif gctx and gctx != cctx and set(P.toks(gctx)) & set(P.toks(tgt)):
        k = "CTX"
    elif P.strip_vn(gr["label"]) in P.strip_vn(cr["label"]) or P.strip_vn(cr["label"]) in P.strip_vn(gr["label"]):
        k = "SUB"
    else:
        k = "OTHER"
    kind[k] += 1
    rowsout.append((k, d["id"], tgt, cov, cr, cctx, gi + 1, gov, gr, gctx))

tot = sum(kind.values())
print(f"So ca dong dung dung hang 2..{MAXR}: {tot}\n")
for k, n in kind.most_common():
    print(f"  {k:<9} {n:>3}  {100*n/max(1,tot):>5.1f}%")
print()
seen = Counter()
for k, qid, tgt, cov, cr, cctx, gi, gov, gr, gctx in sorted(rowsout):
    seen[k] += 1
    if seen[k] > 4:
        continue
    print(f"[{k}] id{qid}  target={tgt!r}")
    print(f"     CHON  h1  ov={cov:.3f} tid={cr['tid']:<4} {cr['label'][:64]!r}" + (f"  ctx={cctx[:40]!r}" if cctx else ""))
    print(f"     DUNG  h{gi}  ov={gov:.3f} tid={gr['tid']:<4} {gr['label'][:64]!r}" + (f"  ctx={gctx[:40]!r}" if gctx else ""))
