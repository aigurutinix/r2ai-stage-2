"""CAN TREN cua MOI tin hieu pha the hoa — va BM25 cap bang sai o dau.

BM25 cap bang da do: nua A -7, nua B +3 => LOAI theo tieu chi dat truoc. Truoc khi bo ca huong
"rang buoc bang", can biet:
  1. CAN TREN: neu trong nhom HOA luon chon dung ung vien (oracle), duoc them bao nhieu cau?
     Thap => ca huong chet, khong phai loi BM25.
  2. BM25 xep bang DUNG hang may trong nhom hoa? Neu thuong xep THAP => BM25 phan xu nguoc.
  3. Cac tin hieu KHAC co san tren row (st = loai bao cao, ctx = nhan cha, note) co phan biet duoc khong?
"""
import os, re, sys, json, collections

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


GAP = float(os.environ.get("GAP_THR", "0.05"))
stat = collections.Counter()
rank_of_right = collections.Counter()
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    (rows, uf, src), tt = ing(str(fr[0]))
    tgt, qdir, gold = P.target_label(d["question"]), P.qdir_of(d["question"]), d["llm_val"]
    cands = P.score_cands(rows, tgt, qdir)
    if not cands:
        continue
    cur_ok = close(P.pick_value({"label": cands[0][1]["label"], "cur": cands[0][1]["cur"],
                                 "prev": cands[0][1]["prev"]}, qdir), gold)
    if len(cands) < 2 or cands[0][0] - cands[1][0] >= GAP:
        stat["ngoai vung hoa"] += 1
        stat["ngoai vung hoa & dung"] += cur_ok
        continue
    tied = [(ov, r) for ov, r in cands if cands[0][0] - ov < GAP]
    stat["TRONG vung hoa"] += 1
    stat["trong vung hoa & hien dung"] += cur_ok
    # oracle: co ung vien nao trong nhom hoa cho dap an DUNG khong
    good = [(ov, r) for ov, r in tied
            if close(P.pick_value({"label": r["label"], "cur": r["cur"], "prev": r["prev"]}, qdir), gold)]
    if good:
        stat["trong vung hoa & CO ung vien dung"] += 1
        if not cur_ok:
            stat["=> ORACLE cuu duoc"] += 1
        # BM25 xep bang cua ung vien DUNG hang may trong so cac bang co mat trong nhom hoa?
        bm = P.bm25_scores(P.toks_seg(tgt), [t["toks"] for t in tt])
        tsc = {t["tid"]: s for t, s in zip(tt, bm)}
        tids = sorted({r["tid"] for _, r in tied}, key=lambda x: -tsc.get(x, 0.0))
        rt = {r["tid"] for _, r in good}
        pos = next((i + 1 for i, t in enumerate(tids) if t in rt), None)
        rank_of_right[f"hang {pos}/{len(tids)}" if pos and len(tids) <= 4 else
                      (f"hang {pos}" if pos else "khong ro")] += 1
    elif not cur_ok:
        stat["trong vung hoa & KHONG ung vien nao dung"] += 1

n_tie = stat["TRONG vung hoa"]
print(f"GAP_THR = {GAP}\n")
print(f"ngoai vung hoa      : {stat['ngoai vung hoa']:>3} cau | dung {stat['ngoai vung hoa & dung']:>3}"
      f" = {100*stat['ngoai vung hoa & dung']/max(1,stat['ngoai vung hoa']):.1f}%")
print(f"TRONG vung hoa      : {n_tie:>3} cau | dung {stat['trong vung hoa & hien dung']:>3}"
      f" = {100*stat['trong vung hoa & hien dung']/max(1,n_tie):.1f}%")
print()
print(f"  co ung vien DUNG trong nhom hoa   : {stat['trong vung hoa & CO ung vien dung']}")
print(f"  => ORACLE (luon chon dung) CUU DUOC: {stat['=> ORACLE cuu duoc']} cau   <== CAN TREN")
print(f"  khong ung vien nao dung (vo phuong): {stat['trong vung hoa & KHONG ung vien nao dung']}")
print()
print("BM25 cap bang xep BANG DUNG hang may (trong so cac bang co mat o nhom hoa):")
for k, v in rank_of_right.most_common():
    print(f"  {k:<14}: {v}")
