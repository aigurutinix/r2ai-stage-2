"""DO CAN TREN cua TIEN TO PHAN CAP truoc khi viet bat cu dong code san xuat nao.

Vi sao do truoc: ba lan trong du an nay da cai roi moi do, ca ba deu phi cong (expand_rows ra 0
tac dung; CTX_W +2 roi vo nhanh RATIO-PAIR; fine-tune reranker co lap 62->83% ma end-to-end te di).
Tien le tot: huong lan theo Thuyet minh bi chan dung cach khi do can tren ra 1.9% roi bo ngay.

GIA THUYET (tu bai OHD, arxiv 2602.01969): nhan dong dung mot minh thi vo nghia ("Trong vong 1 nam",
"Bang VND", "So cuoi nam"); ghep TO TIEN vao truoc thi no moi mang du nghia de khop cau hoi.
Bao cao §7.4 do duoc 57% nhom cau khop nhan thap roi vao dung tinh huong nay.

DO BA THU, theo thu tu tang dan chi phi:
  1. DO PHU cua `ctx` hien co — neu thap thi phan lon ca 57% kia khong co to tien de ghep, va can
     tren tu dong thap. Biet som dieu nay la tiet kiem nhat.
  2. CAN TREN: voi nhan da ghep tien to, co bao nhieu cau SAI chuyen thanh dong dung len HANG 1.
  3. THIET HAI: co bao nhieu cau DANG DUNG bi day xuong — vi tien to lam nhan dai ra, ma cong thuc
     F1 phat token thua. Day dung la co che da khien CTX_W gay hai.

TIEU CHI DAT TRUOC (ghi ra day TRUOC khi nhin so):
  di tiep neu  CUU >= 15 cau  VA  HONG <= CUU/2.
  Duoi nguong -> dung, ghi vao bang ngo cut.

Chay: python diag_ctx_upper.py
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

def expanded(r):
    """Nhan da ghep to tien: 'cha > nhan'. Rong thi giu nguyen nhan."""
    c = (r.get("ctx") or "").strip()
    return f"{c} {r['label']}".strip() if c else r["label"]

# ---------- 1. DO PHU cua ctx ----------
tot_rows = has_ctx = 0
seen = set()
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr or str(fr[0]) in seen:
        continue
    seen.add(str(fr[0]))
    rows, uf, src = ing(str(fr[0]))
    for r in rows:
        if r["cur"] is None or not r["label"]:
            continue
        tot_rows += 1
        has_ctx += bool((r.get("ctx") or "").strip())
print(f"1. DO PHU `ctx` tren {len(seen)} bao cao cua dev set")
print(f"   dong co gia tri + co nhan : {tot_rows}")
print(f"   trong do CO ctx           : {has_ctx}  ({100*has_ctx/max(1,tot_rows):.1f}%)")

# ---------- 2 & 3. CAN TREN va THIET HAI ----------
def rank_of_correct(rows, tgt, qdir, gold, use_ctx):
    """Hang cua dong dung trong danh sach cham diem. None neu khong co dong nao dung."""
    cands = P.score_cands(rows, tgt, qdir) if not use_ctx else score_ctx(rows, tgt, qdir)
    for i, (ov, r) in enumerate(cands):
        if close(P.pick_value(as_loc(r), qdir), gold):
            return i
    return None

def score_ctx(rows, target, qdir):
    """Ban sao score_cands nhung cham tren NHAN DA GHEP TO TIEN. Giu nguyen moi thanh phan khac
    de chi mot bien thay doi."""
    tt = P.toks(target); tn_s = P.strip_vn(target); out = []
    for r in rows:
        if r["cur"] is None or not r["label"]:
            continue
        lab = expanded(r)
        ls = P.strip_vn(lab); lt = P.toks(lab)
        inter = len(tt & lt)
        if inter == 0:
            continue
        prec = inter / max(1, len(lt)); rec = inter / max(1, len(tt))
        ov = 2 * prec * rec / (prec + rec)
        if tn_s and (tn_s in ls): ov += P.CONT_IN
        elif tn_s and (ls in tn_s): ov += P.CONT_OF
        if lt <= tt or tt <= lt: ov += 0.001
        if qdir == "cuoi" and ("dau nam" in ls or "dau ky" in ls): ov -= 0.5
        if qdir == "dau" and ("cuoi nam" in ls or "cuoi ky" in ls): ov -= 0.5
        if ov > 0: out.append((ov, r))
    out.sort(key=lambda x: (-x[0], len(x[1]["label"])))
    return out

cuu, hong, giu_dung, giu_sai, khong_co_dong_dung = 0, 0, 0, 0, 0
vd_cuu, vd_hong = [], []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    tgt = P.target_label(d["question"]); qdir = P.qdir_of(d["question"]); gold = d["llm_val"]
    r0 = rank_of_correct(rows, tgt, qdir, gold, False)
    r1 = rank_of_correct(rows, tgt, qdir, gold, True)
    if r0 is None and r1 is None:
        khong_co_dong_dung += 1; continue
    dung0 = (r0 == 0); dung1 = (r1 == 0)
    if not dung0 and dung1:
        cuu += 1
        if len(vd_cuu) < 5: vd_cuu.append((d["id"], d["question"][:78], tgt))
    elif dung0 and not dung1:
        hong += 1
        if len(vd_hong) < 5: vd_hong.append((d["id"], d["question"][:78], tgt))
    elif dung0: giu_dung += 1
    else: giu_sai += 1

print(f"\n2/3. GHEP TO TIEN vao nhan roi cham lai ({len(D)} cau dev set)")
print(f"   dang SAI -> thanh DUNG   (CUU)  : {cuu}")
print(f"   dang DUNG -> thanh SAI   (HONG) : {hong}")
print(f"   giu nguyen dung                 : {giu_dung}")
print(f"   giu nguyen sai                  : {giu_sai}")
print(f"   khong dong nao mang gia tri dung: {khong_co_dong_dung}")
print(f"\n   RONG: {cuu - hong:+d} cau")
print("\n--- vi du CUU ---")
for i, q, t in vd_cuu: print(f"   id{i}: {q}\n        target={t!r}")
print("--- vi du HONG ---")
for i, q, t in vd_hong: print(f"   id{i}: {q}\n        target={t!r}")

dat = cuu >= 15 and hong <= cuu / 2
print(f"\n{'='*70}\nTIEU CHI DAT TRUOC: CUU >= 15 VA HONG <= CUU/2")
print(f"KET QUA: cuu={cuu}, hong={hong}  ->  {'DAT — di tiep buoc 2' if dat else 'KHONG DAT — dung, ghi vao bang ngo cut'}")
