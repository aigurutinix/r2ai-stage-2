"""THAM DO: doi chieu LIEN NAM co phat hien duoc loi chon dong o nhom Easy khong.

Vi sao can: Auditor hien tai MU voi Easy — dap an tien luon "hop le ve kieu" nen khong the chung
minh sai. Ma Easy chiem 361/1012 cau va la cho con nhieu du dia nhat.

Rang buoc doi chieu: dong ta chon trong bao cao nam N co cot KY TRUOC = X. Cung nhan do trong bao
cao nam N-1 phai co cot KY NAY = X (cung mot con so, do bao cao nam sau in lai so nam truoc).
Lech nhau => it nhat MOT trong hai lan chon dong da sai. Day la rang buoc noi tai cua du lieu,
khong phai doan dap an — cung ban chat voi audit_wrong.py.

Do tren dev set (312 cau co dap an tham chieu doc lap) de biet:
  - do PHU: bao nhieu cau kiem duoc
  - do NHAY: trong so cau BI SAI, bao nhieu bi co
  - do CHINH XAC: trong so cau bi co, bao nhieu that su sai
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
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    return _C[p]


def close(a, b, rel=2e-3):
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    return abs(a - b) <= max(1.0, abs(b) * rel)


stat = collections.Counter()
rows_out = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    qdir = P.qdir_of(d["question"])
    loc = P.locate(rows, P.target_label(d["question"]), qdir)
    if not loc or loc.get("prev") is None:
        stat["khong co cot ky truoc"] += 1
        continue
    fr0 = P.find_report(d["tk"], str(int(d["nam"]) - 1), d["doctype"])
    if not fr0:
        stat["khong co bao cao nam truoc"] += 1
        continue
    rows0, uf0, src0 = ing(str(fr0[0]))
    # cung NHAN trong bao cao nam truoc (khop nguyen van, uu tien cung loai bao cao)
    cands = [r for r in rows0 if r["label"] == loc["label"] and r["cur"] is not None]
    if not cands:
        stat["nam truoc khong co nhan do"] += 1
        continue
    stat["kiem duoc"] += 1
    hit = any(close(loc["prev"], r["cur"]) for r in cands)
    truth = close(P.pick_value(loc, qdir), d["llm_val"])
    stat[("khop" if hit else "LECH") + " / " + ("dung" if truth else "SAI")] += 1
    if not hit:
        rows_out.append((d["id"], truth, loc["label"], loc["prev"], [r["cur"] for r in cands[:3]], d["question"]))

n = stat["kiem duoc"]
print(f"dev set {len(D)} cau | KIEM DUOC bang doi chieu lien nam: {n} ({100*n/len(D):.1f}%)")
for k in ("khong co cot ky truoc", "khong co bao cao nam truoc", "nam truoc khong co nhan do"):
    print(f"  bo qua vi {k}: {stat[k]}")
print()
kd, ks = stat["khop / dung"], stat["khop / SAI"]
ld, ls = stat["LECH / dung"], stat["LECH / SAI"]
print(f"{'':<8}{'thuc te DUNG':>14}{'thuc te SAI':>14}")
print(f"{'khop':<8}{kd:>14}{ks:>14}")
print(f"{'LECH':<8}{ld:>14}{ls:>14}")
print()
if ld + ls:
    print(f"  do CHINH XAC cua co LECH (bao nhieu ca bi co that su sai): {100*ls/(ld+ls):.1f}%")
if ks + ls:
    print(f"  do NHAY (trong so cau SAI, bao nhieu bi co)              : {100*ls/(ks+ls):.1f}%")
if kd + ks:
    print(f"  ty le SAI o nhom 'khop' (nen thap)                       : {100*ks/(kd+ks):.1f}%")
print("\n--- vi du ca bi co LECH ---")
for i, truth, lab, prev, curs, q in rows_out[:10]:
    print(f"  id{i:<5} {'(thuc te DUNG - bao dong nham)' if truth else '(thuc te SAI - bat trung)'}")
    print(f"        nhan={lab[:52]!r} prev={prev} vs nam truoc cur={curs}")
