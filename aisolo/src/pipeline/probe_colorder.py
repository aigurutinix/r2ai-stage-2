"""THAM DO: bang 2 COT co bi DAO THU TU KY khong (dau nam truoc, cuoi nam sau)?

`ingest()` mac dinh vals[0]=cur (ky nay), vals[1]=prev (ky truoc). Nhung header chi duoc doc khi
bang co TREN 2 cot so (`if ncol>2`), nen bang 2 cot KHONG BAO GIO duoc kiem — neu bao cao in theo
thu tu thoi gian tang dan (So dau nam | So cuoi nam) thi ta lay nham ky, va dev set co ca id7, id45
nhan dung ma gia tri sai dung kieu do.

Do: voi moi cau dev set, doc header cua bang chua dong da chon, xem cot dau la ky SOM hay ky MUON,
va neu dao lai thi co bao nhieu cau tu sai thanh dung / tu dung thanh sai.
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
def load(p):
    if p not in _C:
        txt = open(p, encoding="utf-8", errors="replace").read()
        _C[p] = (P.ingest(txt), txt)
    return _C[p]


def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)


EARLY = re.compile(r"dau (nam|ky)|dau ky", re.I)
LATE = re.compile(r"cuoi (nam|ky)|cuoi ky", re.I)

stat = collections.Counter()
flip = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    (rows, uf, src), txt = load(str(fr[0]))
    qdir, tgt, gold = P.qdir_of(d["question"]), P.target_label(d["question"]), d["llm_val"]
    loc = P.locate(rows, tgt, qdir)
    if not loc:
        continue
    ok = close(P.pick_value(loc, qdir), gold)
    r = next((x for x in rows if x["tid"] == loc["tid"] and x["label"] == loc["label"]), None)
    vals = (r or {}).get("vals") or []
    if len(vals) != 2:
        stat["khong phai bang 2 cot"] += 1
        continue
    stat["bang 2 cot"] += 1
    stat["bang 2 cot & dung"] += ok
    # doc header cua bang do
    tabs = [m.group(1) for m in re.finditer(r"<table>(.*?)</table>", txt, re.S)]
    if loc["tid"] >= len(tabs):
        continue
    trows = [P.cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>", tabs[loc["tid"]], re.S)]
    hrow = next((x for x in trows if x and not any(P.money(c) is not None for c in x)), None)
    if not hrow:
        stat["bang 2 cot & KHONG co header"] += 1
        continue
    hs = [P.strip_vn(c) for c in hrow if P.strip_vn(c).strip()]
    yrs = [re.findall(r"\b(20\d{2})\b", c) for c in hs]
    order = None
    if len(hs) >= 2:
        a, b = hs[-2], hs[-1]
        if EARLY.search(a) and LATE.search(b):
            order = "DAO (dau truoc, cuoi sau)"
        elif LATE.search(a) and EARLY.search(b):
            order = "chuan (cuoi truoc)"
    ya = [int(y) for yy in yrs for y in yy]
    if order is None and len(ya) >= 2:
        order = "DAO (nam tang dan)" if ya[-2] < ya[-1] else "chuan (nam giam dan)"
    stat[f"header: {order}"] += 1
    if order and order.startswith("DAO"):
        swapped = {"label": r["label"], "cur": vals[1], "prev": vals[0]}
        ok2 = close(P.pick_value(swapped, qdir), gold)
        if ok2 != ok:
            flip.append((d["id"], ok, ok2, hs[-2:], d["question"]))
        stat["DAO & hien dung" ] += ok
        stat["DAO & dao lai thi dung"] += ok2

print(f"dev set: {len(D)} cau")
for k, v in stat.most_common():
    print(f"  {k:<38}: {v}")
print()
g = sum(1 for _, o, o2, *_ in flip if o2 and not o)
l = sum(1 for _, o, o2, *_ in flip if o and not o2)
print(f"=> Neu DAO cot cho cac bang co header thu tu tang dan: sai->dung {g}, dung->sai {l}")
for i, o, o2, h, q in flip[:12]:
    print(f"  id{i:<5} {'SAI->DUNG' if o2 else 'DUNG->SAI'} header={h} | {q[:70]}")
