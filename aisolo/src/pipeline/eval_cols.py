"""KIEM GIA THUYET: `cur` lay vals[0] (so tien DAU TIEN trong dong) co dung cot ky bao cao khong.

Vi sao: dev set cho trong tai an CHINH cac dong da parse cua ta kem CHINH gia tri cur, nen no chi
cham duoc khau CHON DONG va MU HOAN TOAN voi loi parse sai cot. Do la khoang cach 68% (dev set)
vs 40% (probe). Slide BTC cung ghi ro loi lon nhat la "doc sai o/dong/COT" 54.7%.

Do: voi moi bang duoc dung lam nguon dap an, dem so COT SO trong dong du lieu; neu > 2 thi vals[0]
nhieu kha nang KHONG phai ky bao cao can lay.
"""
import os, re, sys, json, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

D = [d for d in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "devset", "devset.json"), encoding="utf-8"))
     if d["llm_val"] is not None]

_C = {}
def raw_tables(p):
    """-> {tid: (header_cells, [so cot so cua tung dong du lieu])}"""
    if p in _C:
        return _C[p]
    txt = open(p, encoding="utf-8", errors="replace").read()
    out = {}
    for tid, m in enumerate(re.finditer(r"<table>(.*?)</table>", txt, re.S)):
        trows = [P.cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S)]
        ncols = [sum(1 for c in r if P.money(c) is not None) for r in trows if r]
        hdr = next((r for r in trows if r and not any(P.money(c) is not None for c in r)), [])
        out[tid] = (hdr, ncols)
    _C[p] = out
    return out


dist = collections.Counter()
examples = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    rows, uf, src = P.ingest(open(str(fr[0]), encoding="utf-8", errors="replace").read())
    loc = P.locate(rows, P.target_label(d["question"]), P.qdir_of(d["question"]))
    if not loc:
        continue
    T = raw_tables(str(fr[0]))
    hdr, ncols = T.get(loc["tid"], ([], []))
    data = [n for n in ncols if n > 0]
    if not data:
        continue
    mode = collections.Counter(data).most_common(1)[0][0]
    dist[mode] += 1
    if mode > 2 and len(examples) < 12:
        examples.append((d["id"], mode, hdr[:6], loc["label"][:38], d["question"][:60]))

tot = sum(dist.values())
print(f"So COT SO cua bang duoc chon lam nguon dap an ({tot} cau dev set):")
for k in sorted(dist):
    flag = "  <== vals[0] CHUA CHAC dung ky" if k > 2 else ""
    print(f"  {k} cot so : {dist[k]:>4} cau = {100*dist[k]/tot:>5.1f}%{flag}")
risk = sum(v for k, v in dist.items() if k > 2)
print(f"\n=> {risk}/{tot} = {100*risk/tot:.1f}% cau lay dap an tu bang co TREN 2 cot so.")
print("\nVi du bang nhieu cot (header + nhan duoc chon):")
for qid, m, hdr, lab, q in examples:
    print(f"  id{qid:<5} {m} cot | header={hdr}")
    print(f"         nhan={lab!r} | {q}")
