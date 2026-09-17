"""THAM DO: lan theo lien ket THUYET MINH co cuu duoc cac cau chon sai dong o nhom Easy khong.

Y tuong: bang chinh (CDKT/KQKD) co cot "Thuyet minh" tro sang bang chi tiet cua chinh chi tieu do.
Lien ket nay do BAO CAO TU KHAI nen chinh xac tuyet doi va DOC LAP hoan toan voi khop tu vung.
Khi cau hoi cu the hon nhan tren bang chinh ("nguyen lieu vat lieu" nam trong "hang ton kho"),
dap an dung nam o bang thuyet minh chu khong phai dong da chon.

Do CAN TREN: trong so cau dev set dang SAI, bao nhieu cau co bang thuyet minh (tro tu dong da chon)
chua dung gia tri ma trong tai xac nhan. Do la muc toi da huong nay co the cuu.
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


def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(1.0, abs(float(b)) * 2e-4)


stat = collections.Counter()
recover = []
for d in D:
    fr = P.find_report(d["tk"], d["nam"], d["doctype"])
    if not fr:
        continue
    txt = open(str(fr[0]), encoding="utf-8", errors="replace").read()
    rows, uf, src = ing(str(fr[0]))
    qdir = P.qdir_of(d["question"])
    loc = P.locate(rows, P.target_label(d["question"]), qdir)
    if not loc:
        continue
    truth = close(P.pick_value(loc, qdir), d["llm_val"])
    stat["dung" if truth else "SAI"] += 1
    if truth:
        continue
    # dong da chon co so thuyet minh khong
    r = next((x for x in rows if x["tid"] == loc["tid"] and x["label"] == loc["label"]), None)
    note = (r or {}).get("note")
    if not note:
        stat["SAI & dong chon KHONG co thuyet minh"] += 1
        continue
    stat["SAI & dong chon CO thuyet minh"] += 1
    ln = P.note_table_line(txt, note)
    if not ln:
        stat["SAI & co thuyet minh nhung khong tim thay bang"] += 1
        continue
    # bang thuyet minh = bang bat dau tai dong `ln`
    tids = {x["tid"] for x in rows if x["line"] == ln}
    if not tids:
        continue
    cells = [x for x in rows if x["tid"] in tids and x["cur"] is not None]
    hit = [x for x in cells if close(P.pick_value(x, qdir), d["llm_val"])]
    if hit:
        stat["SAI & BANG THUYET MINH CO DAP AN DUNG"] += 1
        recover.append((d["id"], loc["label"], note, hit[0]["label"], d["question"]))

print(f"dev set {len(D)} cau: dung {stat['dung']} | SAI {stat['SAI']}\n")
for k in ("SAI & dong chon KHONG co thuyet minh", "SAI & dong chon CO thuyet minh",
          "SAI & co thuyet minh nhung khong tim thay bang", "SAI & BANG THUYET MINH CO DAP AN DUNG"):
    print(f"  {k:<48}: {stat[k]}")
print(f"\n=> CAN TREN cua huong thuyet minh: {stat['SAI & BANG THUYET MINH CO DAP AN DUNG']}"
      f"/{stat['SAI']} cau sai = {100*stat['SAI & BANG THUYET MINH CO DAP AN DUNG']/max(1,stat['SAI']):.1f}%")
for i, lab, note, good, q in recover[:10]:
    print(f"\n  id{i}: {q[:88]}")
    print(f"     da chon {lab[:44]!r} (TM {note}) -> dap an dung o dong {good[:44]!r}")
