"""IN RA de SOI TAY ket qua exp_rank23_llm.py — khong tu dong ket luan gi.

Vi sao bat buoc phai co buoc nay: gold cua dev set do MOT LLM doc bang tho sinh ra, con thi nghiem
lai do MOT LLM doc bang tho. Neu chi nhin ty le "B_dung_hang23" thi ta dang do DO DONG THUAN GIUA
HAI LLM CUNG PHUONG PHAP, khong phai do do chinh xac — dung cai bay da lam hong ket luan router
self-consistency (dev set +12, nop that -0.002).

Script nay in: cau hoi · nhan pipeline chon (A) · nhan hang 2-3 (B) · nhan model chon.
Nguoi doc tu phan xu B co that su la dap an dung cua cau hoi khong.

Chay: python show_rank23.py            (in cac ca model chon B)
      KIND=A_giu_hang1 python show_rank23.py
      KIND=C_khac python show_rank23.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIND = os.environ.get("KIND", "B_dung_hang23")
N = int(os.environ.get("N", "12"))

recs = json.load(open(os.path.join(HERE, "exp_rank23_llm.json"), encoding="utf-8"))
sel = [r for r in recs if r.get("ket") == KIND]
print(f"{len(sel)} ca thuoc nhom {KIND} (in toi da {N})\n")

for r in sel[:N]:
    print("=" * 96)
    print(f"id{r['id']}  {r['q']}")
    print(f"  A pipeline chon : {r['A_pipeline']!r}")
    print(f"  B hang 2-3      : {r['B_hang23']!r}")
    print(f"  model chon      : {r['model_chon']!r}")
print("=" * 96)
print("\nCau hoi tu dat khi soi: nhan B co that su tra loi DUNG cau hoi khong, hay chi la")
print("dong ma trong tai LLM cua dev set tinh co cung chon? Neu B sai thi ca 'thang' nay la GIA.")
