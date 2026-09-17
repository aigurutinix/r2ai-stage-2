"""Xem vị trí ô Mã số trong dòng tổng tài sản + đo lại độ phủ Mã-số ở MỌI vị trí ô."""
import re
from pathlib import Path
import os as _os
_BASE = _os.environ.get("VIFINQA_ROOT",
                        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data_vifinqa"))
from collections import Counter
ROOT = Path(_BASE) / "financial_statements"
def cells(tr): return [re.sub(r"<[^>]*>","",c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S)]

for f in ["VNM/2023","HPG/2022","FPT/2023","GAS/2021","MWG/2020","ACB/2022"]:
    rps=list((ROOT/f).glob("*/*.txt"))
    if not rps: print(f"--- {f}: no file ---"); continue
    txt=rps[0].read_text(encoding="utf-8",errors="replace")
    rows=re.findall(r"<tr>(.*?)</tr>",txt,re.S)
    hit=None
    for tr in rows:
        c=cells(tr)
        joined=" ".join(c).lower()
        if "tổng cộng tài sản" in joined or ("tổng tài sản" in joined and len(c)>=3):
            hit=c; break
    print(f"--- {f} ---")
    print(f"   {hit}")

# đo lại: bảng có Mã số ở BẤT KỲ ô nào (không chỉ ô đầu)
print("\n=== ĐO LẠI độ phủ Mã-số (code ở bất kỳ cột nào) ===")
reports=list(ROOT.glob("*/*/*/*.txt"))
CODE=re.compile(r"^\d{2,3}$")
maso_any=0; maso_col_pos=Counter(); n=0
for rp in reports:
    txt=rp.read_text(encoding="utf-8",errors="replace")
    tabs=[re.findall(r"<tr>(.*?)</tr>",tb,re.S) for tb in re.findall(r"<table>(.*?)</table>",txt,re.S)]
    rep_has=False
    for tab in tabs:
        # tìm cột mà >=3 dòng là code TT200
        colcodes=Counter()
        for tr in tab[:25]:
            c=cells(tr)
            for j,cell in enumerate(c):
                if CODE.match(cell.strip()): colcodes[j]+=1
        for j,cnt in colcodes.items():
            if cnt>=3: rep_has=True; maso_col_pos[j]+=1; break
    if rep_has: maso_any+=1
    n+=1
print(f"Report có bảng Mã-số (ô bất kỳ): {maso_any}/{n} ({maso_any*100//n}%)")
print(f"Vị trí cột chứa Mã số (col index → #report): {dict(maso_col_pos.most_common())}")
