"""Nghiên cứu toàn diện #1: cấu trúc corpus 1973 report — hiểu không gian retrieval,
độ phủ unit-marker, độ phủ Mã-số, phân loại statement-type từng bảng."""
import re, statistics
from pathlib import Path
import os as _os
_BASE = _os.environ.get("VIFINQA_ROOT",
                        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data_vifinqa"))
from collections import Counter
ROOT = Path(_BASE) / "financial_statements"

reports = list(ROOT.glob("*/*/*/*.txt"))
print(f"Tổng report .txt: {len(reports)}")

def parse_tables(txt):
    return [re.findall(r"<tr>(.*?)</tr>",tb,re.S) for tb in re.findall(r"<table>(.*?)</table>",txt,re.S)]
def cells(tr): return [re.sub(r"<[^>]*>","",c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S)]

TT200_BS={"100","110","270","300","400","410","440"}
TT200_PL={"01","10","11","20","50","60"}
def unit_of(low):
    ctx=r"(số (?:cuối|đầu) (?:năm|kỳ)|đơn vị[:\s]*tính?|làm tròn đến hàng)[^<\n]{0,25}?"
    if re.search(ctx+r"triệu\s*đồng",low): return "triệu"
    if re.search(ctx+r"(?:nghìn|ngàn)\s*đồng",low): return "nghìn"
    if re.search(ctx+r"tỷ\s*đồng",low): return "tỷ"
    # marker lỏng (bất kỳ đâu) — để đo độ phủ khả dĩ
    if re.search(r"triệu\s*đồng",low): return "triệu?"
    if re.search(r"(?:nghìn|ngàn)\s*đồng",low): return "nghìn?"
    return "VND/none"

tabcounts=[]; unit_c=Counter(); has_bs=0; has_pl=0; has_cf=0
maso_tables=0; total_tables=0; maso_reports=0
sample_done=0
for i,rp in enumerate(reports):
    try: txt=rp.read_text(encoding="utf-8",errors="replace")
    except: continue
    low=txt.lower()
    tabs=parse_tables(txt)
    tabcounts.append(len(tabs))
    total_tables+=len(tabs)
    unit_c[unit_of(low)]+=1
    if "tổng cộng tài sản" in low or "tổng tài sản" in low: has_bs+=1
    if "lưu chuyển tiền thuần" in low: has_cf+=1
    if "lợi nhuận sau thuế" in low: has_pl+=1
    rep_has_maso=False
    for tab in tabs:
        codes=0
        for tr in tab[:20]:
            c=cells(tr)
            if c and re.fullmatch(r"\d{2,3}",c[0].strip()): codes+=1
        if codes>=3: maso_tables+=1; rep_has_maso=True
    if rep_has_maso: maso_reports+=1

print(f"\n=== BẢNG / REPORT ===")
print(f"Tổng bảng: {total_tables} | trung bình {total_tables/len(tabcounts):.1f}/report")
print(f"Phân phối #bảng/report: min={min(tabcounts)} p25={statistics.quantiles(tabcounts,n=4)[0]:.0f} "
      f"median={statistics.median(tabcounts):.0f} p75={statistics.quantiles(tabcounts,n=4)[2]:.0f} max={max(tabcounts)}")
print(f"→ Retrieval phải chọn đúng ~1 bảng trong TRUNG BÌNH {total_tables/len(tabcounts):.0f} bảng (median {statistics.median(tabcounts):.0f})")

print(f"\n=== ĐỘ PHỦ STATEMENT (report có chứa) ===")
n=len(tabcounts)
print(f"Có bảng CĐKT (tổng tài sản): {has_bs} ({has_bs*100//n}%)")
print(f"Có KQKD (lợi nhuận sau thuế): {has_pl} ({has_pl*100//n}%)")
print(f"Có LCTT (lưu chuyển tiền thuần): {has_cf} ({has_cf*100//n}%)")

print(f"\n=== ĐỘ PHỦ MÃ SỐ (bảng có cột Mã số TT200) ===")
print(f"Bảng có ≥3 dòng Mã số: {maso_tables} ({maso_tables*100//total_tables}% tổng bảng)")
print(f"Report có ≥1 bảng Mã số: {maso_reports} ({maso_reports*100//n}%)")
print(f"→ Main statement tra thẳng Mã số được; {100-maso_tables*100//total_tables}% bảng còn lại (notes) KHÔNG có Mã số → phải match label")

print(f"\n=== ĐỘ PHỦ UNIT MARKER ===")
for k,v in unit_c.most_common(): print(f"  {k:12}: {v} ({v*100//n}%)")
