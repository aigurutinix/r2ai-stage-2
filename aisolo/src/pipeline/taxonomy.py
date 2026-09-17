"""Nghiên cứu #2: taxonomy 1012 câu theo ĐỘ PHỨC TẠP TRẢ LỜI (quyết định template deterministic phủ được bao nhiêu)."""
import json, re
from pathlib import Path
import os as _os
_BASE = _os.environ.get("VIFINQA_ROOT",
                        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data_vifinqa"))
from collections import Counter
ROOT = Path(_BASE)
Q=[json.loads(l) for l in (ROOT/"questions/questions.jsonl").open(encoding="utf-8") if l.strip()]

import csv, unicodedata
def strip_vn(s):
    s=s.replace("đ","d").replace("Đ","D")
    return "".join(c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c)!="Mn").lower()
tickers=set()
with (ROOT/"code_stock.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f): tickers.add(r["Mã CK"].strip())

def n_tickers(q):
    return len(set(m for m in re.findall(r"\b([A-Z0-9]{2,4})\b",q) if m in tickers))
def years(q): return sorted(set(re.findall(r"\b(20\d{2})\b",q)))

cat=Counter(); samp={}
AGG=["cao nhất","thấp nhất","lớn nhất","nhỏ nhất","tăng nhiều nhất","giảm nhiều nhất","trung bình",
     "tổng cộng các năm","nhiều nhất","ít nhất"]
COND=["trong các năm","trong những năm","trong giai đoạn","các công ty có","doanh nghiệp có mức",
      "vào năm mà","năm ngay sau","xét các","trong ngành"]
RATIO=["tỷ suất","biên lợi nhuận","roe","roa","tỷ lệ nợ","vòng quay","hệ số","tốc độ tăng trưởng","tăng trưởng"]
OWN=["tỷ lệ sở hữu","tỷ lệ biểu quyết","quyền biểu quyết","tỷ lệ lợi ích","tỷ lệ nắm giữ"]
GROWTH=["tăng trưởng","so với năm","yoy","tăng bao nhiêu","giảm bao nhiêu"]

for q in Q:
    t=q["question"]; low=t.lower(); yrs=years(t); nt=n_tickers(t)
    if any(k in low for k in OWN): c="ownership-%"
    elif any(k in low for k in COND): c="conditional-reasoning"
    elif any(k in low for k in AGG): c="aggregation(min/max/avg)"
    elif nt>=2: c="multi-company-compare"
    elif len(yrs)>=2 or any(k in low for k in GROWTH): c="multi-year/growth"
    elif any(k in low for k in RATIO): c="ratio"
    else: c="direct-single-lookup"
    cat[c]+=1
    samp.setdefault(c,[]).append(t[:88])

n=len(Q)
print(f"=== TAXONOMY {n} câu theo độ phức tạp trả lời ===\n")
order=["direct-single-lookup","ratio","multi-year/growth","multi-company-compare",
       "aggregation(min/max/avg)","conditional-reasoning","ownership-%"]
DIFF={"direct-single-lookup":"DỄ (1 lookup)","ratio":"TB (2 metric ÷)",
      "multi-year/growth":"TB (2 kỳ)","multi-company-compare":"TB (union DN)",
      "aggregation(min/max/avg)":"KHÓ (pandas agg nhiều dòng)",
      "conditional-reasoning":"RẤT KHÓ (filter+reason)","ownership-%":"ĐẶC THÙ (%-extractor)"}
for c in order:
    v=cat[c]; print(f"  {c:26} {v:4} ({v*100//n:2}%)  · {DIFF[c]}")
print(f"\n→ Deterministic template dễ phủ: direct + ratio + multi-year + multi-company + ownership")
easy=sum(cat[c] for c in ["direct-single-lookup","ratio","multi-year/growth","multi-company-compare","ownership-%"])
hard=sum(cat[c] for c in ["aggregation(min/max/avg)","conditional-reasoning"])
print(f"   'Template-được' ~{easy} ({easy*100//n}%) · 'Khó/agg/reason' ~{hard} ({hard*100//n}%)")
print("\n=== Mẫu mỗi loại ===")
for c in order:
    print(f"\n[{c}]")
    for s in samp.get(c,[])[:3]: print("   -",s)
