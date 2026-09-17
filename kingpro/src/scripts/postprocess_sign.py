"""Đòn nhỏ cuối từ slide (quy tắc semantic BTC): 'chênh lệch KHÔNG nêu hướng -> giá trị tuyệt đối'.
Trên sub_audit: câu 'chênh lệch/thay đổi/biến động' không nêu hướng mà base trả ÂM -> abs. An toàn (giữ ô).
Chạy: PYTHONUTF8=1 python scripts/postprocess_sign.py sub_audit sub_sign
"""
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.evaluation.metrics import coerce_number

SRC = sys.argv[1] if len(sys.argv) > 1 else "sub_audit"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sub_sign"

# 'chênh lệch/thay đổi/biến động' = không hướng; loại trừ nếu có từ chỉ hướng rõ (giảm/tăng/hơn/kém/thấp/cao)
DIRLESS = re.compile(r"chênh lệch|biến động|thay đổi|tăng hay giảm|tăng giảm", re.I)
HASDIR = re.compile(r"giảm bao nhiêu|tăng bao nhiêu|cao hơn|thấp hơn|nhiều hơn|ít hơn|tăng trưởng", re.I)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))

n_fix = 0
for e in rows:
    q = e["question"]
    if not (DIRLESS.search(q) and not HASDIR.search(q)):
        continue
    av = coerce_number(e.get("answer"))
    code = (e.get("pandas_query") or "").strip()
    if av is None or av >= 0 or not code:
        continue
    e["answer"] = abs(av)
    e["pandas_query"] = code + "\nresult = abs(result)\n"     # grader chạy code -> ép dương
    n_fix += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"SIGN_FIX_chenh_lech_am={n_fix} -> {zp} {os.path.getsize(zp)}")
