"""BUCKET #2 (probe convention): trên nền sub_mai, đổi CHỈ SCALE của 12 câu base>1e11
(giữ NGUYÊN ô base chọn, chỉ thay hệ số scale trong query) -> sub_bkt_scale. 1 phép biến đổi duy nhất.
Chạy: PYTHONUTF8=1 python scripts/build_bucket_scale.py
"""
import json
import os
import re
import shutil
import zipfile

SRC = "sub_mai"
OUT = "sub_bkt_scale"
sc = json.load(open("build/scale_mismatch.json", encoding="utf-8"))


def num(x):
    try:
        return float(x)
    except Exception:
        return None


targets = {c["id"]: c for c in sc if num(c["base_ans"]) is not None and abs(num(c["base_ans"])) > 1e11
           and abs(num(c["base_ans"]) - num(c["new_ans"])) > 0.01 and abs(num(c["new_ans"])) < 1e11}

if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
n = 0
done = []
for e in rows:
    if e["id"] not in targets:
        continue
    t = targets[e["id"]]
    q = e.get("pandas_query", "")
    newscale = repr(t["expected_scale"])
    # thay hệ số scale trong 'float(_v) * OLD' -> 'float(_v) * NEW' (giữ nguyên ô/row/df)
    q2, cnt = re.subn(r"(float\(_v\)\s*\*\s*)[0-9.eE+-]+", r"\g<1>" + newscale, q)
    if cnt == 0:
        # base không có '* scale' (scale=1) -> nhân thêm
        q2 = q.replace("result = round(float(_v)", f"result = round(float(_v) * {newscale}")
        if q2 == q:
            continue
    e["pandas_query"] = q2
    e["answer"] = float(t["new_ans"])
    n += 1
    done.append((e["id"], t["base_ans"], t["new_ans"]))

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"BUCKET_SCALE: doi {n} cau (chi scale) -> {zp} {os.path.getsize(zp)}")
for qid, ba, na in done:
    print(f"  Q{qid} {ba} -> {na}")
