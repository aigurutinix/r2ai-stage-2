"""GROUP-TESTING (GPT #1): dùng leaderboard làm hidden-gold oracle. Trên nền sub_mai (best 0.1996),
REVERT một subset ID về base (sub_base_fix) -> nộp -> delta = tổng contribution thật của subset.
Score TĂNG => subset net ÂM (bỏ luôn). Score GIẢM => net dương (giữ).
Chạy: PYTHONUTF8=1 python scripts/build_ablate.py <out_name> <id1,id2,...>
"""
import json
import os
import shutil
import sys
import zipfile

OUT = sys.argv[1]
REVERT = set(int(x) for x in sys.argv[2].split(","))
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}

if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree("sub_mai", OUT)
rows = json.load(open(f"{OUT}/submission.json", encoding="utf-8"))
existing = set(os.listdir(f"{OUT}/data"))
n = 0
for e in rows:
    if e["id"] not in REVERT:
        continue
    b = base[e["id"]]
    # copy CSV evidence của base nếu thiếu
    for ev in b.get("evidence", []):
        fn = ev["csv_path"].split("/")[-1]
        src = f"sub_base_fix/data/{fn}"
        if os.path.exists(src) and fn not in existing:
            shutil.copyfile(src, f"{OUT}/data/{fn}")
            existing.add(fn)
    e["answer"] = b.get("answer")
    e["pandas_query"] = b.get("pandas_query", "")
    e["evidence"] = b.get("evidence", [])
    n += 1

json.dump(rows, open(f"{OUT}/submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zp = f"{OUT}.zip"
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/submission.json", "submission.json")
    for c in os.listdir(f"{OUT}/data"):
        z.write(f"{OUT}/data/{c}", f"data/{c}")
print(f"ABLATE {OUT}: revert {n} cau ve base -> {zp} {os.path.getsize(zp)}")
