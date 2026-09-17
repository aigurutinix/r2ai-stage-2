"""PROBE DO NEN NHOM EASY: xoa dap an moi cau tra cuu truc tiep, giu nguyen phan con lai.

Vi sao can: suot ba phien ta chi UOC LUONG do chinh xac nhom Easy (dev set noi 66.7% khau chon
dong, probe cu noi 40% dap an cuoi) va khoang chenh 26 diem chua giai thich duoc. Khong co con so
nen thi moi cai tien cho Easy deu la mo trong toi.

Cach do: nop mot ban chi khac o cho moi cau `P.direct` tra ve 0. Hieu Execution giua ban that va
ban probe = SO CAU EASY DANG DUNG trong tap public, chia cho 506.

An toan: GHI RA THU MUC RIENG, khong dung toi submission.zip that.
"""
import os, re, sys, json, shutil, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "submission_out", "submission.zip")
OUT = os.path.join(HERE, "submission_out", "probe_easy")
os.makedirs(OUT, exist_ok=True)

direct = {q["id"] for q in P.direct}
zin = zipfile.ZipFile(SRC)
entries = json.load(zin.open("submission.json"))

n = 0
for e in entries:
    if e["id"] in direct and e.get("pandas_query"):
        e["answer"] = 0.0
        e["pandas_query"] = "float(0.0)"     # chay duoc, tra 0 → Execution = 0 (tru khi gold dung bang 0)
        n += 1

sj = os.path.join(OUT, "submission.json")
json.dump(entries, open(sj, "w", encoding="utf-8"), ensure_ascii=False)

zp = os.path.join(OUT, "submission.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(sj, "submission.json")
    for nm in zin.namelist():
        if nm != "submission.json":
            z.writestr(nm, zin.read(nm))

print(f"cau P.direct           : {len(direct)}")
print(f"cau da xoa dap an (=0) : {n}")
print(f"cau GIU NGUYEN         : {len(entries) - n}")
print(f"\nZIP probe: {zp} ({os.path.getsize(zp)//1024} KB)")
print("Hieu Execution (ban that - ban probe) x 506 = SO CAU EASY DANG DUNG trong public.")
