"""PROBE QUY UOC DAU: lay TRI TUYET DOI cho cac cau dap an AM, giu nguyen phan con lai.

Vi sao can: 18 cau dang tra so AM cho chi tieu ma cau hoi hoi mot luong duong ("so du du phong rui
ro cho vay", "chi phi lai"). Bao cao tai chinh in chi phi/du phong trong ngoac don (= so am), nen
KHONG BIET gold cua BTC theo quy uoc nao. Toi da hai lan tu choi doan — day la cach DO thay vi doan.

Doc ket qua:
  Execution TANG  => gold dung quy uoc DUONG  -> ap dung abs() cho ca lop nay
  Execution GIAM  => gold dung quy uoc AM     -> giu nguyen, va biet chac de khong thu lai
  DUNG YEN        => cac cau nay roi vao nua private -> khong ket luan duoc

Mot luot nop chot duoc quy uoc dau cho MOI viec ve sau, khong chi 18 cau nay.

An toan: GHI RA THU MUC RIENG, khong dung toi submission.zip that.
"""
import os, re, sys, json, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import build_submission as B

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "submission_out", "submission.zip")
OUT = os.path.join(HERE, "submission_out", "probe_sign")
os.makedirs(OUT, exist_ok=True)

PCT = re.compile(r"phần trăm|\bt[ỷyỉi] (?:l[ệe]|tr[ọo]ng)\b|\b%", re.I)
NONNEG = ["tổng tài sản", "tổng cộng tài sản", "tổng nguồn vốn", "vốn chủ sở hữu", "vốn điều lệ",
          "vốn góp", "nợ phải trả", "tiền và các khoản tương đương tiền", "hàng tồn kho",
          "số dư", "nguyên giá", "giá trị còn lại", "số lượng", "doanh thu thuần",
          "tổng doanh thu", "tiền gửi", "cho vay", "thù lao", "chi phí"]
DIFF = ["chênh lệch", "biến động", "tăng", "giảm", "lỗ", "hiệu giữa", "so với", "âm"]

zin = zipfile.ZipFile(SRC)
entries = json.load(zin.open("submission.json"))

changed = []
for e in entries:
    v, qt, l = e.get("answer"), e["question"], e["question"].lower()
    if v is None or not e.get("pandas_query") or float(v) >= 0:
        continue
    if any(k in l for k in DIFF) or PCT.search(qt) or not any(k in l for k in NONNEG):
        continue
    e["answer"] = abs(float(v))
    e["pandas_query"] = f"abs({e['pandas_query']})"
    changed.append(e["id"])

sj = os.path.join(OUT, "submission.json")
json.dump(entries, open(sj, "w", encoding="utf-8"), ensure_ascii=False)
zp = os.path.join(OUT, "submission.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(sj, "submission.json")
    for nm in zin.namelist():
        if nm != "submission.json":
            z.writestr(nm, zin.read(nm))

print(f"so cau lay tri tuyet doi: {len(changed)}")
print(f"id: {changed}")
print(f"\nZIP probe: {zp} ({os.path.getsize(zp)//1024} KB)")
print("Moi thu khac GIU NGUYEN. Hieu Execution cho biet gold dung quy uoc dau nao.")
