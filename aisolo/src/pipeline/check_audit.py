"""KIEM CHAT hon cong Auditor: dap an moi co hop ly voi CHINH cau hoi do khong.

Cong dang dung chi hoi "co dung KIEU khong" (dem thi phai la so nguyen nho, ty le thi trong
[-300,300]...). Day la lop kiem thu hai, rang buoc sat hon va van gold-free:
  - hoi dem trong nhom N ma chung khoan -> dap an khong the LON HON N
  - bien loi nhuan / ty le so huu / ty trong -> nam trong [-100, 100]
  - he so thanh toan, D/E, vong quay -> |v| khong qua 50
  - dap an dung bang 0 -> dang ngo, doi soi tay
Ket qua chi de NGUOI doc quyet dinh, khong tu dong loai bo.
"""
import os, re, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "agent_audit.json"), encoding="utf-8"))
E = {e["id"]: e for e in json.load(open(os.path.join(HERE, "submission_out", "submission.json"), encoding="utf-8"))}
QT = {q["id"]: q["question"] for q in P.Q}

MARGIN = ["biên lợi nhuận", "biên ln", "tỷ suất lợi nhuận", "tỷ lệ sở hữu", "tỷ lệ biểu quyết",
          "tỷ trọng", "tỉ trọng", "tỷ lệ bao phủ", "tỷ lệ dự phòng"]
COEF = ["hệ số", "khả năng thanh toán", "vòng quay", "d/e", "bao nhiêu lần"]

ok, susp = [], []
for x in D:
    v = x.get("answer")
    if v is None:
        continue
    qt = QT[x["id"]]; l = qt.lower(); v = float(v)
    why = []
    # hoi dem trong mot NHOM ma CK -> khong the vuot so ma duoc liet ke
    if re.search(r"(bao nhiêu|có mấy)\s+(doanh nghiệp|công ty|ngân hàng)", l):
        n = len(set(m for m in re.findall(r"\b([A-Z0-9]{2,4})\b", qt) if m in P.tickers))
        if n and v > n:
            why.append(f"dem {v:g} > {n} ma CK duoc liet ke")
        if abs(v - round(v)) > 1e-9:
            why.append("dem khong nguyen")
    if any(k in l for k in MARGIN) and not (-100 <= v <= 100):
        why.append(f"ty le/bien {v:g} ngoai [-100,100]")
    if any(k in l for k in COEF) and abs(v) > 50:
        why.append(f"he so {v:g} qua lon")
    if v == 0.0:
        why.append("dap an = 0, dang ngo")
    if not x.get("refs"):
        why.append("khong co refs (bang chung rong)")
    (susp if why else ok).append((x["id"], v, why, qt))

print(f"Dap an Auditor DA CHAP NHAN: {len(ok)+len(susp)}")
print(f"  qua duoc lop kiem chat : {len(ok)}")
print(f"  DANG NGO               : {len(susp)}\n")
for i, v, why, qt in susp:
    print(f"  id{i:<5} = {v:<14g} | {'; '.join(why)}")
    print(f"         {qt[:96]}")
print("\n--- mau 10 cau qua duoc lop kiem chat ---")
for i, v, _, qt in ok[:10]:
    print(f"  id{i:<5} cu={str(E[i]['answer'])[:18]:<20} -> {v:<12g} | {qt[:76]}")
