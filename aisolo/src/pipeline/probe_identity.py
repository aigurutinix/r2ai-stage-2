"""THAM DO: rang buoc DANG THUC KE TOAN co bat duoc loi chon dong o nhom Easy khong.

Y tuong: moi khoan muc thuoc BANG CAN DOI KE TOAN cua mot cong ty/nam KHONG THE lon hon TONG TAI
SAN cua chinh cong ty/nam do. Day la rang buoc AM CHAC, suy tu ke toan chu khong phai doan dap an —
cung ban chat voi audit_wrong.py nen cung dang tin.

Neu bat duoc nhieu ca thi day la cach mo rong Auditor sang nhom Easy (361 cau, 35.7% de thi) —
nhom ma bo dem hien tai gan nhu mu vi dap an tien luon "hop le ve kieu".
"""
import os, re, sys, json, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P
import build_submission as B

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "submission_out", "submission.json"), encoding="utf-8"))

# khoan muc BANG CAN DOI (khong phai ket qua kinh doanh / luu chuyen tien)
BS = ["tài sản", "nguồn vốn", "nợ phải trả", "vốn chủ sở hữu", "vốn điều lệ", "vốn góp",
      "hàng tồn kho", "tiền và các khoản tương đương tiền", "phải thu", "phải trả",
      "đầu tư", "tài sản cố định", "nguyên giá", "giá trị còn lại", "quỹ", "lợi nhuận chưa phân phối",
      "vay", "tiền gửi", "cho vay", "dự phòng", "trả trước", "chi phí xây dựng cơ bản"]
SKIP = P.AGG + P.COND + ["chênh lệch", "tăng trưởng", "so với năm", "tỷ lệ", "phần trăm", "bao nhiêu lần"]

_C = {}
def ing(p):
    if p not in _C:
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    return _C[p]

stat = collections.Counter()
viol = []
for e in E:
    qt = e["question"]; l = qt.lower()
    v = e.get("answer")
    if v is None or not e.get("pandas_query"):
        continue
    if any(k in l for k in SKIP) or not any(k in l for k in BS):
        continue
    tk = P.resolve(qt); yrs = sorted(set(re.findall(r"\b(20\d{2})\b", qt)))
    if not tk or len(yrs) != 1:
        continue
    fr = P.find_report(tk, yrs[0], P.doctype(qt))
    if not fr:
        continue
    rows, uf, src = ing(str(fr[0]))
    ta = next((r["cur"] for r in rows if r["cur"] is not None
               and ("tong cong tai san" in P.strip_vn(r["label"]) or r["ma"] == "270")), None)
    if not ta or ta <= 0:
        stat["khong tim thay tong tai san"] += 1
        continue
    stat["kiem duoc"] += 1
    vnd = abs(float(v)) * B.q_unit(qt)
    if vnd > abs(ta) * 1.02:                      # bien 2% cho lech lam tron/don vi
        viol.append((e["id"], vnd, ta, qt))

print(f"cau BANG CAN DOI kiem duoc bang rang buoc <= tong tai san: {stat['kiem duoc']}")
print(f"  khong tim thay tong tai san (bo qua)  : {stat['khong tim thay tong tai san']}")
print(f"  VI PHAM (dap an > tong tai san)       : {len(viol)}")
print(f"  => ty le vi pham: {100*len(viol)/max(1,stat['kiem duoc']):.1f}%\n")
for i, vnd, ta, qt in viol[:15]:
    print(f"  id{i:<5} dap an={vnd:.3e} VND  >  tong tai san={abs(ta):.3e}  (gap {vnd/abs(ta):.1f} lan)")
    print(f"         {qt[:96]}")
json.dump(sorted(i for i, *_ in viol), open(os.path.join(HERE, "identity_viol.json"), "w"))
