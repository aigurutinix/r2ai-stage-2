"""ĐẾM ĐÁP ÁN SAI CHẮC CHẮN — gold-free, deterministic, không dùng LLM.

Mỗi câu bị gắn cờ là SAI 100% suy từ chính câu hỏi (miền giá trị hợp lệ của đáp án). Dùng làm
thước đo chấp nhận cho mọi thay đổi: số câu bị gắn cờ phải GIẢM. Không thay thế leaderboard nhưng
là thước đo duy nhất không bị lừa bởi thiên lệch LLM.

Mốc đã đo (bản 0.247, ngày 12/08/2026): 151/1012 = 14.9%.
"""
import os, re, sys, json, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VIFINQA_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_vifinqa"))
import pipeline as P
import build_submission as B

# --- luật bổ sung: vẫn chỉ phát biểu điều SUY ĐƯỢC TỪ CÂU HỎI, không đoán đáp án đúng ---
OWN_RATIO = ["tỷ lệ sở hữu", "tỉ lệ sở hữu", "tỷ lệ biểu quyết", "tỷ lệ lợi ích",
             "tỷ lệ nắm giữ", "tỷ lệ vốn góp", "quyền biểu quyết"]
PART_WHOLE = ["tỷ trọng", "tỉ trọng"]
CHANGE = ["tăng", "giảm", "chênh lệch", "tăng trưởng", "so với", "biến động", "thay đổi"]
# khoản mục BẢNG CÂN ĐỐI: không thể lớn hơn TỔNG TÀI SẢN của chính công ty/năm đó
BS_ITEM = ["tài sản", "nguồn vốn", "nợ phải trả", "vốn chủ sở hữu", "vốn điều lệ", "vốn góp",
           "hàng tồn kho", "tiền và các khoản tương đương tiền", "phải thu", "phải trả",
           "tài sản cố định", "nguyên giá", "giá trị còn lại", "lợi nhuận chưa phân phối",
           "tiền gửi", "cho vay", "trả trước", "chi phí xây dựng cơ bản"]

_C = {}
def _total_assets(qt):
    """Tổng tài sản (VND) của công ty/năm trong câu hỏi, None nếu không xác định được."""
    tk = P.resolve(qt); yrs = sorted(set(re.findall(r"\b(20\d{2})\b", qt)))
    if not tk or len(yrs) != 1:
        return None
    fr = P.find_report(tk, yrs[0], P.doctype(qt))
    if not fr:
        return None
    p = str(fr[0])
    if p not in _C:
        _C[p] = P.ingest(open(p, encoding="utf-8", errors="replace").read())
    rows, uf, src = _C[p]
    ta = next((r["cur"] for r in rows if r["cur"] is not None
               and ("tong cong tai san" in P.strip_vn(r["label"]) or r["ma"] == "270")), None)
    return abs(ta) if ta else None

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "submission_out", "submission.json"), encoding="utf-8"))

# BUG CŨ: `t[ỉi] tr[ọo]ng` khớp "tỉ trọng" nhưng KHÔNG khớp "tỷ trọng" (ỷ khác ỉ) → 25 câu hỏi
# phần trăm chưa bao giờ được kiểm, trong đó 6 câu trả về số tiền hàng nghìn tỷ.
PCT = re.compile(r"phần trăm|\bt[ỷyỉi] (?:l[ệe]|tr[ọo]ng)\b|\b%", re.I)
MON = re.compile(r"tỷ đồng|triệu đồng|nghìn đồng|ngàn đồng|\bvnđ\b|\bđồng\b", re.I)
NONNEG = ["tổng tài sản", "tổng cộng tài sản", "tổng nguồn vốn", "vốn chủ sở hữu", "vốn điều lệ",
          "vốn góp", "nợ phải trả", "tiền và các khoản tương đương tiền", "hàng tồn kho",
          "số dư", "nguyên giá", "giá trị còn lại", "số lượng", "doanh thu thuần",
          "tổng doanh thu", "tiền gửi", "cho vay", "thù lao", "chi phí"]
DIFF = ["chênh lệch", "biến động", "tăng", "giảm", "lỗ", "hiệu giữa", "so với", "âm"]

flag = collections.defaultdict(list)
for e in E:
    qt, l, v = e["question"], e["question"].lower(), e.get("answer")
    if not e.get("pandas_query"):
        flag["KHONG co pandas (chac chan 0 diem)"].append(e["id"]); continue
    if v is None:
        continue
    v = float(v)
    if re.search(r"\bnăm nào\b", l) and not (2014 <= v <= 2026):
        flag["hoi NAM NAO ma tra khong phai nam"].append(e["id"])
    # "bao nhiêu NĂM" / "số năm mà..." cũng là phép ĐẾM — trước đây lọt lưới (id383 trả −730 tỷ cho
    # câu hỏi có bao nhiêu năm). Corpus chỉ trải ~10 năm nên đếm hợp lệ nằm trong 0..30.
    if re.search(r"(bao nhiêu|có mấy)\s+(doanh nghiệp|công ty|ngân hàng|mã|năm)|\bsố năm\b", l) and not (
            0 <= v <= 30 and abs(v - round(v)) < 1e-6):
        flag["hoi DEM (cong ty / nam) ma tra khong phai so nguyen nho"].append(e["id"])
    # câu "năm nào" có thể chứa chữ "tỷ trọng" nhưng đáp án là NĂM → luật % không được áp
    if PCT.search(qt) and not re.search(r"\bnăm nào\b", l) and not (-300 <= v <= 300):
        flag["hoi PHAN TRAM ma tra ngoai [-300,300]"].append(e["id"])
    if re.search(r"bao nhiêu lần\b", l) and not (-200 <= v <= 200):
        flag["hoi SO LAN ma tra ngoai [-200,200]"].append(e["id"])
    if MON.search(qt) and not PCT.search(qt):
        vnd = abs(v) * B.q_unit(qt)
        if 0 < vnd < 1e5:
            flag["TIEN qua nho (<100 nghin dong)"].append(e["id"])
        elif vnd > 2e16:
            flag["TIEN qua lon (>20 trieu ty dong)"].append(e["id"])
    if v < 0 and not any(k in l for k in DIFF) and any(k in l for k in NONNEG) and not PCT.search(qt):
        flag["DAP AN AM cho chi tieu khong the am"].append(e["id"])
    if v == 0.0:
        flag["dap an = 0 (fallback)"].append(e["id"])
    # --- luật mới ---
    if any(k in l for k in OWN_RATIO) and not any(k in l for k in CHANGE) and not (0 <= v <= 100):
        flag["TY LE SO HUU/BIEU QUYET ngoai [0,100]"].append(e["id"])
    # ĐÃ BỎ hai luật vì KHÔNG đứng vững, dù nghe hợp lý (giữ ghi chú để không thử lại):
    #  - "tỷ trọng phần/tổng thể phải trong [-100,100]": SAI khi mẫu số là số THUẦN đã trừ dự phòng,
    #    lúc đó một thành phần vượt 100% là hợp lệ (id688 100.57%, id697 113.84%).
    #  - "số đếm không vượt số mã CK liệt kê": SAI khi câu hỏi gọi tên công ty bằng chữ thay vì mã
    #    (id413 "trong số 5 doanh nghiệp ... MSN, Vinamilk..." trả 5 là hợp lý, bị gắn cờ oan).
    if (MON.search(qt) and not PCT.search(qt) and not any(k in l for k in CHANGE)
            and any(k in l for k in BS_ITEM)):
        ta = _total_assets(qt)
        if ta and abs(v) * B.q_unit(qt) > ta * 1.02:
            flag["KHOAN MUC BCD lon hon TONG TAI SAN"].append(e["id"])

print(f"{'lop loi':<48} {'so cau':>7}")
print("-" * 58)
allids = set()
for k in sorted(flag, key=lambda x: -len(flag[x])):
    print(f"{k:<48} {len(flag[k]):>7}")
    allids |= set(flag[k])
print("-" * 58)
print(f"{'TONG (khong trung lap)':<48} {len(allids):>7}")
print(f"\n= {100*len(allids)/len(E):.1f}% cua {len(E)} | uoc tinh ~{len(allids)//2} cau trong public 506")
print(f"MOC CU (ban 0.247): 151  ->  BAY GIO: {len(allids)}  ({len(allids)-151:+d})")
json.dump(sorted(allids), open(os.path.join(HERE, "wrong_ids.json"), "w"))
