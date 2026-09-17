"""EDGE riêng: Mã số chỉ tiêu chuẩn Thông tư 200/2014/TT-BTC.

BCTC Việt Nam gán MÃ SỐ CỐ ĐỊNH cho từng chỉ tiêu (giống mọi công ty). Câu hỏi "doanh thu
thuần" -> Mã 10 -> tra thẳng dòng Mã số=10, KHÔNG để model đoán dòng (nguồn sai lớn nhất).
Đội quốc tế không có edge này (không biết kế toán VN).

Nguồn: TT200 (B01/B02/B03-DN), MISA/meinvoice/thuvienphapluat. Đẳng thức = kế toán chuẩn.
"""

from __future__ import annotations

import re
import unicodedata


def _norm(s: str) -> str:
    """Bỏ dấu + lowercase (để khớp tên chỉ tiêu bất kể dấu/viết hoa)."""
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


# {mã số: [các cụm từ khoá đã chuẩn hoá, khớp là ra mã]}. Xếp cụm ĐẶC THÙ trước.
_MASO = {
    # --- Kết quả kinh doanh B02-DN ---
    "10": ["doanh thu thuan"],
    "01": ["doanh thu ban hang va cung cap dich vu", "doanh thu ban hang"],
    "11": ["gia von hang ban", "gia von"],
    "20": ["loi nhuan gop", "lai gop", "loi nhuan gop ve ban hang"],
    "21": ["doanh thu hoat dong tai chinh", "doanh thu tai chinh"],
    "22": ["chi phi tai chinh"],
    "23": ["chi phi lai vay", "lai vay"],
    "25": ["chi phi ban hang"],
    "26": ["chi phi quan ly doanh nghiep", "chi phi quan ly"],
    "30": ["loi nhuan thuan tu hoat dong kinh doanh", "loi nhuan thuan"],
    "31": ["thu nhap khac"],
    "32": ["chi phi khac"],
    "40": ["loi nhuan khac"],
    "50": ["tong loi nhuan ke toan truoc thue", "loi nhuan truoc thue", "lai truoc thue"],
    "51": ["chi phi thue tndn hien hanh", "thue tndn hien hanh"],
    "52": ["chi phi thue tndn hoan lai", "thue tndn hoan lai"],
    "60": ["loi nhuan sau thue", "lai sau thue", "lnst", "loi nhuan rong", "lai rong"],
    "70": ["lai co ban tren co phieu", "lai co ban tren mot co phieu", "eps"],
    # --- Cân đối kế toán B01-DN ---
    "100": ["tai san ngan han"],
    "110": ["tien va cac khoan tuong duong tien", "tien va tuong duong tien"],
    "120": ["dau tu tai chinh ngan han"],
    "130": ["cac khoan phai thu ngan han", "phai thu ngan han"],
    "140": ["hang ton kho"],
    "200": ["tai san dai han"],
    "220": ["tai san co dinh"],
    "230": ["bat dong san dau tu"],
    "250": ["dau tu tai chinh dai han"],
    "270": ["tong cong tai san", "tong tai san"],
    "300": ["no phai tra"],
    "310": ["no ngan han"],
    "320": ["vay va no thue tai chinh ngan han", "vay ngan han"],
    "330": ["no dai han"],
    "338": ["vay va no thue tai chinh dai han", "vay dai han"],
    "400": ["von chu so huu", "von chu", "vcsh"],
    "411": ["von gop cua chu so huu", "von co phan", "von dieu le"],
    "421": ["loi nhuan sau thue chua phan phoi", "lnst chua phan phoi"],
    "440": ["tong cong nguon von", "tong nguon von"],
    # --- Lưu chuyển tiền tệ B03-DN ---
    "cf20": ["luu chuyen tien thuan tu hoat dong kinh doanh", "dong tien tu hoat dong kinh doanh", "cfo"],
    "cf30": ["luu chuyen tien thuan tu hoat dong dau tu", "dong tien tu hoat dong dau tu"],
    "cf40": ["luu chuyen tien thuan tu hoat dong tai chinh", "dong tien tu hoat dong tai chinh"],
}

# Tỷ số tài chính -> công thức bằng mã số (để gợi ý cách tính, chống bịa công thức).
_RATIO = {
    "bien loi nhuan gop": "Mã20 / Mã10",
    "bien loi nhuan rong": "Mã60 / Mã10",
    "ros": "Mã60 / Mã10",
    "roa": "Mã60 / Mã270 (hoặc TS bình quân)",
    "roe": "Mã60 / Mã400 (hoặc VCSH bình quân)",
    "he so no": "Mã300 / Mã270",
    "d/e": "Mã300 / Mã400",
    "no tren von chu": "Mã300 / Mã400",
    "thanh toan hien hanh": "Mã100 / Mã310",
    "thanh toan nhanh": "(Mã100 - Mã140) / Mã310",
    "thanh toan tien mat": "Mã110 / Mã310",
    "vong quay hang ton kho": "Mã11 / Mã140 (HTK bình quân)",
}


# Đẳng thức kế toán chuẩn: {mã đích: (mô tả suy, [mã thành phần])}. Suy khi bảng thiếu dòng đích.
_DERIVE = {
    "10": ("Doanh thu thuần = DT bán hàng (01) − Các khoản giảm trừ (02)", ["01", "02"]),
    "20": ("Lợi nhuận gộp = Doanh thu thuần (10) − Giá vốn (11)", ["10", "11"]),
    "50": ("LN trước thuế = LN thuần HĐKD (30) + LN khác (40)", ["30", "40"]),
    "60": ("LN sau thuế = LN trước thuế (50) − Thuế TNDN (51+52)", ["50", "51"]),
    "270": ("Tổng tài sản = TS ngắn hạn (100) + TS dài hạn (200)", ["100", "200"]),
    "440": ("Tổng nguồn vốn = Nợ phải trả (300) + Vốn CSH (400)", ["300", "400"]),
}

# Đẳng thức để KIỂM (verify) tính nhất quán các số đã trích. (a, [b,c]) nghĩa a = tổng/hiệu của b,c.
_IDENTITIES = [
    ("270", ["100", "200"], "cong"),      # Tổng TS = NH + DH
    ("440", ["300", "400"], "cong"),      # Tổng NV = Nợ + VCSH
    ("20", ["10", "11"], "hieu"),         # LN gộp = DT thuần − Giá vốn
    ("50", ["30", "40"], "cong"),         # LN trước thuế = LN thuần + LN khác
]


def verify_identities(vals: dict) -> list[str]:
    """vals={mã: số}. Trả danh sách đẳng thức BỊ LỆCH >1% (dấu hiệu trích sai). Rỗng = ổn."""
    bad = []
    for a, comps, op in _IDENTITIES:
        if a in vals and all(c in vals for c in comps):
            got = vals[comps[0]] + (vals[comps[1]] if op == "cong" else -vals[comps[1]])
            if abs(got - vals[a]) > abs(vals[a]) * 0.01 + 1e-6:
                bad.append(f"Mã{a}({vals[a]:g}) ≠ {op} {comps}({got:g})")
    return bad


def maso_of(question: str) -> tuple[str, str] | None:
    """Trả (mã_số, tên_chỉ_tiêu_khớp) nếu câu hỏi hỏi một chỉ tiêu chuẩn. Khớp cụm DÀI nhất."""
    q = _norm(question)
    best = None
    for code, phrases in _MASO.items():
        for p in phrases:
            if p in q and (best is None or len(p) > best[2]):
                best = (code, p, len(p))
    if best:
        return best[0].replace("cf", ""), best[1]
    return None


def ratio_of(question: str) -> str | None:
    """Trả công thức tỷ số theo mã số nếu câu hỏi hỏi một tỷ số chuẩn."""
    q = _norm(question)
    for name, formula in _RATIO.items():
        if name in q:
            return f"{name} = {formula}"
    return None


def maso_hint(question: str) -> str:
    """Gợi ý NEO cho prompt: chỉ tiêu -> mã số chuẩn -> tra đúng dòng (chống nhầm dòng)."""
    parts = []
    m = maso_of(question)
    if m:
        code, name = m
        parts.append(
            f"NEO MÃ SỐ (Thông tư 200): chỉ tiêu '{name}' có Mã số chuẩn = **{code}**. "
            f"Nếu bảng có cột 'Mã số', LỌC đúng dòng có Mã số = '{code}' "
            f"(vd df1[df1['<cột mã số>'].astype(str).str.strip()=='{code}']) rồi lấy giá trị cột năm câu hỏi. "
            f"Cách này chắc hơn dò theo tên chỉ tiêu."
        )
        if code in _DERIVE:                                # có thể SUY nếu bảng thiếu dòng đích
            parts.append(f"Nếu bảng KHÔNG có dòng Mã {code}, SUY bằng: {_DERIVE[code][0]}.")
    r = ratio_of(question)
    if r:
        parts.append(f"CÔNG THỨC TỶ SỐ chuẩn: {r} (lấy từng thành phần theo mã số rồi tính).")
    return ("\n" + "\n".join(parts)) if parts else ""
