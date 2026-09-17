"""Pipeline OPERAND gần-tất-định (theo phản biện GPT): thay vì bắt 14B sinh code phức tạp,
tách thành: giải CỘT-NĂM tất định + tìm DÒNG lexical (exact-phrase + IDF) + trích ô + compiler.
LLM chỉ CHỌN ô từ candidate (làm sau). File này lo phần TẤT ĐỊNH + test grounding "p" offline.

Cấu trúc bảng thật (pd.read_html -> read_csv header='infer'): cột tên '0','1',...; df.iloc[0]=
HEADER (nhãn cột: 'Mã số', '2017 VND', '31/12/2015'); có thể iloc[1]=section header; data từ sau đó.
Số kiểu Việt, âm dùng ngoặc, đơn vị ghi trong header cột ('VND','Triệu VND').
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_UNITS = [("nghin ty", 1_000_000_000_000), ("nghìn tỷ", 1_000_000_000_000),
          ("ty dong", 1_000_000_000), ("tỷ", 1_000_000_000),
          ("trieu", 1_000_000), ("triệu", 1_000_000),
          ("nghin", 1_000), ("nghìn", 1_000)]

_STOP = set(("la bao nhieu bao nhieu cua nam vao ngay tai thoi diem cuoi dau ky cho biet gia tri "
             "cong ty me tap doan ctcp co phan ngan hang tmcp la tren duoi khoang bao gom va cac "
             "theo tinh den ket thuc dong trieu ty nghin vnd").split())


def _strip_accents(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")   # 'đ' KHÔNG phải combining mark -> phải map tay
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    s = _strip_accents(str(s).lower())
    s = re.sub(r"[^a-z0-9%/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if t and t not in _STOP and len(t) > 1]


def _num(x) -> float | None:
    # OCR can concatenate current/prior-year values in one cell, for example
    # ``(72.193.585.614)(27.471.160.925)``.  The selected statement column is
    # the first value; joining every digit would create a gigantic fake value.
    text = str(x).strip()
    match = re.match(r"\s*(\(?-?\d[\d.,]*\)?%?)", text)
    if not match:
        return None
    s = "".join(ch for ch in match.group(1) if ch in "0123456789.,-()%").replace("%", "")
    if s in ("", "-", "."):
        return None
    neg = ("(" in s) or s.startswith("-")
    s = s.replace("(", "").replace(")", "").replace("-", "").replace(".", "").replace(",", ".")
    if s in ("", "."):
        return None
    try:
        r = float(s)
    except ValueError:
        return None
    return -r if neg else r


_REQ_UNITS = [("nghin ty", 1_000_000_000_000), ("tram ty", 100_000_000_000),
              ("ty", 1_000_000_000), ("trieu", 1_000_000), ("nghin", 1_000)]


def requested_unit(question: str):
    """Đơn vị câu hỏi yêu cầu. PHẢI là '<đơn vị> đồng/vnd' liền nhau — tránh bắt nhầm 'ty' trong 'công ty'."""
    q = _norm(question)
    for name, mult in _REQ_UNITS:                 # dài trước (nghìn tỷ > tỷ)
        if re.search(rf"\b{name}\s*(dong|vnd)\b", q):
            return mult
    return 1  # đồng


def _col_unit(sig: str) -> int:
    n = _norm(sig)
    if "trieu" in n:
        return 1_000_000
    if "ty" in n:
        return 1_000_000_000
    if "nghin" in n:
        return 1_000
    return 1  # VND / không ghi


def resolve_columns(df, n_header: int = 2) -> dict:
    """Mỗi cột -> {years:set, dates:[(d,m,y)], unit_mult, is_maso, is_label, sig}."""
    ncol = df.shape[1]
    nrow = len(df)
    out = {}
    for j in range(ncol):
        sig = " | ".join(str(df.iloc[i, j]) for i in range(min(n_header, nrow)))
        years = set(re.findall(r"20\d{2}", sig))                # KHÔNG \b: năm hay dính 'VND'/'Triệu'
        dates = re.findall(r"(\d{1,2})/(\d{1,2})/(20\d{2})", sig)
        nn = _norm(sig)
        # cột năm TƯƠNG ĐỐI (không ghi năm): 'năm nay'/'cuối năm/kỳ' = kỳ hiện tại; 'năm trước'/'đầu năm/kỳ' = kỳ trước
        rel = None
        if not years:
            if re.search(r"nam nay|cuoi nam|cuoi ky|ky nay| cuoi$|^cuoi ", nn):
                rel = "cur"
            elif re.search(r"nam truoc|dau nam|dau ky|ky truoc| dau$|^dau ", nn):
                rel = "prev"
        is_maso = ("ma so" in nn or nn.strip() == "ma" or "mã" in sig.lower())
        # cột nhãn: header rỗng/'chỉ tiêu', và bên dưới chủ yếu là CHỮ (không phải số)
        col_vals = [str(df.iloc[i, j]) for i in range(min(n_header, nrow), nrow)]
        n_textual = sum(1 for v in col_vals if _num(v) is None and len(_norm(v)) > 3)
        is_label = ("chi tieu" in nn or nn == "") and n_textual >= max(1, len(col_vals) // 2)
        out[j] = {"years": years, "dates": dates, "unit_mult": _col_unit(sig),
                  "is_maso": is_maso, "is_label": is_label, "sig": sig,
                  "rel": rel, "n_textual": n_textual}
    return out


def label_column(df, cols) -> int:
    lbls = [j for j, c in cols.items() if c["is_label"]]
    if lbls:
        return min(lbls)
    # fallback: cột nhiều chữ nhất
    return max(cols, key=lambda j: cols[j]["n_textual"])


def maso_column(df, cols):
    for j, c in cols.items():
        if c["is_maso"]:
            return j
    # fallback: cột có nhiều mã 2-3 chữ số
    best, bj = 0, None
    for j in range(df.shape[1]):
        cnt = sum(1 for i in range(len(df)) if re.fullmatch(r"\d{2,3}", str(df.iloc[i, j]).strip()))
        if cnt > best:
            best, bj = cnt, j
    return bj if best >= 3 else None


def year_column(cols, year: str, label_col: int, maso_col, want_dau: bool = False, report_year=None):
    """Cột ứng với năm hỏi. Ưu tiên date 31/12/year (cuối kỳ) / 01/01/year (đầu kỳ),
    rồi năm trong header, rồi cột TƯƠNG ĐỐI ('năm nay'/'năm trước') theo report_year."""
    year = str(year)
    val_cols = [j for j, c in cols.items() if j != label_col and j != maso_col and not c["is_maso"]]
    # 1) khớp theo date rõ ràng
    for j in val_cols:
        for d, m, y in cols[j]["dates"]:
            if want_dau and d in ("1", "01") and m in ("1", "01") and y == year:
                return j
            if not want_dau and d in ("31", "30") and m == "12" and y == year:
                return j
    # 2) khớp theo năm trong header
    cand = [j for j in val_cols if year in cols[j]["years"]]
    if len(cand) == 1:
        return cand[0]
    if cand:
        return cand[0] if not want_dau else cand[-1]
    # 3) cột tương đối 'năm nay/năm trước' theo report_year
    if report_year is not None:
        ry = int(report_year)
        want_rel = "cur" if int(year) >= ry else "prev"
        if want_dau:                                     # đầu kỳ = kỳ trước
            want_rel = "prev"
        relc = [j for j in val_cols if cols[j].get("rel") == want_rel]
        if relc:
            return relc[0]
    # 4) chỉ 1 cột giá trị -> lấy luôn
    if len(val_cols) == 1:
        return val_cols[0]
    return None


def first_data_row(df, n_header: int = 1) -> int:
    """Bỏ header + dòng section (mọi cột giống hệt nhau)."""
    for i in range(n_header, len(df)):
        row = [str(df.iloc[i, j]).strip() for j in range(df.shape[1])]
        nonempty = [v for v in row if v]
        if len(set(nonempty)) <= 1 and len(nonempty) >= 2:   # section header lặp
            continue
        return i
    return n_header


def _ngrams(s: str, n: int = 3) -> set:
    s = _norm(s)
    return set(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}


def _row_signature(df, i: int, label_col: int) -> str:
    """Chữ ký dòng GIÀU NGỮ CẢNH (theo GPT): mọi ô CHỮ của dòng + tiêu đề mục gần nhất phía trên.
    Cứu note-level ('Thương mại' trần vô nghĩa -> 'Dư nợ cho vay theo ngành > Thương mại') và OCR lệch cột."""
    cells = [str(df.iloc[i, j]) for j in range(df.shape[1])]
    texts = [c for c in cells if _num(c) is None and len(_norm(c)) > 2]
    sig = " ".join(texts)
    for k in range(i - 1, max(-1, i - 6), -1):         # tiêu đề mục = dòng chữ, ít cột phân biệt
        row = [str(df.iloc[k, j]) for j in range(df.shape[1])]
        txt = [c for c in row if _num(c) is None and len(_norm(c)) > 3]
        nonempty = [c for c in row if c.strip()]
        if txt and len(set(nonempty)) <= 2:
            sig = " ".join(txt) + " > " + sig
            break
    return sig


# Từ điển BÍ DANH (TT200) — phá paraphrase (research: Chen FinQA + chuẩn TT200/2014/TT-BTC).
# canonical (đã bỏ dấu) -> các cách gọi. Bồi dần theo câu sai thật.
_ALIAS = {
    "loi nhuan sau thue thu nhap doanh nghiep": ["lnst", "lai rong", "loi nhuan rong", "loi nhuan sau thue", "lai sau thue"],
    "loi nhuan gop ve ban hang va cung cap dich vu": ["loi nhuan gop", "lai gop", "loi nhuan gop ban hang"],
    "doanh thu thuan ve ban hang va cung cap dich vu": ["doanh thu thuan", "doanh thu thuan ban hang"],
    "gia von hang ban": ["gia von", "gia von hang ban"],
    "loi nhuan thuan tu hoat dong kinh doanh": ["loi nhuan thuan hdkd", "lai thuan tu kinh doanh"],
    "tong cong tai san": ["tong tai san", "tong cong tai san"],
    "tong cong nguon von": ["tong nguon von", "tong cong nguon von"],
    "no phai tra": ["tong no phai tra", "tong no"],
    "loi nhuan ke toan truoc thue": ["loi nhuan truoc thue", "lai truoc thue"],
    "loi nhuan sau thue chua phan phoi": ["lndp", "loi nhuan chua phan phoi", "loi nhuan giu lai"],
    "von chu so huu": ["von chu so huu", "vcsh", "tong von chu so huu"],
}


def _alias_target(qn: str):
    """Nếu câu hỏi khớp một BÍ DANH -> trả canonical (để khớp nhãn dòng chuẩn). None nếu không."""
    for canon, al in _ALIAS.items():
        for a in al:
            if re.search(rf"\b{re.escape(a)}\b", qn):
                return canon
    return None


def _row_has_number(df, i) -> bool:
    for j in range(df.shape[1]):
        if _num(df.iloc[i, j]) is not None:
            return True
    return False


def score_rows(df, query: str, label_col: int, idf: dict, start: int, topk: int = 8):
    """Chấm mọi dòng theo CHỮ KÝ giàu ngữ cảnh: token-IDF + exact-substring bonus + char-ngram (OCR)."""
    qn = _norm(query)
    qtok = set(_tokens(query))
    qgr = _ngrams(qn)
    # KỲ cuối/đầu trong nhãn dòng (vd 'Thuế... cuối năm' vs '... đầu năm'): khớp theo câu hỏi
    q_cuoi = bool(re.search(r"cuoi nam|cuoi ky|cuoi quy|cuoi thang|so cuoi|31/12", qn))
    q_dau = bool(re.search(r"dau nam|dau ky|dau quy|dau thang|so dau|01/01|1/1", qn))
    wants_total = bool(re.search(r"\btong\b|\btong cong\b|toan bo|tong so", qn))
    alias_canon = _alias_target(qn)                       # phá paraphrase (TT200)
    ranked = []
    for i in range(start, len(df)):
        lbl = str(df.iloc[i, label_col])
        sig = _row_signature(df, i, label_col)
        sn = _norm(sig)
        if len(sn) < 2:
            continue
        if not _row_has_number(df, i):                    # dòng chỉ có nhãn, trống số = tiêu đề mục -> loại
            continue
        ltok = set(_tokens(sig))
        if not ltok:
            continue
        inter = qtok & ltok
        idf_score = sum(idf.get(t, 1.0) for t in inter)
        cover = len(inter) / max(1, len(ltok))
        s = idf_score + 2.0 * cover
        ln = _norm(lbl)
        if ln and ln in qn:                            # nhãn là chuỗi con của câu -> thưởng lớn
            s += 8.0 + len(_tokens(lbl))
        elif qtok and set(_tokens(lbl)) <= qtok and _tokens(lbl):
            s += 3.0
        lgr = _ngrams(sn)                              # char-ngram: OCR, viết dính, biến thể
        s += 3.0 * len(qgr & lgr) / max(1, len(qgr | lgr))
        s += 0.3 * SequenceMatcher(None, qn, ln).ratio()
        # KỲ cuối/đầu: câu hỏi 'cuối năm' mà nhãn ghi 'đầu năm' -> phạt (vd Q46), khớp -> thưởng
        l_cuoi = bool(re.search(r"cuoi (nam|ky|quy|thang)", sn))
        l_dau = bool(re.search(r"dau (nam|ky|quy|thang)", sn))
        if q_cuoi and l_dau and not l_cuoi:
            s -= 4.0
        if q_dau and l_cuoi and not l_dau:
            s -= 4.0
        if (q_cuoi and l_cuoi) or (q_dau and l_dau):
            s += 2.5
        if re.search(r"\btong\b", qn) and re.search(r"tong cong|tong$|^tong ", ln):
            s += 2.0
        # BÍ DANH TT200: nhãn dòng khớp câu-chuẩn của bí danh câu hỏi -> thưởng mạnh (phá paraphrase)
        if alias_canon:
            ratio = SequenceMatcher(None, alias_canon, ln).ratio()
            atok = set(alias_canon.split())
            if ratio > 0.85 or (atok and atok <= set(ln.split())):
                s += 6.0
            elif ratio > 0.6:
                s += 2.5 * ratio
        # HẠ dòng-tổng khi câu hỏi KHÔNG hỏi tổng (research #4: total-vs-detail)
        if not wants_total and re.search(r"^(cong|tong cong|tong)\b|luy ke|cong phat sinh", ln):
            s -= 2.0
        ranked.append((s, i, lbl))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:topk]


def reconciled_total_row(df, cols, lc, mc, value_col):
    """ĐỐI SOÁT SỐ (numeric reconciliation): dòng có giá trị = Σ các dòng chi tiết còn lại ->
    đó là dòng TỔNG. Cứu note-table mà dòng tổng nhãn rỗng/'Cộng' (vd Q14/Q47). None nếu không rõ."""
    start = first_data_row(df)
    vals = {}
    for i in range(start, len(df)):
        v = _num(df.iloc[i, value_col])
        if v is not None and v != 0:
            vals[i] = v
    if len(vals) < 3:
        return None
    idxs = list(vals.keys())
    best = None
    for i in idxs:
        others = [vals[j] for j in idxs if j != i]
        # tổng của TẤT CẢ dòng khác (note 1 mục) — dung sai 0.5%
        s = sum(others)
        if s != 0 and abs(vals[i] - s) <= 0.005 * abs(s):
            if best is None:
                best = i
            else:
                return None       # >1 dòng khớp -> mơ hồ, bỏ
    return best


_NUM_SRC = (
    "def _num(x):\n"
    "    s = ''.join(ch for ch in str(x) if ch in '0123456789.,-()%')\n"
    "    s = s.replace('%', '')\n"
    "    if s == '' or s == '-' or s == '.': return None\n"
    "    neg = ('(' in s) or s.startswith('-')\n"
    "    s = s.replace('(', '').replace(')', '').replace('-', '').replace('.', '').replace(',', '.')\n"
    "    if s == '' or s == '.': return None\n"
    "    return -float(s) if neg else float(s)\n"
)


def deterministic_answer(question: str, tables: list[dict], idf: dict | None = None) -> dict | None:
    """Sinh đáp án TẤT ĐỊNH (không LLM) + pandas_query compiler cho câu lookup đơn.
    tables = [{'table_ref','csv_path'}]. Trả None nếu không đủ tự tin. conf: 3=mã số exact, 2/1=lexical."""
    import pandas as pd
    req = requested_unit(question)
    ys = re.findall(r"\b(20\d{2})\b", question)
    year = ys[0] if ys else None
    want_dau = bool(re.search(r"đầu năm|đầu kỳ|01/01", question, re.I))
    try:
        from kingpro.answering.ma_so_tt200 import maso_of
        mm = maso_of(question)
    except Exception:
        mm = None
    best = None
    for t in tables:
        try:
            df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        cols = resolve_columns(df)
        lc = label_column(df, cols)
        mc = maso_column(df, cols)
        rid = str(t["table_ref"]).split("|")[0]
        rry = re.findall(r"20\d{2}", rid)
        yc = year_column(cols, year, lc, mc, want_dau, rry[0] if rry else None) if year else None
        if yc is None:
            vc = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
            yc = vc[0] if len(vc) == 1 else None
        if yc is None:
            continue
        row, conf = None, 0
        if mm and mc is not None:
            code = str(mm[0]).strip()
            idxs = [i for i in range(len(df)) if str(df.iloc[i, mc]).strip() == code]
            if idxs:
                row, conf = idxs[0], 3
        if row is None:
            cand = score_rows(df, question, lc, idf or {}, first_data_row(df), topk=2)
            if cand and (len(cand) == 1 or cand[0][0] >= 1.6 * cand[1][0]):
                row, conf = cand[0][1], (2 if cand[0][0] > 6 else 1)
        if row is None:                       # fallback ĐỐI SOÁT SỐ: note-table -> dòng TỔNG (Q14/Q47)
            agg = reconciled_total_row(df, cols, lc, mc, yc)
            if agg is not None:
                row, conf = agg, 2
        if row is None:
            continue
        v = _num(df.iloc[row, yc])
        if v is None:
            continue
        factor = cols[yc]["unit_mult"] / req
        if best is None or conf > best[0]:
            best = (conf, t, int(row), int(yc), factor, round(v * factor, 2))
    if best is None or best[0] < 1:
        return None
    conf, t, row, yc, factor, ans = best
    q = _NUM_SRC + "df1 = list(dfs.values())[0]\n" + f"result = round(_num(df1.iloc[{row}, {yc}]) * {factor!r}, 2)\n"
    return {"ok": True, "answer": ans, "conf": conf, "pandas_query": q,
            "evidence": [{"variable": "df1", "csv_path": t["csv_path"], "table_ref": t["table_ref"]}]}


_OP_STRIP = re.compile(
    r"tăng trưởng|tốc độ tăng|tăng bao nhiêu|giảm bao nhiêu|chênh lệch|biến động|thay đổi|"
    r"so với|cùng kỳ|phần trăm|giữa|và năm|năm trước|bao nhiêu|là bao nhiêu|%", re.I)


def _classify_op(q: str):
    ql = q.lower()
    if re.search(r"tăng trưởng|tốc độ tăng|tỷ lệ tăng|tăng.{0,6}(bao nhiêu )?(phần trăm|%)|(phần trăm|%).{0,6}so với", ql):
        return "GROWTH"
    if re.search(r"chênh lệch|biến động|thay đổi|tăng.{0,8}bao nhiêu|giảm.{0,8}bao nhiêu|tăng hay giảm", ql):
        return "DIFF"
    return None


def analytic_answer(question: str, tables: list[dict], idf: dict | None = None) -> dict | None:
    """Compiler ANALYTICAL cùng-chỉ-tiêu/2-năm (chênh lệch, tăng trưởng): 1 dòng × 2 cột-năm.
    Base ~0 điểm ở nhóm này nên đây là NET GAIN. Trả None nếu không phải mẫu này / không đủ tự tin."""
    import pandas as pd
    op = _classify_op(question)
    if op is None:
        return None
    yrs = sorted(set(re.findall(r"20\d{2}", question)))
    try:
        from kingpro.answering.ma_so_tt200 import maso_of
        mm = maso_of(question)
    except Exception:
        mm = None
    mq = re.sub(r"20\d{2}", " ", _OP_STRIP.sub(" ", question))
    req = requested_unit(question)
    best = None
    for t in tables:
        try:
            df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        cols = resolve_columns(df)
        lc = label_column(df, cols)
        mc = maso_column(df, cols)
        rid = str(t["table_ref"]).split("|")[0]
        rry = re.findall(r"20\d{2}", rid)
        report_year = rry[0] if rry else None
        if len(yrs) >= 2:
            yA, yB = yrs[-1], yrs[0]
        elif len(yrs) == 1:
            yA, yB = yrs[0], str(int(yrs[0]) - 1)
        else:
            continue
        cA = year_column(cols, yA, lc, mc, False, report_year)
        cB = year_column(cols, yB, lc, mc, False, report_year)
        if cA is None or cB is None or cA == cB:
            continue
        row, conf = None, 0
        if mm and mc is not None:
            code = str(mm[0]).strip()
            idxs = [i for i in range(len(df)) if str(df.iloc[i, mc]).strip() == code]
            if idxs:
                row, conf = idxs[0], 3
        if row is None:
            cand = score_rows(df, mq, lc, idf or {}, first_data_row(df), topk=2)
            if cand and (len(cand) == 1 or cand[0][0] >= 1.6 * cand[1][0]):
                row, conf = cand[0][1], (2 if cand[0][0] > 6 else 1)
        if row is None:
            continue
        a, b = _num(df.iloc[row, cA]), _num(df.iloc[row, cB])
        if a is None or b is None:
            continue
        pre = _NUM_SRC + "df1 = list(dfs.values())[0]\n" + f"a = _num(df1.iloc[{row}, {cA}])\nb = _num(df1.iloc[{row}, {cB}])\n"
        if op == "GROWTH":
            if b == 0:
                continue
            ans = round((a - b) / b * 100, 2)
            q_code = pre + "result = round((a - b) / b * 100, 2)\n"
        else:
            factor = cols[cA]["unit_mult"] / req
            ans = round(abs(a - b) * factor, 2)
            q_code = pre + f"result = round(abs(a - b) * {factor!r}, 2)\n"
        if best is None or conf > best[0]:
            best = (conf, t, q_code, ans)
    if best is None or best[0] < 1:
        return None
    conf, t, q_code, ans = best
    return {"ok": True, "answer": ans, "conf": conf, "pandas_query": q_code,
            "evidence": [{"variable": "df1", "csv_path": t["csv_path"], "table_ref": t["table_ref"]}]}


def analytic_cross(question: str, tables: list[dict], idf: dict | None = None) -> dict | None:
    """CHÉO-BẢNG: chỉ tiêu M năm A (báo cáo-năm-A) vs năm B (báo cáo-năm-B) — thủ phạm Medium chính.
    Với mỗi năm tìm ĐÚNG bảng-năm đó + ô cùng nhãn/mã-số, rồi tính GROWTH/DIFF. Trả None nếu không chắc."""
    import pandas as pd
    op = _classify_op(question)
    if op is None:
        return None
    yrs = sorted(set(re.findall(r"20\d{2}", question)))
    # CHỈ chéo-bảng khi cách >=2 năm: năm liền kề thì 1 báo cáo đã có đủ 2 cột (số so sánh/
    # trình bày lại) -> để base/within-table lo, tránh lệch do restatement (vd Q640 2020 vs 2021).
    if len(yrs) < 2 or int(yrs[-1]) - int(yrs[0]) < 2:
        return None
    yA, yB = yrs[-1], yrs[0]
    try:
        from kingpro.answering.ma_so_tt200 import maso_of
        mm = maso_of(question)
    except Exception:
        mm = None
    code = str(mm[0]).strip() if mm else None
    mq = re.sub(r"20\d{2}", " ", _OP_STRIP.sub(" ", question))
    req = requested_unit(question)

    def cells_for_year(y):
        res = []
        for ti, t in enumerate(tables):
            try:
                df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
            except Exception:
                continue
            cols = resolve_columns(df)
            lc = label_column(df, cols)
            mc = maso_column(df, cols)
            rid = str(t["table_ref"]).split("|")[0]
            rry = re.findall(r"20\d{2}", rid)
            report_year = rry[0] if rry else None
            yc = year_column(cols, y, lc, mc, False, report_year)
            if yc is None:
                continue
            row, conf = None, 0
            if code and mc is not None:
                idxs = [i for i in range(len(df)) if str(df.iloc[i, mc]).strip() == code]
                if idxs:
                    row, conf = idxs[0], 3
            if row is None:
                cand = score_rows(df, mq, lc, idf or {}, first_data_row(df), topk=2)
                if cand and (len(cand) == 1 or cand[0][0] >= 1.4 * cand[1][0]):
                    row, conf = cand[0][1], (2 if cand[0][0] > 6 else 1)
            if row is None:
                continue
            v = _num(df.iloc[row, yc])
            if v is None:
                continue
            rep_bonus = 1.0 if report_year == y else 0.0
            res.append({"s": conf + rep_bonus, "conf": conf, "ti": ti, "t": t, "row": int(row),
                        "yc": int(yc), "v": v, "unit": cols[yc]["unit_mult"], "lbl": _norm(str(df.iloc[row, lc]))})
        return res

    CA, CB = cells_for_year(yA), cells_for_year(yB)
    if not CA or not CB:
        return None
    best = None
    for a in CA:
        for b in CB:
            if a["ti"] == b["ti"]:          # BẮT BUỘC chéo 2 BẢNG khác nhau (cùng bảng = việc của
                continue                    # analytic_answer; tránh lấy nhầm đầu/cuối-năm nội bảng, vd Q617)
            same_maso = a["conf"] >= 3 and b["conf"] >= 3
            same_lbl = a["lbl"] and b["lbl"] and a["lbl"] == b["lbl"]
            if not (same_maso or same_lbl):
                continue
            cross = 1.0 if a["ti"] != b["ti"] else 0.0     # thưởng chéo-bảng (đúng mẫu)
            sc = a["s"] + b["s"] + cross
            if best is None or sc > best[0]:
                best = (sc, a, b, same_maso, cross)
    if best is None:
        return None
    _, a, b, same_maso, cross = best
    conf = 3 if same_maso else (2 if min(a["conf"], b["conf"]) >= 2 else 1)
    va, vb = a["v"], b["v"]
    if op == "GROWTH":
        if vb == 0:
            return None
        ans = round((va - vb) / vb * 100, 2)
        if abs(ans) > 50000:      # % tăng trưởng vô lý = nổ đơn vị/nhầm cột (vd Q596 lấy '18,00%' sở hữu)
            return None
        formula = "result = round((a - b) / b * 100, 2)\n"
    else:
        factor = a["unit"] / req
        ans = round(abs(va - vb) * factor, 2)
        formula = f"result = round(abs(a - b) * {factor!r}, 2)\n"
    # pandas_query + evidence theo khuôn base đã chứng minh: _dfvals=list(dfs.values()) (bất-biến-key).
    same_tbl = a["t"]["table_ref"] == b["t"]["table_ref"]
    if same_tbl:
        pre = _NUM_SRC + "_dfvals = list(dfs.values())\ndf1 = _dfvals[0]\n"
        pre += f"a = _num(df1.iloc[{a['row']}, {a['yc']}])\nb = _num(df1.iloc[{b['row']}, {b['yc']}])\n"
        evidence = [{"variable": "df1", "table_ref": a["t"]["table_ref"]}]
    else:
        pre = _NUM_SRC + "_dfvals = list(dfs.values())\ndf1 = _dfvals[0]\ndf2 = _dfvals[1]\n"
        pre += f"a = _num(df1.iloc[{a['row']}, {a['yc']}])\nb = _num(df2.iloc[{b['row']}, {b['yc']}])\n"
        evidence = [{"variable": "df1", "table_ref": a["t"]["table_ref"]},
                    {"variable": "df2", "table_ref": b["t"]["table_ref"]}]
    return {"ok": True, "answer": ans, "conf": conf, "cross": bool(cross),
            "pandas_query": pre + formula, "evidence": evidence,
            "a": {"table_ref": a["t"]["table_ref"], "row": a["row"], "col": a["yc"]},
            "b": {"table_ref": b["t"]["table_ref"], "row": b["row"], "col": b["yc"]}}


def count_observed_operands(code: str):
    """UNDER-READING (theo GPT): đếm số leaf-read scalar DUY NHẤT (.values[0]/.iloc/.iat/.at)
    có dataflow tới `result`. None nếu không phân tích chắc. Dùng để chứng minh base thiếu operand."""
    import ast
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    assigns = {}          # var -> RHS node (lần gán cuối)
    result_rhs = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node.value
            if node.targets[0].id == "result":
                result_rhs = node.value
    if result_rhs is None:
        return None

    def is_leaf_read(n):
        # X.values[0] | X.iloc[..] | X.iat[..] | X.at[..]
        if isinstance(n, ast.Subscript):
            v = n.value
            if isinstance(v, ast.Attribute) and v.attr in ("values", "iloc", "iat", "at"):
                return True
        return False

    seen_vars = set()
    leaf_sigs = set()

    def walk(n, depth=0):
        if n is None or depth > 40:
            return
        if is_leaf_read(n):
            try:
                leaf_sigs.add(ast.dump(n))
            except Exception:
                leaf_sigs.add(id(n))
            # vẫn đi tiếp vào bên trong để bắt read lồng nhau
        for child in ast.iter_child_nodes(n):
            walk(child, depth + 1)
        # theo biến -> định nghĩa của nó
        if isinstance(n, ast.Name) and n.id in assigns and n.id not in seen_vars:
            seen_vars.add(n.id)
            walk(assigns[n.id], depth + 1)

    walk(result_rhs)
    return len(leaf_sigs)


_OWNERSHIP = re.compile(r"tỷ lệ\s+sở hữu|tỷ lệ\s+lợi ích|quyền\s+biểu quyết|phần\s+vốn\s+(góp|sở hữu)|"
                        r"tỷ lệ\s+nắm giữ|tỷ lệ\s+biểu quyết", re.I)
_RATIO_PCT = re.compile(r"\bchiếm\b|\btỷ trọng\b|tỷ lệ.{0,50}\b(trên|so với)\b|"
                        r"bao nhiêu\s*%\s*(so với|của)|bằng\s+bao nhiêu\s*%", re.I)
_RATIO_X = re.compile(r"gấp\s+(bao nhiêu|mấy)\s+lần", re.I)


def classify_operation(question: str):
    """(op, required_operand_count) — ownership bắt TRƯỚC ratio (theo GPT). Chỉ dùng cho câu ĐƠN thực thể."""
    q = question
    if _OWNERSHIP.search(q):
        return ("LOOKUP_PERCENT", 1)
    if _RATIO_X.search(q):
        return ("RATIO_X", 2)
    if _RATIO_PCT.search(q):
        return ("RATIO_PERCENT", 2)
    op = _classify_op(q)
    if op == "GROWTH":
        return ("GROWTH", 2)
    if op == "DIFF":
        return ("DIFF", 2)
    return (None, 1)


def _split_ratio(question: str):
    """Tách (tử A, mẫu B) từ câu tỷ trọng/tỷ lệ/gấp-lần. None nếu không tách sạch."""
    q = re.sub(r"20\d{2}|ngày|tháng|năm|cuối|đầu|kỳ|\bcủa công ty mẹ\b|\bcông ty mẹ\b|là bao nhiêu|\?|\.", " ", question)
    q = re.sub(r"\s+", " ", q).strip()
    pats = [
        r"tỷ trọng (?:của )?(.+?) tr(?:ong|ên) (?:tổng )?(.+)",
        r"(.+?) chiếm (?:bao nhiêu ?%|.*?phần trăm)? ?(?:trong|của|so với) (?:tổng )?(.+)",
        r"tỷ lệ (?:giữa )?(.+?) (?:trên|so với|/) (.+)",
        r"(.+?) gấp (?:bao nhiêu|mấy) lần (.+)",
        r"(.+?) tr(?:ên|ong) (?:tổng )?(.+)",
    ]
    for p in pats:
        m = re.search(p, q, re.I)
        if m:
            a, b = m.group(1).strip(" ,"), m.group(2).strip(" ,")
            if len(_norm(a)) >= 3 and len(_norm(b)) >= 3 and _norm(a) != _norm(b):
                return a, b
    return None


def ratio_answer(question: str, tables: list[dict], idf: dict | None = None) -> dict | None:
    """RATIO A/B (chiếm/tỷ trọng/tỷ lệ trên/gấp lần) — 2 chỉ tiêu, ground độc lập, cùng bảng+năm.
    Chỉ để override khi base UNDER-READ (observed<2). Trả None nếu không đủ chắc."""
    import pandas as pd
    op, req = classify_operation(question)
    if op not in ("RATIO_PERCENT", "RATIO_X"):
        return None
    sp = _split_ratio(question)
    if not sp:
        return None
    textA, textB = sp
    yrs = sorted(set(re.findall(r"20\d{2}", question)))
    year = yrs[0] if yrs else None
    best = None
    for t in tables:
        try:
            df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        cols = resolve_columns(df)
        lc = label_column(df, cols)
        mc = maso_column(df, cols)
        rid = str(t["table_ref"]).split("|")[0]
        rry = re.findall(r"20\d{2}", rid)
        yc = year_column(cols, year, lc, mc, False, rry[0] if rry else None) if year else None
        if yc is None:
            vc = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
            yc = vc[0] if len(vc) == 1 else None
        if yc is None:
            continue
        candA = score_rows(df, textA, lc, idf or {}, first_data_row(df), topk=2)
        candB = score_rows(df, textB, lc, idf or {}, first_data_row(df), topk=2)
        if not candA or not candB:
            continue
        # yêu cầu tách bạch: mỗi bên trội hơn á quân, và 2 dòng KHÁC nhau
        okA = len(candA) == 1 or candA[0][0] >= 1.5 * candA[1][0]
        okB = len(candB) == 1 or candB[0][0] >= 1.5 * candB[1][0]
        rA, rB = candA[0][1], candB[0][1]
        if not (okA and okB) or rA == rB:
            continue
        a, b = _num(df.iloc[rA, yc]), _num(df.iloc[rB, yc])
        if a is None or b is None or b == 0:
            continue
        conf = 2 if (candA[0][0] > 6 and candB[0][0] > 6) else 1
        ans = round(a / b * (100 if op == "RATIO_PERCENT" else 1), 2)
        if 0 <= abs(ans) <= (100000 if op == "RATIO_PERCENT" else 1e6):
            q_code = (_NUM_SRC + "_dfvals = list(dfs.values())\ndf1 = _dfvals[0]\n"
                      + f"a = _num(df1.iloc[{rA}, {yc}])\nb = _num(df1.iloc[{rB}, {yc}])\n"
                      + f"result = round(a / b * {100 if op == 'RATIO_PERCENT' else 1}, 2)\n")
            if best is None or conf > best[0]:
                best = (conf, t, q_code, ans, rA, rB, yc)
    if best is None or best[0] < 2:
        return None
    conf, t, q_code, ans, rA, rB, yc = best
    return {"ok": True, "answer": ans, "conf": conf, "pandas_query": q_code,
            "evidence": [{"variable": "df1", "table_ref": t["table_ref"]}]}


_CMP_DIFF = re.compile(r"chênh lệch|hiệu (giữa|số)|so với|cao hơn|thấp hơn|kém hơn|nhiều hơn|ít hơn|lớn hơn|nhỏ hơn", re.I)


def crosscompany_diff(question: str, tables: list[dict], idf: dict | None = None) -> dict | None:
    """DIFF CHÉO 2 CÔNG TY (base thiếu năng lực đọc chéo báo cáo): cùng chỉ tiêu+năm, 2 công ty khác nhau.
    |A - B| về đơn vị hỏi. Chỉ khi ground được cả 2 bên ở 2 bảng khác ticker. None nếu không chắc."""
    import pandas as pd
    if not _CMP_DIFF.search(question):
        return None
    try:
        from kingpro.retrieval.bm25_index import extract_all_facets
        ticks = [t.upper() for t in extract_all_facets(question).get("tickers", [])]
    except Exception:
        return None
    if len(ticks) < 2:
        return None
    yrs = sorted(set(re.findall(r"20\d{2}", question)))
    year = yrs[0] if yrs else None
    try:
        from kingpro.answering.ma_so_tt200 import maso_of
        mm = maso_of(question)
    except Exception:
        mm = None
    code = str(mm[0]).strip() if mm else None
    mq = re.sub(r"20\d{2}", " ", _OP_STRIP.sub(" ", question))
    req = requested_unit(question)

    def best_cell(tk):
        best = None
        for t in tables:
            if not str(t["table_ref"]).upper().startswith(tk + "_"):
                continue
            try:
                df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
            except Exception:
                continue
            cols = resolve_columns(df)
            lc = label_column(df, cols)
            mc = maso_column(df, cols)
            rid = str(t["table_ref"]).split("|")[0]
            rry = re.findall(r"20\d{2}", rid)
            yc = year_column(cols, year, lc, mc, False, rry[0] if rry else None) if year else None
            if yc is None:
                vc = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
                yc = vc[0] if len(vc) == 1 else None
            if yc is None:
                continue
            row, conf = None, 0
            if code and mc is not None:
                idxs = [i for i in range(len(df)) if str(df.iloc[i, mc]).strip() == code]
                if idxs:
                    row, conf = idxs[0], 3
            if row is None:
                cand = score_rows(df, mq, lc, idf or {}, first_data_row(df), topk=2)
                if cand and (len(cand) == 1 or cand[0][0] >= 1.5 * cand[1][0]):
                    row, conf = cand[0][1], (2 if cand[0][0] > 6 else 1)
            if row is None:
                continue
            v = _num(df.iloc[row, yc])
            if v is None:
                continue
            if best is None or conf > best[0]:
                best = (conf, t, int(row), int(yc), v, cols[yc]["unit_mult"])
        return best

    grounded = [(tk, best_cell(tk)) for tk in ticks]
    grounded = [(tk, g) for tk, g in grounded if g is not None]
    if len(grounded) < 2:
        return None
    grounded.sort(key=lambda x: -x[1][0])          # 2 bên tự tin nhất
    (tkA, A), (tkB, B) = grounded[0], grounded[1]
    if A[1]["table_ref"] == B[1]["table_ref"]:
        return None
    conf = 2 if min(A[0], B[0]) >= 2 else 1
    if conf < 2:
        return None
    fa, fb = A[5] / req, B[5] / req
    ans = round(abs(A[4] * fa - B[4] * fb), 2)
    q_code = (_NUM_SRC + "_dfvals = list(dfs.values())\ndf1 = _dfvals[0]\ndf2 = _dfvals[1]\n"
              + f"a = _num(df1.iloc[{A[2]}, {A[3]}]) * {fa!r}\nb = _num(df2.iloc[{B[2]}, {B[3]}]) * {fb!r}\n"
              + "result = round(abs(a - b), 2)\n")
    return {"ok": True, "answer": ans, "conf": conf,
            "pandas_query": q_code,
            "evidence": [{"variable": "df1", "table_ref": A[1]["table_ref"]},
                         {"variable": "df2", "table_ref": B[1]["table_ref"]}]}


def build_idf(tables_dfs: list) -> dict:
    import math
    dfc: dict = {}
    N = 0
    for df in tables_dfs:
        seen = set()
        for i in range(len(df)):
            for t in _tokens(str(df.iloc[i, 0])):
                seen.add(t)
        for t in seen:
            dfc[t] = dfc.get(t, 0) + 1
        N += 1
    return {t: math.log((N + 1) / (c + 0.5)) + 1.0 for t, c in dfc.items()}
