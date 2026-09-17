"""TỰ SINH bộ data SFT TẤT ĐỊNH (FREE) cho fine-tune — bản đầy đủ.
Loại câu: lookup, ratio(%), growth, conditional-selection(argmax), multi-entity(aggregate).
Mọi mẫu VALIDATE qua grader (run_one: builtins giới hạn, df/dfs, result) -> chỉ giữ mẫu ra SỐ.
Xuất SFT: {system, user=build_user(...), assistant=pandas df1..dfN}.
Chạy:  python scripts/gen_sft_data.py --n-tables 4000 --out build/sft_data.jsonl
"""
import sys, json, re, random, argparse, collections
sys.path.insert(0, "src")
import pandas as pd
from kingpro.answering.pandas_answer import (
    maso_answer, ratio_answer, build_user, SYSTEM, _with_preamble, _locate_maso_cell,
    detect_table_unit, _flatten_header, _is_value)
from kingpro.evaluation.metrics import coerce_number
from kingpro.answering.ma_so_tt200 import _MASO
sys.path.insert(0, "scripts")
from grader_check import run_one   # validate y hệt máy chấm

STD_CODES = set(c.replace("cf", "") for c in _MASO)
CODE_NAME = {}                     # code -> tên hiển thị đẹp (từ _MASO phrase đầu, có hoa đầu câu)
_DISPLAY = {  # tên đẹp cho vài code hay dùng (phần còn lại lấy từ nhãn bảng thật)
    "10": "Doanh thu thuần", "01": "Doanh thu bán hàng và cung cấp dịch vụ", "11": "Giá vốn hàng bán",
    "20": "Lợi nhuận gộp", "21": "Doanh thu hoạt động tài chính", "25": "Chi phí bán hàng",
    "26": "Chi phí quản lý doanh nghiệp", "50": "Tổng lợi nhuận kế toán trước thuế",
    "60": "Lợi nhuận sau thuế", "100": "Tài sản ngắn hạn", "200": "Tài sản dài hạn",
    "270": "Tổng cộng tài sản", "300": "Nợ phải trả", "400": "Vốn chủ sở hữu", "440": "Tổng cộng nguồn vốn",
}

# ---------- helpers ----------
_NUM_HELPER = (
    "def _num(s):\n"
    "    s = str(s).strip()\n"
    "    neg = s.startswith('(') or s.startswith('-')\n"
    "    s = s.replace('(','').replace(')','').replace('-','').replace('%','').replace(' ','')\n"
    "    s = s.replace('.','').replace(',','.')\n"
    "    return -float(s) if neg else float(s)\n"
)

def clean_label(s):
    s = re.sub(r"\s*\([^)]*[=+][^)]*\)", "", str(s))                       # bỏ '(100 = 110 + ...)'
    s = re.sub(r"^\s*(?:[IVX]{1,4}|[A-E]|\d{1,3}|[a-e]\d?)[.)]\s+", "", s)  # bỏ tiền tố 'I.' 'A.' '1.' 'b1)'
    s = re.sub(r"^[\s\-–▪•*+]+", "", s)                                     # bỏ bullet đầu dòng
    s = re.sub(r"\s+", " ", s).strip(" .:-")
    return s

# Nhãn RÁC: tên tổ chức / dòng tiêu đề nhóm / dòng meta -> KHÔNG phải chỉ tiêu số
_JUNK_LABEL = re.compile(
    r"công ty|ctcp|tnhh|tổng công ty|xí nghiệp|ngân hàng tmcp|chi nhánh|tập đoàn|"
    r"nơi thành lập|hoạt động chính|trụ sở|địa chỉ|người đại diện|"
    r"^(thu nhập|chi phí|tài sản|nguồn vốn|công ty con|công ty liên kết|các khoản|trong đó|cộng|tổng)\b",
    re.I)
# Header cột / dòng là TỶ LỆ %/SỞ HỮU (không phải tiền) -> chặn %-giả-tiền
_PCT_COL = re.compile(r"tỷ lệ|%|sở hữu|biểu quyết|phần trăm|lãi suất|dự trữ", re.I)
# Chỉ tiêu KHÔNG THỂ ÂM (chặn giá trị âm sai)
_NONNEG = re.compile(r"tài sản|tổng|doanh thu|vốn|nguồn vốn|tiền gửi|cho vay|hàng tồn kho", re.I)

def csv_full(row): return "build/tables/" + row["csv_path"]
def company_of(st):
    m = re.search(r"Công ty:\s*(.+?)\s*\(mã", st or ""); return m.group(1) if m else None

# ---------- PHRASING đa dạng (khớp đề thật, chống học vẹt) ----------
def cref(company, tk, scope):
    """Cách gọi công ty đa dạng + đúng scope (công ty mẹ vs hợp nhất). ~36% đề dùng 'công ty mẹ'."""
    base = random.choice([f"{company} ({tk})", f"{company} ({tk})", f"{company}", f"{tk}", f"{company} (mã {tk})"])
    if "mẹ" in scope or "riêng" in scope:
        return random.choice([f"công ty mẹ {base}", f"công ty mẹ {tk}", f"công ty mẹ {company} ({tk})"])
    return random.choice([f"{base}", f"công ty {base}", f"{base} (hợp nhất)"])

def yref(year):
    return random.choice([f"năm {year}", f"năm {year}", f"trong năm {year}", f"vào năm {year}", f"cuối năm {year}"])

def q_lookup(label, comp, year, unit):
    yr = yref(year)
    return random.choice([
        f"{label} của {comp} {yr} là bao nhiêu {unit}?",
        f"{label} {yr} của {comp} là bao nhiêu {unit}?",
        f"Cho biết {label} của {comp} {yr} (đơn vị: {unit}).",
        f"{comp} có {label} {yr} là bao nhiêu {unit}?",
        f"{label} của {comp} {yr} bằng bao nhiêu {unit}?",
    ])

def _extract(dfvar, mc, yc, code):
    return f"_num({dfvar}[{dfvar}['{mc}'].astype(str).str.strip()=='{code}']['{yc}'].values[0])"

def _extract_dong(dfvar, mc, yc, code, mult):
    """Trích + quy về ĐỒNG (nhân hệ số đơn vị bảng)."""
    return f"({_extract(dfvar, mc, yc, code)} * {mult:g})"

def mult_dong(df, table_ref):
    """Hệ số quy đổi giá trị bảng -> ĐỒNG (triệu=1e6...). None nếu không rõ đơn vị -> bỏ (giữ chất lượng)."""
    tu = detect_table_unit(df, table_ref)
    return tu[1] if tu else None

def _fill_mults(entries):
    """Hệ số đơn vị mỗi entry; điền chỗ None bằng đơn vị PHỔ BIẾN NHẤT nhóm.
    Cùng công ty (conditional/count) -> cùng đơn vị nên điền an toàn; nhóm khác công ty (multi) -> giả định gần đúng.
    Trả None nếu KHÔNG entry nào detect được (giữ chất lượng tối thiểu)."""
    ms = [mult_dong(e["df"], e["row"]["table_ref"]) for e in entries]
    known = [m for m in ms if m is not None]
    if not known:
        return None
    common = collections.Counter(known).most_common(1)[0][0]
    return [m if m is not None else common for m in ms]

def validate(code, tables):
    """Chạy qua grader thật -> trả số hoặc None."""
    n = len(tables)
    full = _with_preamble(code, n)
    csv_paths = {f"df{i+1}": t["csv_path"] for i, t in enumerate(tables)}
    try:
        r = run_one(full, csv_paths)
        rv = float(r)
        if rv != rv or abs(rv) == float("inf"):
            return None
        return round(rv, 2)
    except Exception:
        return None

def to_sft(question, tables, code, answer, typ):
    user, _, _ = build_user(question, tables)
    return {"system": SYSTEM, "user": user, "assistant": code.strip(),
            "answer": answer, "type": typ,
            "meta": {"table_refs": [t["table_ref"] for t in tables]}}

# ---------- build index: bảng có Mã số + báo cáo-type + ô đã định vị ----------
def report_type(codes):
    s = set(codes)
    if s & {"10", "20", "60"}: return "KQKD"
    if s & {"100", "270", "400"}: return "CDKT"
    if s & {"20", "30", "40"}: return "LCTT"
    return "?"

def build_index(cat, n_tables):
    """Đọc n_tables bảng 1 LẦN -> giữ MỌI bảng có cột giá trị số; đánh dấu Mã số (has_maso) nếu có.
    Dùng chung: lookup/ratio (bảng có Mã số) + lookup_label (mọi bảng — ngân hàng/thuyết minh)."""
    idx = []
    random.shuffle(cat)
    for r in cat[:n_tables]:
        try:
            df = pd.read_csv(csv_full(r), dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        if len(df) < 5:
            continue
        vcol, _ = _first_valcol(df)
        if vcol is None:                              # không có cột số -> bỏ (bảng bìa/mục lục)
            continue
        mj = _maso_col_idx(df)
        codes = set(str(v).strip() for v in df.iloc[:, mj] if str(v).strip() in STD_CODES) if mj is not None else set()
        idx.append({"row": r, "df": df, "codes": codes, "has_maso": mj is not None,
                    "company": company_of(r["search_text"]) or r["ticker"], "rtype": report_type(codes)})
    return idx

def build_label_index(cat, n_tables):
    """Bảng CÓ cột giá trị số (KHÔNG cần Mã số) -> lookup theo nhãn (ngân hàng/chứng khoán/thuyết minh)."""
    idx = []
    random.shuffle(cat)
    for r in cat[:n_tables]:
        try:
            df = pd.read_csv(csv_full(r), dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        if len(df) < 5:
            continue
        vcol, _ = _first_valcol(df)
        if vcol is not None:
            idx.append({"row": r, "df": df, "company": company_of(r["search_text"]) or r["ticker"]})
    return idx

def _maso_col_idx(df):
    for j in range(len(df.columns)):
        if sum(1 for v in df.iloc[:, j] if str(v).strip() in STD_CODES) >= 3:
            return j
    return None

def build_kqkd_index(cat, max_companies):
    """GOM theo công ty (ticker,scope) có >=3 năm -> đọc CHỈ bảng KQKD mỗi năm.
    Đảm bảo có liệu cho conditional/growth/multi-entity (income metrics)."""
    comp = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in cat:
        comp[(r["ticker"], r.get("scope", ""))][r["year"]].append(r)
    multi = [(k, yrs) for k, yrs in comp.items() if len(yrs) >= 3]
    random.shuffle(multi)
    idx = []
    for (ticker, scope), yrs in multi[:max_companies]:
        for year, rows in yrs.items():
            for r in rows:                      # tìm bảng KQKD trong report này
                try:
                    df = pd.read_csv(csv_full(r), dtype=str, keep_default_na=False, encoding="utf-8-sig")
                except Exception:
                    continue
                mj = _maso_col_idx(df)
                if mj is None:
                    continue
                codes = set(str(v).strip() for v in df.iloc[:, mj] if str(v).strip() in STD_CODES)
                if report_type(codes) == "KQKD" and {"10", "60"} <= codes:
                    idx.append({"row": r, "df": df, "codes": codes,
                                "company": company_of(r["search_text"]) or ticker, "rtype": "KQKD"})
                    break                        # 1 KQKD/năm là đủ
    return idx

# ---------- generators ----------
def gen_lookup(e, out, cap):
    r, df = e["row"], e["df"]
    col_ms = None
    for j in range(len(df.columns)):
        if sum(1 for v in df.iloc[:, j] if str(v).strip() in STD_CODES) >= 3:
            col_ms = df.iloc[:, j].astype(str).str.strip(); break
    tables = [{"table_ref": r["table_ref"], "csv_path": csv_full(r)}]
    tk, scope, year = r["ticker"], r.get("scope", ""), r["year"]
    for i in range(len(df)):
        if len(out) >= cap: return
        if col_ms.iat[i] not in STD_CODES: continue
        label = clean_label(df.iat[i, 0])
        if len(label) < 4: continue
        unit = random.choice(["triệu đồng", "tỷ đồng"])
        q = q_lookup(label, cref(e["company"], tk, scope), year, unit)
        try: res = maso_answer(q, tables)
        except Exception: res = None
        if res and res.get("answer") is not None:
            out.append(to_sft(q, tables, res["pandas_query"], round(float(res["answer"]), 2), "lookup"))

def _first_valcol(df):
    """Cột giá trị: bảng SEGMENT (có cột 'Tổng cộng') -> cột đó; else cột ĐẦU (năm hiện tại). None nếu không có."""
    names, first_data = _flatten_header(df)
    valcols = []
    for j in range(1, len(df.columns)):
        ne = [v for v in (str(df.iat[i, j]).strip() for i in range(first_data, len(df))) if v and v.lower() != "nan"]
        if ne and sum(_is_value(v) for v in ne) >= len(ne) * 0.6:
            valcols.append(j)
    if not valcols:
        return None, first_data
    for j in valcols:                                    # bảng segment -> lấy cột 'Tổng cộng', không lấy 1 phân khúc
        if re.search(r"tổng cộng|tong cong|\btotal\b", str(names[j]), re.I):
            return j, first_data
    return valcols[0], first_data

def gen_lookup_label(e, out, cap):
    """Lookup THEO NHÃN (KHÔNG cần Mã số) -> NGÂN HÀNG/CHỨNG KHOÁN (19% đề) + THUYẾT MINH.
    ĐÃ LỌC: nhãn rác (tên tổ chức/tiêu đề nhóm/ghép), %-giả-tiền, cột segment, sanity dấu/bậc."""
    r, df = e["row"], e["df"]
    tk, scope, year = r["ticker"], r.get("scope", ""), r["year"]
    vcol, first_data = _first_valcol(df)
    if vcol is None:
        return
    mu = mult_dong(df, r["table_ref"])
    if mu is None:                                            # đơn vị không rõ -> bỏ (giữ chất lượng)
        return
    names, _ = _flatten_header(df)
    if _PCT_COL.search(str(names[vcol])):                     # cột giá trị là %/tỷ lệ -> không phải tiền
        return
    vcol_name = df.columns[vcol]
    tables = [{"table_ref": r["table_ref"], "csv_path": csv_full(r)}]
    col0 = [str(df.iat[k, 0]) for k in range(len(df))]
    ncol = len(df.columns)
    for i in range(first_data, len(df)):
        if len(out) >= cap:
            return
        raw = str(df.iat[i, vcol]).strip()
        if "%" in raw or not _is_value(raw):                  # ô %/rỗng
            continue
        if any("%" in str(df.iat[i, j]) for j in range(1, ncol)):   # dòng có % ở cột khác -> dòng tỷ lệ
            continue
        label = clean_label(df.iat[i, 0])
        if len(label) < 8 or len(label) > 80 or " " not in label:   # ngắn/dài/ghép-1-từ (OCR merge)
            continue
        if _JUNK_LABEL.search(label):                         # tên tổ chức / tiêu đề nhóm
            continue
        if str(df.iat[i, 0]).strip() == str(df.iat[i, 1]).strip():  # dòng tiêu đề (col0 lặp sang cột kế)
            continue
        x = coerce_number(raw)
        if x is None or abs(x) < 1000 or abs(x) > 1e15:       # số nhỏ (note-ref) / vô lý
            continue
        if x < 0 and _NONNEG.search(label):                   # âm ở chỉ tiêu không thể âm
            continue
        if sum(1 for c in col0 if label in c) != 1:           # nhãn khớp DUY NHẤT 1 dòng
            continue
        unit = random.choice(["triệu đồng", "tỷ đồng"])
        req = 1e6 if "triệu" in unit else 1e9
        code = (_NUM_HELPER + "# tra theo nhan (cot gia tri), doi don vi\n"
                + f"_v = _num(df1[df1['0'].astype(str).str.contains({label!r}, case=False, na=False, regex=False)]['{vcol_name}'].values[0])\n"
                + f"result = round(_v * {mu:g} / {req:g}, 2)\n")
        a = validate(code, tables)
        if a is not None:
            q = q_lookup(label, cref(e["company"], tk, scope), year, unit)
            out.append(to_sft(q, tables, code, a, "lookup_label"))

def gen_ratio(e, out, cap):
    r = e["row"]; tables = [{"table_ref": r["table_ref"], "csv_path": csv_full(r)}]
    tk, scope, year = r["ticker"], r.get("scope", ""), r["year"]
    for rn in ["ROE", "ROA", "biên lợi nhuận gộp", "biên lợi nhuận ròng"]:
        if len(out) >= cap: return
        comp = cref(e["company"], tk, scope)
        q = random.choice([f"{rn} của {comp} {yref(year)} là bao nhiêu %?",
                           f"{rn} {yref(year)} của {comp} là bao nhiêu %?",
                           f"Tính {rn} của {comp} {yref(year)} (đơn vị %)."])
        try: res = ratio_answer(q, tables)
        except Exception: res = None
        if res and res.get("answer") is not None:
            a = round(float(res["answer"]), 2)
            if -100 <= a <= 300:                       # PLAUSIBILITY FILTER (bỏ bug chỉ lấy tử số)
                out.append(to_sft(q, tables, res["pandas_query"], a, "ratio"))

# Tỷ số từ 1 bảng CDKT -> câu "lần" (KHÔNG ×100). (tên, tử, mẫu, trừ-khỏi-tử). Phủ 5.3% đề "lần".
_CDKT_RATIOS = [
    ("Hệ số nợ trên vốn chủ sở hữu (D/E)", "300", "400", None),
    ("Hệ số nợ", "300", "270", None),
    ("Hệ số khả năng thanh toán hiện hành", "100", "310", None),
    ("Hệ số khả năng thanh toán nhanh", "100", "310", "140"),
]

def gen_ratio_cdkt(e, out, cap):
    """Tỷ số thanh khoản/đòn bẩy từ 1 bảng CDKT -> câu 'lần' (dimensionless, KHÔNG ×100, KHÔNG đổi đơn vị)."""
    if e["rtype"] != "CDKT":
        return
    r = e["row"]
    tables = [{"table_ref": r["table_ref"], "csv_path": csv_full(r)}]
    for name, num, den, sub in _CDKT_RATIOS:
        if len(out) >= cap:
            return
        if not ({num, den} <= e["codes"] and (sub is None or sub in e["codes"])):
            continue
        ln, ld = _cell(e, num), _cell(e, den)
        ls = _cell(e, sub) if sub else None
        if ln is None or ld is None or (sub and ls is None):
            continue
        num_expr = _extract("df1", ln[0], ln[1], num)
        if sub:
            num_expr = f"({num_expr} - {_extract('df1', ls[0], ls[1], sub)})"
        code = (_NUM_HELPER + f"# {name} = ma{num}{'-'+sub if sub else ''} / ma{den}, don vi LAN (khong x100)\n"
                + f"_num_v = {num_expr}\n_den_v = {_extract('df1', ld[0], ld[1], den)}\n"
                + "result = round(_num_v / _den_v, 2)\n")
        a = validate(code, tables)
        if a is not None and 0 <= a <= 100:                   # tỷ số 'lần' hợp lý
            q = random.choice([
                f"{name} của {cref(e['company'], r['ticker'], r.get('scope',''))} {yref(r['year'])} là bao nhiêu lần?",
                f"{name} {yref(r['year'])} của {cref(e['company'], r['ticker'], r.get('scope',''))} là bao nhiêu lần?"])
            out.append(to_sft(q, tables, code, a, "ratio_cdkt"))

_SUM_PAIRS = [  # (mã1, mã2, tên tổng) — tổng 2 chỉ tiêu KQKD trong 1 công ty
    ("25", "26", "Tổng chi phí bán hàng và chi phí quản lý doanh nghiệp"),
    ("21", "23", "Tổng doanh thu hoạt động tài chính và chi phí lãi vay"),
]

def gen_sum(e, out, cap):
    """Tổng 2 chỉ tiêu KQKD trong 1 công ty (vd CPBH+CPQLDN). Quy đổi đơn vị."""
    if e["rtype"] != "KQKD":
        return
    r = e["row"]
    mu = mult_dong(e["df"], r["table_ref"])
    if mu is None:
        return
    tables = [{"table_ref": r["table_ref"], "csv_path": csv_full(r)}]
    for c1, c2, name in _SUM_PAIRS:
        if len(out) >= cap:
            return
        if not ({c1, c2} <= e["codes"]):
            continue
        l1, l2 = _cell(e, c1), _cell(e, c2)
        if l1 is None or l2 is None:
            continue
        unit = random.choice(["triệu đồng", "tỷ đồng"])
        req = 1e6 if "triệu" in unit else 1e9
        code = (_NUM_HELPER + f"# tong ma{c1} + ma{c2}, doi don vi\n"
                + f"_a = {_extract('df1', l1[0], l1[1], c1)}\n_b = {_extract('df1', l2[0], l2[1], c2)}\n"
                + f"result = round((_a + _b) * {mu:g} / {req:g}, 2)\n")
        a = validate(code, tables)
        if a is not None:
            q = f"{name} của {cref(e['company'], r['ticker'], r.get('scope',''))} {yref(r['year'])} là bao nhiêu {unit}?"
            out.append(to_sft(q, tables, code, a, "sum"))

def gen_count(group, out, cap):
    """ĐẾM số công ty có chỉ tiêu DƯƠNG/ÂM trong 1 năm (>=3 công ty). Trả SỐ NGUYÊN (numeric, grader chấm được)."""
    if len(group) < 3:
        return
    g = group[:6]
    common = STD_CODES & set.intersection(*[e["codes"] for e in g])
    for code in common:
        if len(out) >= cap:
            return
        if code not in _DISPLAY:
            continue
        year = g[0]["row"]["year"]
        locs = [_cell(e, code, year) for e in g]
        if any(l is None for l in locs):
            continue
        cname, cop = random.choice([("dương", ">"), ("âm", "<")])
        lines = [_NUM_HELPER, f"# dem cong ty co ma{code} {cname}\n"]
        for i, l in enumerate(locs, 1):
            lines.append(f"_x{i} = {_extract(f'df{i}', l[0], l[1], code)}\n")
        terms = " + ".join(f"(1 if _x{i} {cop} 0 else 0)" for i in range(1, len(g) + 1))
        lines.append(f"result = {terms}\n")
        code_str = "".join(lines)
        tables = [{"table_ref": e["row"]["table_ref"], "csv_path": csv_full(e["row"])} for e in g]
        a = validate(code_str, tables)
        if a is not None:
            names = ", ".join(e["row"]["ticker"] for e in g)
            q = (f"Trong các công ty {names} năm {year}, có bao nhiêu công ty có "
                 f"{_DISPLAY[code]} {cname}?")
            out.append(to_sft(q, tables, code_str, a, "count"))

def _cell(e, code, year=None):
    """(cột_mã, cột_giá_trị_ĐẦU, dòng) — cột trái nhất = NĂM BÁO CÁO (chuẩn TT200), robust hơn đoán năm."""
    df = e["df"]
    mj = _maso_col_idx(df)
    if mj is None:
        return None
    col_ms = df.iloc[:, mj].astype(str).str.strip()
    idxs = [i for i in range(len(df)) if col_ms.iat[i] == code]
    if not idxs:
        return None
    row_i = idxs[0]
    _, first_data = _flatten_header(df)
    for j in range(1, len(df.columns)):
        if j == mj:
            continue
        ne = [v for v in (str(df.iat[i, j]).strip() for i in range(first_data, len(df))) if v and v.lower() != "nan"]
        if ne and sum(_is_value(v) for v in ne) >= len(ne) * 0.6:
            return (df.columns[mj], df.columns[j], row_i)
    return None

def gen_growth(group, out, cap):
    """group = list entry cùng (ticker,scope,rtype) khác năm. Sinh tăng trưởng 1 chỉ tiêu qua 2 năm."""
    if len(group) < 2: return
    g = sorted(group, key=lambda x: x["row"]["year"])
    for code in (STD_CODES & set.intersection(*[e["codes"] for e in g])):
        if len(out) >= cap: return
        if code not in _DISPLAY: continue
        e1, e2 = g[0], g[-1]              # cũ nhất, mới nhất
        y1, y2 = e1["row"]["year"], e2["row"]["year"]
        if y1 == y2: continue
        l1 = _cell(e1, code, y1); l2 = _cell(e2, code, y2)
        if not l1 or not l2: continue
        code_str = (_NUM_HELPER
                    + f"# tang truong Ma {code} tu {y1}->{y2}: (moi/cu-1)*100\n"
                    + f"_v_new = {_extract('df1', l2[0], l2[1], code)}\n"
                    + f"_v_old = {_extract('df2', l1[0], l1[1], code)}\n"
                    + "result = round((_v_new / _v_old - 1) * 100, 2)\n")
        tables = [{"table_ref": e2["row"]["table_ref"], "csv_path": csv_full(e2["row"])},
                  {"table_ref": e1["row"]["table_ref"], "csv_path": csv_full(e1["row"])}]
        a = validate(code_str, tables)
        if a is not None and -100 <= a <= 5000:
            comp = cref(e2["company"], e2["row"]["ticker"], e2["row"].get("scope", ""))
            q = random.choice([
                f"Tăng trưởng {_DISPLAY[code]} của {comp} từ năm {y1} đến năm {y2} là bao nhiêu %?",
                f"{_DISPLAY[code]} của {comp} tăng trưởng bao nhiêu % từ năm {y1} sang năm {y2}?",
                f"Tốc độ tăng trưởng {_DISPLAY[code]} của {comp} giai đoạn {y1}-{y2} là bao nhiêu %?"])
            out.append(to_sft(q, tables, code_str, a, "growth"))

_COND_PAIRS = [("60", "10"), ("20", "10"), ("60", "20"), ("50", "10")]   # (M, C) — M tại năm C cao nhất

def gen_conditional(group, out, cap):
    """Nhiều mẫu/công ty: M tại năm có C lớn nhất + biến thể 'năm nào C cao nhất'. Quy về ĐỒNG rồi so."""
    g = [e for e in group if e["rtype"] == "KQKD"]
    if len(g) < 3: return
    g = sorted(g, key=lambda x: x["row"]["year"])[:6]
    mults = _fill_mults(g)                                # nới đơn vị: detect >=1 năm là dùng cả nhóm
    if mults is None: return
    years = [e["row"]["year"] for e in g]
    company, tk, scope = g[0]["company"], g[0]["row"]["ticker"], g[0]["row"].get("scope", "")
    tables = [{"table_ref": e["row"]["table_ref"], "csv_path": csv_full(e["row"])} for e in g]
    did_year = set()
    for M, C in _COND_PAIRS:
        if len(out) >= cap: return
        if M not in _DISPLAY or C not in _DISPLAY: continue
        if not all(M in e["codes"] and C in e["codes"] for e in g): continue
        lm = [_cell(e, M) for e in g]; lc = [_cell(e, C) for e in g]
        if any(x is None for x in lm + lc): continue
        base = [_NUM_HELPER, f"# {_DISPLAY[M]} tai nam co {_DISPLAY[C]} lon nhat\n"]
        for i in range(len(g)):
            base.append(f"_m{i+1} = {_extract_dong(f'df{i+1}', lm[i][0], lm[i][1], M, mults[i])}\n")
            base.append(f"_c{i+1} = {_extract_dong(f'df{i+1}', lc[i][0], lc[i][1], C, mults[i])}\n")
        argmax = ["_bc = _c1\n_bm = _m1\n_bi = 0\n"] + [
            f"if _c{i} > _bc:\n    _bc = _c{i}\n    _bm = _m{i}\n    _bi = {i-1}\n" for i in range(2, len(g)+1)]
        # (a) lấy M tại năm C cao nhất
        code_m = "".join(base + argmax + ["result = round(_bm / 1e6, 2)\n"])
        a = validate(code_m, tables)
        if a is not None:
            comp = cref(company, tk, scope)
            q = random.choice([
                f"{_DISPLAY[M]} của {comp} vào năm có {_DISPLAY[C]} cao nhất (trong các năm {', '.join(years)}) là bao nhiêu triệu đồng?",
                f"Trong các năm {', '.join(years)}, {_DISPLAY[M]} của {comp} tại năm có {_DISPLAY[C]} lớn nhất là bao nhiêu triệu đồng?"])
            out.append(to_sft(q, tables, code_m, a, "conditional"))
        # (b) 'năm nào C cao nhất' -> trả năm (mỗi C 1 lần)
        if C not in did_year and len(out) < cap:
            did_year.add(C)
            code_y = "".join(base + argmax + [f"_yrs = [{','.join(years)}]\n", "result = float(_yrs[_bi])\n"])
            a2 = validate(code_y, tables)
            if a2 is not None:
                q2 = (f"Năm nào {cref(company, tk, scope)} có {_DISPLAY[C]} cao nhất "
                      f"(trong các năm {', '.join(years)})?")
                out.append(to_sft(q2, tables, code_y, a2, "conditional"))

def gen_multi_entity(group, out, cap):
    """Trung bình M của >=3 công ty cùng năm+scope+rtype. Quy về triệu đồng."""
    if len(group) < 3: return
    g = group[:5]
    for code in (STD_CODES & set.intersection(*[e["codes"] for e in g])):
        if len(out) >= cap: return
        if code not in _DISPLAY: continue
        year = g[0]["row"]["year"]
        mults = [mult_dong(e["df"], e["row"]["table_ref"]) for e in g]   # multi: KHÁC công ty -> đơn vị RIÊNG, KHÔNG nới
        if any(m is None for m in mults): continue
        locs = [_cell(e, code, year) for e in g]
        if any(l is None for l in locs): continue
        req = 1e6
        lines = [_NUM_HELPER, f"# trung binh Ma {code} cua {len(g)} cong ty (quy ve trieu dong)\n"]
        terms = []
        for i, (l, mu) in enumerate(zip(locs, mults), 1):
            lines.append(f"_x{i} = {_extract_dong(f'df{i}', l[0], l[1], code, mu)}\n"); terms.append(f"_x{i}")
        lines.append(f"result = round(({' + '.join(terms)}) / {len(g)} / {req:g}, 2)\n")
        code_str = "".join(lines)
        tables = [{"table_ref": e["row"]["table_ref"], "csv_path": csv_full(e["row"])} for e in g]
        a = validate(code_str, tables)
        if a is not None:
            names = ", ".join(e["row"]["ticker"] for e in g)
            sc = " (công ty mẹ)" if "mẹ" in g[0]["row"].get("scope", "") else ""
            q = random.choice([
                f"Trung bình {_DISPLAY[code]} của các công ty {names}{sc} năm {year} là bao nhiêu triệu đồng?",
                f"{_DISPLAY[code]} trung bình của {len(g)} công ty {names}{sc} năm {year} là bao nhiêu triệu đồng?",
                f"Tính trung bình {_DISPLAY[code]} năm {year} của các công ty {names}{sc} (triệu đồng)."])
            out.append(to_sft(q, tables, code_str, a, "multi_entity"))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tables", type=int, default=4000)
    ap.add_argument("--companies", type=int, default=60, help="số công ty (>=3 năm) đọc cho multi-table")
    ap.add_argument("--out", default="build/sft_data.jsonl")
    ap.add_argument("--cap", type=int, default=200, help="tối đa mỗi loại (pilot)")
    a = ap.parse_args()
    random.seed(11)
    cat = [json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8")]
    print(f"catalog {len(cat)} bảng", flush=True)
    idx = build_index(cat, a.n_tables)                       # 1 lần: mọi bảng có cột số
    n_maso = sum(1 for e in idx if e["has_maso"])
    print(f"index: {len(idx)} bảng có cột số ({n_maso} có Mã số TT200, {len(idx)-n_maso} theo nhãn/ngân hàng)", flush=True)
    kidx = build_kqkd_index(cat, a.companies)                # gom công ty: multi-table
    print(f"index KQKD (gom công ty): {len(kidx)} bảng", flush=True)

    by_company = collections.defaultdict(list)     # (ticker,scope) -> [KQKD entry] khác năm
    by_year = collections.defaultdict(list)        # (year,scope) -> [KQKD entry] khác công ty
    for e in kidx:
        r = e["row"]
        by_company[(r["ticker"], r.get("scope", ""))].append(e)
        by_year[(r["year"], r.get("scope", ""))].append(e)

    # target theo PHÂN BỐ đề thật (lookup nhiều nhất; conditional cao vì 24%; lookup_label phủ ngân hàng 19%)
    # cap cân theo phân bố đề: giảm lookup_label (đừng để 54%), tăng nhóm KHÓ (conditional 24% + multi 18%)
    T = {"lookup": int(a.cap * 5), "lookup_label": int(a.cap * 3.5), "ratio": int(a.cap * 1),
         "ratio_cdkt": int(a.cap * 2), "sum": int(a.cap * 1), "growth": int(a.cap * 1.5),
         "conditional": int(a.cap * 5), "multi_entity": int(a.cap * 4), "count": int(a.cap * 2)}
    out = {k: [] for k in T}
    for e in idx:
        if e["has_maso"] and len(out["lookup"]) < T["lookup"]: gen_lookup(e, out["lookup"], T["lookup"])
        if e["has_maso"] and len(out["ratio"]) < T["ratio"] and e["rtype"] == "KQKD": gen_ratio(e, out["ratio"], T["ratio"])
        if e["has_maso"] and len(out["ratio_cdkt"]) < T["ratio_cdkt"] and e["rtype"] == "CDKT": gen_ratio_cdkt(e, out["ratio_cdkt"], T["ratio_cdkt"])
        if e["has_maso"] and len(out["sum"]) < T["sum"] and e["rtype"] == "KQKD": gen_sum(e, out["sum"], T["sum"])
        if len(out["lookup_label"]) < T["lookup_label"]: gen_lookup_label(e, out["lookup_label"], T["lookup_label"])
    for grp in by_company.values():
        if len(out["growth"]) < T["growth"]: gen_growth(grp, out["growth"], T["growth"])
        if len(out["conditional"]) < T["conditional"]: gen_conditional(grp, out["conditional"], T["conditional"])
    for grp in by_year.values():
        if len(out["multi_entity"]) < T["multi_entity"]: gen_multi_entity(grp, out["multi_entity"], T["multi_entity"])
        if len(out["count"]) < T["count"]: gen_count(grp, out["count"], T["count"])

    # LỌC BẬC BẤT KHẢ THI (ô OCR ghép số / sai đơn vị): không giá trị tiền nào lớn cỡ này
    def _implausible(s):
        try:
            av = abs(float(s["answer"]))
            if "triệu đồng" in s["user"] and (av > 1e10 or 0 < av < 0.001): return True
            if "tỷ đồng" in s["user"] and (av > 1e7 or 0 < av < 0.001): return True
            if "%" in s["user"].split("?")[0][-6:] and av > 1e4: return True   # % vô lý
        except Exception:
            pass
        return False
    removed = sum(1 for v in out.values() for s in v if _implausible(s))
    alls = [s for v in out.values() for s in v if not _implausible(s)]
    print(f"[lọc bậc bất khả thi (OCR ghép/sai đơn vị): bỏ {removed} mẫu]")
    with open(a.out, "w", encoding="utf-8") as f:
        for s in alls:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n=== SINH ĐƯỢC (đã validate qua grader) ===")
    for k, v in out.items():
        print(f"  {k:14} {len(v)}")
    print(f"  TỔNG          {len(alls)}  -> {a.out}")
    print("\n=== MẪU mỗi loại ===")
    for k, v in out.items():
        if v:
            s = v[0]
            print(f"[{k}] Q: {s['question'] if 'question' in s else ''}")
            # câu nằm trong user; in ngắn
            uq = s["user"].split(chr(10))[0][:95]
            print(f"    {uq}")
            print(f"    → answer={s['answer']} | code: {s['assistant'][:90].replace(chr(10),' ⏎ ')}")


if __name__ == "__main__":
    main()
