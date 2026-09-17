"""ĐÒN #1 (đã prototype trên grader thật): GHÉP MẢNH -> statement ĐẦY ĐỦ.
Bảng bị chẻ ~70 mảnh/report; mảnh cùng statement có line-ref LIỀN + cùng schema 5 cột
[Mã số | Chỉ tiêu | Thuyết minh | Số cuối năm | Số đầu năm]. Ghép lại -> mọi toán hạng CÙNG 1 df
đánh chỉ mục theo Mã số -> hạ bức tường chọn-mảnh (Tables-F2 0.325).

merged_statements(report_id) -> {'CDKT': df, 'KQKD': df, 'LCTT': df} (df nào có).
Chỉ áp cho DN theo TT200 (844/1012); ngân hàng/CK/BH trả rỗng -> caller fallback retrieval cũ.
"""
import json
import re
from collections import defaultdict

import pandas as pd

_STD_CDKT = set("100 110 120 130 140 150 200 210 220 230 240 250 260 270 300 310 320 330 340 400 410 411 420 430 440".split())
_STD_KQKD = set("01 02 10 11 20 21 22 23 24 25 26 30 31 32 40 50 51 60 70 71".split())
_STD = _STD_CDKT | _STD_KQKD

_CAT_CACHE = None


def _catalog():
    global _CAT_CACHE
    if _CAT_CACHE is None:
        byrep = defaultdict(list)
        for l in open("build/catalog.jsonl", encoding="utf-8"):
            r = json.loads(l)
            byrep[r["report_id"]].append(r)
        _CAT_CACHE = byrep
    return _CAT_CACHE


def _maso_col(df):
    """Cột chứa mã số (thử 3 cột đầu, chọn cột nhiều mã chuẩn nhất). KBC cột0, AAA cột1."""
    best_j, best_n = -1, 0
    for j in range(min(3, df.shape[1])):
        vals = set(str(df.iloc[i, j]).strip() for i in range(len(df)))
        n = len(vals & _STD)
        if n > best_n:
            best_j, best_n = j, n
    return best_j, best_n


def _classify(df, mc):
    codes = set(str(df.iloc[i, mc]).strip() for i in range(len(df))) if mc >= 0 else set()
    txt = " ".join(str(df.iloc[i, j]) for i in range(min(3, len(df))) for j in range(df.shape[1])).lower()
    if "lưu chuyển" in txt:
        return "LCTT"
    if (codes & {"100", "270", "300", "400", "440"}) or "tài sản" in txt or "nguồn vốn" in txt or "nguồn vòn" in txt:
        return "CDKT"
    if "năm nay" in txt or (codes & {"01", "10", "11", "20", "50", "60"}):
        return "KQKD"
    return None


def _is_header_row(df, i, mc):
    v = str(df.iloc[i, mc]).strip().lower()
    return v in ("mã số", "ma so", "mã", "")


def merged_statements(report_id):
    """Trả dict statement_type -> DataFrame ghép (cột 0..N, iloc[0]=header như hợp đồng grader)."""
    frs = sorted(_catalog().get(report_id, []), key=lambda r: r.get("line", 0))
    groups = defaultdict(list)   # type -> [(df, mc)]
    for r in frs:
        if r.get("n_cols", 0) not in (4, 5):
            continue
        try:
            df = pd.read_csv("build/tables/" + r["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        mc, n = _maso_col(df)
        if n < 2:                # không phải mảnh statement (ít mã chuẩn)
            continue
        typ = _classify(df, mc)
        if typ:
            groups[typ].append((df, mc))

    out = {}
    for typ, items in groups.items():
        # header = row 0 của mảnh đầu (giữ 1 header duy nhất); data = mọi dòng có mã ở mọi mảnh
        first_df, first_mc = items[0]
        ncol = first_df.shape[1]
        rows = [[str(first_df.iloc[0, j]) for j in range(ncol)]]   # header row
        seen = set()
        for df, mc in items:
            if df.shape[1] != ncol:
                continue
            for i in range(len(df)):
                if _is_header_row(df, i, mc):
                    continue
                code = str(df.iloc[i, mc]).strip()
                row = [str(df.iloc[i, j]) for j in range(ncol)]
                key = (code, row[1] if ncol > 1 else "")
                if code and code in _STD and key in seen:   # tránh lặp mã trùng
                    continue
                seen.add(key)
                rows.append(row)
        if len(rows) > 1:
            merged = pd.DataFrame(rows[1:], columns=[str(j) for j in range(ncol)])
            # đặt header làm dòng 0 (hợp đồng grader: iloc[0]=header)
            hdr = pd.DataFrame([rows[0]], columns=[str(j) for j in range(ncol)])
            out[typ] = pd.concat([hdr, merged], ignore_index=True)
    return out
