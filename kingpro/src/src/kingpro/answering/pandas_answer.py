"""Text-to-Pandas: câu hỏi + bảng đã truy hồi -> sinh pandas -> chạy -> đáp án.

Ráp: prompt (bê hợp đồng program_system baseline) + schema-harness (đưa cột+mẫu để model
KHÔNG bịa cột) + code_check (chặn cột bịa trước khi chạy) + sandbox (timeout) + SELF-REPAIR
(baseline không có: crash/không ra số thì đưa lỗi ngược cho model sửa 1-2 vòng).

llm_fn(system, user)->text được tiêm vào để đổi model dễ + test bằng mock.
"""

from __future__ import annotations

import re

import pandas as pd

from kingpro.answering.code_check import check_columns
from kingpro.answering.provenance import bind_evidence
from kingpro.answering.sandbox import run_pandas_code, _sanitize
from kingpro.evaluation.metrics import coerce_number
from kingpro.answering.ma_so_tt200 import maso_of

_STD_CODES = {"01", "02", "10", "11", "20", "21", "22", "23", "25", "26", "30", "31", "32",
              "40", "50", "51", "52", "60", "70", "100", "110", "120", "130", "140", "200",
              "220", "270", "300", "310", "330", "400", "410", "421", "440"}

# Đơn vị tiền câu hỏi HỎI (hệ số so với đồng). Có 'trăm tỷ' (fix bug sai 100×). Dài trước.
_UNITS = [("nghìn tỷ", 1_000_000_000_000), ("trăm tỷ", 100_000_000_000),
          ("tỷ", 1_000_000_000), ("triệu", 1_000_000), ("nghìn", 1_000)]


def requested_unit(question: str) -> tuple[str, int]:
    """Return the answer unit; use the final explicit currency unit in mixed-unit questions."""
    q = question.lower()
    matches: list[tuple[int, int, str, int]] = []

    # A question can contain a filter threshold in one unit and request its
    # answer in another, e.g. ``biên lợi nhuận > 10%, doanh thu thấp nhất là
    # bao nhiêu nghìn tỷ đồng``.  Selecting percent merely because it appears
    # anywhere makes the product replay a numerically correct metric at the
    # wrong scale.  Collect every explicit answer-unit cue and keep the final
    # one in question order; longer phrases win an exact-position tie.
    cue_patterns = (
        (r"điểm\s*phần\s*trăm", "điểm phần trăm", 1),
        (r"phần\s*trăm|%", "phần trăm", 1),
        (r"(?:bao nhiêu|mấy)\s+lần|hệ số", "lần", 1),
        (r"(?:bao nhiêu|mấy)\s+(?:cổ phiếu|cp)\b", "cổ phiếu", 1),
        (r"(?:bao nhiêu|mấy)\s+năm\b|năm\s+nào\b", "năm", 1),
    )
    for pattern, name, mult in cue_patterns:
        for match in re.finditer(pattern, q):
            matches.append((match.end(), len(match.group(0)), name, mult))
    for name, mult in _UNITS:
        for match in re.finditer(rf"{name}\s*(đồng|vnđ|vnd)", q):
            matches.append((match.end(), len(match.group(0)), name + " đồng", mult))
    if matches:
        _end, _length, name, mult = max(matches)
        return name, mult
    return "đồng", 1


def _scan_unit(blob: str) -> tuple[str, int] | None:
    """Tìm ĐVT trong 1 đoạn text ĐÃ FOLD. Chống bắt nhầm 'dong' từ 'hoat dong'/'cong ty dong':
    - 'trieu/nghin ty + dong/vnd' = RÕ đơn vị -> khớp bất kỳ đâu.
    - 'ty/nghin/dong/vnd' MƠ HỒ -> chỉ khớp trong ngữ cảnh 'don vi'/'dvt' (đơn vị tính)."""
    # (1) cụm KHÔNG mơ hồ
    if re.search(r"nghin ty\s*(dong|vnd|vnđ)", blob): return "nghìn tỷ đồng", 1_000_000_000_000
    if re.search(r"tram ty\s*(dong|vnd|vnđ)", blob):  return "trăm tỷ đồng", 100_000_000_000
    if re.search(r"trieu\s*(dong|vnd|vnđ)", blob):    return "triệu đồng", 1_000_000
    # (2) cụm mơ hồ -> cần ngữ cảnh 'don vi'
    m = re.search(r"(don vi tinh|don vi|dvt)\s*:?\s*([a-z0-9% /]{0,22})", blob)
    ctx = m.group(2) if m else ""
    if re.search(r"\bnghin ty\b", ctx):                          return "nghìn tỷ đồng", 1_000_000_000_000
    # ``ty VND`` must be resolved within one cell by ``detect_table_unit``.
    # Searching the whole flattened table created false positives such as
    # ``công ty`` at the end of one cell followed by ``VND`` in the next cell.
    if re.search(r"\bty\b", ctx):                                  return "tỷ đồng", 1_000_000_000
    if re.search(r"\bnghin\b", ctx):                             return "nghìn đồng", 1_000
    if re.search(r"\b(dong|vnd|vnđ)\b", ctx):                    return "đồng", 1
    return None


def detect_table_unit(df: pd.DataFrame, table_ref: str | None = None) -> tuple[str, int] | None:
    """Phát hiện ĐƠN VỊ bảng (triệu/tỷ/nghìn tỷ/đồng) — fix lỗi mất điểm #1 (quên đơn vị, FinQA Chen 2021).
    ĐÃ VÁ: không bắt nhầm 'dong' từ 'hoạt động'; nhận 'Triệu VND'; 3 tầng (quanh bảng -> header -> cấp báo cáo)."""
    from kingpro.retrieval.bm25_index import fold

    # 2) Header đầu bảng (nhanh, không I/O). Scan từng ô trước để ranh giới
    # cột không ghép nhầm ``công ty`` + ``VND`` thành đơn vị ``tỷ VND``.
    header_cells = [fold(str(v)) for v in df.head(8).values.flatten()]
    for col_idx in range(df.shape[1]):
        column_header = [fold(str(df.iloc[row_idx, col_idx])) for row_idx in range(min(3, len(df)))]
        if any(re.search(r"20\d{2}", cell) for cell in column_header) and any(
            re.fullmatch(r"(?:vnd|dong)", cell) for cell in column_header
        ):
            return "đồng", 1
    for cell in header_cells:
        if re.search(r"\bty\s*(dong|vnd)\b", cell) and "cong ty" not in cell:
            return "tỷ đồng", 1_000_000_000
        u = _scan_unit(cell)
        if u:
            return u
    # An explicit VND value-column header means raw đồng, even if the report
    # later contains unrelated text mentioning a tỷ lệ.
    for cell in header_cells:
        if re.search(r"(?:20\d{2}|\d{1,2}/\d{1,2}/20\d{2}).*vnd\b", cell):
            return "đồng", 1
    # OCR frequently merges adjacent header cells as ``VNDNăm trước``.  It is
    # still an explicit raw-VND marker.  Resolve it before the surrounding
    # report fallback can contribute an unrelated "tỷ đồng" mention.
    if any(("vnd" in cell or "vnđ" in cell) for cell in header_cells):
        return "đồng", 1
    u = _scan_unit(fold(" ".join(str(v) for v in df.head(8).values.flatten())))
    if u:
        return u
    if not (table_ref and "|" in table_ref):
        return None
    try:
        rid, line = table_ref.split("|"); line = int(line)
        ticker = rid.split("_")[0]
        ys = re.findall(r"20\d{2}", rid); year = ys[0] if ys else ""
        from pathlib import Path
        p = Path(f"data/financial_statements/{ticker}/{year}/{rid}/{rid}_extracted.txt")
        if not p.exists():
            return None
        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        # 1) OCR quanh bảng (±20 dòng)
        u = _scan_unit(fold(" ".join(lines[max(0, line - 20):line + 2])))
        if u:
            return u
        # 3) FALLBACK cấp BÁO CÁO: dòng 'Đơn vị'/'ĐVT' đầu tiên trong cả file (đơn vị chung 1 báo cáo)
        for ln in lines:
            fl = fold(ln)
            if "don vi" in fl or "dvt" in fl:
                u = _scan_unit(fl)
                if u:
                    return u
    except Exception:
        pass
    return None

# Preamble dán trước mọi pandas_query. Máy chấm GIỚI HẠN builtins (chỉ abs/round/len/min/max/
# sum/sorted/float/int/str/bool/list/dict/set/range/enumerate/zip/all/any/isinstance — KHÔNG có
# globals/getattr/NameError...) và cấp `df` khi ĐÚNG 1 bảng, `dfs` (dict) khi nhiều bảng.
# Vì code LLM/maso dùng df1/df2, ta ALIAS chúng CHỈ bằng list/len (được phép), không globals/try-except.
def _preamble(n_tables: int) -> str:
    n = max(int(n_tables), 1)
    if n == 1:
        return "df1 = list(dfs.values())[0]\n"           # 1 bảng: máy chấm CHỈ cấp `dfs` (KHÔNG df!) -> bind từ dfs
    lines = ["_dfvals = list(dfs.values())"]             # nhiều bảng: máy chấm cấp `dfs` (theo thứ tự evidence)
    for i in range(1, n + 1):
        lines.append(f"if len(_dfvals) >= {i}:")
        lines.append(f"    df{i} = _dfvals[{i - 1}]")
    return "\n".join(lines) + "\n"


def _with_preamble(code: str, n_tables: int) -> str:
    """Dán preamble alias df1..dfN để code chạy dưới đúng hợp đồng máy chấm (builtins giới hạn)."""
    return _preamble(n_tables) + code


def _neutralize_exc(code: str) -> str:
    """Đổi `except <Tên>[ as e]:` -> `except:` (bare) vì máy chấm KHÔNG có lớp exception
    trong builtins (ValueError/Exception... -> NameError). Bare except không cần tên."""
    code = re.sub(r'except\s+[\w.]+(?:\s*,\s*[\w.]+)*(?:\s+as\s+\w+)?\s*:', 'except:', code)
    code = re.sub(r'except\s*\([^)]*\)(?:\s+as\s+\w+)?\s*:', 'except:', code)
    return code


SYSTEM = """Bạn viết code pandas (NHIỀU DÒNG) để tính ra MỘT con số trả lời câu hỏi tài chính.
HỢP ĐỒNG THỰC THI (máy chấm chạy y hệt — sai là 0 điểm):
- `pd` đã sẵn. KHÔNG import gì. Mỗi bảng là DataFrame tên df1, df2, ... đúng như SCHEMA. Dùng đúng các biến đó.
- CHỈ được dùng builtin: abs, round, len, min, max, sum, sorted, float, int, str, bool, list, dict, set, range, enumerate, zip, all, any, isinstance. TUYỆT ĐỐI KHÔNG dùng map, filter, getattr, print, eval, globals hay builtin khác (máy chấm chặn -> NameError).
- TÊN CỘT là CHUỖI (vd '0','1','2'): dùng df1['1'] HOẶC df1.iloc[:, 1]. KHÔNG dùng df1[1] (số) — KeyError.
- MỌI Ô là CHUỖI. Tự parse số kiểu Việt: '.'=ngăn nghìn, ','=thập phân, '(...)'=âm, bỏ '%','$'.
  Vd '1.234,5'->1234.5 ; '(54)'->-54 ; '48,8%'->48.8.
- CHỌN ĐÚNG CỘT theo năm câu hỏi (mỗi cột có chú thích, vd '1'=Năm 2023). Đừng lấy nhầm cột năm so sánh.
- KHỚP DÒNG theo nhãn: `.str.contains('<từ khoá>', case=False, na=False, regex=False)` (KHÔNG dùng ==).
  Có NEO MÃ SỐ bên dưới thì LỌC theo mã số (chính xác hơn dò tên).
- CÁCH LÀM (bắt buộc): MỞ ĐẦU bằng 1–3 dòng CHÚ THÍCH `#` nêu rõ: (1) lấy DÒNG nào (mã số/nhãn), (2) CỘT năm/kỳ nào, (3) CÔNG THỨC. Rồi mới viết code. Suy nghĩ trong chú thích giúp lấy ĐÚNG ô (đừng đoán).
- max/min/sắp xếp: ép float TRƯỚC rồi sort_values(kind='mergesort'). Trước .iloc[0]/.iloc[-1] phải kiểm bảng lọc KHÔNG rỗng.
- Câu TĂNG/GIẢM/THAY ĐỔI không nêu chiều: result = trị tuyệt đối |mới - cũ|. Có nêu chiều thì giữ dấu.
- ĐƠN VỊ & LÀM TRÒN (đọc kỹ — sai là mất câu):
  * TIỀN (doanh thu, lợi nhuận, tài sản, dòng tiền...): quy về đúng đơn vị câu hỏi rồi result = round(giá_trị, 2). KHÔNG nhân 100.
  * PHẦN TRĂM (ROE, ROA, tỷ suất sinh lời, biên lợi nhuận, tỷ trọng, TĂNG TRƯỞNG/tốc độ tăng, câu ghi '%'):
    tính tỉ số thập phân rồi NHÂN 100: result = round(tỉ_số * 100, 2). Vd ROE = LNST/VCSH = 0.1534 -> 15.34.
    Tăng trưởng = (mới/cũ - 1) * 100. Nếu ô đã có sẵn '%' ('15,34%') thì result = 15.34.
  * HỆ SỐ/SỐ LẦN (thanh toán nhanh/hiện hành, vòng quay, D/E hỏi 'lần'): result = round(tỉ_số, 2). KHÔNG nhân 100.
  * ROE/ROA dùng MẪU BÌNH QUÂN: chia cho (đầu_kỳ + cuối_kỳ)/2, KHÔNG chia riêng số cuối kỳ.
- CHỈ dùng đúng tên cột trong SCHEMA, KHÔNG bịa cột. Lấy số TỪ Ô rồi tính bằng code, cấm nhẩm.
- Gán MỘT số vô hướng vào biến `result` (không phải Series/chuỗi/DataFrame). Thiếu dữ kiện: result = None.
- CHỈ trả code Python, không giải thích, không markdown."""


from kingpro.retrieval.bm25_index import fold  # noqa: E402

def _is_value(s: str) -> bool:
    """Ô GIÁ TRỊ = toàn số sau khi bỏ . , % ( ) - khoảng trắng (phân biệt với nhãn 'Năm 2023')."""
    t = re.sub(r"[\s.,%()\-]", "", str(s))
    return t.isdigit() and len(t) > 0


def _flatten_header(df: pd.DataFrame, max_head: int = 4) -> tuple[list[str], int]:
    """Làm phẳng header đa tầng -> tên cột rõ ('Năm 2023 (Triệu đồng)'). Trả (tên_cột, dòng_data_đầu).

    Nguồn: NormTab (Nahid&Rafiei EMNLP 2024) + benchmark định dạng (CSV 44% < Markdown-KV 61%).
    Data = dòng mà PHẦN LỚN ô (trừ cột 0) là GIÁ TRỊ SỐ; header = các dòng nhãn phía trên.
    """
    cols = list(df.columns)
    first_data = 0
    for i in range(min(max_head, len(df))):
        vals = [df.iat[i, j] for j in range(1, len(cols))]
        nonempty = [v for v in vals if str(v).strip() and str(v).strip().lower() != "nan"]
        if nonempty and sum(_is_value(v) for v in nonempty) >= max(1, len(nonempty) // 2):
            first_data = i
            break
    else:
        first_data = min(max_head, len(df))
    names = []
    for j, c in enumerate(cols):
        parts = []
        for i in range(first_data):
            v = str(df.iat[i, j]).strip()
            if v and v.lower() != "nan" and v not in parts:
                parts.append(v)
        nm = " ".join(parts).strip()
        if j == 0:
            nm = nm or "Chỉ tiêu"
        else:
            nm = nm or f"cột{c}"
        names.append(nm)
    return names, first_data


def _render_kv(df: pd.DataFrame, names: list[str], rows: list[int], cols: list) -> str:
    """Mỗi dòng -> Markdown-KV, GIỮ tên cột thật để code dùng: "nhãn | '1'(Năm 2023): val"."""
    out = []
    for i in rows:
        label = str(df.iat[i, 0]).strip()
        kvs = []
        for j in range(1, len(cols)):
            v = str(df.iat[i, j]).strip()
            if v and v.lower() != "nan":
                kvs.append(f"'{cols[j]}'({names[j]}): {v}")
        if kvs:
            out.append(f"- [{label}] " + " | ".join(kvs))
    return "\n".join(out)


def _table_view(var: str, df: pd.DataFrame, question: str, n_head: int = 3, n_rel: int = 12) -> str:
    """NormTab/TAP4LLM-lite: gán nghĩa cột (tiêu đề/năm) + chọn DÒNG liên quan câu hỏi.

    Giữ nguyên tên cột thật ('0','1'...) để code chạy được, nhưng CHÚ THÍCH mỗi cột
    bằng giá trị tiêu đề, và trích các dòng có nhãn khớp từ khoá câu hỏi (bớt nhiễu, đỡ sai ô).
    """
    cols = list(df.columns)
    try:
        names, first_data = _flatten_header(df)
    except Exception:
        names, first_data = [f"cột{c}" for c in cols], 0
    # bản đồ cột GIÀU (AILS-NTUA): tên thật -> nghĩa + kiểu (số/chữ) + số ô có giá trị
    n_data = max(1, len(df) - first_data)
    colmap_parts = []
    for j, c in enumerate(cols):
        vals = [str(df.iat[i, j]).strip() for i in range(first_data, len(df))]
        nonempty = [v for v in vals if v and v.lower() != "nan"]
        n_num = sum(_is_value(v) for v in nonempty)
        if nonempty and n_num >= len(nonempty) * 0.6:      # cột SỐ -> kèm miền min/max (chống lấy sai bậc)
            nums = sorted(x for x in (coerce_number(v) for v in nonempty) if x is not None)
            rng = f", min={nums[0]:g} max={nums[-1]:g}" if nums else ""
            colmap_parts.append(f"'{c}'={names[j]} [SỐ, {len(nonempty)}/{n_data} ô{rng}]")
        else:
            colmap_parts.append(f"'{c}'={names[j]} [chữ, {len(nonempty)}/{n_data} ô]")
    colmap = ", ".join(colmap_parts)
    # dòng liên quan: nhãn cột đầu chứa từ khoá câu hỏi (>=4 ký tự, đã bỏ dấu)
    qterms = {t for t in fold(question).split() if len(t) >= 4}
    rel_idx = []
    col0 = df.iloc[:, 0].astype(str)
    for i in range(first_data, len(df)):
        fl = fold(col0.iat[i])
        if qterms and sum(1 for t in qterms if t in fl) >= 1:
            rel_idx.append(i)
    n_data = len(df) - first_data
    parts = [f"# Bảng `{var}` — CỘT (tên thật để dùng trong code -> nghĩa): {colmap}"]
    # UIT_BlackCoffee (VLSP 2025): lọc context quá tay phản tác dụng -> bảng NHỎ thì đưa TOÀN BỘ dòng.
    if n_data <= 22 or not rel_idx:
        parts.append(f"# TOÀN BỘ {n_data} dòng (mỗi số kèm cột nguồn):")
        parts.append(_render_kv(df, names, list(range(first_data, len(df))), cols))
    else:
        parts.append(f"# DÒNG KHỚP CÂU HỎI ({len(rel_idx)}/{n_data}) — mỗi số kèm cột nguồn, nhiều khả năng chứa đáp án:")
        parts.append(_render_kv(df, names, rel_idx[:n_rel], cols))
    return "\n".join(parts)


# Few-shot ICL (winner Anotheroption: "cải thiện đáng kể"). 2 ví dụ pandas GRADER-SẠCH:
# nhãn-lọc (tiền) + Mã số (%). Model học mẫu parse số VN + round 2 + ×100 cho %.
FEWSHOT = (
    "\n\nHỌC THEO 2 VÍ DỤ (thay tên cột ĐÚNG theo SCHEMA của câu bạn giải):\n"
    "# VD1 — 'Doanh thu thuần năm 2023 (triệu đồng)': lọc dòng theo NHÃN, cột năm 2023, tiền -> round 2\n"
    "_r = df1[df1['0'].str.contains('Doanh thu thuần', case=False, na=False, regex=False)]\n"
    "_v = str(_r['2'].values[0]).replace('(','-').replace(')','').replace('.','').replace(',','.')\n"
    "result = round(float(_v), 2)\n"
    "# VD2 — 'Biên lợi nhuận gộp năm 2023 (%)': LN gộp(Mã 20)/DT thuần(Mã 10) rồi ×100 -> round 2\n"
    "_g = str(df1[df1['1'].astype(str).str.strip()=='20']['2'].values[0]).replace('.','').replace(',','.')\n"
    "_d = str(df1[df1['1'].astype(str).str.strip()=='10']['2'].values[0]).replace('.','').replace(',','.')\n"
    "result = round(float(_g) / float(_d) * 100, 2)\n"
)

_MONEY_RE = re.compile(r"tỷ|triệu|nghìn|đồng|vnđ|vnd", re.IGNORECASE)
_RATIO_RE = re.compile(r"tỷ lệ|tỉ lệ|%|phần trăm|biên|hệ số|vòng quay|số ngày|roe|roa|r\s*o\s*e|d/e|lần", re.IGNORECASE)

# ×100 (percentage) vs không ×100 (number/lần) — theo gold BTC (intermediate_compiler + format_terminal):
# ROA/ROE + tăng trưởng + tỷ suất/biên/tỷ trọng/% = ×100 ; "lần"/hệ số/vòng quay = giữ nguyên.
_PCT_RE = re.compile(r"\broe\b|\broa\b|\bros\b|tỷ suất|tỉ suất|biên lợi nhuận|\bbiên\b|phần trăm|%|tăng trưởng|tốc độ tăng|tỷ trọng|tỉ trọng", re.IGNORECASE)
_LAN_RE = re.compile(r"bao nhiêu lần|\blần\b|hệ số thanh toán|hệ số khả năng|vòng quay|số ngày", re.IGNORECASE)


def percent_hint(question: str) -> str:
    """Gợi ý dạng kết quả (×100 hay không) theo tín hiệu câu hỏi — khớp cách BTC sinh gold."""
    if _LAN_RE.search(question):
        return ("\n\nDẠNG KẾT QUẢ: câu HỆ SỐ/LẦN -> result = round(tỉ_số, 2), KHÔNG nhân 100.")
    if _PCT_RE.search(question):
        return ("\n\nDẠNG KẾT QUẢ: câu PHẦN TRĂM -> result = round(tỉ_số * 100, 2) "
                "(vd biên/ROE 0.1534 -> 15.34; tăng trưởng = (mới/cũ - 1)*100). "
                "ROE/ROA chia mẫu BÌNH QUÂN (đầu+cuối)/2.")
    return ""


def build_user(question: str, tables: list[dict]) -> tuple[str, dict, set]:
    """Trả (user_prompt, csv_paths, schema_columns). Có phát hiện+quy đổi ĐƠN VỊ tất định."""
    csv_paths: dict = {}
    schema_cols: set = set()
    blocks = []
    unit_lines = []
    uname, umult = requested_unit(question)
    is_money = bool(_MONEY_RE.search(question)) and not (umult == 1 and _RATIO_RE.search(question) and not re.search(r"đồng|vnđ|vnd", question, re.I))
    for i, t in enumerate(tables):
        var = f"df{i + 1}"
        csv_paths[var] = t["csv_path"]
        df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        schema_cols |= {str(c) for c in df.columns}
        blocks.append(_table_view(var, df, question))
        if is_money:
            tu = detect_table_unit(df, t.get("table_ref"))
            if tu:
                tname, tmult = tu
                factor = tmult / umult
                if abs(factor - 1.0) < 1e-9:
                    unit_lines.append(f"- `{var}` theo **{tname}** = đúng đơn vị câu hỏi ({uname}). GIỮ NGUYÊN giá trị tiền.")
                else:
                    unit_lines.append(f"- `{var}` theo **{tname}**. Để ra {uname}: NHÂN giá trị tiền lấy từ `{var}` với {factor:g}.")
            else:
                unit_lines.append(f"- `{var}`: KHÔNG rõ ĐVT — nếu số rất lớn (hàng tỉ) thì bảng theo đồng, chia {umult} ra {uname}.")
    if is_money:
        unit_note = ("\n\nĐƠN VỊ (BẮT BUỘC làm đúng — sai đơn vị là sai đáp án):\n"
                     + f"Câu hỏi cần kết quả theo **{uname}**.\n" + "\n".join(unit_lines)
                     + "\nChỉ quy đổi GIÁ TRỊ TIỀN, KHÔNG quy đổi tỉ lệ/%/hệ số. result = round(giá_trị, 2).")
    else:
        unit_note = ("\n\nĐÂY LÀ CÂU TỈ LỆ/HỆ SỐ — KHÔNG quy đổi đơn vị tiền.")
    unit_note += percent_hint(question)                    # ×100 hay không, theo gold BTC
    from kingpro.answering.ma_so_tt200 import maso_hint
    user = (
        f"Câu hỏi: {question}\n\nSCHEMA (chỉ dùng đúng các cột này):\n"
        + "\n\n".join(blocks)
        + unit_note
        + maso_hint(question)                              # EDGE: neo Mã số TT200 (53% câu) chống nhầm dòng
        + FEWSHOT                                          # Nhóm A: few-shot ICL (mẫu parse + round 2 + ×100)
        + "\n\nViết code gán `result` (mở đầu bằng chú thích # nêu dòng/cột/công thức)."
    )
    return user, csv_paths, schema_cols


def maso_answer(question: str, tables: list[dict]) -> dict | None:
    """TRẢ LỜI TẤT ĐỊNH bằng Mã số TT200 — KHÔNG gọi LLM. None nếu không áp dụng/không chắc.

    Phủ ~53% câu (chỉ tiêu chuẩn), $0, gần như chắc đúng. Conservative: chỉ trả khi xác định
    được ĐÚNG cột năm + đúng dòng mã số; mơ hồ thì trả None để đường LLM lo (giữ precision cao).
    """
    m = maso_of(question)
    if not m:
        return None
    code, _name = m
    # GUARD: câu TĂNG TRƯỞNG/BIẾN ĐỘNG/TỔNG-HỢP -> tra 1 giá trị là SAI (val: growth 26/26 sai) -> để LLM lo
    if re.search(r"tăng trưởng|tốc độ tăng|tỷ lệ tăng|tăng.{0,8}bao nhiêu|giảm.{0,8}bao nhiêu|"
                 r"biến động|chênh lệch|thay đổi|so với.{0,12}(năm|cùng kỳ)|tổng .{2,30} và ", question, re.I):
        return None
    yrs = re.findall(r"\b(20\d{2})\b", question)
    year = yrs[-1] if yrs else None
    want_dau = bool(re.search(r"đầu năm|đầu kỳ|01/01|1/1", question, re.I))
    want_cuoi = bool(re.search(r"cuối năm|cuối kỳ|31/12", question, re.I))
    uname, umult = requested_unit(question)
    is_money = bool(_MONEY_RE.search(question)) and not (umult == 1 and _RATIO_RE.search(question) and not re.search(r"đồng|vnđ|vnd", question, re.I))
    for t in tables:
        try:
            df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        cols = list(df.columns)
        # 1) cột Mã số = cột có nhiều mã chuẩn nhất (>=3)
        maso_j, best = None, 2
        for j in range(len(cols)):
            cnt = sum(1 for v in df.iloc[:, j] if str(v).strip() in _STD_CODES)
            if cnt > best:
                best, maso_j = cnt, j
        if maso_j is None:
            continue
        # 2) dòng có Mã số == code
        col_ms = df.iloc[:, maso_j].astype(str).str.strip()
        idxs = [i for i in range(len(df)) if col_ms.iat[i] == code]
        if not idxs:
            continue
        row_i = idxs[0]
        # 3) cột năm: các cột SỐ (khác cột nhãn 0 và cột mã số)
        names, first_data = _flatten_header(df)
        val_cols = []
        for j in range(1, len(cols)):
            if j == maso_j:
                continue
            nn = [str(df.iat[i, j]).strip() for i in range(first_data, len(df))]
            ne = [v for v in nn if v and v.lower() != "nan"]
            if ne and sum(_is_value(v) for v in ne) >= len(ne) * 0.6:
                val_cols.append(j)
        if not val_cols:
            continue
        year_j = None
        if year:
            cand = [j for j in val_cols if year in names[j]]
            if want_dau:
                cand = [j for j in cand if "đầu" in names[j].lower()] or cand
            if want_cuoi:
                cand = [j for j in cand if "cuối" in names[j].lower() or "31/12" in names[j]] or cand
            if len(cand) == 1:
                year_j = cand[0]
        if year_j is None and len(val_cols) == 1:          # chỉ 1 cột số -> chắc chắn
            year_j = val_cols[0]
        if year_j is None and year and len(val_cols) <= 3:  # NỚI: chọn cột theo VỊ TRÍ (TT200: trái=năm báo cáo, phải=năm trước)
            ryrs = re.findall(r"20\d{2}", t.get("table_ref", ""))
            ry = int(ryrs[0]) if ryrs else None
            if ry is not None:
                off = ry - int(year) + (1 if want_dau else 0)   # 0=năm báo cáo(cột trái); 1=năm trước; đầu-kỳ lùi 1 cột
                if 0 <= off < len(val_cols):
                    year_j = val_cols[off]
        if year_j is None:                                 # vẫn mơ hồ -> để LLM lo (giữ precision)
            continue
        if coerce_number(str(df.iat[row_i, year_j])) is None:
            continue
        factor = 1.0
        if is_money:                                       # hệ số quy đổi đơn vị
            tu = detect_table_unit(df, t.get("table_ref"))
            if tu:
                factor = tu[1] / umult
        # SINH PANDAS THẬT (Text-to-Pandas hợp lệ, execution tính) — tra dòng theo Mã số:
        mc, yc = cols[maso_j], cols[year_j]
        code_str = _with_preamble(
            f"_v = str(df1[df1['{mc}'].astype(str).str.strip()=='{code}']['{yc}'].values[0]).strip()\n"
            f"_neg = '(' in _v\n"
            f"_v = _v.replace('(','').replace(')','').replace('%','').replace(' ','').replace('.','').replace(',','.')\n"
            f"_x = float(_v) * (-1 if _neg else 1)\n"
            f"result = round(_x * {factor:g}, 2)",
            1,                                             # maso luôn 1 bảng -> alias df1 = df
        )
        res = run_pandas_code(code_str, {"df1": t["csv_path"]}, timeout=5.0)
        rv = coerce_number(res.get("result")) if res.get("ok") else None
        if rv is None:
            continue
        return {
            "ok": True, "answer": rv, "pandas_query": code_str,
            "evidence": [{"variable": "df1", "csv_path": t["csv_path"], "table_ref": t["table_ref"]}],
            "method": "maso",
        }
    return None


def _locate_maso_cell(df, code, year, want_dau, want_cuoi, table_ref):
    """Trả (tên_cột_mã, tên_cột_năm, dòng) cho dòng Mã=code + cột năm khớp. None nếu mơ hồ.
    Bê nguyên logic dò của maso_answer (giữ precision cao)."""
    cols = list(df.columns)
    maso_j, best = None, 2
    for j in range(len(cols)):
        cnt = sum(1 for v in df.iloc[:, j] if str(v).strip() in _STD_CODES)
        if cnt > best:
            best, maso_j = cnt, j
    if maso_j is None:
        return None
    col_ms = df.iloc[:, maso_j].astype(str).str.strip()
    idxs = [i for i in range(len(df)) if col_ms.iat[i] == code]
    if not idxs:
        return None
    row_i = idxs[0]
    names, first_data = _flatten_header(df)
    val_cols = []
    for j in range(1, len(cols)):
        if j == maso_j:
            continue
        ne = [v for v in (str(df.iat[i, j]).strip() for i in range(first_data, len(df))) if v and v.lower() != "nan"]
        if ne and sum(_is_value(v) for v in ne) >= len(ne) * 0.6:
            val_cols.append(j)
    if not val_cols:
        return None
    year_j = None
    if year:
        cand = [j for j in val_cols if year in names[j]]
        if want_dau:
            cand = [j for j in cand if "đầu" in names[j].lower()] or cand
        if want_cuoi:
            cand = [j for j in cand if "cuối" in names[j].lower() or "31/12" in names[j]] or cand
        if len(cand) == 1:
            year_j = cand[0]
    if year_j is None and len(val_cols) == 1:
        year_j = val_cols[0]
    if year_j is None and year and len(val_cols) == 2:
        ryrs = re.findall(r"20\d{2}", table_ref or "")
        if (ryrs[0] if ryrs else None) == year and not want_dau:
            year_j = val_cols[0]
    if year_j is None:
        return None
    return cols[maso_j], cols[year_j], row_i


# Tỷ số theo Mã số: (regex, tử_mã, mẫu_mã). Đều ×100, round 2 (đúng gold BTC). Cùng 1 bảng KQKD.
_RATIO_MASO = {
    "biên lợi nhuận gộp": (re.compile(r"biên lợi nhuận gộp|tỷ suất lợi nhuận gộp", re.I), "20", "10"),
    "biên lợi nhuận ròng": (re.compile(r"biên lợi nhuận (ròng|thuần)|tỷ suất lợi nhuận (ròng|thuần|sau thuế)", re.I), "60", "10"),
}

# ROE/ROA: tử = LNST(60) ở KQKD ; mẫu = BÌNH QUÂN (đầu+cuối)/2 của Mã ở CĐKT ; ×100 (đúng gold).
_RATIO_AVG = {
    "roe": (re.compile(r"\broe\b|tỷ suất.*vốn chủ|lợi nhuận.*trên vốn chủ|sinh lời.*vốn chủ", re.I), "60", "400"),
    "roa": (re.compile(r"\broa\b|tỷ suất.*(tổng )?tài sản|lợi nhuận.*trên.*tài sản|sinh lời.*tài sản", re.I), "60", "270"),
}


def _locate_maso_avg(df, code):
    """Tìm dòng Mã=code + trả (cột_mã, [cột_val_a, cột_val_b], dòng) khi bảng có ĐÚNG 2 cột số
    (đầu kỳ + cuối kỳ) để tính bình quân. None nếu không đúng 2 cột (giữ precision)."""
    cols = list(df.columns)
    maso_j, best = None, 2
    for j in range(len(cols)):
        cnt = sum(1 for v in df.iloc[:, j] if str(v).strip() in _STD_CODES)
        if cnt > best:
            best, maso_j = cnt, j
    if maso_j is None:
        return None
    col_ms = df.iloc[:, maso_j].astype(str).str.strip()
    idxs = [i for i in range(len(df)) if col_ms.iat[i] == code]
    if not idxs:
        return None
    _, first_data = _flatten_header(df)
    val_cols = []
    for j in range(1, len(cols)):
        if j == maso_j:
            continue
        ne = [v for v in (str(df.iat[i, j]).strip() for i in range(first_data, len(df))) if v and v.lower() != "nan"]
        if ne and sum(_is_value(v) for v in ne) >= len(ne) * 0.6:
            val_cols.append(j)
    if len(val_cols) != 2:
        return None
    return cols[maso_j], [cols[val_cols[0]], cols[val_cols[1]]], idxs[0]


def _parse_cell_lines(prefix, dfvar, mc, yc, code):
    """Sinh các dòng pandas: lấy ô [Mã=code, cột yc] -> parse số VN -> biến {prefix}."""
    return (
        f"{prefix}_s = str({dfvar}[{dfvar}['{mc}'].astype(str).str.strip()=='{code}']['{yc}'].values[0]).strip()\n"
        f"{prefix}_neg = '(' in {prefix}_s\n"
        f"{prefix}_s = {prefix}_s.replace('(','').replace(')','').replace('%','').replace(' ','').replace('.','').replace(',','.')\n"
        f"{prefix} = float({prefix}_s) * (-1 if {prefix}_neg else 1)\n"
    )


def ratio_answer(question: str, tables: list[dict]) -> dict | None:
    """TẤT ĐỊNH: biên lợi nhuận gộp/ròng = Mã(tử)/Mã(10) ×100, round 2 — sinh pandas THẬT.
    Chỉ trả khi tìm được CẢ tử lẫn mẫu trong CÙNG 1 bảng KQKD (giữ precision). None nếu không chắc."""
    yrs = re.findall(r"\b(20\d{2})\b", question)
    year = yrs[-1] if yrs else None
    want_dau = bool(re.search(r"đầu năm|đầu kỳ|01/01|1/1", question, re.I))
    want_cuoi = bool(re.search(r"cuối năm|cuối kỳ|31/12", question, re.I))
    spec = next((v for v in _RATIO_MASO.values() if v[0].search(question)), None)
    num_code, den_code = (spec[1], spec[2]) if spec else (None, None)
    for i, t in (enumerate(tables) if spec else []):      # nhánh margin; không match -> rơi xuống ROE/ROA
        try:
            df = pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        loc_n = _locate_maso_cell(df, num_code, year, want_dau, want_cuoi, t.get("table_ref"))
        loc_d = _locate_maso_cell(df, den_code, year, want_dau, want_cuoi, t.get("table_ref"))
        if not loc_n or not loc_d:
            continue
        (mc_n, yc_n, _), (mc_d, yc_d, _) = loc_n, loc_d
        code_str = _with_preamble(
            _parse_cell_lines("_num", "df1", mc_n, yc_n, num_code)
            + _parse_cell_lines("_den", "df1", mc_d, yc_d, den_code)
            + "result = round(_num / _den * 100, 2)\n",
            1,
        )
        res = run_pandas_code(code_str, {"df1": t["csv_path"]}, timeout=5.0)
        rv = coerce_number(res.get("result")) if res.get("ok") else None
        if rv is None:
            continue
        return {
            "ok": True, "answer": rv, "pandas_query": code_str,
            "evidence": [{"variable": "df1", "csv_path": t["csv_path"], "table_ref": t["table_ref"]}],
            "method": "ratio",
        }

    # --- ROE/ROA: mẫu BÌNH QUÂN (đầu+cuối)/2, tử LNST(60) ở KQKD + mẫu ở CĐKT (CÙNG báo cáo) ---
    spec2 = next((v for v in _RATIO_AVG.values() if v[0].search(question)), None)
    if spec2:
        _, num_code, den_code = spec2

        def _rd(t):
            try:
                return pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
            except Exception:
                return None

        num_loc = None
        for t in tables:
            df = _rd(t)
            ln = _locate_maso_cell(df, num_code, year, want_dau, want_cuoi, t.get("table_ref")) if df is not None else None
            if ln:
                num_loc = (t, ln[0], ln[1])
                break
        den_loc = None
        if num_loc:
            num_rid = str(num_loc[0].get("table_ref", "")).split("|")[0]
            for t in tables:
                if str(t.get("table_ref", "")).split("|")[0] != num_rid:   # phải CÙNG báo cáo
                    continue
                df = _rd(t)
                ld = _locate_maso_avg(df, den_code) if df is not None else None
                if ld:
                    den_loc = (t, ld[0], ld[1])
                    break
        if num_loc and den_loc:
            t_n, mc_n, yc_n = num_loc
            t_d, mc_d, (cd_a, cd_b) = den_loc
            same = t_n["csv_path"] == t_d["csv_path"]
            dv = "df1" if same else "df2"
            body = (
                _parse_cell_lines("_num", "df1", mc_n, yc_n, num_code)
                + _parse_cell_lines("_da", dv, mc_d, cd_a, den_code)
                + _parse_cell_lines("_db", dv, mc_d, cd_b, den_code)
                + "_den = (_da + _db) / 2\n"
                + "result = round(_num / _den * 100, 2)\n"
            )
            csv_paths = {"df1": t_n["csv_path"]} if same else {"df1": t_n["csv_path"], "df2": t_d["csv_path"]}
            code_str = _with_preamble(body, len(csv_paths))
            res = run_pandas_code(code_str, csv_paths, timeout=5.0)
            rv = coerce_number(res.get("result")) if res.get("ok") else None
            if rv is not None:
                ev = [{"variable": "df1", "csv_path": t_n["csv_path"], "table_ref": t_n["table_ref"]}]
                if not same:
                    ev.append({"variable": "df2", "csv_path": t_d["csv_path"], "table_ref": t_d["table_ref"]})
                return {"ok": True, "answer": rv, "pandas_query": code_str, "evidence": ev, "method": "ratio"}
    return None


def maso_rescue(question: str, tables: list[dict], res: dict) -> dict:
    """Option C: LLM sinh pandas (in-spirit). Mã số tất định làm lớp VERIFY/CỨU.
    - LLM khớp giá trị tra-mã-số -> giữ LLM (đồng thuận, tin cao).
    - LLM fail HOẶC lệch -> dùng Mã số (kèm pandas THẬT tra mã số) -> cứu.
    """
    mres = ratio_answer(question, tables) or maso_answer(question, tables)   # tỷ số (biên) trước, rồi mã số
    if not mres:
        return res
    lv = coerce_number(res.get("answer")) if res.get("ok") else None
    mv = mres["answer"]
    if lv is not None and abs(lv - mv) <= abs(mv) * 0.01 + 1e-6:
        return res                                         # đồng thuận -> giữ LLM (đúng tinh thần)
    mres["rescued"] = res.get("ok")                        # đánh dấu để đếm
    return mres                                            # LLM sai/lệch -> cứu bằng Mã số (pandas thật)


def answer_question(question: str, tables: list[dict], llm_fn, *, max_fix: int = 3,
                    timeout: float = 5.0, python_exe: str | None = None, hint: str = "") -> dict:
    user0, csv_paths, schema_cols = build_user(question, tables)
    user = user0 + hint          # hint = chiến lược đa dạng (diverse candidate)
    err = ""
    code = ""
    for attempt in range(max_fix):
        prompt = user if not err else user + f"\n\nLần trước LỖI: {err}\nSửa lại code cho đúng."
        code = _neutralize_exc(_sanitize(llm_fn(SYSTEM, prompt)))   # bỏ except <Tên> (builtins giới hạn)
        ok_cols, bad = check_columns(code, schema_cols)
        if not ok_cols:
            err = f"code dùng cột không có trong schema: {bad}"
            continue
        full = _with_preamble(code, len(tables))          # máy chấm chỉ có df/dfs -> preamble alias df1..dfN
        res = run_pandas_code(full, csv_paths, timeout=timeout, python_exe=python_exe)
        _r = res.get("result")
        _is_nan = isinstance(_r, float) and _r != _r
        _num = coerce_number(_r) if (res["ok"] and _r is not None and not _is_nan) else None
        if _num is not None:                              # PHẢI là 1 SỐ VÔ HƯỚNG (Series/list -> None -> self-repair)
            evidence = bind_evidence(code, tables)
            if not evidence:
                err = "code không đọc bảng dữ liệu nào trong retrieval trace"
                continue
            return {
                "ok": True,
                "answer": _num,
                "pandas_query": full,
                "evidence": evidence,
                "attempts": attempt + 1,
            }
        if res["ok"] and _r is not None and not _is_nan:
            err = "kết quả KHÔNG phải 1 số vô hướng (có thể là Series/nhiều giá trị) — gán result = MỘT số duy nhất"
        else:
            err = res["error"] or ("kết quả là NaN (không tìm/không tính được) — thử ô/cách khác" if _is_nan else "code không ra số")
    return {"ok": False, "answer": None, "pandas_query": _with_preamble(code, len(tables)) if code else "",
            "evidence": [], "attempts": max_fix, "error": err}
