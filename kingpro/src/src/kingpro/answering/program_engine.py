"""Engine 'program' — bê KỶ LUẬT answering của vendor baseline (0.36) vào pipeline mình.

Khác engine cũ (answer_question) ĐÚNG 3 điểm mà workflow chẩn đoán là nút thắt trích số:
  1. Render bảng = CSV THÔ (df.to_csv) thay vì _table_view FLATTEN header — bảng thật header đa dòng
     làm _flatten_header mất cột năm (bug xác nhận). CSV thô cho model thấy ĐÚNG cấu trúc thật.
  2. Bơm METADATA tường minh mỗi bảng: ticker, năm, và ĐƠN VỊ bảng + HỆ SỐ QUY ĐỔI ra đơn vị câu hỏi
     (fix lỗi #1 FinQA: sai đơn vị = sai bậc 10^6). Đây là chỗ mình làm TỐT HƠN vendor (vendor để model
     tự đoán đơn vị từ text; mình trích thẳng dòng ĐVT + ra lệnh nhân hệ số).
  3. Giữ nguyên hợp đồng máy chấm đã chạy được: preamble alias df1..dfN từ dfs, evidence variable df1..dfN,
     builtins giới hạn, self-debug đưa traceback vào sửa (Self-Debug, Chen et al. 2023).

Nguồn kỷ luật: vendor prompts/answering/program_system.txt (data_contract/reliability_rules) — ta đã có
SYSTEM tiếng Việt tương đương (mạnh hơn: có %×100, mẫu bình quân ROE) trong pandas_answer, tái dùng luôn.
"""
from __future__ import annotations

import re

import pandas as pd

from kingpro.answering.pandas_answer import (
    SYSTEM,
    _preamble,
    _neutralize_exc,
    detect_table_unit,
    requested_unit,
)
from kingpro.answering.code_check import check_columns
from kingpro.answering.provenance import bind_evidence
from kingpro.answering.sandbox import run_pandas_code, _sanitize
from kingpro.evaluation.metrics import coerce_number

# Hàm parse SỐ KIỂU VIỆT nhúng thẳng vào pandas_query (grader chạy nó luôn). CHỈ dùng builtin cho phép
# (str/float/join/genexpr) — KHÔNG re, KHÔNG import. Fix 3 lỗi fail chính: ValueError float('1.234.567'),
# KeyError df[0], ImportError. Model được lệnh gọi _num(ô) thay vì tự float().
_NUM_HELPER = (
    "def _num(x):\n"
    "    s = ''.join(ch for ch in str(x) if ch in '0123456789.,-()%')\n"
    "    s = s.replace('%', '')\n"
    "    if s == '' or s == '-': return None\n"
    "    neg = ('(' in s) or s.startswith('-')\n"
    "    s = s.replace('(', '').replace(')', '').replace('-', '')\n"
    "    s = s.replace('.', '').replace(',', '.')\n"
    "    if s == '' or s == '.': return None\n"
    "    r = float(s)\n"
    "    return -r if neg else r\n"
)


def _program_preamble(n_tables: int) -> str:
    """_num helper + alias df1..dfN từ dfs (đúng hợp đồng máy chấm)."""
    return _NUM_HELPER + _preamble(n_tables)


def _read(csv_path):
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _meta_line(table_ref: str, df: pd.DataFrame, req_name: str, req_mult: int) -> str:
    """Dòng metadata tường minh: ticker, năm, ĐƠN VỊ bảng + hệ số quy đổi ra đơn vị câu hỏi."""
    rid = str(table_ref).split("|")[0]
    ticker = rid.split("_")[0]
    ys = re.findall(r"20\d{2}", rid)
    year = ys[0] if ys else "?"
    try:
        u = detect_table_unit(df, table_ref)
    except Exception:
        u = None
    if u:
        name, mult = u
        factor = mult / req_mult
        # hệ số nhân trực tiếp ô -> ra đơn vị câu hỏi (fix sai bậc 10^6)
        fac_txt = f"{factor:g}"
        unit = (f"đơn vị bảng = {name}; để ra '{req_name}' NHÂN mỗi giá trị tiền với {fac_txt}")
    else:
        unit = "đơn vị bảng KHÔNG khai báo (suy từ độ lớn số/ngữ cảnh; nếu là 'triệu đồng' thì ×1e6 ra đồng)"
    return f"ticker={ticker}; năm báo cáo={year}; {unit}"


def build_user_raw(question: str, tables: list[dict]) -> tuple[str, dict]:
    """User prompt kiểu vendor: câu hỏi + đơn vị yêu cầu + từng bảng dạng CSV THÔ kèm metadata/đơn vị."""
    req_name, req_mult = requested_unit(question)
    blocks, csv_paths = [], {}
    for i, t in enumerate(tables, 1):
        var = f"df{i}"
        df = _read(t["csv_path"])
        csv_paths[var] = t["csv_path"]
        raw = df.to_csv(index=False)
        if len(raw) > 6500:                       # cắt bớt bảng quá dài (giữ đủ phần đầu chứa các mã/nhãn chính)
            raw = raw[:6500] + "\n...(bảng bị cắt bớt)"
        blocks.append(
            f"### {var}  [{_meta_line(t['table_ref'], df, req_name, req_mult)}]\n"
            f"(CSV thô — MỌI ô là chuỗi; tên cột là dòng đầu)\n{raw}"
        )
    rules = (
        "QUY TẮC BẮT BUỘC (sai là 0 điểm):\n"
        "- ĐÃ CÓ SẴN hàm _num(ô): parse số kiểu Việt ('.'=ngăn nghìn, ','=thập phân, '(x)'=âm, bỏ '%'). "
        "LUÔN dùng _num(cell) để lấy số — TUYỆT ĐỐI KHÔNG float(chuỗi) trực tiếp (float('1.234.567') sẽ lỗi).\n"
        "- Cột là CHUỖI: dùng df1.iloc[:, i] HOẶC df1['<tên cột CHÍNH XÁC copy từ CSV>']. "
        "KHÔNG viết df1[0]/df1[1] (số nguyên) -> KeyError.\n"
        "- KHÔNG import gì, KHÔNG dùng re/__import__/globals/getattr. Chỉ có pd, các df, và builtin cơ bản.\n"
        "- Lọc dòng theo nhãn: .str.contains('từ khoá', case=False, na=False, regex=False); hoặc so cột Mã số == '60'.\n"
        "- Kiểm bảng lọc KHÔNG rỗng trước .iloc[0]; gán MỘT số vào result; round(result, 2) ở bước cuối.\n"
    )
    user = (
        f"<câu_hỏi>\n{question}\n</câu_hỏi>\n"
        f"ĐƠN VỊ CÂU HỎI YÊU CẦU: {req_name}. (nếu là phần trăm/tỷ suất thì nhân 100 như luật)\n\n"
        + rules +
        f"\nCó {len(tables)} bảng, đã nạp sẵn thành các DataFrame df1..df{len(tables)} "
        f"(ô là CHUỖI thô):\n\n" + "\n\n".join(blocks)
    )
    return user, csv_paths


def _vote(samples: list[tuple]) -> tuple:
    """samples = [(value, full_code, generated_code)]. Chọn cụm đồng thuận (rel tol 0.5%);
    đại diện = phần tử gần trung bình cụm nhất (giá trị + CODE khớp nhau, vì grader chạy code)."""
    best = []
    for sample_i in samples:
        vi = sample_i[0]
        cl = [sample_j for sample_j in samples
              if abs(vi - sample_j[0]) <= 0.005 * max(abs(vi), abs(sample_j[0]), 1.0)]
        if len(cl) > len(best):
            best = cl
    mean = sum(sample[0] for sample in best) / len(best)
    rep = min(best, key=lambda sample: abs(sample[0] - mean))
    return rep[0], rep[1], rep[2], len(best), len(samples)


def run_program(question: str, tables: list[dict], llm_fn, *, max_fix: int = 2, n_vote: int = 1,
                timeout: float = 6.0, python_exe: str | None = None) -> dict:
    """Sinh code -> chạy đúng hợp đồng grader -> self-debug. n_vote>1: sinh N mẫu (temp>0),
    gom kết quả, lấy cụm đồng thuận đông nhất (Self-Consistency, Wang 2022)."""
    if not tables:
        return {"ok": False, "answer": None, "pandas_query": "", "evidence": [], "attempts": 0}
    user0, csv_paths = build_user_raw(question, tables)
    schema_columns: set[str] = set()
    for path in csv_paths.values():
        try:
            schema_columns.update(str(c) for c in _read(path).columns)
        except Exception:
            pass
    samples, last_full = [], ""
    for _s in range(max(1, n_vote)):
        err, code = "", ""
        for _attempt in range(max_fix):
            prompt = user0 if not err else user0 + f"\n\nLần trước LỖI khi chạy: {err}\nSửa code cho chạy đúng, gán result = MỘT số."
            code = _neutralize_exc(_sanitize(llm_fn(SYSTEM, prompt)))
            ok_columns, invented = check_columns(code, schema_columns)
            if not ok_columns:
                err = f"code dùng cột không tồn tại trong các bảng nguồn: {invented}"
                continue
            full = _program_preamble(len(tables)) + code
            last_full = full
            res = run_pandas_code(full, csv_paths, timeout=timeout, python_exe=python_exe)
            r = res.get("result")
            is_nan = isinstance(r, float) and r != r
            num = coerce_number(r) if (res["ok"] and r is not None and not is_nan) else None
            if num is not None:
                samples.append((num, full, code))
                break
            if res["ok"] and r is not None and not is_nan:
                err = "result KHÔNG phải 1 số vô hướng (Series/DataFrame?) — gán result = MỘT số duy nhất"
            else:
                err = res["error"] or ("result là NaN — thử ô/cột khác" if is_nan else "code không ra số")
    if not samples:
        return {"ok": False, "answer": None, "pandas_query": last_full,
                "evidence": [], "attempts": max(1, n_vote), "error": err if 'err' in dir() else ""}
    value, full, generated_code, agree, total = _vote(samples)
    evidence = bind_evidence(generated_code, tables)
    if not evidence:
        return {"ok": False, "answer": None, "pandas_query": full,
                "evidence": [], "attempts": total,
                "error": "GroundingError: code không đọc bảng dữ liệu nào trong retrieval trace"}
    return {
        "ok": True,
        "answer": value,
        "pandas_query": full,
        "evidence": evidence,
        "attempts": total, "agree": agree,
    }
