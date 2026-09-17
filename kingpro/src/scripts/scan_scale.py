"""AUDITOR #3 (GPT): SAME-CELL SCALE mismatch theo HEADER LITERAL. Giữ nguyên ô base chọn;
nếu header cột ghi literal đơn vị (Triệu/Tỷ/Nghìn đồng) và scale base DÙNG trái với (header_unit/requested_unit)
-> sửa scale. Không reground. GPU-free. Chạy: PYTHONUTF8=1 python scripts/scan_scale.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.answering.operand_pipeline import resolve_columns, _num, requested_unit, _norm
import pandas as pd

base = json.load(open("sub_base_fix/submission.json", encoding="utf-8"))
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def header_unit(df, col_i, nheader=2):
    sig = " ".join(str(df.iloc[i, col_i]) for i in range(min(nheader, len(df))))
    n = _norm(sig)
    # cả nhãn cột lẫn tiêu đề bảng (dòng 0 cột nhãn) có thể chứa đơn vị
    n2 = _norm(" ".join(str(df.iloc[i, 0]) for i in range(min(nheader, len(df)))))
    s = n + " " + n2
    if "trieu" in s:
        return 1_000_000
    if re.search(r"\bty\b|ty dong|ty vnd", s):
        return 1_000_000_000
    if "nghin" in s:
        return 1_000
    return 1


cands = []
for e in base:
    q = (e.get("pandas_query") or "").strip()
    if not q or "str.contains" not in q:
        continue
    m_df = re.search(r"_r\s*=\s*df(\d+)\[", q)
    m_col = re.search(r"_r\['(\w+)'\]\.values\[0\]", q)
    scale_m = re.search(r"float\(_v\)\s*\*\s*([0-9.eE+-]+)", q)
    if not (m_df and m_col):
        continue
    dfk = int(m_df.group(1))
    try:
        col_i = int(m_col.group(1))
    except ValueError:
        continue
    base_scale = float(scale_m.group(1)) if scale_m else 1.0
    ev = e.get("evidence", [])
    if dfk - 1 >= len(ev):
        continue
    tref = ev[dfk - 1].get("table_ref")
    csvp = ("build/tables/" + CAT[tref]["csv_path"]) if (tref and tref in CAT) else ("sub_base_fix/" + ev[dfk - 1]["csv_path"])
    try:
        df = pd.read_csv(csvp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        continue
    cols = resolve_columns(df)
    if col_i not in cols:
        continue
    hu = header_unit(df, col_i)
    req = requested_unit(e["question"])
    expected = hu / req                         # scale đúng = header_unit / requested_unit
    if expected <= 0:
        continue
    # base_scale khác expected theo bội 10 (>=10x) -> nghi sai scale
    ratio = base_scale / expected
    if 0.5 <= abs(ratio) <= 2:                   # coi như khớp
        continue
    # chỉ nhận khi lệch đúng bội 10^k
    import math
    k = round(math.log10(abs(base_scale) / expected)) if base_scale and expected else 0
    if abs(abs(base_scale) / expected - 10.0 ** k) > 0.05 * (10.0 ** k) or k == 0:
        continue
    lab_m = re.search(r"str\.contains\('([^']*)'", q)
    label = lab_m.group(1) if lab_m else ""
    idxs = [i for i in range(len(df)) if label.lower() in str(df.iloc[i, 0]).lower()]
    if not idxs:
        continue
    cell = _num(df.iloc[idxs[0], col_i])
    if cell is None:
        continue
    cands.append({"id": e["id"], "base_ans": e.get("answer"), "base_scale": base_scale,
                  "header_unit": hu, "req_unit": req, "expected_scale": expected,
                  "new_ans": round(cell * expected, 2), "label": label[:28]})

print(f"SCALE-MISMATCH candidates = {len(cands)}")
json.dump(cands, open("build/scale_mismatch.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for c in cands[:30]:
    print(f"  Q{c['id']} base={c['base_ans']}(scale{c['base_scale']}) -> {c['new_ans']}(scale{c['expected_scale']}) hu={c['header_unit']} req={c['req_unit']} | {c['label']}")
