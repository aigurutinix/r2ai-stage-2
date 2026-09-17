"""AUDITOR #1 (GPT): YEAR-COLUMN MISMATCH. base GIỮ row nhưng đọc SAI cột-năm.
Câu hỏi có đúng 1 năm Y; cột base đọc map về năm khác Y; có ĐÚNG 1 cột map về Y -> sửa MỖI index cột.
Provably structural mismatch (giống unit 10^k). GPU-free. Chạy: PYTHONUTF8=1 python scripts/scan_year_mismatch.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
from kingpro.answering.operand_pipeline import resolve_columns, _num
import pandas as pd

base = json.load(open("sub_base_fix/submission.json", encoding="utf-8"))
CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def col_year(cols, j):
    """Năm rõ ràng của cột j (từ date 31/12/YYYY hoặc year trong header). None nếu không có/nhiều."""
    c = cols.get(j)
    if not c:
        return None
    ys = set(y for _, _, y in c["dates"]) | set(c["years"])
    return list(ys)[0] if len(ys) == 1 else None


cands = []
for e in base:
    q = (e.get("pandas_query") or "").strip()
    if not q or "str.contains" not in q:
        continue
    ys = re.findall(r"\b(20\d{2})\b", e["question"])
    if len(set(ys)) != 1:                       # đúng 1 năm explicit
        continue
    Y = ys[0]
    # df dùng + cột đọc + scale (dòng result)
    m_df = re.search(r"_r\s*=\s*df(\d+)\[", q)
    m_col = re.search(r"_r\['(\w+)'\]\.values\[0\]", q)
    if not (m_df and m_col):
        continue
    dfk = int(m_df.group(1))
    col = m_col.group(1)
    try:
        col_i = int(col)
    except ValueError:
        continue
    ev = e.get("evidence", [])
    if dfk - 1 >= len(ev):
        continue
    tref = ev[dfk - 1].get("table_ref") or None
    # map evidence csv_path (data/xxx.csv) -> build/tables path qua catalog: dùng table_ref nếu có
    csvp = None
    if tref and tref in CAT:
        csvp = "build/tables/" + CAT[tref]["csv_path"]
    if not csvp:
        # thử suy từ csv_path trong evidence (đã copy vào submission/data) — dùng chính file đó
        csvp = "sub_base_fix/" + ev[dfk - 1]["csv_path"]
    try:
        df = pd.read_csv(csvp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        continue
    cols = resolve_columns(df)
    if col_i not in cols:
        continue
    ybase = col_year(cols, col_i)
    if ybase is None or ybase == Y:            # cột base không rõ năm, hoặc đã đúng
        continue
    # có ĐÚNG 1 cột map về Y (khác cột base)
    ycols = [j for j in cols if j != col_i and col_year(cols, j) == Y]
    if len(ycols) != 1:
        continue
    jfix = ycols[0]
    # tính lại: cùng row (lấy từ contains), cột jfix
    lab_m = re.search(r"str\.contains\('([^']*)'", q)
    if not lab_m:
        continue
    label = lab_m.group(1)
    lc = 0
    idxs = [i for i in range(len(df)) if label.lower() in str(df.iloc[i, lc]).lower()]
    if not idxs:
        continue
    row = idxs[0]
    old = _num(df.iloc[row, col_i])
    new = _num(df.iloc[row, jfix])
    if new is None:
        continue
    scale_m = re.search(r"float\(_v\)\s*\*\s*([0-9.eE+-]+)", q)
    scale = float(scale_m.group(1)) if scale_m else 1.0
    cands.append({"id": e["id"], "Y": Y, "col_base": col_i, "yr_base": ybase, "col_fix": jfix,
                  "base_ans": e.get("answer"), "old_cell": old, "new_ans": round(new * scale, 2),
                  "label": label[:30]})

print(f"YEAR-MISMATCH candidates = {len(cands)}")
json.dump(cands, open("build/year_mismatch.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for c in cands[:30]:
    print(f"  Q{c['id']} hoi {c['Y']} | base doc cot{c['col_base']}(nam {c['yr_base']})={c['base_ans']} -> cot{c['col_fix']}={c['new_ans']} | {c['label']}")
