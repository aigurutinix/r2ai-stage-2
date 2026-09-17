"""PILOT tự sinh data SFT (TẤT ĐỊNH, FREE) — bê code deterministic của đội làm bộ sinh pandas.
Ý tưởng: duyệt bảng thật -> chọn dòng có Mã số + cột năm -> template câu hỏi từ NHÃN THẬT ->
cho maso_answer/ratio_answer sinh pandas + đáp án (chạy thật) -> chỉ giữ mẫu ra SỐ.
Chạy: python scripts/gen_sft_pilot.py
"""
import sys, json, re, random
sys.path.insert(0, "src")
import pandas as pd
from kingpro.answering.pandas_answer import maso_answer, ratio_answer, build_user, SYSTEM
from kingpro.answering.ma_so_tt200 import _MASO

CAT = [json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8")]
STD_CODES = set(c.replace("cf", "") for c in _MASO)


def csv_full(row): return "build/tables/" + row["csv_path"]
def company_of(st):
    m = re.search(r"Công ty:\s*(.+?)\s*\(mã", st or ""); return m.group(1) if m else None


def maso_col(df):
    for j in range(len(df.columns)):
        if sum(1 for v in df.iloc[:, j] if str(v).strip() in STD_CODES) >= 3:
            return j
    return None


UNITS = ["triệu đồng", "tỷ đồng"]
RATIOS = ["ROE", "ROA", "biên lợi nhuận gộp", "biên lợi nhuận ròng"]

def gen():
    random.seed(7)
    cand = [r for r in CAT if r["n_rows"] >= 8 and r["n_cols"] >= 2]
    random.shuffle(cand)
    lookup, ratio = [], []
    for r in cand:
        if len(lookup) >= 25 and len(ratio) >= 8:
            break
        p = csv_full(r)
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        mj = maso_col(df)
        if mj is None:
            continue
        company = company_of(r["search_text"]) or r["ticker"]
        year, scope, tk = r["year"], r.get("scope", ""), r["ticker"]
        tables = [{"table_ref": r["table_ref"], "csv_path": p}]
        # LOOKUP: mỗi dòng có mã số -> câu hỏi từ nhãn thật
        col_ms = df.iloc[:, mj].astype(str).str.strip()
        for i in range(len(df)):
            if len(lookup) >= 25:
                break
            if col_ms.iat[i] not in STD_CODES:
                continue
            label = str(df.iat[i, 0]).strip()
            if len(label) < 4 or label.lower() in ("nan", ""):
                continue
            unit = random.choice(UNITS)
            q = f"{label} của công ty {company} ({tk}) {scope} năm {year} là bao nhiêu {unit}?"
            try:
                res = maso_answer(q, tables)
            except Exception:
                res = None
            if res and res.get("answer") is not None:
                lookup.append({"question": q, "pandas_query": res["pandas_query"],
                               "answer": res["answer"], "type": "lookup", "table_ref": r["table_ref"]})
        # RATIO: thử ROE/ROA/biên
        for rn in RATIOS:
            if len(ratio) >= 8:
                break
            q = f"{rn} của công ty {company} ({tk}) {scope} năm {year} là bao nhiêu %?"
            try:
                res = ratio_answer(q, tables)
            except Exception:
                res = None
            if res and res.get("answer") is not None:
                ratio.append({"question": q, "pandas_query": res["pandas_query"],
                              "answer": res["answer"], "type": "ratio", "table_ref": r["table_ref"]})
    return lookup, ratio


if __name__ == "__main__":
    lookup, ratio = gen()
    print(f"=== SINH ĐƯỢC: {len(lookup)} lookup + {len(ratio)} ratio ===\n")
    for s in (lookup[:6] + ratio[:4]):
        print(f"[{s['type']}] Q: {s['question'][:95]}")
        print(f"    → answer = {s['answer']}")
        print(f"    pandas: {s['pandas_query'][:140].replace(chr(10),' ⏎ ')}")
        print()
    # lưu ra file để bước sau transform + grader_check
    alls = lookup + ratio
    with open("build/sft_pilot.jsonl", "w", encoding="utf-8") as f:
        for s in alls:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"đã lưu {len(alls)} mẫu -> build/sft_pilot.jsonl")
