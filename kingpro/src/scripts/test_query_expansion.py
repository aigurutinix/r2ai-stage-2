"""Test đòn SEMANTIC của GPT: dùng Qwen mở rộng truy vấn (paraphrase nhãn chỉ tiêu) rồi tìm lại dòng.
Chỉ chạy trên các câu đang MISS (đo trực tiếp RESCUE, rẻ). Nếu recall vọt -> semantic phá được trần lexical.
Chạy: PYTHONUTF8=1 python scripts/test_query_expansion.py
"""
import json
import os
import re
import sys

sys.path.insert(0, "src")
import pandas as pd

from kingpro.answering.llm_client import chat
from kingpro.answering.operand_pipeline import (
    resolve_columns, label_column, maso_column, year_column,
    first_data_row, score_rows, build_idf, _num,
)
try:
    from kingpro.answering.ma_so_tt200 import maso_of
except Exception:
    def maso_of(q):
        return None

base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
gold = [json.loads(l) for l in open("build/dev_gold_clean.jsonl", encoding="utf-8")]
KEY = os.environ["KINGPRO_LLM_API_KEY"]
BASE = os.environ["BASE_CODER"]
MODEL = os.environ["MODEL_CODER"]


def rd(p):
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        return None


def uagn(raw, g):
    if raw is None or raw == 0:
        return False
    return any(abs(raw * (10 ** e) - g) <= 0.01 * max(abs(g), 1.0) for e in range(-13, 14))


def cand_vals(q, df, idf, ry, topk=8):
    cols = resolve_columns(df)
    lc = label_column(df, cols)
    mc = maso_column(df, cols)
    ys = re.findall(r"\b(20\d{2})\b", q)
    year = ys[0] if ys else None
    yc = year_column(cols, year, lc, mc, bool(re.search(r"đầu|01/01", q)), ry) if year else None
    if yc is None:
        vc = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
        yc = vc[0] if len(vc) == 1 else None
    if yc is None:
        return []
    rows = []
    mm = maso_of(q)
    if mm and mc is not None:
        code = str(mm[0]).strip()
        rows += [i for i in range(len(df)) if str(df.iloc[i, mc]).strip() == code]
    for _s, i, _l in score_rows(df, q, lc, idf, first_data_row(df), topk=topk):
        if i not in rows:
            rows.append(i)
    return [_num(df.iloc[i, yc]) for i in rows[:topk]]


def expand(question):
    sys_p = "Bạn là kế toán. Chỉ trả các cụm từ, mỗi dòng một cụm, không giải thích."
    usr = (f"Câu hỏi: {question}\n\nChỉ tiêu tài chính được hỏi có thể xuất hiện trong bảng BCTC dưới những "
           f"NHÃN nào? Liệt kê 3-6 cách gọi/nhãn đầy đủ theo mẫu BCTC Việt Nam (TT200). Mỗi dòng một nhãn.")
    try:
        out = chat(sys_p, usr, base_url=BASE, api_key=KEY, model=MODEL, temperature=0.3, max_tokens=160, timeout=60)
        return " ".join(l.strip(" -•\t") for l in out.splitlines() if l.strip())
    except Exception:
        return ""


dfs = [rd("sub_base_fix/" + base[g["id"]]["evidence"][0]["csv_path"]) for g in gold if base.get(g["id"], {}).get("evidence")]
idf = build_idf([d for d in dfs if d is not None])

n = hit0 = hit_qe = rescued = 0
for g in gold:
    ev = base.get(g["id"], {}).get("evidence", [])
    if not ev:
        continue
    df = rd("sub_base_fix/" + ev[0]["csv_path"])
    if df is None:
        continue
    n += 1
    ry = (re.search(r"_(20\d{2})_", ev[0]["csv_path"]) or [None, None])[1]
    base_hit = any(uagn(v, g["gold"]) for v in cand_vals(g["question"], df, idf, ry))
    if base_hit:
        hit0 += 1
        hit_qe += 1
        continue
    # MISS -> thử query expansion
    exp = expand(g["question"])
    q2 = g["question"] + " " + exp
    if any(uagn(v, g["gold"]) for v in cand_vals(q2, df, idf, ry)):
        hit_qe += 1
        rescued += 1

print(f"N={n}")
print(f"RECALL@8 lexical      = {hit0}/{n} = {hit0/max(n,1):.3f}")
print(f"RECALL@8 + query-exp  = {hit_qe}/{n} = {hit_qe/max(n,1):.3f}  (rescued {rescued} câu miss)")
