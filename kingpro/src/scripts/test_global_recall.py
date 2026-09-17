"""Thí nghiệm GPT đề xuất: so RECALL@8 khi tìm dòng trên 1 BẢNG base vs trên TẤT CẢ bảng trong
document retrieved. Nếu nhảy vọt -> nút thắt là SAI BẢNG (không phải scorer), fix = global row search.
Chạy: PYTHONUTF8=1 python scripts/test_global_recall.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
import pandas as pd

from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.evaluation.metrics import doc_of
from kingpro.answering.operand_pipeline import (
    resolve_columns, label_column, maso_column, year_column,
    first_data_row, score_rows, build_idf, _num,
)
try:
    from kingpro.answering.ma_so_tt200 import maso_of
except Exception:
    def maso_of(q):
        return None

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
gold = [json.loads(l) for l in open("build/dev_gold_clean.jsonl", encoding="utf-8")]


def csv_full(tref):
    r = CAT.get(tref)
    return "build/tables/" + r["csv_path"] if r else None


def read(p):
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        return None


def uagn(raw, g):
    if raw is None or raw == 0:
        return False
    return any(abs(raw * (10 ** e) - g) <= 0.01 * max(abs(g), 1.0) for e in range(-13, 14))


def cands_in_table(q, df, idf, ry, per=4):
    cols = resolve_columns(df)
    lc = label_column(df, cols)
    mc = maso_column(df, cols)
    ys = re.findall(r"\b(20\d{2})\b", q)
    year = ys[0] if ys else None
    want_dau = bool(re.search(r"đầu năm|đầu kỳ|01/01", q, re.I))
    yc = year_column(cols, year, lc, mc, want_dau, ry) if year else None
    if yc is None:
        vc = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
        yc = vc[0] if len(vc) == 1 else None
    if yc is None:
        return []
    out = []
    mm = maso_of(q)
    if mm and mc is not None:
        code = str(mm[0]).strip()
        for i in range(len(df)):
            if str(df.iloc[i, mc]).strip() == code:
                out.append((100.0, _num(df.iloc[i, yc])))
    for s, i, _l in score_rows(df, q, lc, idf, first_data_row(df), topk=per):
        out.append((s, _num(df.iloc[i, yc])))
    return out


retrieve_decomposed("khoi dong")
# IDF từ bảng base (xấp xỉ)
idf = build_idf([d for d in (read("sub_base_fix/" + base[g["id"]]["evidence"][0]["csv_path"])
                             if base.get(g["id"], {}).get("evidence") else None for g in gold) if d is not None])

n = rec_base = rec_global = 0
for g in gold:
    ev = base.get(g["id"], {}).get("evidence", [])
    if not ev:
        continue
    n += 1
    ry = (re.search(r"_(20\d{2})_", ev[0]["csv_path"]) or [None, None])[1]
    # base table
    dfb = read("sub_base_fix/" + ev[0]["csv_path"])
    cb = cands_in_table(g["question"], dfb, idf, ry) if dfb is not None else []
    if any(uagn(v, g["gold"]) for _s, v in sorted(cb, key=lambda x: -x[0])[:8]):
        rec_base += 1
    # all tables in retrieved document
    try:
        hits = retrieve_decomposed(g["question"])
        rel_docs = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
        atabs = tables_in_reports(g["question"], rel_docs, n=25)
        allc = []
        for h in atabs:
            p = csv_full(h["table_ref"])
            df = read(p) if p else None
            if df is not None:
                allc += cands_in_table(g["question"], df, idf, ry)
        if any(uagn(v, g["gold"]) for _s, v in sorted(allc, key=lambda x: -x[0])[:8]):
            rec_global += 1
    except Exception:
        pass

print(f"N = {n}")
print(f"RECALL@8 trên 1 BẢNG base        = {rec_base}/{n} = {rec_base/max(n,1):.3f}")
print(f"RECALL@8 trên TẤT CẢ bảng in-doc = {rec_global}/{n} = {rec_global/max(n,1):.3f}")
