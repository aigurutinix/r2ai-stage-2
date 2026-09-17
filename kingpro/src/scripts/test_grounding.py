"""Đo ĐỘ CHÍNH XÁC GROUNDING tất định (đúng dòng + đúng cột-năm) trên dev-gold, cho sẵn bảng đúng
của base (cô lập row+column recall). Đây là "p" mà GPT bảo phải đo — vì analytical ~ p^2.
Chạy: PYTHONUTF8=1 python scripts/test_grounding.py
"""
import json
import re
import sys

sys.path.insert(0, "src")
import pandas as pd

from kingpro.answering.operand_pipeline import (
    resolve_columns, label_column, maso_column, year_column,
    first_data_row, score_rows, build_idf, requested_unit, _num,
)

base = {r["id"]: r for r in json.load(open("sub_base_fix/submission.json", encoding="utf-8"))}
gold = [json.loads(l) for l in open("build/dev_gold_clean.jsonl", encoding="utf-8")]


def load_ev(qid):
    r = base.get(qid, {})
    ev = r.get("evidence", [])
    if not ev:
        return None
    try:
        return pd.read_csv("sub_base_fix/" + ev[0]["csv_path"], dtype=str,
                           keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        return None


def det_lookup(q, df, idf, dbg=None):
    cols = resolve_columns(df)
    lc = label_column(df, cols)
    mc = maso_column(df, cols)
    ys = re.findall(r"\b(20\d{2})\b", q)
    year = ys[0] if ys else None
    want_dau = bool(re.search(r"đầu năm|đầu kỳ|01/01|\b1/1\b", q, re.I))
    yc = year_column(cols, year, lc, mc, want_dau) if year else None
    if yc is None:
        val_cols = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
        yc = val_cols[0] if len(val_cols) == 1 else None
    start = first_data_row(df)
    cands = score_rows(df, q, lc, idf, start)
    if dbg is not None:
        dbg.update(dict(lc=lc, mc=mc, yc=yc, year=year, top=[(round(s, 1), i, l[:30]) for s, i, l in cands[:3]]))
    if yc is None or not cands:
        return None
    row = cands[0][1]
    v = _num(df.iloc[row, yc])
    if v is None:
        return None
    return round(v * cols[yc]["unit_mult"] / requested_unit(q), 2)


def report_year_of(qid):
    ev = base.get(qid, {}).get("evidence", [])
    if ev:
        m = re.search(r"_(20\d{2})_", ev[0]["csv_path"])
        if m:
            return m.group(1)
    return None


try:
    from kingpro.answering.ma_so_tt200 import maso_of
except Exception:
    def maso_of(q):
        return None


def candidates(q, df, idf, ry, topk=8):
    """Trả (yc, [raw values của top-k candidate rows tại cột năm]). Ưu tiên dòng khớp MÃ SỐ."""
    cols = resolve_columns(df)
    lc = label_column(df, cols)
    mc = maso_column(df, cols)
    ys = re.findall(r"\b(20\d{2})\b", q)
    year = ys[0] if ys else None
    want_dau = bool(re.search(r"đầu năm|đầu kỳ|01/01|\b1/1\b", q, re.I))
    yc = year_column(cols, year, lc, mc, want_dau, ry) if year else None
    if yc is None:
        val_cols = [j for j in cols if j != lc and j != mc and not cols[j]["is_maso"]]
        yc = val_cols[0] if len(val_cols) == 1 else None
    if yc is None:
        return yc, []
    rows = []
    # 1) MÃ SỐ exact (nếu câu map ra mã chuẩn) — precision cao
    mm = maso_of(q)
    if mm and mc is not None:
        code = str(mm[0]).strip()
        for i in range(len(df)):
            if str(df.iloc[i, mc]).strip() == code:
                rows.append(i)
    # 2) lexical
    start = first_data_row(df)
    for _s, i, _l in score_rows(df, q, lc, idf, start, topk=topk):
        if i not in rows:
            rows.append(i)
    return yc, [_num(df.iloc[i, yc]) for i in rows[:topk]]


def uagn(raw, gold):
    if raw is None or raw == 0:
        return False
    return any(abs(raw * (10 ** e) - gold) <= 0.01 * max(abs(gold), 1.0) for e in range(-13, 14))


dfs = [d for d in (load_ev(g["id"]) for g in gold) if d is not None]
idf = build_idf(dfs)

n = top1 = rec8 = yc_ok = 0
for g in gold:
    df = load_ev(g["id"])
    if df is None:
        continue
    n += 1
    ry = report_year_of(g["id"])
    yc, vals = candidates(g["question"], df, idf, ry, topk=8)
    if yc is not None:
        yc_ok += 1
    if vals and uagn(vals[0], g["gold"]):
        top1 += 1
    if any(uagn(v, g["gold"]) for v in vals):
        rec8 += 1

print(f"CỘT-NĂM tìm được   = {yc_ok}/{n} = {yc_ok/max(n,1):.3f}")
print(f"TOP-1 đúng ô        = {top1}/{n} = {top1/max(n,1):.3f}")
print(f"RECALL@8 (ô đúng trong top-8, LLM sẽ chọn) = {rec8}/{n} = {rec8/max(n,1):.3f}")
