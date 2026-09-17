"""ĐÒN #1 áp vào submission: evidence = STATEMENT GHÉP đầy đủ + deterministic_answer (tìm dòng mã số
trên statement đủ -> conf 3 chuẩn, hạ tường chọn-mảnh). Không GPU.

Với câu DN theo TT200: ghép statement của report (dùng relevant_docs base) -> feed các statement (CDKT/KQKD/LCTT)
cho deterministic_answer -> nếu conf>=2 dùng, else GIỮ base. Ngân hàng/CK hoặc không ghép được -> giữ base.

Chạy:
  PYTHONUTF8=1 python -u scripts/build_merged_sub.py --limit 30      # test
  PYTHONUTF8=1 python -u scripts/build_merged_sub.py --out sub_merged2   # full + ráp
"""
import argparse
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.tables.merge_statements import merged_statements
from kingpro.answering.operand_pipeline import deterministic_answer, analytic_answer, ratio_answer, build_idf
from kingpro.answering.ma_so_tt200 import maso_of
from kingpro.evaluation.metrics import coerce_number
import pandas as pd

BASE = "sub_ctxB"
MERGED_DIR = "build/merged"
os.makedirs(MERGED_DIR, exist_ok=True)
QS = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
_MERGE_CACHE = {}


def merged_csv_paths(report_id):
    """Ghép statement của report, ghi CSV vào build/merged/, trả list (table_ref, csv_path)."""
    if report_id in _MERGE_CACHE:
        return _MERGE_CACHE[report_id]
    out = []
    try:
        m = merged_statements(report_id)
    except Exception:
        m = {}
    for typ, df in m.items():
        p = os.path.join(MERGED_DIR, f"{report_id}__{typ}.csv")
        if not os.path.exists(p):
            df.to_csv(p, index=False, encoding="utf-8-sig")
        out.append({"table_ref": f"{report_id}|{typ}", "csv_path": p})
    _MERGE_CACHE[report_id] = out
    return out


def solve(qid, base_entry):
    q = QS[qid]
    docs = base_entry.get("relevant_docs", []) or []
    # gom statement ghép của các report liên quan (thường 1 DN 1 năm 1 scope)
    tabs = []
    for d in docs[:3]:
        tabs += merged_csv_paths(d)
    if not tabs:
        return None
    dfs = []
    for t in tabs:
        try:
            dfs.append(pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig"))
        except Exception:
            pass
    idf = build_idf(dfs)
    for fn in (ratio_answer, analytic_answer, deterministic_answer):
        try:
            r = fn(q, tabs, idf)
        except Exception:
            r = None
        if r and r.get("ok") and r.get("answer") is not None and r.get("conf", 0) >= 2:
            av = coerce_number(r["answer"])
            if av is None:
                continue
            return {"answer": float(av), "pandas_query": r.get("pandas_query", ""),
                    "evidence_refs": [ev.get("table_ref") for ev in r.get("evidence", [])],
                    "conf": r.get("conf")}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--fresh", action="store_true", help="FRESH: câu không giải được -> RỖNG (không giữ base) -> đo deterministic-on-merged standalone")
    a = ap.parse_args()
    base = {e["id"]: e for e in json.load(open(f"{BASE}/submission.json", encoding="utf-8"))}
    ids = list(QS.keys())[: a.limit]
    solved = {}
    n_conf = 0
    for i, qid in enumerate(ids, 1):
        r = solve(qid, base.get(qid, {}))
        if r:
            solved[qid] = r
            n_conf += 1
        if i % 50 == 0:
            print(f"  ...{i}/{len(ids)} solved={n_conf}", flush=True)
    print(f"MERGED-DET giải conf>=2: {n_conf}/{len(ids)} câu", flush=True)
    # đối chiếu với base: bao nhiêu KHÁC base
    diff = 0
    for qid, r in solved.items():
        ba = coerce_number(base.get(qid, {}).get("answer"))
        if ba is None or abs(r["answer"] - ba) > max(0.01, abs(ba) * 0.005):
            diff += 1
    print(f"  trong đó KHÁC base: {diff}", flush=True)
    if not a.out:
        # test mode: in vài mẫu
        for qid in list(solved)[:12]:
            print(f"  Q{qid} conf{solved[qid]['conf']} ans={solved[qid]['answer']} | {QS[qid][:55]}")
        return
    # ráp submission: override câu solved (evidence = merged CSV), giữ base phần còn lại
    out = a.out
    ddir = os.path.join(out, "data")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(ddir, exist_ok=True)
    for f in os.listdir(f"{BASE}/data"):
        shutil.copyfile(f"{BASE}/data/{f}", os.path.join(ddir, f))
    copied = {}
    rows = []
    n_ov = 0
    for e in json.load(open(f"{BASE}/submission.json", encoding="utf-8")):
        e2 = dict(e)
        if a.fresh:                       # FRESH: bỏ đáp án base, chỉ giữ format
            e2["answer"] = 0.0
            e2["pandas_query"] = ""
            e2["evidence"] = []
        r = solved.get(e["id"])
        if r and r.get("evidence_refs"):
            evidence = []
            ok = True
            for j, ref in enumerate(r["evidence_refs"], 1):
                src = os.path.join(MERGED_DIR, ref.replace("|", "__") + ".csv")
                if not os.path.exists(src):
                    ok = False
                    break
                if ref not in copied:
                    dst = re.sub(r"[^0-9A-Za-z_]+", "_", ref) + ".csv"
                    shutil.copyfile(src, os.path.join(ddir, dst))
                    copied[ref] = f"data/{dst}"
                evidence.append({"variable": f"df{j}", "csv_path": copied[ref]})
            if ok and evidence:
                e2["answer"] = r["answer"]
                e2["pandas_query"] = r["pandas_query"]
                e2["evidence"] = evidence
                n_ov += 1
        rows.append(e2)
    json.dump(rows, open(os.path.join(out, "submission.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"RÁP {out}: override {n_ov} câu bằng merged-det, copy {len(copied)} statement ghép", flush=True)


if __name__ == "__main__":
    main()
