"""⭐ Lấp 240 câu rỗng bằng bộ TẤT ĐỊNH (KHÔNG GPU). Risk-free: câu rỗng=0, sai=0, đúng=+1 -> lấp HẾT.
Compute (analytic_cross > ratio > analytic > deterministic) -> merge base 772 + fill -> copy CSV -> zip.
Chạy: PYTHONUTF8=1 python -u scripts/build_deterministic_fill.py --out sub_detfill
"""
import argparse
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve_decomposed, tables_in_reports
from kingpro.answering.operand_pipeline import (
    deterministic_answer, analytic_answer, ratio_answer, analytic_cross, build_idf)
from kingpro.evaluation.metrics import doc_of, coerce_number
import pandas as pd

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
BASE_DIR = "sub_base_fix"  # override bằng --base


def csv_full(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def safe_name(tr):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tr) + ".csv"


def solve(q):
    """Trả (answer, pandas_query, [table_refs theo thứ tự], rel_docs, engine, conf) hoặc None."""
    hits = retrieve_decomposed(q)
    rel = list(dict.fromkeys(doc_of(h["table_ref"]) for h in hits))
    cands = tables_in_reports(q, rel, n=8)
    tabs = [{"table_ref": c["table_ref"], "csv_path": csv_full(c["table_ref"])}
            for c in cands if csv_full(c["table_ref"])]
    if not tabs:
        return None
    dfs = []
    for t in tabs:
        try:
            dfs.append(pd.read_csv(t["csv_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig"))
        except Exception:
            pass
    idf = build_idf(dfs)
    for fn, name in [(analytic_cross, "cross"), (ratio_answer, "ratio"),
                     (analytic_answer, "analytic"), (deterministic_answer, "det")]:
        try:
            r = fn(q, tabs, idf)
        except Exception:
            r = None
        if r and r.get("ok") and r.get("answer") is not None:
            av = coerce_number(r.get("answer"))
            if av is None:
                continue
            refs = [ev["table_ref"] for ev in r.get("evidence", []) if ev.get("table_ref")]
            if not refs:
                continue
            return (float(av), r.get("pandas_query", ""), refs, rel, name, r.get("conf", "?"))
    return None


def main():
    global BASE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sub_detfill")
    ap.add_argument("--base", default="sub_base_fix", help="thư mục base để lấp (mặc định sub_base_fix; dùng sub_ctxB cho best)")
    a = ap.parse_args()
    BASE_DIR = a.base
    base = json.load(open(f"{BASE_DIR}/submission.json", encoding="utf-8"))
    empty = [e for e in base if not (e.get("pandas_query") or "").strip()]
    retrieve_decomposed("warm")

    fills = {}
    import collections
    eng_cnt = collections.Counter()
    for i, e in enumerate(empty, 1):
        r = solve(e["question"])
        if r:
            fills[e["id"]] = r
            eng_cnt[r[4]] += 1
        if i % 40 == 0:
            print(f"  ...{i}/{len(empty)} lấp {len(fills)}", flush=True)
    print(f"LẤP {len(fills)}/{len(empty)} câu | engine={dict(eng_cnt)}", flush=True)

    # build submission
    out = a.out
    data_dir = os.path.join(out, "data")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(data_dir, exist_ok=True)
    for f in os.listdir(os.path.join(BASE_DIR, "data")):
        shutil.copyfile(os.path.join(BASE_DIR, "data", f), os.path.join(data_dir, f))
    copied = {}
    rows = []
    for e in base:
        e2 = dict(e)
        if e["id"] in fills:
            ans, pq, refs, rel, eng, conf = fills[e["id"]]
            evidence = []
            for j, tref in enumerate(refs, 1):
                if tref not in copied:
                    src = csv_full(tref)
                    if not src or not os.path.exists(src):
                        continue
                    dst = safe_name(tref)
                    shutil.copyfile(src, os.path.join(data_dir, dst))
                    copied[tref] = f"data/{dst}"
                evidence.append({"variable": f"df{j}", "csv_path": copied[tref]})
            if evidence:
                e2["answer"] = ans
                e2["pandas_query"] = pq
                e2["evidence"] = evidence
                e2["relevant_tables"] = refs
                e2["relevant_docs"] = rel
        rows.append(e2)
    json.dump(rows, open(os.path.join(out, "submission.json"), "w", encoding="utf-8"), ensure_ascii=False)
    zip_path = f"{out}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out, "submission.json"), "submission.json")
        for f in os.listdir(data_dir):
            z.write(os.path.join(data_dir, f), f"data/{f}")
    n_filled = sum(1 for e in rows if (e.get("pandas_query") or "").strip()) - 772
    print(f"ZIP -> {zip_path} | lấp {len(fills)} câu, copy thêm {len(copied)} CSV | tổng có code = {772+len(fills)}", flush=True)


if __name__ == "__main__":
    main()
