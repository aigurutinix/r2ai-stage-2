"""Chấm 1 submission trên dev-gold BẠC (không tốn lượt nộp). Chạy query đúng hợp đồng grader
(pandas 1.1.5, dfs keyed variable, df khi 1 bảng) rồi so với nhãn bạc (rel tol 0.5%).

Dùng để XẾP HẠNG các bản ứng viên trước khi nộp:
  .venv-grader/Scripts/python.exe scripts/eval_offline.py <sub_dir> [build/dev_gold.jsonl]
"""
import builtins
import json
import sys
from pathlib import Path

import pandas as pd

_WL = ("abs round len min max sum sorted float int str bool list dict set "
       "range enumerate zip all any isinstance").split()
_SAFE = {n: getattr(builtins, n) for n in _WL}


def run_one(code, subdir, evidence):
    cp = {ev["variable"]: str((subdir / ev["csv_path"]).resolve()) for ev in evidence}
    dfs = {k: pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False, index_col=None)
           for k, p in cp.items()}
    ns = {"pd": pd, "dfs": dfs, "__builtins__": _SAFE}
    if len(dfs) == 1:
        ns["df"] = next(iter(dfs.values()))
    exec(compile(code, "<q>", "exec"), ns)
    r = ns.get("result")
    if hasattr(r, "item"):
        try:
            r = r.item()
        except Exception:
            pass
    return float(r)


def main():
    subdir = Path(sys.argv[1])
    gold_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("build/dev_gold.jsonl")
    gold = {g["id"]: g for g in (json.loads(l) for l in open(gold_path, encoding="utf-8"))}
    rows = {r["id"]: r for r in json.load(open(subdir / "submission.json", encoding="utf-8"))}
    n = correct = ran = 0
    for qid, g in gold.items():
        r = rows.get(qid)
        if not r:
            continue
        n += 1
        q = (r.get("pandas_query") or "").strip()
        if not q or not r.get("evidence"):
            continue
        try:
            v = run_one(q, subdir, r["evidence"])
            ran += 1
            gv = g["gold"]
            if abs(v - gv) <= 0.005 * max(abs(v), abs(gv), 1.0):
                correct += 1
        except Exception:
            pass
    acc = correct / n if n else 0
    print(json.dumps({
        "dev_gold_n": n, "ran": ran, "khop_gold": correct,
        "do_chinh_xac_devgold": round(acc, 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
