"""Merge AN TOÀN (chỉ lấp rỗng): giữ NGUYÊN base 0.1719, chỉ thay câu base RỖNG query bằng
đáp án program hợp lệ. Rỗng đang = 0 điểm nên thay vào chỉ có thể GIỮ NGUYÊN (0) hoặc TĂNG (+1),
không bao giờ giảm. => điểm >= base.

Chạy: .venv-grader/Scripts/python.exe scripts/merge_fill.py sub_program sub_base_fix sub_fill
"""
import builtins
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

_WL = ("abs round len min max sum sorted float int str bool list dict set "
       "range enumerate zip all any isinstance").split()
_SAFE = {n: getattr(builtins, n) for n in _WL}


def _valid_number(code, subdir, evidence):
    code = (code or "").strip()
    if not code or not evidence:
        return False
    try:
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
        float(r)
        return True
    except Exception:
        return False


def main():
    prog_dir, base_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    prog = {r["id"]: r for r in json.loads((prog_dir / "submission.json").read_text(encoding="utf-8"))}
    base = json.loads((base_dir / "submission.json").read_text(encoding="utf-8"))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True)
    existing = set()

    def copy_ev(src_dir, evidence):
        out = []
        for ev in evidence:
            name = Path(ev["csv_path"]).name
            if name not in existing:
                src = src_dir / ev["csv_path"]
                if src.exists():
                    shutil.copyfile(src, out_dir / "data" / name)
                    existing.add(name)
            out.append({"variable": ev["variable"], "csv_path": f"data/{name}"})
        return out

    rows, n_fill = [], 0
    for b in base:
        qid = b["id"]
        base_empty = not (b.get("pandas_query") or "").strip()
        p = prog.get(qid)
        fill = base_empty and p is not None and _valid_number(p.get("pandas_query"), prog_dir, p.get("evidence", []))
        if fill:
            n_fill += 1
            rows.append({
                "id": qid, "question": b["question"],
                "answer": float(p["answer"]),
                "relevant_docs": b.get("relevant_docs", []),
                "relevant_tables": b.get("relevant_tables", []),
                "evidence": copy_ev(prog_dir, p.get("evidence", [])),
                "pandas_query": p.get("pandas_query", ""),
            })
        else:
            rows.append({
                "id": qid, "question": b["question"],
                "answer": float(b.get("answer") or 0.0),
                "relevant_docs": b.get("relevant_docs", []),
                "relevant_tables": b.get("relevant_tables", []),
                "evidence": copy_ev(base_dir, b.get("evidence", [])),
                "pandas_query": b.get("pandas_query", ""),
            })

    (out_dir / "submission.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not any((out_dir / "data").glob("*.csv")):
        (out_dir / "data" / "_placeholder.csv").write_text("0\n0\n", encoding="utf-8")
    zp = out_dir.parent / f"{out_dir.name}.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "submission.json", "submission.json")
        for c in (out_dir / "data").glob("*.csv"):
            z.write(c, f"data/{c.name}")
    print(f"FILL {len(rows)} câu | lấp rỗng bằng program: {n_fill} | ZIP {zp} {zp.stat().st_size} bytes")


if __name__ == "__main__":
    main()
