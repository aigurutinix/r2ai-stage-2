"""Best-of merge (Tier 2) — GHÉP sub_program vào sub_base_fix để KHÔNG BAO GIỜ tụt dưới nền cũ.

Luật (chống regression): với mỗi câu,
  - nếu pandas_query của PROGRAM chạy ĐƯỢC trên hợp đồng grader (dfs keyed variable, builtins giới hạn,
    ra biến result là 1 SỐ) -> lấy đáp án PROGRAM (trích số đã nâng cấp);
  - ngược lại (program rỗng/crash/không ra số) -> GIỮ đáp án BASE_FIX.
=> tập câu program hỏng luôn = base cũ; tập câu program chạy được = bản nâng cấp. Sàn = base_fix.

Chạy bằng .venv-grader (pandas 1.1.5) cho khớp máy chấm:
  .venv-grader/Scripts/python.exe scripts/merge_best.py sub_program sub_base_fix sub_merged
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


def _valid_number(code: str, subdir: Path, evidence: list) -> bool:
    """True nếu code chạy đúng hợp đồng grader và ra 1 số float được."""
    code = (code or "").strip()
    if not code or not evidence:
        return False
    try:
        csv_paths = {ev["variable"]: str((subdir / ev["csv_path"]).resolve()) for ev in evidence}
        dfs = {k: pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False, index_col=None)
               for k, p in csv_paths.items()}
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

    def copy_ev(src_dir: Path, evidence: list) -> list:
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

    rows, n_prog, n_base = [], 0, 0
    for b in base:
        qid = b["id"]
        p = prog.get(qid)
        use_prog = p is not None and _valid_number(p.get("pandas_query"), prog_dir, p.get("evidence", []))
        if use_prog:
            n_prog += 1
            rows.append({
                "id": qid, "question": b["question"],
                "answer": float(p["answer"]),
                "relevant_docs": b.get("relevant_docs", []),
                "relevant_tables": b.get("relevant_tables", []),
                "evidence": copy_ev(prog_dir, p.get("evidence", [])),
                "pandas_query": p.get("pandas_query", ""),
            })
        else:
            n_base += 1
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
    print(f"MERGED {len(rows)} câu | program={n_prog} base={n_base} | ZIP {zp} {zp.stat().st_size} bytes")


if __name__ == "__main__":
    main()
