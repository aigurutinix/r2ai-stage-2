"""Đo OFFLINE trung thực với máy chấm R2AI: chạy pandas_query của submission bằng ĐÚNG hợp đồng
grader (chỉ cấp `df` khi 1 bảng + `dfs`; exec; đọc biến `result`) — bê nguyên vendor sandbox.

CHẠY BẰNG .venv-grader (pandas 1.1.5) để giống hệt máy chấm:
  .venv-grader/Scripts/python.exe scripts/grader_check.py <thu_muc_submission>

Báo: bao nhiêu câu CHẠY được (không exception) / ra số / khớp answer đã lưu (abs_tol 0.01), và
phân loại lỗi (NameError=thiếu preamble df1, ValueError=pandas rỗng, KeyError=sai cột...).
"""
import builtins
import json
import sys
from pathlib import Path

import pandas as pd

# Máy chấm GIỚI HẠN builtins đúng danh sách trong prompt gốc BTC (program_system.txt).
# globals/getattr/print/__import__/NameError... KHÔNG có -> code dùng chúng sẽ NameError.
_WHITELIST = ("abs round len min max sum sorted float int str bool list dict set "
              "range enumerate zip all any isinstance").split()
_SAFE_BUILTINS = {n: getattr(builtins, n) for n in _WHITELIST}


def _read_csv(p, typed=False):
    if typed:
        return pd.read_csv(p)
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False, index_col=None)


def run_one(code: str, csv_paths: dict, official=False, typed_dfs=False, return_namespace=False):
    """Y HỆT máy chấm: chỉ df (1 bảng) + dfs, builtins GIỚI HẠN, exec, đọc result."""
    dfs = {
        ref: _read_csv(p, typed=(official or typed_dfs))
        for ref, p in csv_paths.items()
    }
    # Current Stage-2 scoring demonstrably supplies ``dfs`` even for a
    # one-table submission; otherwise 761 one-table programs in the measured
    # v102 artifact could not have produced its 321/506 execution score.
    # Keep ``df`` as a compatibility alias without removing ``dfs``.
    ns = {"pd": pd, "dfs": dfs, "__builtins__": _SAFE_BUILTINS}
    if len(dfs) == 1:
        ns["df"] = next(iter(dfs.values()))
    exec(compile(code, "<pandas_query>", "exec"), ns)      # noqa: S102
    if "result" not in ns:
        raise RuntimeError("no-result-var")
    r = ns["result"]
    if hasattr(r, "item"):
        try:
            r = r.item()
        except Exception:
            pass
    return (r, ns) if return_namespace else r


def main():
    subdir = Path(sys.argv[1])
    selected_ids = None
    official = "--official" in sys.argv
    typed_dfs = "--typed-dfs" in sys.argv
    if len(sys.argv) >= 4 and sys.argv[2] == "--ids":
        selected_ids = {int(value) for value in sys.argv[3].split(",") if value.strip()}
    rows = json.loads((subdir / "submission.json").read_text(encoding="utf-8"))
    if selected_ids is not None:
        rows = [row for row in rows if int(row.get("id", -1)) in selected_ids]
    n = ran = num = match = empty = 0
    errs: dict = {}
    samples = []
    error_ids = []
    mismatch_ids = []
    for e in rows:
        q = (e.get("pandas_query") or "").strip()
        if not q:
            empty += 1
            continue
        n += 1
        csv_paths = {ev["variable"]: str((subdir / ev["csv_path"]).resolve()) for ev in e.get("evidence", [])}
        if not csv_paths:                                  # không có evidence -> không thể chạy
            errs["no-evidence"] = errs.get("no-evidence", 0) + 1
            continue
        try:
            r = run_one(q, csv_paths, official=official, typed_dfs=typed_dfs)
            ran += 1
            try:
                rv = float(r)
                num += 1
                stored = e.get("answer")
                if stored is not None and abs(rv - float(stored)) <= 0.01 + 1e-9:
                    match += 1
                else:
                    mismatch_ids.append(e.get("id"))
            except (TypeError, ValueError):
                errs["result-not-number"] = errs.get("result-not-number", 0) + 1
        except Exception as ex:
            t = type(ex).__name__
            errs[t] = errs.get(t, 0) + 1
            error_ids.append(e.get("id"))
            if len(samples) < 8:
                samples.append({"id": e.get("id"), "err": f"{t}: {str(ex)[:120]}"})
    print(json.dumps({
        "entries": len(rows), "with_query": n, "empty_query": empty,
        "ran_no_exception": ran, "numeric_result": num, "match_stored_answer": match,
        "mode": "official-typed-dfs" if official else ("typed-dfs" if typed_dfs else "string-dfs"),
        "errors": errs, "error_ids": error_ids, "mismatch_ids": mismatch_ids,
        "err_samples": samples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
