"""Repair an EXISTING submission: rewrite every multi-line pandas_query into a
single expression, verified against the submitted CSVs. No GPU re-run needed.

WHY: the grader evaluates pandas_query as an EXPRESSION. Multi-line scripts are
SyntaxErrors -> scored as crashes. Submission #6 shipped 233 scripts and lost
EXECUTION_ACCURACY (0.085 -> 0.0613) even though ANSWER_ACCURACY rose to 0.1047.

  python scripts/08_repair_expressions.py --submission artifacts/submission
  python scripts/08_repair_expressions.py --submission artifacts/submission \
      --out-dir artifacts/submission_fixed          # keep the original intact

Also works on a codegen_results.jsonl so the fix persists upstream:
  python scripts/08_repair_expressions.py --codegen artifacts/codegen_results.jsonl
"""
import argparse
import json

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from vifinqa.codegen.to_expression import try_to_expression
from vifinqa.utils.io import setup_stdout, read_jsonl, write_jsonl, write_json


def repair_submission(sub_dir: Path, out_dir: Path, json_name: str = "results.json") -> None:
    entries = json.loads((sub_dir / json_name).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSVs are never rewritten - read/zip them straight from the source folder
    # (copying ~1k files is slow and pointless).

    n_script = n_fixed = n_value_ok = n_left = 0
    failures = []
    for e in entries:
        q = (e.get("pandas_query") or "").strip()
        if not q:
            continue
        if _is_expression(q):
            continue
        n_script += 1
        expr, err = try_to_expression(q)
        if err:
            n_left += 1
            failures.append((e["id"], err))
            continue
        value, exec_err = _eval_with_evidence(expr, e, sub_dir)
        if exec_err is not None:
            n_left += 1
            failures.append((e["id"], f"eval failed: {exec_err}"))
            continue
        if abs(float(value) - float(e["answer"])) > 0.011:
            n_left += 1
            failures.append((e["id"], f"value drift {value} != {e['answer']}"))
            continue
        e["pandas_query"] = expr
        n_fixed += 1
        n_value_ok += 1

    write_json(out_dir / json_name, entries)
    zip_path = out_dir / "submission.zip"
    csv_names = sorted({Path(ev["csv_path"]).name
                        for e in entries for ev in e.get("evidence", [])})
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / json_name, json_name)
        for name in csv_names:
            z.write(sub_dir / "data" / name, f"data/{name}")

    still_bad = [e["id"] for e in entries if not _is_expression(e["pandas_query"])]
    print(f"scripts found      : {n_script}")
    print(f"converted+verified : {n_fixed} (value identical to `answer`: {n_value_ok})")
    print(f"left as script     : {n_left}")
    print(f"eval-compilable now: {len(entries) - len(still_bad)}/{len(entries)}")
    if failures:
        print("un-inlinable (kept as-is, ANSWER still counts):")
        for qid, err in failures[:10]:
            print(f"  id={qid}: {err}")
    print(f"-> {zip_path}")


def repair_codegen(path: Path, out_path: Path) -> None:
    rows = read_jsonl(path)
    n = 0
    for r in rows:
        q = (r.get("pandas_query") or "").strip()
        if q and not _is_expression(q):
            expr, err = try_to_expression(q)
            if not err:
                r["pandas_query"] = expr
                n += 1
    write_jsonl(out_path, rows)
    print(f"codegen: converted {n}/{len(rows)} queries -> {out_path}")


def _is_expression(code: str) -> bool:
    try:
        compile(code, "<q>", "eval")
        return True
    except SyntaxError:
        return False


def _eval_with_evidence(expr: str, entry: dict, root: Path):
    ns = {"pd": pd}
    try:
        import numpy as np
        ns["np"] = np
    except ImportError:
        pass
    try:
        for ev in entry.get("evidence", []):
            ns[ev["variable"]] = pd.read_csv(root / ev["csv_path"])
        return float(eval(compile(expr, "<q>", "eval"), ns)), None  # noqa: S307
    except Exception as ex:  # noqa: BLE001
        return None, f"{type(ex).__name__}: {ex}"[:120]


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="", help="folder with results.json + data/")
    ap.add_argument("--out-dir", default="", help="default: repair in place")
    ap.add_argument("--codegen", default="", help="also/instead repair a codegen jsonl")
    ap.add_argument("--json-name", default="results.json")
    args = ap.parse_args()

    if not args.submission and not args.codegen:
        raise SystemExit("pass --submission and/or --codegen")
    if args.codegen:
        p = Path(args.codegen)
        repair_codegen(p, Path(args.out_dir or p))
    if args.submission:
        sub = Path(args.submission)
        repair_submission(sub, Path(args.out_dir) if args.out_dir else sub, args.json_name)


if __name__ == "__main__":
    main()
