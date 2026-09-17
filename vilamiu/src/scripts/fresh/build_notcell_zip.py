"""Phase 1: splice prog answers onto vote3 where incumbent is not-a-cell.

vote3 answers 179 questions with values that do not match any cell in the
report at any scale — replacing them with verified programs is free upside.

Usage:
  python scripts/fresh/build_notcell_zip.py
  python scripts/fresh/build_notcell_zip.py --ids artifacts/fresh/splice_notcell.json
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from answer_gate import verdict  # noqa: E402
from build_from_programs import PRELUDE  # noqa: E402
from num_helper import SOURCE as NUM_SOURCE  # noqa: E402


def prog_row(entry: dict, question: str) -> tuple[dict, dict[str, bytes]] | None:
    """Build submission fields + csv blobs from one prog_results record."""

    used = {ref: rel for ref, rel in entry["csvs"].items()
            if ref in entry["pandas_query"]} or entry["csvs"]
    evidence, mapping, files = [], [], {}
    for position, (ref, csv_rel) in enumerate(sorted(used.items()), start=1):
        name = (f"data/{Path(csv_rel).parent.parent.name}_table_"
                f"{Path(csv_rel).stem.split('_')[-1]}.csv")
        source = ROOT / csv_rel
        if not source.exists():
            return None
        files[name] = source.read_bytes()
        evidence.append({"variable": f"df{position}", "csv_path": name})
        mapping.append(f"    {ref!r}: df{position},")
    if not evidence:
        return None
    code = (NUM_SOURCE + PRELUDE.format(mapping="\n".join(mapping))
            + entry["pandas_query"])
    record = {
        "answer": entry["answer"],
        "pandas_query": code,
        "evidence": evidence,
        "relevant_docs": entry.get("docs") or [],
        "relevant_tables": entry.get("refs") or [],
    }
    return record, files


def reexec_ok(record: dict, blobs: dict[str, bytes]) -> bool:
    namespace: dict = {"pd": pd}
    try:
        for item in record["evidence"]:
            blob = blobs.get(item["csv_path"])
            if blob is None:
                return False
            namespace[item["variable"]] = pd.read_csv(
                io.BytesIO(blob), dtype=str, keep_default_na=False,
            )
        exec(record["pandas_query"], namespace, namespace)  # noqa: S102
        return abs(float(record["answer"]) - float(namespace["result"])) <= 0.01
    except Exception:
        return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--dest", default="submissions/vote3_notcell.zip")
    parser.add_argument("--results", default="artifacts/fresh/prog_results.jsonl")
    parser.add_argument("--ids", default="artifacts/fresh/splice_notcell.json")
    args = parser.parse_args()

    splice_ids = set(json.loads((ROOT / args.ids).read_text(encoding="utf-8")))
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    prog = {}
    for line in (ROOT / args.results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            prog[record["id"]] = record

    with zipfile.ZipFile(ROOT / args.base) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}

    spliced = dropped = reexec_fail = 0
    for qid in sorted(splice_ids):
        if qid not in prog:
            dropped += 1
            continue
        question = qs[qid]
        ans = float(prog[qid]["answer"])
        if abs(ans - float(rows[qid].get("answer") or 0)) <= 0.01:
            continue
        reason = verdict(question, ans)
        if reason:
            dropped += 1
            print(qid, "GATE", reason)
            continue
        built = prog_row(prog[qid], question)
        if built is None:
            dropped += 1
            print(qid, "NO_CSV")
            continue
        patch, new_files = built
        if not reexec_ok(patch, new_files):
            reexec_fail += 1
            dropped += 1
            print(qid, "reexec FAIL")
            continue
        old = float(rows[qid]["answer"])
        rows[qid].update(patch)
        rows[qid]["question"] = question
        files.update(new_files)
        spliced += 1
        print(f"{qid} {patch['answer']} (was {old})")

    dest = ROOT / args.dest
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    print(f"spliced {spliced} / {len(splice_ids)}  "
          f"dropped {dropped}  reexec_fail {reexec_fail} -> {dest}")


if __name__ == "__main__":
    main()
