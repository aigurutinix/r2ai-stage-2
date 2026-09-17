"""Layered overlay on vote3_hardhop — more splices, no scale-fix traps.

vote3_dual (30 splices) scored 0.4209 because it replaced vote3 answers with
tab/cot "scale fixes" (1000× ratio). hardhop was already right on those rows.

This build keeps all 45 hardhop splices and adds only on untouched questions:
  1. tab verified + prog reexec agree
  2. tab verified + |Δ| ≤ 3× vs incumbent
  3. cot k3 + rev agree + |Δ| ≤ 3×
  4. maso/greedy/model plan + |Δ| ≤ 3× (reachable blocks only)

Large magnitude changes (>3× or <⅓×) are dropped unless tab and prog both agree.

Usage:
  python scripts/fresh/build_overlay_zip.py
  python scripts/fresh/build_overlay_zip.py --dest submissions/vote3_overlay.zip
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import parse_statements as ps  # noqa: E402
from answer_gate import verdict  # noqa: E402
from build_dualread_zip import (  # noqa: E402
    build_tab_patch,
    locate_cot_cell,
    load_cot_answers,
    load_jsonl,
    tab_answer,
)
from build_notcell_zip import prog_row, reexec_ok  # noqa: E402
from build_submission import MAGNITUDE_CODES, unit_of  # noqa: E402

REACHABLE = (
    "tien — MOT O", "khac — don gian", "ty le — mot nam", "nam nao",
    "tien — nhieu nam", "ty le — nhieu nam", "khac — nhieu nam",
    "tong hop nhieu o", "dem cong ty",
)
MAG_MIN, MAG_MAX = 1.0 / 3.0, 3.0


def mag_ratio(new: float, old: float) -> float:
    if old == 0 or new == 0:
        return 999.0
    return max(abs(new / old), abs(old / new))


def ensure_wide_zip(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fresh/build_submission.py"),
            "--greedy-plan", "artifacts/fresh/greedy_plan.jsonl",
            "--model-plan", "artifacts/fresh/model_plan.jsonl",
            "--cohort-plan", "artifacts/fresh/cohort_plan.jsonl",
            "--out", str(path.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )


def row_patch(source_row: dict, blobs: dict[str, bytes]) -> tuple[dict, dict[str, bytes]] | None:
    evidence = source_row.get("evidence") or []
    if not evidence or not source_row.get("pandas_query"):
        return None
    patch = {
        "answer": float(source_row["answer"]),
        "pandas_query": source_row["pandas_query"],
        "evidence": evidence,
        "relevant_docs": source_row.get("relevant_docs") or [],
        "relevant_tables": source_row.get("relevant_tables") or [],
    }
    files = {item["csv_path"]: blobs[item["csv_path"]]
             for item in evidence if item["csv_path"] in blobs}
    if len(files) != len(evidence):
        return None
    return patch, files


def pick_layers(
    hop_ids: set[int],
    allowed: set[int],
    single: set[int],
    hard: dict[int, float],
    qs: dict[int, str],
    tab: dict[int, dict],
    prog: dict[int, dict],
    k3: dict[int, float],
    rev: dict[int, float],
    wide_rows: dict[int, dict],
) -> dict[int, str]:
    chosen: dict[int, str] = {}

    def eligible(qid: int, answer: float) -> bool:
        if qid in hop_ids or qid not in allowed:
            return False
        if verdict(qs[qid], answer):
            return False
        if abs(answer - hard[qid]) <= 0.01:
            return False
        return True

    def accept(qid: int, answer: float, layer: str) -> bool:
        if qid in chosen or not eligible(qid, answer):
            return False
        ratio = mag_ratio(answer, hard[qid])
        if not (MAG_MIN <= ratio <= MAG_MAX):
            return False
        chosen[qid] = layer
        return True

    for qid, entry in tab.items():
        if qid not in single:
            continue
        ta = tab_answer(entry, qs[qid])
        if ta is None:
            continue
        pa = prog.get(qid)
        if pa is None:
            continue
        built = prog_row(pa, qs[qid])
        if built is None or not reexec_ok(built[0], built[1]):
            continue
        if abs(float(pa["answer"]) - ta) > 0.01:
            continue
        accept(qid, ta, "tab_prog")

    for qid, entry in tab.items():
        if qid not in single:
            continue
        ta = tab_answer(entry, qs[qid])
        if ta is not None:
            accept(qid, ta, "tab_small")

    cot_dual = {qid for qid in set(k3) & set(rev) if abs(k3[qid] - rev[qid]) <= 0.01}
    for qid in cot_dual:
        if qid in single:
            accept(qid, k3[qid], "cot_dual_small")

    for qid, row in wide_rows.items():
        ans = float(row.get("answer") or 0)
        if ans == 0:
            continue
        accept(qid, ans, "wide_small")

    return chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/vote3_hardhop.zip")
    parser.add_argument("--dest", default="submissions/vote3_overlay.zip")
    parser.add_argument("--wide-zip", default="submissions/_wide_maso.zip")
    parser.add_argument("--tab", default="artifacts/fresh/tab_plan.jsonl")
    parser.add_argument("--prog", default="artifacts/fresh/prog_results.jsonl")
    parser.add_argument("--blocks", default="artifacts/fresh/blocks.json")
    args = parser.parse_args()

    wide_path = ROOT / args.wide_zip
    ensure_wide_zip(wide_path)

    blocks = json.loads((ROOT / args.blocks).read_text(encoding="utf-8"))
    allowed: set[int] = set()
    for name in REACHABLE:
        allowed |= set(blocks.get(name, []))
    single = set(blocks.get("tien — MOT O", [])) | set(
        blocks.get("khac — don gian", []))

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    with zipfile.ZipFile(ROOT / args.base) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}
        hard = {qid: float(row.get("answer") or 0) for qid, row in rows.items()}

    with zipfile.ZipFile(ROOT / "submissions/vote3.zip") as src:
        vote3 = {r["id"]: float(r.get("answer") or 0)
                 for r in json.loads(src.read("submission.json"))}
    hop_ids = {qid for qid in vote3 if abs(vote3[qid] - hard[qid]) > 0.01}

    with zipfile.ZipFile(wide_path) as wide_archive:
        wide_rows = {r["id"]: r for r in json.loads(wide_archive.read("submission.json"))}
        wide_blobs = {n: wide_archive.read(n) for n in wide_archive.namelist()
                      if n.startswith("data/")}

    tab = load_jsonl(ROOT / args.tab)
    prog = {}
    for line in (ROOT / args.prog).read_text(encoding="utf-8").splitlines():
        if line.strip():
            prog[json.loads(line)["id"]] = json.loads(line)

    k3 = load_cot_answers(ROOT / "artifacts/fresh/cot_k3.jsonl")
    rev = load_cot_answers(ROOT / "artifacts/fresh/cot_rev_results.jsonl")
    meta = {
        record["id"]: record["meta"]
        for record in load_jsonl(ROOT / "artifacts/fresh/prompts_progall.jsonl").values()
        if record.get("meta")
    }
    tab_meta = {
        record["id"]: record["meta"]
        for record in load_jsonl(ROOT / "artifacts/fresh/prompts_tab.jsonl").values()
        if record.get("meta")
    }

    layers = pick_layers(hop_ids, allowed, single, hard, qs, tab, prog, k3, rev,
                         wide_rows)

    stats: dict[str, int] = {}
    spliced = dropped = reexec_fail = 0
    for qid in sorted(layers):
        layer = layers[qid]
        question = qs[qid]
        built = None

        if layer == "tab_prog" or layer == "tab_small":
            entry = tab[qid]
            built = build_tab_patch(entry, question)
        elif layer == "cot_dual_small":
            built = locate_cot_cell(qid, k3[qid], question, meta, tab_meta, tab.get(qid))
        elif layer == "wide_small":
            built = row_patch(wide_rows[qid], wide_blobs)

        if built is None:
            dropped += 1
            print(qid, layer, "NO_PATCH")
            continue
        patch, new_files = built
        if not reexec_ok(patch, new_files):
            reexec_fail += 1
            dropped += 1
            print(qid, layer, "reexec FAIL")
            continue

        old = hard[qid]
        rows[qid].update(patch)
        rows[qid]["question"] = question
        files.update(new_files)
        spliced += 1
        stats[layer] = stats.get(layer, 0) + 1
        print(f"{qid} {layer} {patch['answer']} (was {old})")

    dest = ROOT / args.dest
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    print(f"\nprotected hardhop hops: {len(hop_ids)}")
    print(f"candidates={len(layers)}  spliced={spliced}  "
          f"dropped={dropped}  reexec_fail={reexec_fail}")
    for name, count in sorted(stats.items()):
        print(f"  {count:4d}  {name}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
