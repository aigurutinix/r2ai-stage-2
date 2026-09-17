"""Prepare reproducible target cohorts for the final private-test GPU runs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.utils.io import read_jsonl, setup_stdout
from vifinqa.utils.viet_text import norm


_BANK_NOTE_MARKERS = (
    "ngan hang", "lai va phi", "lai suat", "ngoai hoi", "tien gui",
    "cho vay", "tin dung", "chung khoan", "du phong rui ro", "thanh khoan",
    "no co kha nang mat von", "tai san phai sinh", "nhay cam voi lai suat",
)


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--all-out", required=True)
    parser.add_argument("--schema-out", required=True)
    args = parser.parse_args()

    checkpoint = {int(row["id"]): row for row in read_jsonl(args.checkpoint)}
    retrieval = {int(row["id"]): row for row in read_jsonl(args.retrieval)}
    if set(checkpoint) != set(retrieval):
        raise ValueError("checkpoint/retrieval id sets differ")

    single_vote = sorted(
        qid for qid, row in checkpoint.items()
        if row.get("source") == "llm_select"
        and int(row.get("votes") or 0) == 1
        and int(row.get("n_ok") or 0) == 1
    )
    single_vote_set = set(single_vote)
    schema = []
    reasons = Counter()
    for qid, row in checkpoint.items():
        rec = retrieval[qid]
        route = rec.get("route") or {}
        op = str((route.get("plan") or {}).get("op") or "lookup")
        no_requirements = not route.get("evidence_requirements")
        bank_note = any(marker in norm(rec.get("question") or "")
                        for marker in _BANK_NOTE_MARKERS)
        failed = row.get("status") != "ok"
        in_single_vote = qid in single_vote_set
        if failed or (in_single_vote and (op != "lookup" or no_requirements or bank_note)):
            schema.append(qid)
            if failed:
                reasons["failed"] += 1
            if op != "lookup":
                reasons[f"op:{op}"] += 1
            if no_requirements:
                reasons["no_requirements"] += 1
            if bank_note:
                reasons["bank_note"] += 1

    all_payload = {"name": "v29_llm_single_vote", "count": len(single_vote),
                   "ids": single_vote}
    schema_payload = {"name": "v30_formula_ranking_bank_schema",
                      "count": len(schema), "ids": sorted(schema)}
    Path(args.all_out).write_text(
        json.dumps(all_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.schema_out).write_text(
        json.dumps(schema_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"all single-vote targets: {len(single_vote)} -> {args.all_out}")
    print(f"schema targets: {len(schema)} -> {args.schema_out}")
    print("schema reasons:", dict(reasons))


if __name__ == "__main__":
    main()
