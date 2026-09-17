"""Conservatively merge high-confidence rule results into a codegen checkpoint.

The default policy only fills failed checkpoint rows.  It deliberately rejects
ambiguous matches and unit warnings, so a broader solver run cannot overwrite a
known working Qwen answer without an explicit ``--replace-ok`` opt-in.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl


REJECT_MARKERS = ("AMBIGUOUS", "UNIT-WARN")


def _by_id(rows: list[dict], label: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for row in rows:
        qid = int(row["id"])
        if qid in indexed:
            raise ValueError(f"{label} contains duplicate id {qid}")
        indexed[qid] = row
    return indexed


def merge_codegen(base_rows: list[dict], candidate_rows: list[dict], *,
                  min_confidence: float = 90.0,
                  replace_ok: bool = False,
                  allow_ids: set[int] | None = None) -> tuple[list[dict], list[int]]:
    base = _by_id(base_rows, "base")
    candidates = _by_id(candidate_rows, "candidate")
    if set(base) != set(candidates):
        missing = sorted(set(base) ^ set(candidates))
        raise ValueError(f"base/candidate id sets differ: {missing[:10]}")

    merged: list[dict] = []
    accepted: list[int] = []
    for qid in sorted(base):
        current = base[qid]
        candidate = candidates[qid]
        detail = str(candidate.get("detail", ""))
        eligible = (
            candidate.get("status") == "ok"
            and float(candidate.get("detail_conf", 0.0) or 0.0) >= min_confidence
            and not any(marker in detail for marker in REJECT_MARKERS)
            and (replace_ok or current.get("status") != "ok")
            and (allow_ids is None or qid in allow_ids)
        )
        if not eligible:
            merged.append(current)
            continue

        replacement = dict(candidate)
        original_source = str(candidate.get("source", "unknown"))
        replacement["source"] = (
            original_source if original_source.startswith("canonical_v2_blend:")
            else f"canonical_v2_blend:{original_source}"
        )
        replacement["detail"] = (
            f"canonical-v2 safe fill over {current.get('source', 'unknown')}; "
            f"{detail}"
        )
        merged.append(replacement)
        accepted.append(qid)
    return merged, accepted


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-confidence", type=float, default=90.0)
    parser.add_argument("--replace-ok", action="store_true",
                        help="also replace successful base rows; off by default")
    parser.add_argument("--allow-ids", default="",
                        help="optional comma-separated audited id allowlist")
    args = parser.parse_args()

    allow_ids = ({int(value) for value in args.allow_ids.split(",") if value.strip()}
                 if args.allow_ids else None)

    merged, accepted = merge_codegen(
        read_jsonl(args.base), read_jsonl(args.candidate),
        min_confidence=args.min_confidence, replace_ok=args.replace_ok,
        allow_ids=allow_ids,
    )
    write_jsonl(args.out, merged)
    print(f"merged {len(merged)} rows; accepted={len(accepted)} ids={accepted}")
    print("sources:", dict(Counter(row.get("source", "") for row in merged)))


if __name__ == "__main__":
    main()
