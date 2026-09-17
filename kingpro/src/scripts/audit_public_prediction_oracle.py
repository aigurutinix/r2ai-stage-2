"""Compare a submission with a public, non-gold prediction artifact.

This tool deliberately treats the external predictions as a weak oracle.  It
does not turn disagreements into submission edits.  Instead it separates
agreement, likely unit-scale mistakes, and same-document semantic conflicts so
the latter can be verified against the original tables in batches.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_COMMIT = "48c2201"
DEFAULT_SHARDS = tuple(range(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("oracle_repo", type=Path)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--review-log", type=Path, default=Path("knowledge/vothuong/experiments.jsonl")
    )
    return parser.parse_args()


def git_show(repo: Path, revision_path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", revision_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8")


def load_oracle(repo: Path, commit: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for shard in DEFAULT_SHARDS:
        path = f"rescue/shard_{shard}/predictions.jsonl"
        for line in git_show(repo, f"{commit}:{path}").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row["id"])
            if qid in rows:
                raise ValueError(f"duplicate external prediction id: {qid}")
            row["oracle_shard"] = shard
            rows[qid] = row
    return rows


CELL_READ = re.compile(
    r"(?P<variable>df\d+)\.loc\[\s*\(\1?[^\]]*?\['row_index'\]\s*==\s*(?P<row>\d+)\)"
    r"\s*&\s*\([^\]]*?\['column_index'\]\s*==\s*(?P<column>\d+)\)",
    re.DOTALL,
)


def external_cells(repo: Path, commit: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return metadata for cells explicitly read by the external program."""
    evidence = {
        str(entry.get("variable")): str(entry.get("csv_path"))
        for entry in item.get("evidence", [])
    }
    frames: dict[str, list[dict[str, str]]] = {}
    result: list[dict[str, Any]] = []
    query = str(item.get("pandas_query", ""))
    for match in CELL_READ.finditer(query):
        variable = match.group("variable")
        csv_path = evidence.get(variable)
        if not csv_path:
            continue
        if variable not in frames:
            shard = int(item["oracle_shard"])
            repository_path = f"rescue/shard_{shard}/{csv_path}"
            raw = git_show(repo, f"{commit}:{repository_path}")
            frames[variable] = list(csv.DictReader(io.StringIO(raw)))
        row_index = int(match.group("row"))
        column_index = int(match.group("column"))
        selected = [
            row
            for row in frames[variable]
            if int(row["row_index"]) == row_index
            and int(row["column_index"]) == column_index
        ]
        if selected:
            cell = selected[0]
            result.append(
                {
                    "variable": variable,
                    "row_index": row_index,
                    "column_index": column_index,
                    "table_ref": cell.get("table_ref"),
                    "row_label": cell.get("row_label"),
                    "column_label": cell.get("column_label"),
                    "raw_value": cell.get("raw_value"),
                    "numeric_value": cell.get("numeric_value"),
                    "base_value": cell.get("base_value"),
                }
            )
    return result


def reviewed_ids(path: Path) -> set[int]:
    result: set[int] = set()
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("kind") != "review":
            continue
        for value in item.get("question_ids", []):
            try:
                result.add(int(value))
            except (TypeError, ValueError):
                pass
    return result


def doc_set(row: dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("relevant_docs", [])}


def scale_bucket(current: float, external: float) -> str:
    if current == 0 or external == 0:
        return "zero"
    ratio = abs(external / current)
    for exponent in range(-12, 13, 3):
        target = 10.0**exponent
        if math.isclose(ratio, target, rel_tol=1e-6, abs_tol=1e-12):
            return f"10^{exponent}"
    return "other"


def priority(record: dict[str, Any]) -> tuple[int, int, float, int]:
    # Same-document, not previously reviewed and not an obvious 10^3 unit
    # conflict is the most useful queue for source verification.
    same_doc = bool(record["same_docs"])
    unreviewed = not bool(record["reviewed"])
    semantic = record["scale_bucket"] in {"other", "zero", "10^0"}
    return (
        int(same_doc and unreviewed and semantic),
        int(same_doc and semantic),
        -float(record["relative_gap"]),
        -int(record["id"]),
    )


def main() -> int:
    args = parse_args()
    submission_path = args.submission / "submission.json"
    current_rows = {
        int(row["id"]): row
        for row in json.loads(submission_path.read_text(encoding="utf-8"))
    }
    oracle_rows = load_oracle(args.oracle_repo, args.commit)
    reviewed = reviewed_ids(args.review_log)

    agreements: list[int] = []
    conflicts: list[dict[str, Any]] = []
    for qid, external in sorted(oracle_rows.items()):
        current = current_rows.get(qid)
        if current is None:
            continue
        current_answer = float(current["answer"])
        external_answer = float(external["answer"])
        if math.isclose(current_answer, external_answer, abs_tol=0.01, rel_tol=0.0):
            agreements.append(qid)
            continue
        denominator = max(abs(current_answer), abs(external_answer), 1.0)
        conflicts.append(
            {
                "id": qid,
                "question": current.get("question", ""),
                "current_answer": current_answer,
                "external_answer": external_answer,
                "absolute_gap": abs(current_answer - external_answer),
                "relative_gap": abs(current_answer - external_answer) / denominator,
                "scale_bucket": scale_bucket(current_answer, external_answer),
                "same_docs": sorted(doc_set(current) & doc_set(external)),
                "current_docs": sorted(doc_set(current)),
                "external_docs": sorted(doc_set(external)),
                "external_tables": external.get("relevant_tables", []),
                "external_evidence": external.get("evidence", []),
                "external_query": external.get("pandas_query", ""),
                "external_cells": [],
                "oracle_shard": external["oracle_shard"],
                "reviewed": qid in reviewed,
            }
        )

    conflicts.sort(key=priority, reverse=True)
    counts = Counter(item["scale_bucket"] for item in conflicts)
    same_doc = [item for item in conflicts if item["same_docs"]]
    actionable = [
        item
        for item in conflicts
        if item["same_docs"]
        and not item["reviewed"]
        and item["scale_bucket"] in {"other", "zero", "10^0"}
    ]
    # Reading historical blobs is comparatively expensive.  Enrich only the
    # queue that will actually be source-reviewed, not every weak conflict.
    for item in actionable:
        item["external_cells"] = external_cells(
            args.oracle_repo, args.commit, oracle_rows[int(item["id"])]
        )
    output = {
        "schema_version": 1,
        "warning": "External predictions are weak evidence, not gold labels.",
        "submission": str(args.submission.resolve()),
        "oracle_repo": str(args.oracle_repo.resolve()),
        "oracle_commit": args.commit,
        "summary": {
            "external_predictions": len(oracle_rows),
            "compared": len(agreements) + len(conflicts),
            "agreements_abs_tol_0_01": len(agreements),
            "conflicts": len(conflicts),
            "same_document_conflicts": len(same_doc),
            "actionable_unreviewed_same_document": len(actionable),
            "conflict_scale_buckets": dict(sorted(counts.items())),
        },
        "agreement_ids": agreements,
        "actionable_ids": [item["id"] for item in actionable],
        "conflicts": conflicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print("actionable_ids=" + ",".join(str(item["id"]) for item in actionable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
