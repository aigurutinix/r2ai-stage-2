"""Mine high-throughput answer-review batches from independent solver views.

This is a triage tool, not an answer oracle.  It compares the locked candidate with
historical MBR/program/merged/silver outputs, removes questions already closed by
the durable Vô Thượng review ledger, and groups the remaining disagreements by a
repeatable error family.  Nothing is uploaded or mutated.

Example:
  python scripts/mine_counterfactual_batches.py \
    --candidate sub_top123_candidate_v207_semantic_batch6_final \
    --out build/v209_counterfactual_batches_v207.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def agrees(a: float, b: float, relative_tolerance: float) -> bool:
    return abs(a - b) <= relative_tolerance * max(abs(a), abs(b), 1.0)


def source_refs(row: dict[str, Any]) -> list[str]:
    refs = list(row.get("table_refs") or [])
    for item in row.get("ev") or []:
        if isinstance(item, dict) and item.get("ref"):
            refs.append(str(item["ref"]))
    return sorted(set(refs))


def dependency_group(view_name: str, row: dict[str, Any]) -> str:
    """Collapse views that reuse the same generator/retrieval ancestry.

    ``dev_gold`` is derived from program/base/maso agreement and therefore must
    not be counted as a fresh vote.  MBR and full-program caches also share the
    compact-table generator family.  The merged-statement cache is the only
    materially different representation among the local historical caches.
    """
    if view_name == "merged":
        return "merged_statement_llm"
    if view_name == "silver" and "maso" in set(row.get("sources") or []):
        return "derived_maso_consensus"
    return "compact_table_llm"


def has_independent_representation(groups: list[str]) -> bool:
    """Require two genuinely different source representations.

    ``derived_maso_consensus`` is a downstream silver label, not a fresh model
    observation.  Counting it beside its parent caches creates a false quorum.
    The local archive currently contains only two materially independent
    representations: compact tables and merged statements.
    """
    return {"compact_table_llm", "merged_statement_llm"}.issubset(set(groups))


def family(question: str) -> str:
    q = question.casefold()
    if re.search(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất|ít nhất|năm nào", q):
        return "selector_extrema"
    if re.search(r"bao nhiêu (công ty|doanh nghiệp|ngân hàng)|có bao nhiêu", q):
        return "threshold_count"
    if re.search(r"bình quân|trung bình|trung vị", q):
        return "aggregation_mean"
    if re.search(r"tăng trưởng|tốc độ tăng|cagr", q):
        return "growth"
    if re.search(r"chênh lệch|thay đổi|biến động|nhiều hơn|ít hơn", q):
        return "difference_sign"
    if re.search(r"tỷ lệ|tỉ lệ|tỷ trọng|tỷ suất|biên|%|roe|roa", q):
        return "ratio_percent"
    if re.search(r"tổng|cộng lại|tổng cộng", q):
        return "aggregation_total"
    return "direct_lookup"


def discrepancy_kind(baseline: float, alternative: float) -> str:
    if abs(abs(baseline) - abs(alternative)) <= 0.005 * max(abs(baseline), abs(alternative), 1.0):
        if baseline * alternative < 0:
            return "sign"
    if baseline == 0 or alternative == 0:
        return "zero_or_blank"
    ratio = abs(alternative / baseline)
    for scale in (100.0, 1_000.0, 1_000_000.0, 1_000_000_000.0, 1_000_000_000_000.0):
        if abs(ratio - scale) / scale <= 0.005 or abs((1.0 / ratio) - scale) / scale <= 0.005:
            return "unit_scale"
    return "semantic_or_operator"


def reviewed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["id"]) for row in payload.get("reviewed", []) if "id" in row}


def public_forced_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["id"])
        for row in payload.get("forced_one", [])
        if row.get("is_baseline")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="sub_top123_candidate_v207_semantic_batch6_final")
    parser.add_argument(
        "--forensics",
        type=Path,
        default=ROOT / "build/v209_answer_forensics_after_batch11_v3.json",
    )
    parser.add_argument(
        "--public-constraints",
        type=Path,
        default=ROOT / "build/v209_public_score_constraints_v207.json",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    parser.add_argument("--min-independent-views", type=int, default=2)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "build/v209_counterfactual_batches_v207.json"
    )
    args = parser.parse_args()

    candidate_dir = ROOT / args.candidate
    baseline_rows = json.loads((candidate_dir / "submission.json").read_text(encoding="utf-8"))
    baseline = {int(row["id"]): row for row in baseline_rows}
    questions = {
        int(row["id"]): row["question"]
        for row in load_jsonl(ROOT / "data/questions/questions.jsonl")
    }
    reviewed = reviewed_ids(args.forensics)
    forced = public_forced_ids(args.public_constraints)

    views: dict[str, dict[int, dict[str, Any]]] = {}
    for name, relpath in (
        ("mbr", "build/llm_mbr_cache.jsonl"),
        ("program", "build/full_program_cache.jsonl"),
        ("merged", "build/llm_merged_cache.jsonl"),
    ):
        views[name] = {
            int(row["id"]): row
            for row in load_jsonl(ROOT / relpath)
            if row.get("ok") and number(row.get("answer")) is not None
        }
    views["silver"] = {
        int(row["id"]): {
            "id": row["id"],
            "answer": row["gold"],
            "agree": row.get("n_agree", 0),
            "attempts": 3,
            "sources": row.get("sources", []),
        }
        for row in load_jsonl(ROOT / "build/dev_gold.jsonl")
        if number(row.get("gold")) is not None
    }

    candidates: list[dict[str, Any]] = []
    for qid, base_row in sorted(baseline.items()):
        base_answer = number(base_row.get("answer"))
        if base_answer is None:
            continue
        alternatives: list[dict[str, Any]] = []
        for view_name, rows in views.items():
            row = rows.get(qid)
            if not row:
                continue
            alt = number(row.get("answer"))
            if alt is None or agrees(base_answer, alt, args.relative_tolerance):
                continue
            attempts = int(row.get("attempts") or 0)
            agree_count = int(row.get("agree") or 0)
            confidence = agree_count / attempts if attempts else 0.0
            alternatives.append(
                {
                    "view": view_name,
                    "dependency_group": dependency_group(view_name, row),
                    "answer": alt,
                    "agree": agree_count,
                    "attempts": attempts,
                    "confidence": confidence,
                    "table_refs": source_refs(row),
                }
            )
        if not alternatives:
            continue

        clusters: list[dict[str, Any]] = []
        for alt in alternatives:
            target = next(
                (
                    cluster
                    for cluster in clusters
                    if agrees(cluster["center"], alt["answer"], args.relative_tolerance)
                ),
                None,
            )
            if target is None:
                target = {"center": alt["answer"], "members": []}
                clusters.append(target)
            target["members"].append(alt)
            target["center"] = statistics.median(
                item["answer"] for item in target["members"]
            )

        best = max(
            clusters,
            key=lambda cluster: (
                len({row["dependency_group"] for row in cluster["members"]}),
                sum(row["confidence"] for row in cluster["members"]),
            ),
        )
        independent_views = sorted({row["view"] for row in best["members"]})
        dependency_groups = sorted(
            {row["dependency_group"] for row in best["members"]}
        )
        alt_refs = sorted({ref for row in best["members"] for ref in row["table_refs"]})
        base_refs = sorted(set(base_row.get("relevant_tables") or []))
        table_overlap = sorted(set(base_refs) & set(alt_refs))
        question = questions.get(qid, base_row.get("question", ""))
        is_reviewed = qid in reviewed
        is_forced = qid in forced
        score = 10 * len(dependency_groups)
        score += 2 * sum(row["confidence"] for row in best["members"])
        score += 3 if alt_refs and not table_overlap else 0
        score -= 20 if is_reviewed else 0
        score -= 50 if is_forced else 0
        candidates.append(
            {
                "id": qid,
                "question": question,
                "family": family(question),
                "discrepancy": discrepancy_kind(base_answer, best["center"]),
                "baseline_answer": base_answer,
                "alternative_answer": best["center"],
                "independent_views": independent_views,
                "dependency_groups": dependency_groups,
                "view_details": best["members"],
                "baseline_tables": base_refs,
                "alternative_tables": alt_refs,
                "table_overlap": table_overlap,
                "durably_reviewed": is_reviewed,
                "public_forced_baseline": is_forced,
                "priority": round(score, 3),
            }
        )

    candidates.sort(key=lambda row: (-row["priority"], row["id"]))
    actionable = [
        row
        for row in candidates
        if not row["durably_reviewed"]
        and not row["public_forced_baseline"]
        and len(row["dependency_groups"]) >= args.min_independent_views
        and has_independent_representation(row["dependency_groups"])
    ]
    grouped: dict[str, list[int]] = defaultdict(list)
    grouped_discrepancy: dict[str, list[int]] = defaultdict(list)
    for row in actionable:
        grouped[row["family"]].append(row["id"])
        grouped_discrepancy[row["discrepancy"]].append(row["id"])

    report = {
        "schema_version": 1,
        "candidate": args.candidate,
        "policy": {
            "kind": "triage_only",
            "relative_tolerance": args.relative_tolerance,
            "minimum_independent_views": args.min_independent_views,
            "required_representations": [
                "compact_table_llm",
                "merged_statement_llm",
            ],
            "derived_silver_is_independent": False,
            "claim_limit": "Every answer change still requires exact BTC source-cell and runtime verification.",
        },
        "counts": {
            "questions": len(baseline),
            "durably_reviewed": len(reviewed),
            "public_forced_baseline": len(forced),
            "any_disagreement": len(candidates),
            "actionable_multi_view": len(actionable),
        },
        "family_batches": [
            {"family": key, "count": len(ids), "question_ids": ids}
            for key, ids in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
        "discrepancy_batches": [
            {"kind": key, "count": len(ids), "question_ids": ids}
            for key, ids in sorted(
                grouped_discrepancy.items(), key=lambda item: (-len(item[1]), item[0])
            )
        ],
        "actionable": actionable,
        "all_disagreements": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = args.out.with_suffix(".md")
    lines = [
        "# Counterfactual batch miner",
        "",
        f"- Candidate: `{args.candidate}`",
        f"- Any disagreement: {len(candidates)}",
        f"- Actionable multi-view, not durably reviewed: {len(actionable)}",
        "- Triage only: source verification is mandatory before changing an answer.",
        "",
        "## Batches",
        "",
    ]
    for key, ids in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.append(f"- `{key}`: {len(ids)} — {', '.join('q' + str(qid) for qid in ids)}")
    lines.extend(["", "## Top candidates", ""])
    for row in actionable[:50]:
        lines.append(
            f"- q{row['id']} [{row['family']}/{row['discrepancy']}] "
            f"{row['baseline_answer']} -> {row['alternative_answer']:.10g}; "
            f"views={','.join(row['independent_views'])}; priority={row['priority']}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "family_batches": report["family_batches"],
                "top_ids": [row["id"] for row in actionable[:30]],
                "out": str(args.out),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
