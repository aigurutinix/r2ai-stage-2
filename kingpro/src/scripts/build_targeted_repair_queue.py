"""Build a dependency-aware, high-throughput repair queue for Finance Tabular QA.

The queue is deliberately not an answer oracle. It merges independent risk
signals, removes durable source reviews, collapses stale ensemble ancestry and
produces operator-homogeneous batches that can be verified in parallel.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text).casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


def family(question: str) -> str:
    q = fold(question)
    if re.search(r"cao nhat|thap nhat|lon nhat|nho nhat|nhieu nhat|it nhat|nam nao", q):
        return "selector_extrema"
    if re.search(r"bao nhieu (cong ty|doanh nghiep|ngan hang)|co bao nhieu", q):
        return "threshold_count"
    if re.search(r"binh quan|trung binh|trung vi", q):
        return "aggregation_mean"
    if re.search(r"tang truong|toc do tang|toc do giam|cagr", q):
        return "growth"
    if re.search(
        r"chenh lech|hieu so|\bhieu (?:giua|cua)\b|thay doi|bien dong|nhieu hon|it hon|lon hon|nho hon|"
        r"tru di|so voi|hon bao nhieu|kem bao nhieu",
        q,
    ):
        return "difference_sign"
    if re.search(r"ty le|ti le|ty trong|ty suat|phan tram|%|\broe\b|\broa\b|gap .* lan", q):
        return "ratio_percent"
    if re.search(
        r"tong cong(?! ty)|cong lai|tong (?:gia tri|so du|so tien|tai san|nguon von|"
        r"no |chi phi|doanh thu|loi nhuan|von |cac khoan)",
        q,
    ):
        return "aggregation_total"
    return "direct_lookup"


def plausible_counterfactual(row: dict[str, Any]) -> bool:
    try:
        baseline = float(row["baseline_answer"])
        alternative = float(row["alternative_answer"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (baseline, alternative)):
        return False
    if baseline == 0 or alternative == 0:
        return False
    ratio = abs(alternative / baseline)
    return 0.05 <= ratio <= 20.0


def provenance_sets(candidate: Path) -> tuple[set[int], set[int], set[int]]:
    source = {int(row["id"]) for row in load_json(candidate / "source_audit.json")}
    panel = {int(row["id"]) for row in load_json(candidate / "panel_source_audit.json")}
    legacy_payload = load_json(ROOT / "build/v209_legacy_query_sources_v207.json")
    legacy = {int(row["id"]) for row in legacy_payload.get("records", [])}
    return source, panel, legacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="sub_top123_candidate_v207_semantic_batch6_final")
    parser.add_argument("--forensics", type=Path, default=ROOT / "build/v209_answer_forensics_after_counterfactual_batch1_v2.json")
    parser.add_argument("--risk", type=Path, default=ROOT / "build/v209_cross_task_risk_v207_top300.json")
    parser.add_argument("--counterfactuals", type=Path, default=ROOT / "build/v209_counterfactual_batches_v207_v2.json")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--out", type=Path, default=ROOT / "build/v209_targeted_repair_queue.json")
    args = parser.parse_args()

    candidate = ROOT / args.candidate
    rows = load_json(candidate / "submission.json")
    by_id = {int(row["id"]): row for row in rows}
    reviewed = {int(row["id"]) for row in load_json(args.forensics).get("reviewed", [])}
    risk_rows = {int(row["id"]): row for row in load_json(args.risk).get("unreviewed_top", [])}
    counter_rows = {
        int(row["id"]): row
        for row in load_json(args.counterfactuals).get("all_disagreements", [])
        if not row.get("public_forced_baseline")
    }
    mbr = {int(row["id"]): row for row in load_jsonl(ROOT / "build/llm_mbr_cache.jsonl")}
    source_ids, panel_ids, legacy_ids = provenance_sets(candidate)

    queue: list[dict[str, Any]] = []
    complex_families = {"selector_extrema", "threshold_count", "aggregation_mean", "growth", "difference_sign", "ratio_percent"}
    for qid, row in by_id.items():
        if qid in reviewed:
            continue
        question = row.get("question", "")
        op_family = family(question)
        risk = risk_rows.get(qid, {})
        counter = counter_rows.get(qid)
        sample = mbr.get(qid, {})
        attempts = int(sample.get("attempts") or 0)
        agree = int(sample.get("agree") or 0)
        consensus = agree / attempts if attempts else None
        plausible = bool(counter and plausible_counterfactual(counter))
        if qid in source_ids:
            provenance = "compact_source"
        elif qid in panel_ids:
            provenance = "panel_source"
        elif qid in legacy_ids:
            provenance = "legacy_traced"
        else:
            provenance = "unresolved"

        score = float(risk.get("risk_score") or 0.0)
        score += 8.0 if op_family in complex_families else 0.0
        score += 9.0 if plausible else 0.0
        # Compact/panel manifests already carry curated semantic notes and are
        # lower-yield than legacy programs that only have runtime cell traces.
        # Spend review budget where semantic attestation is still weakest.
        if provenance in {"compact_source", "panel_source"}:
            score -= 6.0
        elif provenance == "legacy_traced":
            score += 6.0
        else:
            score += 10.0
        score += 8.0 * (1.0 - consensus) if consensus is not None else 2.0
        if not risk and not counter and consensus is None:
            score -= 4.0
        signals = []
        if risk:
            signals.append("cross_task_risk")
        if plausible:
            signals.append("plausible_counterfactual")
        if consensus is not None and consensus < 0.6:
            signals.append("weak_mbr_consensus")
        if provenance == "legacy_traced":
            signals.append("legacy_semantic_attestation_needed")
        elif provenance == "unresolved":
            signals.append("source_provenance_recovery_needed")
        queue.append(
            {
                "id": qid,
                "priority": round(score, 3),
                "family": op_family,
                "question": question,
                "baseline_answer": row.get("answer"),
                "provenance": provenance,
                "signals": signals,
                "risk_reasons": risk.get("reasons", []),
                "mbr_consensus": round(consensus, 4) if consensus is not None else None,
                "counterfactual_answer": counter.get("alternative_answer") if plausible else None,
                "counterfactual_groups": counter.get("dependency_groups", []) if plausible else [],
                "claim_limit": "Triage only; exact source-cell and runtime verification required before mutation.",
            }
        )

    queue.sort(key=lambda item: (-item["priority"], item["id"]))
    queue = queue[: args.limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queue:
        grouped[row["family"]].append(row)
    batches = []
    for op_family, family_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        for start in range(0, len(family_rows), args.batch_size):
            chunk = family_rows[start : start + args.batch_size]
            batches.append(
                {
                    "batch": f"{op_family}_{start // args.batch_size + 1}",
                    "family": op_family,
                    "count": len(chunk),
                    "question_ids": [row["id"] for row in chunk],
                }
            )

    payload = {
        "schema_version": 1,
        "candidate": args.candidate,
        "policy": {
            "kind": "dependency-aware targeted repair queue",
            "durable_reviews_excluded": len(reviewed),
            "batch_size": args.batch_size,
            "automatic_answer_changes": False,
            "source_verification_required": True,
        },
        "counts": {
            "questions": len(rows),
            "unreviewed": len(rows) - len(reviewed),
            "queued": len(queue),
            "plausible_counterfactuals": sum("plausible_counterfactual" in row["signals"] for row in queue),
            "weak_consensus": sum("weak_mbr_consensus" in row["signals"] for row in queue),
        },
        "batches": batches,
        "queue": queue,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "batches": batches, "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
