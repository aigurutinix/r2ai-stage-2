"""Find near-duplicate test questions whose stored answers disagree.

Pairs are compared only after exact agreement on ticker set, year set, report
scope, requested unit, operator signature and time basis.  Lexical similarity
is then calculated over finance terms with company boilerplate removed.  The
audit is read-only and treats every disagreement as a review lead, never as a
gold label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.product.service import ProductService  # noqa: E402


@dataclass(frozen=True)
class PairFinding:
    left_id: int
    right_id: int
    score: float
    shared_terms: list[str]
    left_only_terms: list[str]
    right_only_terms: list[str]
    left_answer: float
    right_answer: float
    left_unit: str
    right_unit: str
    left_canonical: float
    right_canonical: float
    absolute_difference: float
    left_question: str
    right_question: str
    left_tables: list[str]
    right_tables: list[str]


def f1_terms(left: set[str], right: set[str]) -> tuple[float, set[str]]:
    shared = left & right
    if not shared:
        return 0.0, shared
    precision = len(shared) / len(left)
    recall = len(shared) / len(right)
    return 2 * precision * recall / (precision + recall), shared


def materially_different(left: float, right: float, *, abs_tol: float = 0.005) -> bool:
    return not math.isclose(left, right, rel_tol=1e-7, abs_tol=abs_tol)


def currency_unit(unit: object) -> tuple[str, int] | None:
    if not isinstance(unit, tuple) or len(unit) != 2:
        return None
    name, multiplier = str(unit[0]), int(unit[1])
    return (name, multiplier) if "đồng" in name else None


def cluster_key(item: dict, *, cross_currency: bool = False) -> tuple:
    facets = item["facets"]
    unit = currency_unit(item.get("unit"))
    unit_key: object = "currency" if cross_currency and unit else item.get("unit")
    return (
        tuple(sorted(str(value) for value in facets.get("tickers", []))),
        tuple(sorted(str(value) for value in facets.get("years", []))),
        str(facets.get("scope", "")),
        tuple(unit_key) if isinstance(unit_key, tuple) else str(unit_key),
        tuple(sorted(item.get("intent", []))),
        str(item.get("time_basis", "")),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--min-score", type=float, default=0.82)
    parser.add_argument("--min-shared", type=int, default=4)
    parser.add_argument(
        "--cross-currency",
        action="store_true",
        help="compare equivalent currency questions after scaling answers back to VND",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    service = ProductService(
        root=ROOT,
        replay_submission=args.submission_dir / "submission.json",
    )
    index = service._load_replay_paraphrase_index()
    clusters: dict[tuple, list[dict]] = defaultdict(list)
    for item in index:
        clusters[cluster_key(item, cross_currency=args.cross_currency)].append(item)

    findings: list[PairFinding] = []
    pairs_compared = 0
    for items in clusters.values():
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                pairs_compared += 1
                score, shared = f1_terms(set(left["terms"]), set(right["terms"]))
                if score < args.min_score or len(shared) < args.min_shared:
                    continue
                left_row = left["row"]
                right_row = right["row"]
                left_answer = float(left_row["answer"])
                right_answer = float(right_row["answer"])
                left_unit = currency_unit(left.get("unit"))
                right_unit = currency_unit(right.get("unit"))
                if args.cross_currency and left_unit and right_unit:
                    left_value = left_answer * left_unit[1]
                    right_value = right_answer * right_unit[1]
                    # Stored answers are rounded to two decimals in their own
                    # units.  Half a cent of the coarser unit is expected.
                    tolerance = 0.0051 * max(left_unit[1], right_unit[1])
                else:
                    left_value = left_answer
                    right_value = right_answer
                    tolerance = 0.005
                if not materially_different(left_value, right_value, abs_tol=tolerance):
                    continue
                findings.append(PairFinding(
                    left_id=int(left_row["id"]),
                    right_id=int(right_row["id"]),
                    score=round(score, 4),
                    shared_terms=sorted(shared),
                    left_only_terms=sorted(set(left["terms"]) - shared),
                    right_only_terms=sorted(set(right["terms"]) - shared),
                    left_answer=left_answer,
                    right_answer=right_answer,
                    left_unit=left_unit[0] if left_unit else str(left.get("unit")),
                    right_unit=right_unit[0] if right_unit else str(right.get("unit")),
                    left_canonical=left_value,
                    right_canonical=right_value,
                    absolute_difference=round(abs(left_value - right_value), 6),
                    left_question=str(left_row["question"]),
                    right_question=str(right_row["question"]),
                    left_tables=[str(value) for value in left_row.get("relevant_tables", [])],
                    right_tables=[str(value) for value in right_row.get("relevant_tables", [])],
                ))

    findings.sort(key=lambda item: (-item.score, item.left_id, item.right_id))
    payload = {
        "submission": str(args.submission_dir.resolve()),
        "indexed_questions": len(index),
        "clusters": len(clusters),
        "pairs_compared_after_hard_facets": pairs_compared,
        "min_score": args.min_score,
        "min_shared": args.min_shared,
        "cross_currency": args.cross_currency,
        "finding_count": len(findings),
        "findings": [asdict(item) for item in findings],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
