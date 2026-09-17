"""Mine near-paraphrase questions with inconsistent answers or source operands.

The test set contains many human/templated paraphrases.  Exact facet grouping
is intentionally conservative and misses pairs when their operator wording is
parsed differently.  This audit keeps the hard entity/year/scope/unit-family
contract, then uses character n-gram similarity to produce a broad review
queue.  It is read-only: a close pair is evidence for review, never a gold
label by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_legacy_source_scope import fold  # noqa: E402
from kingpro.product.service import ProductService  # noqa: E402


def unit_family(unit: object) -> tuple[str, int]:
    if isinstance(unit, tuple) and len(unit) == 2:
        name, multiplier = str(unit[0]), int(unit[1])
    elif isinstance(unit, list) and len(unit) == 2:
        name, multiplier = str(unit[0]), int(unit[1])
    else:
        return (str(unit), 1)
    if "đồng" in name:
        return ("currency", multiplier)
    if "phần trăm" in name or "%" in name:
        return ("percent", 1)
    if "điểm phần trăm" in name:
        return ("percentage_point", 1)
    return (name, multiplier)


def canonical_answer(answer: object, family: tuple[str, int]) -> float:
    value = float(answer)
    return value * family[1] if family[0] == "currency" else value


def source_signatures(path: Path | None) -> dict[int, set[str]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, set[str]] = {}
    for record in payload.get("records", []):
        values = set()
        for cell in record.get("cells", []):
            values.add(
                "{}:{}:{}:{}".format(
                    cell.get("source_table", ""),
                    cell.get("row_idx", ""),
                    cell.get("col_idx", ""),
                    cell.get("raw_physical", cell.get("raw_manifest", "")),
                )
            )
        result[int(record["id"])] = values
    return result


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--min-char-score", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    submission = args.submission_dir.resolve()
    service = ProductService(root=ROOT, replay_submission=submission / "submission.json")
    index = service._load_replay_paraphrase_index()
    questions = [fold(item["row"]["question"]) for item in index]
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    ).fit_transform(questions)
    similarities = (matrix @ matrix.T).tocsr()
    sources = source_signatures(args.lineage.resolve() if args.lineage else None)

    findings: list[dict[str, Any]] = []
    hard_compatible_pairs = 0
    for left_index, left in enumerate(index):
        left_facets = left["facets"]
        left_family = unit_family(left.get("unit"))
        left_tickers = set(left_facets.get("tickers", []))
        left_years = set(left_facets.get("years", []))
        if not left_tickers:
            continue
        start, end = similarities.indptr[left_index:left_index + 2]
        for offset in range(start, end):
            right_index = int(similarities.indices[offset])
            if right_index <= left_index:
                continue
            score = float(similarities.data[offset])
            if score < args.min_char_score:
                continue
            right = index[right_index]
            right_facets = right["facets"]
            right_family = unit_family(right.get("unit"))
            if left_tickers != set(right_facets.get("tickers", [])):
                continue
            if left_years != set(right_facets.get("years", [])):
                continue
            if str(left_facets.get("scope", "")) != str(right_facets.get("scope", "")):
                continue
            if left_family[0] != right_family[0]:
                continue
            hard_compatible_pairs += 1

            left_row, right_row = left["row"], right["row"]
            left_value = canonical_answer(left_row["answer"], left_family)
            right_value = canonical_answer(right_row["answer"], right_family)
            tolerance = (
                0.0051 * max(left_family[1], right_family[1])
                if left_family[0] == "currency"
                else 0.0051
            )
            answer_differs = not math.isclose(
                left_value, right_value, rel_tol=1e-9, abs_tol=tolerance
            )
            left_sources = sources.get(int(left_row["id"]), set())
            right_sources = sources.get(int(right_row["id"]), set())
            source_overlap = jaccard(left_sources, right_sources)
            term_overlap = jaccard(set(left["terms"]), set(right["terms"]))
            intent_equal = set(left.get("intent", [])) == set(right.get("intent", []))

            # Keep answer conflicts and very-close source-lineage conflicts.
            if not answer_differs and (score < 0.72 or source_overlap >= 0.999):
                continue
            priority = (
                100 * int(answer_differs)
                + 30 * score
                + 10 * term_overlap
                + 8 * int(intent_equal)
                + 6 * source_overlap
            )
            findings.append(
                {
                    "left_id": int(left_row["id"]),
                    "right_id": int(right_row["id"]),
                    "priority": round(priority, 4),
                    "char_score": round(score, 4),
                    "term_jaccard": round(term_overlap, 4),
                    "intent_equal": intent_equal,
                    "source_jaccard": round(source_overlap, 4),
                    "answer_differs": answer_differs,
                    "left_answer": left_row["answer"],
                    "right_answer": right_row["answer"],
                    "left_canonical": left_value,
                    "right_canonical": right_value,
                    "unit_family": left_family[0],
                    "tickers": sorted(left_tickers),
                    "years": sorted(left_years),
                    "scope": left_facets.get("scope", ""),
                    "left_intent": sorted(left.get("intent", [])),
                    "right_intent": sorted(right.get("intent", [])),
                    "left_question": left_row["question"],
                    "right_question": right_row["question"],
                    "left_tables": left_row.get("relevant_tables", []),
                    "right_tables": right_row.get("relevant_tables", []),
                    "claim_limit": "near-paraphrase review lead; verify operator and source before mutation",
                }
            )

    findings.sort(
        key=lambda item: (
            -int(item["answer_differs"]),
            -float(item["char_score"]),
            -float(item["term_jaccard"]),
            int(item["left_id"]),
            int(item["right_id"]),
        )
    )
    findings = findings[: args.limit]
    payload = {
        "kind": "broad_near_paraphrase_consistency",
        "submission": str(submission),
        "lineage": str(args.lineage.resolve()) if args.lineage else None,
        "indexed_questions": len(index),
        "hard_compatible_pairs_above_threshold": hard_compatible_pairs,
        "finding_count": len(findings),
        "answer_conflict_count": sum(bool(item["answer_differs"]) for item in findings),
        "question_ids": sorted(
            {
                int(item[key])
                for item in findings
                for key in ("left_id", "right_id")
            }
        ),
        "policy": "Read-only; source/operator verification is required before any repair.",
        "findings": findings,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "findings"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
