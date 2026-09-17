"""Turn a measured v265 score vector into a deterministic promotion verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SIZE = 506
BASELINE = {
    "execution": 0.7115,
    "answer": 0.7115,
    "tables_f2": 0.6104,
    "tables_precision": 0.5912,
    "tables_recall": 0.6244,
    "tables_mrr5": 0.6514,
    "docs_f2": 0.9611,
    "docs_precision": 0.9580,
    "docs_recall": 0.9672,
    "docs_mrr5": 0.9806,
}


def adjudicate(measured: dict[str, float], tolerance: float = 0.00011) -> dict:
    missing = sorted(set(BASELINE) - set(measured))
    if missing:
        raise ValueError(f"missing score fields: {missing}")
    deltas = {key: round(float(measured[key]) - value, 6) for key, value in BASELINE.items()}
    answer_count = int(round(float(measured["answer"]) * PUBLIC_SIZE))
    if abs(float(measured["answer"]) - answer_count / PUBLIC_SIZE) > 0.00006:
        raise ValueError("answer score is inconsistent with a 506-question public split")

    if answer_count >= 361:
        q24 = "new_answer_public_correct"
    elif answer_count <= 359:
        q24 = "new_answer_public_wrong"
    else:
        q24 = "public_neutral_or_unresolved"

    docs_regression = any(
        deltas[key] < -tolerance
        for key in ("docs_f2", "docs_precision", "docs_recall", "docs_mrr5")
    )
    table_regression = deltas["tables_f2"] < -tolerance
    table_improvement = deltas["tables_f2"] > tolerance
    execution_regression = deltas["execution"] < -tolerance
    answer_regression = deltas["answer"] < -tolerance

    if q24 == "new_answer_public_wrong" or answer_regression:
        verdict = "reject_rollback_v217"
    elif docs_regression or execution_regression:
        verdict = "reject_rollback_v217"
    elif q24 == "new_answer_public_correct" and not table_regression:
        verdict = "promote_candidate"
    elif q24 == "new_answer_public_correct" and table_regression:
        verdict = "split_q24_from_retrieval_ablation"
    elif table_improvement and not answer_regression:
        verdict = "promote_retrieval_candidate_or_measure_v239"
    elif table_regression:
        verdict = "reject_retrieval_ablation"
    else:
        verdict = "public_neutral_keep_v217_selected"

    return {
        "baseline": BASELINE,
        "measured": measured,
        "deltas": deltas,
        "public_answer_count": answer_count,
        "q24_inference": q24,
        "gates": {
            "execution_regression": execution_regression,
            "answer_regression": answer_regression,
            "tables_f2_regression": table_regression,
            "tables_f2_improvement": table_improvement,
            "docs_regression": docs_regression,
        },
        "verdict": verdict,
        "automatic_external_action": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in BASELINE:
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, required=True)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "build/v265_public_score_verdict.json"
    )
    args = parser.parse_args()
    measured = {key: float(getattr(args, key)) for key in BASELINE}
    report = adjudicate(measured)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
