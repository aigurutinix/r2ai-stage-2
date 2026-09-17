"""Measure the retrieval refusal threshold on local positive/negative cases.

Positive cases come from clean_gold. Synthetic negatives deliberately keep a
valid company and year while asking for an out-of-domain fact, making them more
useful than empty or malformed questions. This is a development calibration
aid, not a claim about the hidden BTC test set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.product import ProductService  # noqa: E402
from kingpro.product.service import _out_of_corpus_topic  # noqa: E402
from kingpro.retrieval.bm25_index import extract_all_facets  # noqa: E402


NEGATIVE_TEMPLATES = {
    "absurd": "Công ty {ticker} năm {year} nuôi bao nhiêu thú cưng kỳ lân trong văn phòng?",
    "visual_brand": "Màu logo chủ đạo của công ty {ticker} năm {year} có mã hex là gì?",
    "employee_survey": (
        "Điểm hài lòng nhân viên theo khảo sát nội bộ của {ticker} năm {year} "
        "là bao nhiêu phần trăm?"
    ),
    "market_price": (
        "Giá cổ phiếu đóng cửa của {ticker} trong ngày giao dịch cuối cùng năm {year} "
        "là bao nhiêu đồng?"
    ),
    "mobile_usage": "Ứng dụng di động của {ticker} có bao nhiêu lượt tải trong năm {year}?",
    "esg_external": (
        "Điểm xếp hạng ESG quốc tế của {ticker} năm {year} là bao nhiêu điểm?"
    ),
}


def metrics_at(scores: list[dict], threshold: float) -> dict:
    positives = [row for row in scores if row["label"] == 1]
    negatives = [row for row in scores if row["label"] == 0]
    tp = sum(row["score"] >= threshold for row in positives)
    fn = len(positives) - tp
    fp = sum(row["score"] >= threshold for row in negatives)
    tn = len(negatives) - fp
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "positive_accept": recall,
        "negative_reject": specificity,
        "grounded_precision": precision,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selected-threshold", type=float, default=0.90)
    args = parser.parse_args()
    service = ProductService(root=ROOT, llm_fn=lambda _s, _u: "result = 0")
    positives = [json.loads(line) for line in (ROOT / "build" / "clean_gold.jsonl").open(encoding="utf-8")]
    scores: list[dict] = []
    for row in positives:
        facets = extract_all_facets(row["question"])
        _facets, _docs, _tables, checks = service._retrieve(row["question"], facets)
        scores.append({
            "score": checks["confidence"] if _out_of_corpus_topic(row["question"]) is None else 0.0,
            "retrieval_score": checks["confidence"],
            "domain_block": _out_of_corpus_topic(row["question"]),
            "label": 1, "kind": "positive",
            "question": row["question"],
        })
        if facets["tickers"] and facets["years"]:
            values = {"ticker": facets["tickers"][0], "year": facets["years"][0]}
            for kind, template in NEGATIVE_TEMPLATES.items():
                negative = template.format(**values)
                nfac = extract_all_facets(negative)
                _f, _d, _t, nchecks = service._retrieve(negative, nfac)
                domain_block = _out_of_corpus_topic(negative)
                scores.append({
                    "score": 0.0 if domain_block is not None else nchecks["confidence"],
                    "retrieval_score": nchecks["confidence"],
                    "domain_block": domain_block,
                    "label": 0, "kind": kind,
                    "question": negative,
                })

    positive_count = sum(row["label"] for row in scores)
    negative_count = len(scores) - positive_count
    print(f"cases={len(scores)} positives={positive_count} negatives={negative_count}")
    print(
        f"{'threshold':>10} {'positive_accept':>16} {'negative_reject':>16} "
        f"{'precision':>10} {'balanced':>10}"
    )
    curve = []
    for threshold in [x / 100 for x in range(60, 97, 2)]:
        row = metrics_at(scores, threshold)
        curve.append(row)
        print(
            f"{threshold:10.2f} {row['positive_accept']:16.1%} "
            f"{row['negative_reject']:16.1%} {row['grounded_precision']:10.1%} "
            f"{row['balanced_accuracy']:10.1%}"
        )

    selected = metrics_at(scores, args.selected_threshold)
    false_accepts = sorted(
        (
            row for row in scores
            if row["label"] == 0 and row["score"] >= args.selected_threshold
        ),
        key=lambda row: (-row["score"], row["kind"], row["question"]),
    )
    false_refusals = sorted(
        (
            row for row in scores
            if row["label"] == 1 and row["score"] < args.selected_threshold
        ),
        key=lambda row: (row["score"], row["question"]),
    )
    print(
        f"selected={args.selected_threshold:.2f} false_accepts={len(false_accepts)} "
        f"false_refusals={len(false_refusals)}"
    )
    for row in false_accepts[:12]:
        print(f"  ACCEPTED_NEGATIVE {row['score']:.4f} {row['kind']}: {row['question']}")
    for row in false_refusals[:12]:
        print(f"  REFUSED_POSITIVE {row['score']:.4f}: {row['question']}")

    if args.out:
        payload = {
            "case_count": len(scores),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "negative_templates": NEGATIVE_TEMPLATES,
            "selected": selected,
            "curve": curve,
            "false_accepts": false_accepts,
            "false_refusals": false_refusals,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
