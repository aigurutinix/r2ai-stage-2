"""Stratified source-bound proxy validation for private-set robustness.

The public leaderboard is never used as a label.  Instead, this audit slices
the 1,012 verified registry questions by stable entity folds, source years and
question archetypes.  It reconstructs per-question document recall from the
full retrieval report and measures deterministic-compiler precision in every
cohort.  The result detects concentrated weaknesses that a single global score
can hide.

These are diagnostic holdout-like partitions, not true unseen BTC labels: the
underlying corpus and verified registry remain available to the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kingpro.evaluation.metrics import coerce_number  # noqa: E402
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler  # noqa: E402
from kingpro.retrieval.bm25_index import extract_all_facets, fold  # noqa: E402


DEFAULT_REGISTRY = (
    ROOT
    / "sub_v290_scope2"
    / "submission.json"
)
DEFAULT_RETRIEVAL = (
    ROOT
    / "build"
    / "demo_compliance"
    / "document_retrieval_coverage_semantic_v22_final.json"
)
DEFAULT_OUTPUT = (
    ROOT / "build" / "demo_compliance" / "private_proxy_cohorts_hanoi_v290_v50.json"
)
_YEAR_RE = re.compile(r"_financial_statements_(20\d{2})(?:_|$)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_dimensions(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    tickers: set[str] = set()
    years: set[str] = set()
    for document in row.get("relevant_docs", []) or []:
        name = str(document)
        marker = "_financial_statements_"
        if marker in name:
            tickers.add(name.split(marker, 1)[0])
        match = _YEAR_RE.search(name)
        if match:
            years.add(match.group(1))
    return sorted(tickers), sorted(years)


def _archetype(question: str) -> str:
    normalized = fold(question)
    if re.search(
        r"\b(?:cao nhat|thap nhat|lon nhat|nho nhat|trong nhom|"
        r"trong so|chi xet|thoa dieu kien)\b",
        normalized,
    ):
        return "conditional_extreme"
    if re.search(
        r"\b(?:trung binh|binh quan|tong cong|tong gia tri|"
        r"bao nhieu cong ty|bao nhieu doanh nghiep|dem)\b",
        normalized,
    ):
        return "aggregation"
    if re.search(
        r"\b(?:ty le|ti le|ty so|ti so|ty trong|ti trong|he so|"
        r"bien loi nhuan|phan tram)\b",
        normalized,
    ):
        return "ratio"
    if re.search(
        r"\b(?:tang truong|tang bao nhieu|giam bao nhieu|chenh lech|"
        r"thay doi|so voi)\b|\btu\b.{0,80}\bsang\b",
        normalized,
    ):
        return "change"
    return "direct_lookup"


def _entity_cohort(tickers: list[str]) -> str:
    if len(tickers) != 1:
        return "multi_entity_or_unresolved"
    digest = hashlib.sha256(tickers[0].encode("utf-8")).digest()
    return f"single_entity_fold_{int.from_bytes(digest[:4], 'big') % 5}"


def _year_cohort(years: list[str]) -> str:
    return years[0] if len(years) == 1 else "multi_year_or_unresolved"


def _document_recalls(
    rows: list[dict[str, Any]], retrieval: dict[str, Any]
) -> dict[int, float]:
    mode = retrieval.get("modes", {}).get("1", {})
    misses = {int(item["id"]): item for item in mode.get("misses", [])}
    recalls: dict[int, float] = {}
    for row in rows:
        question_id = int(row["id"])
        miss = misses.get(question_id)
        if miss is None:
            recalls[question_id] = 1.0
            continue
        targets = set(str(value) for value in miss.get("targets", []))
        predictions = set(str(value) for value in miss.get("predictions", []))
        recalls[question_id] = (
            len(targets & predictions) / len(targets) if targets else 0.0
        )
    reconstructed = sum(recalls.values()) / len(rows) if rows else 0.0
    reported = float(mode.get("macro_recall", 0.0))
    if not math.isclose(reconstructed, reported, rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            "per-question recall reconstruction disagrees with retrieval report"
        )
    return recalls


def _summarize(
    members: list[dict[str, Any]],
    recalls: dict[int, float],
    compiler_rows: dict[int, tuple[bool, bool]],
) -> dict[str, Any]:
    values = [recalls[int(row["id"])] for row in members]
    effective_values = [
        1.0 if compiler_rows[int(row["id"])][1] else recalls[int(row["id"])]
        for row in members
    ]
    accepted = sum(compiler_rows[int(row["id"])][0] for row in members)
    matched = sum(compiler_rows[int(row["id"])][1] for row in members)
    return {
        "questions": len(members),
        "document_macro_recall": round(sum(values) / len(values), 6),
        "document_full_recall_questions": sum(value == 1.0 for value in values),
        # A source-cell-bound compiler does not consume the ordinary retrieval
        # context.  Count a registry-matched compiled route as fully grounded,
        # while preserving raw document recall beside it so retrieval defects
        # remain visible instead of being hidden by the aggregate.
        "effective_grounded_macro_recall": round(
            sum(effective_values) / len(effective_values), 6
        ),
        "effective_full_grounded_questions": sum(
            value == 1.0 for value in effective_values
        ),
        "compiler_accepted": accepted,
        "compiler_matched": matched,
        "compiler_coverage": round(accepted / len(members), 6),
        "compiler_precision": round(matched / accepted, 6) if accepted else None,
    }


def audit(
    registry: Path = DEFAULT_REGISTRY,
    retrieval_report: Path = DEFAULT_RETRIEVAL,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    rows = json.loads(registry.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_report.read_text(encoding="utf-8"))
    recalls = _document_recalls(rows, retrieval)
    compiler = DeterministicFinancialCompiler(root)
    compiler_rows: dict[int, tuple[bool, bool]] = {}
    compiler_mismatches: list[int] = []
    for row in rows:
        question_id = int(row["id"])
        question = str(row.get("question", ""))
        compiled = compiler.compile(question, extract_all_facets(question))
        accepted = compiled is not None
        expected = coerce_number(row.get("answer"))
        matched = bool(
            accepted
            and expected is not None
            and abs(float(compiled.answer) - expected) <= 0.0050001
        )
        compiler_rows[question_id] = (accepted, matched)
        if accepted and not matched:
            compiler_mismatches.append(question_id)

    dimensions: dict[str, tuple[Callable[[dict[str, Any]], str], float]] = {
        "entity_folds": (
            lambda row: _entity_cohort(_source_dimensions(row)[0]),
            0.98,
        ),
        "source_years": (
            lambda row: _year_cohort(_source_dimensions(row)[1]),
            0.97,
        ),
        "question_archetypes": (
            lambda row: _archetype(str(row.get("question", ""))),
            0.96,
        ),
    }
    cohorts: dict[str, dict[str, Any]] = {}
    risk_flags: list[dict[str, Any]] = []
    retrieval_observations: list[dict[str, Any]] = []
    for dimension, (key_fn, threshold) in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[key_fn(row)].append(row)
        cohorts[dimension] = {}
        for key, members in sorted(grouped.items()):
            summary = _summarize(members, recalls, compiler_rows)
            cohorts[dimension][key] = summary
            if summary["questions"] >= 30 and summary["document_macro_recall"] < threshold:
                retrieval_observations.append(
                    {
                        "dimension": dimension,
                        "cohort": key,
                        "kind": "raw_document_recall_below_floor",
                        "value": summary["document_macro_recall"],
                        "floor": threshold,
                    }
                )
            if (
                summary["questions"] >= 30
                and summary["effective_grounded_macro_recall"] < threshold
            ):
                risk_flags.append(
                    {
                        "dimension": dimension,
                        "cohort": key,
                        "kind": "effective_grounded_recall_below_floor",
                        "value": summary["effective_grounded_macro_recall"],
                        "floor": threshold,
                    }
                )
            precision = summary["compiler_precision"]
            if precision is not None and precision < 1.0:
                risk_flags.append(
                    {
                        "dimension": dimension,
                        "cohort": key,
                        "kind": "compiler_precision_below_one",
                        "value": precision,
                        "floor": 1.0,
                    }
                )

    global_summary = _summarize(rows, recalls, compiler_rows)
    passed = bool(
        len(rows) == 1012
        and global_summary["effective_grounded_macro_recall"] >= 0.98
        and global_summary["compiler_accepted"] >= 101
        and global_summary["compiler_precision"] == 1.0
        and not compiler_mismatches
        and not risk_flags
    )
    return {
        "schema_version": 1,
        "kind": "source_bound_private_proxy_not_btc_hidden_gold",
        "registry": str(registry.resolve()),
        "retrieval_report": str(retrieval_report.resolve()),
        "input_hashes": {
            "registry_sha256": _sha256(registry),
            "retrieval_report_sha256": _sha256(retrieval_report),
            "compiler_sha256": _sha256(
                root / "src" / "kingpro" / "product" / "deterministic_compiler.py"
            ),
        },
        "global": global_summary,
        "cohorts": cohorts,
        "risk_floors": {
            "entity_folds_min_questions_30": 0.98,
            "source_years_min_questions_30": 0.97,
            "question_archetypes_min_questions_30": 0.96,
            "compiler_precision_when_accepted": 1.0,
        },
        "risk_flags": risk_flags,
        "retrieval_observations": retrieval_observations,
        "compiler_mismatch_ids": compiler_mismatches,
        "passed": passed,
        "claim_limits": [
            "No public leaderboard score or question ID is used as a feature or label.",
            "Document targets are audited executable source bindings, not BTC hidden gold.",
            "Effective grounded recall credits only compiler outputs that match the verified local registry and retain exact source-cell bindings; raw document recall remains separately reported.",
            "Entity folds are diagnostic slices; this is not a trained leave-one-company-out model.",
            "Passing does not predict private score or unrestricted natural-language accuracy.",
        ],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-on-risk", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.registry.resolve(),
        args.retrieval_report.resolve(),
        root=args.root.resolve(),
    )
    output = args.out.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit("output must remain inside the project") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_risk and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
