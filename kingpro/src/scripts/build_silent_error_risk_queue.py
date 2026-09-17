"""Rank plausible silent errors by contradiction, exposure, and assurance.

Passing execution only proves that a program ran.  This script joins the
independent audits already produced by the repository and makes missing tests
visible.  It is deliberately read-only: a high score is review priority, not
authority to change an answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
DEFAULT_OUTPUT = ROOT / "build" / "v245_silent_error_risk"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def artifact_compatibility(
    payload: dict[str, Any], submission: Path
) -> dict[str, Any]:
    """Prove global or row-local compatibility with an older audit artifact.

    Audit families consumed by this queue are deterministic per question.  A
    newer candidate may therefore reuse an older result only for IDs whose
    complete submission row is unchanged.  Changed rows remain stale even when
    their answer happens to match.
    """
    reference = payload.get("candidate") or payload.get("submission")
    current_file = submission / "submission.json"
    if not isinstance(reference, str) or not reference.strip():
        return {"status": "unknown", "reference": reference}
    referenced = Path(reference)
    if not referenced.is_absolute():
        referenced = ROOT / referenced
    referenced_file = referenced / "submission.json" if referenced.is_dir() else referenced
    if not current_file.is_file() or not referenced_file.is_file():
        return {"status": "unverifiable", "reference": str(reference)}
    current_sha = sha256_file(current_file)
    referenced_sha = sha256_file(referenced_file)
    if current_sha == referenced_sha:
        return {
            "status": "current",
            "reference": str(reference),
            "current_submission_sha256": current_sha,
            "artifact_submission_sha256": referenced_sha,
            "changed_ids": [],
        }
    try:
        current_rows = keyed(load_json(current_file, []))
        referenced_rows = keyed(load_json(referenced_file, []))
    except (OSError, ValueError, TypeError):
        current_rows, referenced_rows = {}, {}
    same_universe = bool(current_rows) and set(current_rows) == set(referenced_rows)
    changed_ids = sorted(
        qid for qid in current_rows if current_rows[qid] != referenced_rows.get(qid)
    ) if same_universe else []
    return {
        "status": "row_local" if same_universe else "stale",
        "reference": str(reference),
        "current_submission_sha256": current_sha,
        "artifact_submission_sha256": referenced_sha,
        "changed_ids": changed_ids,
        "changed_count": len(changed_ids),
    }


def keyed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in rows if "id" in row}


def id_set(payload: dict[str, Any], key: str) -> set[int]:
    return {int(value) for value in payload.get(key, [])}


def record_ids(payload: dict[str, Any], key: str) -> set[int]:
    return {
        int(row["id"])
        for row in payload.get(key, [])
        if isinstance(row, dict) and "id" in row
    }


def add_signal(
    signals: list[dict[str, Any]], family: str, points: int, detail: str
) -> None:
    signals.append({"family": family, "points": points, "detail": detail})


def axis(status: str, evidence: str) -> dict[str, str]:
    return {"status": status, "evidence": evidence}


def build_record(
    row: dict[str, Any],
    submission: Path,
    fast: dict[int, dict[str, Any]],
    deep: dict[int, dict[str, Any]],
    reviewed: dict[str, Any],
    missing: dict[str, Any],
    cross_report: dict[str, Any],
    rounding: dict[str, Any],
    unit: dict[str, Any],
    panel: dict[str, Any],
    independent: dict[str, Any],
    artifact_statuses: dict[str, dict[str, Any]] | None = None,
    filter_cardinality: dict[str, Any] | None = None,
    shape_null: dict[str, Any] | None = None,
    denominators: dict[str, Any] | None = None,
    extrema_ties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qid = int(row["id"])
    code = str(row.get("pandas_query", ""))
    question = str(row.get("question", ""))
    docs = list(row.get("relevant_docs", []))
    tables = list(row.get("relevant_tables", []))
    tickers = {str(doc).split("_financial_statements_", 1)[0] for doc in docs}
    years = set(re.findall(r"\b20\d{2}\b", question))
    compact = submission / "data" / f"q{qid}_source_cells.csv"
    factory = deep.get(qid) or fast.get(qid) or {}
    deep_record = deep.get(qid)
    verdict = reviewed.get(str(qid))

    contradiction: list[dict[str, Any]] = []
    exposure: list[dict[str, Any]] = []
    assurance: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, str]] = {}
    artifact_statuses = artifact_statuses or {}

    def current_artifact(name: str) -> bool:
        # Unit tests may pass synthetic payloads without provenance. Production
        # always supplies the strict statuses from main().
        status = artifact_statuses.get(name, {"status": "current"})
        if status.get("status") == "current":
            return True
        return status.get("status") == "row_local" and qid not in set(status.get("changed_ids", []))

    runtimes = (deep_record or {}).get("runtime", {})
    runtime_states = {item.get("status") for item in runtimes.values()}
    if runtimes and runtime_states == {"pass"}:
        matrix["dual_runtime"] = axis("pass", "string and official-typed replay")
        add_signal(assurance, "dual_runtime", 8, "both execution representations agree")
    elif runtimes:
        matrix["dual_runtime"] = axis("fail", f"states={sorted(runtime_states)}")
        add_signal(contradiction, "runtime", 50, "runtime replay failed or disagreed")
    else:
        matrix["dual_runtime"] = axis("untested", "deep factory record absent")

    physical_audit = (deep_record or {}).get("physical_audit", {})
    physical = physical_audit.get("status")
    if physical == "pass":
        matrix["physical_lineage"] = axis("pass", "cell manifest matches physical CSV")
        add_signal(assurance, "physical_lineage", 16, "physical cells verified")
    elif physical in {"failure", "error", "positional_failure", "positional_risk"}:
        failures = physical_audit.get("field_failures", [])
        hard = [
            item
            for item in failures
            if item.get("source_exists") is False
            or item.get("raw_match") is False
        ]
        scope_failures = [item for item in failures if item.get("scope_ok") is False]
        year_failures = [item for item in failures if item.get("year_header_ok") is False]
        semantic_failures = [item for item in failures if item.get("semantic_ok") is False]
        if hard or physical in {"error", "positional_failure"}:
            matrix["physical_lineage"] = axis("fail", f"hard={len(hard)} status={physical}")
            add_signal(contradiction, "physical_lineage", 50, "source/raw/scope identity failed")
        else:
            matrix["physical_lineage"] = axis(
                "challenge",
                f"scope={len(scope_failures)} year={len(year_failures)} semantic={len(semantic_failures)} status={physical}",
            )
            if scope_failures:
                add_signal(exposure, "scope_parser_challenge", 25, f"{len(scope_failures)} scope mismatches")
            if year_failures:
                add_signal(exposure, "period_header_challenge", 25, f"{len(year_failures)} header mismatches")
            if semantic_failures:
                add_signal(exposure, "semantic_label_challenge", 18, f"{len(semantic_failures)} label mismatches")
            if not scope_failures and not year_failures and not semantic_failures:
                add_signal(exposure, "positional_lineage_challenge", 15, str(physical))
    elif compact.is_file():
        matrix["physical_lineage"] = axis("untested", "compact manifest exists")
    else:
        matrix["physical_lineage"] = axis("blocked", "compact source manifest absent")
        add_signal(exposure, "missing_cell_lineage", 15, "no q<ID> source-cell manifest")

    intent_flags = factory.get("intent_source_flags", [])
    arithmetic_flags = factory.get("arithmetic_invariant_flags", [])
    if intent_flags:
        matrix["semantic_intent"] = axis("challenge", "factory intent/source mismatch")
        add_signal(exposure, "semantic_intent_challenge", 35, "rule-based mismatch needs source proof")
    elif deep_record:
        matrix["semantic_intent"] = axis("pass_rule", "deterministic intent rules clear")
        add_signal(assurance, "semantic_rule", 4, "known intent traps absent")
    else:
        matrix["semantic_intent"] = axis("untested", "deep semantic screen absent")
    if arithmetic_flags:
        add_signal(contradiction, "arithmetic_invariant", 45, "financial invariant violated")

    uses_iloc = "iloc[" in code.replace(" ", "")
    if uses_iloc:
        matrix["schema_mutation"] = axis("untested", "positional program needs permutation test")
        add_signal(exposure, "positional_schema", 8, "iloc depends on table layout")
    else:
        matrix["schema_mutation"] = axis("not_applicable", "no iloc access")

    if "None" in code:
        if not current_artifact("missing"):
            matrix["missing_operand"] = axis("untested_stale", "counterfactual artifact is not candidate-identical")
            add_signal(exposure, "missing_operand", 10, "literal None lacks a current stress test")
        elif qid in id_set(missing, "sensitive_ids"):
            matrix["missing_operand"] = axis("sensitive", "answer changed under operand stress")
            add_signal(exposure, "missing_operand_sensitivity", 30, "counterfactual changes answer")
        elif qid in id_set(missing, "stable_ids"):
            matrix["missing_operand"] = axis("pass", "stable under row-aware stress")
            add_signal(assurance, "missing_operand", 8, "missing operand was stress-tested")
        else:
            matrix["missing_operand"] = axis("untested", "literal None without stress record")
            add_signal(exposure, "missing_operand", 10, "literal None is untested")
    else:
        matrix["missing_operand"] = axis("not_applicable", "no literal None")

    round_applicable = code.count("round(") > 1
    rounding_failed = id_set(rounding, "question_ids")
    if round_applicable and not current_artifact("rounding"):
        matrix["rounding_order"] = axis("untested_stale", "rounding artifact is not candidate-identical")
        add_signal(exposure, "rounding_gap", 4, "intermediate rounding lacks a current test")
    elif round_applicable and qid in rounding_failed:
        matrix["rounding_order"] = axis("fail", "result changed after removing intermediate round")
        add_signal(contradiction, "rounding_order", 15, "intermediate rounding is material")
    elif round_applicable and rounding:
        matrix["rounding_order"] = axis("pass", "global rounding sensitivity audit")
        add_signal(assurance, "rounding_order", 4, "rounding order was tested")
    else:
        matrix["rounding_order"] = axis("not_applicable", "no intermediate rounding pattern")

    cross_rows = keyed(cross_report.get("records", [])) if current_artifact("cross_report") else {}
    if qid in cross_rows:
        changed = bool(cross_rows[qid].get("answer_changes"))
        matrix["cross_report"] = axis("fail" if changed else "pass", "later-report comparative replay")
        if changed:
            add_signal(exposure, "cross_report_sensitivity", 20, "later comparative changed answer")
        else:
            add_signal(assurance, "cross_report", 6, "stable to later comparative")
    elif len(docs) > 1 or len(years) > 1:
        status = "untested_stale" if not current_artifact("cross_report") else "untested"
        matrix["cross_report"] = axis(status, "multi-period/document question")
        add_signal(exposure, "cross_report_gap", 5, "comparative restatement not tested")
    else:
        matrix["cross_report"] = axis("not_applicable", "single-period/document question")

    unit_failed = id_set(unit, "question_ids") if current_artifact("unit") else set()
    if qid in unit_failed:
        matrix["unit_dimension"] = axis("fail", "unit-dimension contract finding")
        add_signal(contradiction, "unit_dimension", 30, "operand/output dimensions disagree")
    else:
        matrix["unit_dimension"] = axis("untested", "no per-question pass attestation")

    independent_rows = (
        keyed(independent.get("all_disagreements", []))
        if current_artifact("independent")
        else {}
    )
    if qid in independent_rows:
        dependency_groups = independent_rows[qid].get("dependency_groups", [])
        matrix["independent_solver"] = axis(
            "disagree", f"dependency_groups={len(set(dependency_groups))}"
        )
        points = 20 if len(set(dependency_groups)) >= 2 else 10
        add_signal(contradiction, "independent_solver", points, "independent answer disagreement")
    else:
        matrix["independent_solver"] = axis("untested", "absence of disagreement is not agreement")

    panel_rows = keyed(panel.get("disagreements", [])) if current_artifact("panel") else {}
    if qid in panel_rows:
        status = str(panel_rows[qid].get("status", "needs_source_adjudication"))
        if status == "needs_source_adjudication":
            add_signal(contradiction, "panel_disagreement", 20, status)
        elif status == "keep_current":
            add_signal(assurance, "panel_adjudication", 8, "alternative was source-adjudicated")

    filter_current = bool(filter_cardinality) and current_artifact("filter_cardinality")
    filter_rows = (
        keyed((filter_cardinality or {}).get("findings", []))
        if filter_current
        else {}
    )
    if qid in filter_rows:
        finding = filter_rows[qid]
        matrix["selector_cardinality"] = axis(
            "challenge", f"match_count={finding.get('match_count')}"
        )
        add_signal(exposure, "selector_cardinality", 18, "selector matches multiple rows")
    elif filter_current:
        matrix["selector_cardinality"] = axis("pass_rule", "filter-cardinality audit clear")
        add_signal(assurance, "selector_cardinality", 2, "no ambiguous executed filter found")
    else:
        matrix["selector_cardinality"] = axis("untested", "no unique-match attestation")

    shape_current = bool(shape_null) and current_artifact("shape_null")
    shape_rows = (
        keyed((shape_null or {}).get("findings", []))
        if shape_current
        else {}
    )
    if qid in shape_rows:
        finding = shape_rows[qid]
        score = int(finding.get("score", 0) or 0)
        matrix["shape_null_additivity"] = axis("challenge", f"static_score={score}")
        add_signal(
            exposure,
            "shape_null_additivity",
            24 if score >= 5 else min(12, 3 + score),
            "; ".join(finding.get("reasons", [])[:2]),
        )
    elif shape_current:
        matrix["shape_null_additivity"] = axis("pass_rule", "global static audit clear")
        add_signal(assurance, "shape_rule", 3, "shape/null static rules clear")
    else:
        matrix["shape_null_additivity"] = axis("untested", "candidate-identical audit absent")

    denominators_current = bool(denominators) and current_artifact("denominators")
    denominator_ids = (
        id_set(denominators or {}, "question_ids")
        if denominators_current
        else set()
    )
    if qid in denominator_ids:
        matrix["runtime_denominator"] = axis("challenge", "zero/NaN/non-finite value reached division")
        add_signal(exposure, "runtime_denominator", 20, "risky denominator executed")
    elif denominators_current:
        matrix["runtime_denominator"] = axis("pass", "runtime denominator audit clear")
        add_signal(assurance, "runtime_denominator", 4, "executed divisions are finite")
    else:
        matrix["runtime_denominator"] = axis("untested", "candidate-identical audit absent")

    extrema_current = bool(extrema_ties) and current_artifact("extrema_ties")
    extrema_ids = (
        id_set(extrema_ties or {}, "question_ids")
        if extrema_current
        else set()
    )
    if qid in extrema_ids:
        matrix["extrema_tie"] = axis("challenge", "executed max/min has tied candidates")
        add_signal(exposure, "extrema_tie", 16, "tie policy may select first row")
    elif extrema_current:
        matrix["extrema_tie"] = axis("pass", "runtime extrema audit clear")
        add_signal(assurance, "extrema_tie", 3, "no executed tie found")
    else:
        matrix["extrema_tie"] = axis("untested", "candidate-identical audit absent")

    durable_terminal = False
    if verdict is not None:
        proposed = verdict.get("answer")
        current_answer = row.get("answer")
        mutation = str(verdict.get("mutation", ""))
        known_source_gold_conflict = mutation.startswith("not_applied_public_gold_conflict")
        changed = False
        try:
            changed = proposed is not None and abs(float(proposed) - float(current_answer)) > 0.005
        except (TypeError, ValueError):
            changed = proposed is not None and proposed != current_answer
        if changed and not known_source_gold_conflict:
            matrix["durable_source_review"] = axis("fail", "source-proven answer change not present in candidate")
            add_signal(contradiction, "durable_answer_change", 60, "ledger answer differs from candidate")
        else:
            evidence = str(verdict.get("status", "reviewed"))
            if known_source_gold_conflict:
                evidence += "; documented source/public-gold conflict"
            matrix["durable_source_review"] = axis("pass", evidence)
            high = str(verdict.get("confidence", "")).lower() == "high"
            add_signal(assurance, "durable_review", 25 if high else 20, "ledger source review")
            durable_terminal = mutation == "none" or mutation.startswith("applied_") or mutation.startswith("not_applied_public_gold_conflict")
    else:
        matrix["durable_source_review"] = axis("untested", "no durable verdict")
        add_signal(exposure, "unreviewed", 12, "not yet source-adjudicated")

    if len(tickers) > 1:
        add_signal(exposure, "multi_entity", 12, f"{len(tickers)} entities")
    if len(years) > 1:
        add_signal(exposure, "multi_period", 6, f"{len(years)} explicit years")
    if len(tables) > 4:
        add_signal(exposure, "wide_retrieval", min(10, 4 + len(tables) // 4), f"{len(tables)} tables")
    if re.search(r"\b(groupby|median|idxmax|idxmin|nlargest|nsmallest)\b", code):
        add_signal(exposure, "selector_program", 10, "selection/aggregation can be plausibly wrong")
    if re.search(r"\b(merge|concat)\s*\(", code):
        add_signal(exposure, "join_program", 6, "join/concat can mix scopes")

    contradiction_score = sum(item["points"] for item in contradiction)
    exposure_score = sum(item["points"] for item in exposure)
    assurance_score = sum(item["points"] for item in assurance)
    residual = max(0, min(100, contradiction_score + exposure_score - min(35, assurance_score)))
    tested = sum(
        item["status"] in {"pass", "pass_rule", "fail", "disagree", "challenge", "sensitive"}
        for item in matrix.values()
    )
    applicable = sum(item["status"] != "not_applicable" for item in matrix.values())
    coverage = round(tested / applicable, 3) if applicable else 1.0
    hard_families = {
        "runtime",
        "physical_lineage",
        "arithmetic_invariant",
        "rounding_order",
        "unit_dimension",
        "durable_answer_change",
    }
    has_hard_signal = any(item["family"] in hard_families for item in contradiction)
    if durable_terminal:
        state = "source_adjudicated"
    elif has_hard_signal:
        state = "hard_signal_needs_source_proof"
    elif contradiction_score:
        state = "challenged_needs_adjudication"
    elif coverage < 0.6:
        state = "under_tested"
    else:
        state = "unfalsified_not_proven"

    return {
        "id": qid,
        "question": question,
        "answer": row.get("answer"),
        "state": state,
        "residual_risk": residual,
        "contradiction_score": contradiction_score,
        "exposure_score": exposure_score,
        "assurance_score": assurance_score,
        "challenge_coverage": coverage,
        "durably_reviewed": verdict is not None,
        "contradictions": contradiction,
        "exposures": exposure,
        "assurances": assurance,
        "challenge_matrix": matrix,
        "artifact_statuses": artifact_statuses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--fast-records", type=Path, default=ROOT / "build/v238_audit_factory_pipeline/fast/records.jsonl")
    parser.add_argument("--deep-records", type=Path, default=ROOT / "build/v238_audit_factory_pipeline/deep/records.jsonl")
    parser.add_argument("--ledger", type=Path, default=ROOT / "knowledge/vothuong/question_source_verdicts.json")
    parser.add_argument("--missing", type=Path, default=ROOT / "build/v226_missing_operand_counterfactuals_v225.json")
    parser.add_argument("--cross-report", type=Path, default=ROOT / "build/v226_cross_report_counterfactuals_v225.json")
    parser.add_argument("--rounding", type=Path, default=ROOT / "build/v226_rounding_order_sensitivity_v225.json")
    parser.add_argument("--unit", type=Path, default=ROOT / "build/v226_unit_dimension_contract_v225.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "build/v227_panel_disagreement_audit.json")
    parser.add_argument("--independent", type=Path, default=ROOT / "build/v227_counterfactual_batches_v217_all.json")
    parser.add_argument("--filter-cardinality", type=Path, default=ROOT / "build/v248_filter_cardinality_v239.json")
    parser.add_argument("--shape-null", type=Path, default=ROOT / "build/v248_shape_null_additivity_v239.json")
    parser.add_argument("--denominators", type=Path, default=ROOT / "build/v248_runtime_denominators_v239.json")
    parser.add_argument("--extrema-ties", type=Path, default=ROOT / "build/v248_extrema_ties_v239.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_json(args.submission / "submission.json", [])
    fast = keyed(load_jsonl(args.fast_records))
    deep = keyed(load_jsonl(args.deep_records))
    ledger = load_json(args.ledger, {}).get("verdicts", {})
    artifacts = {
        "missing": load_json(args.missing, {}),
        "cross_report": load_json(args.cross_report, {}),
        "rounding": load_json(args.rounding, {}),
        "unit": load_json(args.unit, {}),
        "panel": load_json(args.panel, {}),
        "independent": load_json(args.independent, {}),
        "filter_cardinality": load_json(args.filter_cardinality, {}),
        "shape_null": load_json(args.shape_null, {}),
        "denominators": load_json(args.denominators, {}),
        "extrema_ties": load_json(args.extrema_ties, {}),
    }
    artifact_statuses = {
        name: artifact_compatibility(payload, args.submission)
        for name, payload in artifacts.items()
    }
    records = [
        build_record(
            row,
            args.submission,
            fast,
            deep,
            ledger,
            artifacts["missing"],
            artifacts["cross_report"],
            artifacts["rounding"],
            artifacts["unit"],
            artifacts["panel"],
            artifacts["independent"],
            artifact_statuses=artifact_statuses,
            filter_cardinality=artifacts["filter_cardinality"],
            shape_null=artifacts["shape_null"],
            denominators=artifacts["denominators"],
            extrema_ties=artifacts["extrema_ties"],
        )
        for row in rows
    ]
    records.sort(
        key=lambda item: (
            item["state"] != "hard_signal_needs_source_proof",
            item["durably_reviewed"],
            -item["residual_risk"],
            item["challenge_coverage"],
            item["id"],
        )
    )
    selected = records[: max(0, args.top)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    (args.out / "SILENT_RISK_QUEUE.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = Counter(item["state"] for item in records)
    summary = {
        "schema_version": "1.0",
        "submission": str(args.submission),
        "question_count": len(records),
        "top_count": len(selected),
        "state_counts": dict(counts),
        "reviewed_count": len(ledger),
        "artifact_statuses": artifact_statuses,
        "policy": "Risk ranking only; answer mutation requires independent physical-source proof.",
        "top_ids": [item["id"] for item in selected[:50]],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Silent-error residual-risk queue",
        "",
        f"Questions: {len(records)} · reviewed: {len(ledger)} · top queue: {len(selected)}",
        "",
        "`unfalsified_not_proven` means no current contradiction was found; it is not a correctness label.",
        "",
        "| ID | Risk | Coverage | State | Main signals |",
        "|---:|---:|---:|---|---|",
    ]
    for item in selected[:50]:
        signals = (item["contradictions"] + item["exposures"])[:3]
        labels = ", ".join(signal["family"] for signal in signals) or "assurance gap"
        lines.append(
            f"| q{item['id']} | {item['residual_risk']} | {item['challenge_coverage']:.1%} | "
            f"{item['state']} | {labels} |"
        )
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
