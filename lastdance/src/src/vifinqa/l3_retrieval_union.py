"""Build a conservative L3 retrieval-only union from masked LLM reranking."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import token_set_ratio

from .experiments import SubmissionArchive
from .financial_ir import FinancialPlan
from .grounded_plans import GroundedBinder, ROW_REF_RE
from .l1_fact_layer import normalize


def semantic_similarity(metric: str, baseline_label: str, candidate_label: str) -> float:
    """Require the candidate row to agree with both semantic descriptions."""

    return min(
        token_set_ratio(normalize(metric), normalize(candidate_label)),
        token_set_ratio(normalize(baseline_label), normalize(candidate_label)),
    )


def _scope_matches(requested: str, actual: str) -> bool:
    return requested == "any" or requested == actual or actual in {"unspecified", "aggregated"}


def build_l3_retrieval_union(
    *,
    baseline_zip: Path,
    teachers_path: Path,
    legacy_plans_path: Path,
    reranks_path: Path,
    database: Path,
    output_zip: Path,
    audit_path: Path,
    minimum_label_similarity: float = 90.0,
) -> dict[str, Any]:
    """Union exact-value alternate tables while preserving execution fields."""

    if output_zip.exists():
        raise FileExistsError(output_zip)
    baseline = SubmissionArchive.load(baseline_zip)
    teachers = {
        plan.question_id: plan
        for plan in (
            FinancialPlan.from_dict(json.loads(line))
            for line in teachers_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    legacy = {
        int(plan["question"]["id"]): plan
        for plan in (
            json.loads(line)
            for line in legacy_plans_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    reranks = [
        json.loads(line)
        for line in reranks_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    additions: dict[int, list[dict[str, str]]] = {}
    audit: list[dict[str, Any]] = []
    with GroundedBinder(database) as binder:
        for rerank in reranks:
            question_id = int(rerank["question_id"])
            if rerank.get("status") != "VALID":
                audit.append(
                    {
                        "question_id": question_id,
                        "status": "REJECTED_RERANK",
                        "reason": rerank.get("validation_error", "invalid rerank"),
                    }
                )
                continue
            if question_id not in teachers or question_id not in legacy:
                raise ValueError(f"Missing L3 plan for q{question_id}")
            plan = teachers[question_id]
            baseline_facts = [
                fact
                for item in legacy[question_id]["items"]
                for fact in item["facts"].values()
            ]
            if len(baseline_facts) != len(plan.facts):
                raise ValueError(f"q{question_id}: teacher/legacy fact count mismatch")
            original = baseline.by_id[question_id]
            for selection_rank, selection in enumerate(
                rerank["selection"]["selected"], 1
            ):
                row_ref = selection["row_ref"]
                match = ROW_REF_RE.fullmatch(row_ref)
                if match is None:  # The rerank validator should already reject this.
                    raise ValueError(f"q{question_id}: invalid validated row_ref {row_ref}")
                table = binder._table(int(match.group("table_id")))
                matching_indices = [
                    index
                    for index, fact in enumerate(plan.facts)
                    if fact.ticker == table["ticker"]
                    and fact.year == int(table["year"])
                    and _scope_matches(fact.scope, str(table["report_scope"]))
                ]
                record: dict[str, Any] = {
                    "question_id": question_id,
                    "selection_rank": selection_rank,
                    "row_ref": row_ref,
                    "role": selection.get("role"),
                }
                if len(matching_indices) == 0:
                    audit.append(
                        {
                            **record,
                            "status": "REJECTED_DIMENSION",
                            "reason": "matched 0 facts",
                        }
                    )
                    continue
                if len(matching_indices) == 1:
                    fact_index = matching_indices[0]
                    fact = replace(plan.facts[fact_index], row_ref=row_ref)
                    baseline_fact = baseline_facts[fact_index]
                    try:
                        binding = binder.bind_fact(plan, fact)
                    except (IndexError, KeyError, TypeError, ValueError) as error:
                        audit.append(
                            {**record, "status": "REJECTED_BIND", "reason": str(error)}
                        )
                        continue
                else:
                    # Multiple facts share the same dimension; disambiguate by
                    # trying each and checking which matches baseline value.
                    resolved = None
                    for candidate_index in matching_indices:
                        candidate_fact = replace(
                            plan.facts[candidate_index], row_ref=row_ref
                        )
                        try:
                            candidate_binding = binder.bind_fact(plan, candidate_fact)
                        except (IndexError, KeyError, TypeError, ValueError):
                            continue
                        baseline_val = float(baseline_facts[candidate_index]["answer"])
                        if math.isclose(
                            candidate_binding.value, baseline_val,
                            rel_tol=1e-9, abs_tol=1e-7,
                        ):
                            resolved = (candidate_index, candidate_fact, candidate_binding)
                            break
                    if resolved is None:
                        # Try label similarity as fallback
                        best_score, best_candidate = 0.0, None
                        for candidate_index in matching_indices:
                            candidate_fact = replace(
                                plan.facts[candidate_index], row_ref=row_ref
                            )
                            try:
                                candidate_binding = binder.bind_fact(plan, candidate_fact)
                            except (IndexError, KeyError, TypeError, ValueError):
                                continue
                            sim = semantic_similarity(
                                candidate_fact.metric,
                                str(baseline_facts[candidate_index].get("row_text", "")),
                                candidate_binding.row_text,
                            )
                            if sim > best_score:
                                best_score = sim
                                best_candidate = (candidate_index, candidate_fact, candidate_binding)
                        if best_candidate is None or best_score < minimum_label_similarity:
                            audit.append(
                                {
                                    **record,
                                    "status": "REJECTED_DIMENSION",
                                    "reason": f"ambiguous among {len(matching_indices)} facts, best_sim={best_score:.1f}",
                                }
                            )
                            continue
                        resolved = best_candidate
                    fact_index, fact, binding = resolved
                    baseline_fact = baseline_facts[fact_index]
                table_key = f"{binding.document_id}|{binding.source_line_1}"
                similarity = semantic_similarity(
                    fact.metric, str(baseline_fact["row_text"]), binding.row_text
                )
                record.update(
                    {
                        "fact_id": fact.id,
                        "ticker": fact.ticker,
                        "year": fact.year,
                        "baseline_value": float(baseline_fact["answer"]),
                        "candidate_value": binding.value,
                        "baseline_label": baseline_fact["row_text"],
                        "candidate_label": binding.row_text,
                        "label_similarity": round(similarity, 4),
                        "document_id": binding.document_id,
                        "table_key": table_key,
                    }
                )
                if not math.isclose(
                    binding.value,
                    float(baseline_fact["answer"]),
                    rel_tol=1e-9,
                    abs_tol=1e-7,
                ):
                    audit.append({**record, "status": "REJECTED_VALUE"})
                    continue
                if (
                    binding.table_id == int(baseline_fact["table_id"])
                    and binding.row_index == int(baseline_fact["row_index"])
                ):
                    audit.append({**record, "status": "BASELINE_ROW"})
                    continue
                if table_key in original["relevant_tables"]:
                    audit.append({**record, "status": "ALREADY_RETRIEVED"})
                    continue
                if abs(float(baseline_fact["answer"])) < 1e-12:
                    audit.append({**record, "status": "REJECTED_ZERO"})
                    continue
                if similarity < minimum_label_similarity:
                    audit.append({**record, "status": "REJECTED_LABEL"})
                    continue
                additions.setdefault(question_id, []).append(
                    {"document_id": binding.document_id, "table_key": table_key}
                )
                audit.append({**record, "status": "ACCEPTED"})

    replacements: dict[int, dict[str, Any]] = {}
    for question_id, candidates in additions.items():
        original = baseline.by_id[question_id]
        documents = list(original["relevant_docs"])
        tables = list(original["relevant_tables"])
        for candidate in candidates:
            if candidate["document_id"] not in documents:
                documents.append(candidate["document_id"])
            if candidate["table_key"] not in tables:
                tables.append(candidate["table_key"])
        if tables != original["relevant_tables"]:
            replacements[question_id] = {
                **original,
                "relevant_docs": documents,
                "relevant_tables": tables,
            }
    if not replacements:
        raise ValueError("No conservative L3 retrieval additions passed")

    items = [replacements.get(int(item["id"]), item) for item in baseline.items]
    required_paths = {
        evidence["csv_path"] for item in items for evidence in item["evidence"]
    }
    evidence_bytes = {path: baseline.members[path] for path in required_paths}
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            (json.dumps(items, ensure_ascii=False, indent=2) + "\n").encode(),
        )
        for path in sorted(evidence_bytes):
            archive.writestr(path, evidence_bytes[path])
    SubmissionArchive.load(output_zip)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit),
        encoding="utf-8",
    )
    return {
        "candidate_zip": str(output_zip),
        "questions_changed": len(replacements),
        "changed_ids": sorted(replacements),
        "tables_added": sum(
            len(replacements[question_id]["relevant_tables"])
            - len(baseline.by_id[question_id]["relevant_tables"])
            for question_id in replacements
        ),
        "accepted_rows": sum(row["status"] == "ACCEPTED" for row in audit),
        "audit": str(audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--legacy-plans", type=Path, required=True)
    parser.add_argument("--reranks", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--minimum-label-similarity", type=float, default=90.0)
    args = parser.parse_args()
    result = build_l3_retrieval_union(
        baseline_zip=args.baseline_zip,
        teachers_path=args.teachers,
        legacy_plans_path=args.legacy_plans,
        reranks_path=args.reranks,
        database=args.database,
        output_zip=args.output_zip,
        audit_path=args.audit,
        minimum_label_similarity=args.minimum_label_similarity,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
