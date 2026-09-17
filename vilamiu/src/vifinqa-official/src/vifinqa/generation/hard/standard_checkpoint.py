"""Consolidate reviewed Hard-standard ledgers into a replayed checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from vifinqa.generation.hard.standard_pool import (
    _basic_hard_fingerprint_proof,
    _is_published_checkpoint,
    _published_checkpoint_proof,
    _records,
    _write_jsonl,
    audit_standard_pool,
)
from vifinqa.generation.hard.template_intents import INTENTS_BY_ID


def _source_manifest(path: Path, *, row_count: int) -> list[dict[str, object]]:
    if _is_published_checkpoint(path):
        proof_error = _published_checkpoint_proof(path, expected_count=row_count)
        if proof_error:
            raise ValueError(f"{path}: {proof_error}")
        payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return [dict(source) for source in payload["sources"]]

    proof_error = _basic_hard_fingerprint_proof(path, expected_count=row_count)
    if proof_error:
        raise ValueError(f"{path}: {proof_error}")
    fingerprints = json.loads(
        path.with_suffix(".fingerprints.json").read_text(encoding="utf-8")
    )
    manual_review = Path(str(fingerprints.get("manual_review", "")))
    if not manual_review.is_file():
        raise ValueError(f"{path}: manual review evidence is missing")
    return [
        {
            "path": str(path),
            "row_count": row_count,
            "replay_dependency_pass": True,
            "manual_review": str(manual_review),
        }
    ]


def consolidate_standard_checkpoint(
    source_paths: tuple[Path, ...],
    *,
    output_path: Path,
    report_prefix: Path,
) -> dict[str, object]:
    if not source_paths:
        raise ValueError("at least one reviewed source is required")

    report = audit_standard_pool(source_paths)
    if report.summary["auto_reject_count"]:
        first = next(d for d in report.decisions if d.auto_status == "reject")
        raise ValueError(f"{first.source}: {';'.join(first.reasons)}")

    combined: list[dict[str, object]] = []
    source_manifest: list[dict[str, object]] = []
    template_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for path in source_paths:
        records = _records(path)
        source_manifest.extend(_source_manifest(path, row_count=len(records)))
        for record in records:
            promoted = dict(record)
            promoted["id"] = len(combined) + 1
            combined.append(promoted)
            template_id = str(record["template_id"])
            template_counts[template_id] += 1
            operation_counts[INTENTS_BY_ID[template_id].operation_grammar] += 1

    decisions = report.decisions
    summary = {
        "row_count": len(combined),
        "source_count": len(source_manifest),
        "distinct_question_fingerprint_count": len(
            {decision.question_fingerprint for decision in decisions}
        ),
        "distinct_query_fingerprint_count": len(
            {decision.query_fingerprint for decision in decisions}
        ),
        "distinct_semantic_query_fingerprint_count": len(
            {decision.semantic_query_fingerprint for decision in decisions}
        ),
        "replay_pass_count": sum(decision.actual is not None for decision in decisions),
        "dependency_pass_count": sum(
            decision.accessed_refs == decision.declared_refs for decision in decisions
        ),
        "hardness": "hard_standard",
        "all_pass": True,
    }
    if any(
        value != len(combined)
        for key, value in summary.items()
        if key.endswith("_count") and key != "source_count"
    ):
        raise ValueError("checkpoint proof count mismatch")

    _write_jsonl(output_path, tuple(combined))
    payload: dict[str, object] = {
        "summary": summary,
        "sources": source_manifest,
        "template_counts": dict(sorted(template_counts.items())),
        "operation_grammar_counts": dict(sorted(operation_counts.items())),
    }
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    report_prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_prefix.with_suffix(".md").write_text(
        "\n".join(
            (
                f"# Hard-standard checkpoint {len(combined):03d}",
                "",
                f"- Accepted records: **{len(combined)}/{len(combined)}**",
                f"- Evidence sources: **{len(source_manifest)}**",
                f"- Distinct question fingerprints: **{summary['distinct_question_fingerprint_count']}**",
                f"- Distinct query fingerprints: **{summary['distinct_query_fingerprint_count']}**",
                f"- Distinct semantic-query fingerprints: **{summary['distinct_semantic_query_fingerprint_count']}**",
                "- Replay, exact dependency, basic-Hard proof, manual rubric and dedup gates: **PASS**",
                "",
                "## Sources",
                "",
                *(f"- `{source['path']}`" for source in source_manifest),
                "",
            )
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-prefix", required=True, type=Path)
    args = parser.parse_args()
    payload = consolidate_standard_checkpoint(
        tuple(args.sources), output_path=args.out, report_prefix=args.report_prefix
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
