"""Consolidate reviewed Deep Hard shards into an immutable checkpoint ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vifinqa.generation.hard.deep_shard import (
    MIN_ADAPTIVE_EDGES,
    MIN_REASONING_DEPTH,
    _normalized_question,
    _query_fingerprint,
    _sha256,
)
from vifinqa.generation.hard.template_batch_verification import verify_template_batch


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def consolidate_deep_checkpoint(
    shard_paths: tuple[Path, ...],
    *,
    output_path: Path,
    report_prefix: Path,
) -> dict[str, object]:
    if not shard_paths:
        raise ValueError("at least one shard is required")

    combined: list[dict[str, object]] = []
    question_fingerprints: set[str] = set()
    query_fingerprints: set[str] = set()
    template_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    source_summaries: list[dict[str, object]] = []

    for shard_path in shard_paths:
        verification = verify_template_batch(shard_path)
        if not verification.ok:
            raise ValueError(f"{shard_path}: replay/dependency verification failed")

        fingerprint_path = shard_path.with_suffix(".fingerprints.json")
        fingerprint_payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        proofs = fingerprint_payload.get("rows")
        if not isinstance(proofs, list):
            raise ValueError(f"{fingerprint_path}: missing row proofs")
        proofs_by_id = {proof.get("id"): proof for proof in proofs if isinstance(proof, dict)}

        manual_path = shard_path.with_suffix(".manual_review.md")
        manual_text = manual_path.read_text(encoding="utf-8")
        if "Accepted: **10/10**" not in manual_text:
            raise ValueError(f"{manual_path}: missing accepted manual-review marker")

        shard_records = _records(shard_path)
        if len(shard_records) != len(proofs_by_id):
            raise ValueError(f"{shard_path}: record/proof row-count mismatch")

        for record in shard_records:
            source_id = record.get("id")
            proof = proofs_by_id.get(source_id)
            if proof is None:
                raise ValueError(f"{fingerprint_path}: missing proof for id={source_id}")
            if proof.get("template_id") != record.get("template_id"):
                raise ValueError(f"{shard_path}: template/proof mismatch for id={source_id}")
            reasoning_depth = proof.get(
                "reasoning_depth_lower_bound", proof.get("reasoning_depth", 0)
            )
            adaptive_edges = proof.get(
                "adaptive_edges_lower_bound", proof.get("adaptive_edges", 0)
            )
            if int(reasoning_depth) < MIN_REASONING_DEPTH:
                raise ValueError(f"{shard_path}: insufficient reasoning depth for id={source_id}")
            if int(adaptive_edges) < MIN_ADAPTIVE_EDGES:
                raise ValueError(f"{shard_path}: insufficient adaptive edges for id={source_id}")

            question_fp = _sha256(_normalized_question(str(record.get("question", ""))))
            query_fp = _query_fingerprint(str(record.get("pandas_query", "")))
            if proof.get("question_fingerprint") != question_fp:
                raise ValueError(f"{shard_path}: stale question proof for id={source_id}")
            if proof.get("query_fingerprint") != query_fp:
                raise ValueError(f"{shard_path}: stale query proof for id={source_id}")
            if question_fp in question_fingerprints:
                raise ValueError(f"duplicate question fingerprint at {shard_path}#{source_id}")
            if query_fp in query_fingerprints:
                raise ValueError(f"duplicate query fingerprint at {shard_path}#{source_id}")
            question_fingerprints.add(question_fp)
            query_fingerprints.add(query_fp)

            template_id = str(record.get("template_id", ""))
            operation_grammar = str(proof.get("operation_grammar", ""))
            template_counts[template_id] = template_counts.get(template_id, 0) + 1
            operation_counts[operation_grammar] = operation_counts.get(operation_grammar, 0) + 1
            promoted = dict(record)
            promoted["id"] = len(combined) + 1
            combined.append(promoted)

        source_summaries.append(
            {
                "path": str(shard_path),
                "row_count": len(shard_records),
                "replay_dependency_pass": True,
                "manual_review": str(manual_path),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in combined),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)

    payload: dict[str, object] = {
        "summary": {
            "row_count": len(combined),
            "shard_count": len(shard_paths),
            "distinct_question_fingerprint_count": len(question_fingerprints),
            "distinct_query_fingerprint_count": len(query_fingerprints),
            "all_pass": True,
        },
        "sources": source_summaries,
        "template_counts": dict(sorted(template_counts.items())),
        "operation_grammar_counts": dict(sorted(operation_counts.items())),
    }
    report_prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_prefix.with_suffix(".md").write_text(
        "\n".join(
            (
                f"# Deep Hard checkpoint {len(combined):03d}",
                "",
                f"- Accepted records: **{len(combined)}/{len(combined)}**",
                f"- Source shards: **{len(shard_paths)}**",
                f"- Distinct question fingerprints: **{len(question_fingerprints)}**",
                f"- Distinct canonical query fingerprints: **{len(query_fingerprints)}**",
                "- Replay, exact dependency, Deep proof and manual-review gates: **PASS**",
                "",
                "## Sources",
                "",
                *(f"- `{summary['path']}`" for summary in source_summaries),
                "",
            )
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-prefix", required=True, type=Path)
    args = parser.parse_args()
    payload = consolidate_deep_checkpoint(
        tuple(args.shards), output_path=args.out, report_prefix=args.report_prefix
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
