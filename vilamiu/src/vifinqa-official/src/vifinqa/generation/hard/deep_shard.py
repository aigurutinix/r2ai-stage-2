"""Promote individually generated Deep Hard candidates into a verified shard.

The production generator writes one distribution sidecar next to each JSONL output.  Promotion
rechecks that proof, rejects duplicate template/operation/question/query fingerprints, normalizes
row IDs, and replays every query with exact dependency tracking before atomically publishing the
shard.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from vifinqa.generation.hard.template_batch_verification import (
    verify_template_batch,
    write_verification_report,
)
from vifinqa.generation.hard.template_intents import INTENTS_BY_ID

MIN_REASONING_DEPTH = 3
MIN_ADAPTIVE_EDGES = 2


class AutoSelectCapacityError(ValueError):
    def __init__(self, payload: dict[str, object]) -> None:
        summary = payload["summary"]
        assert isinstance(summary, dict)
        super().__init__(
            f"auto-select capacity {summary['selected']}/{summary['requested']}; "
            "inspect selection report"
        )
        self.payload = payload


def _selected_json_record(path: Path, row_id: int | None) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if row_id is None:
        if len(records) != 1:
            raise ValueError(
                f"{path}: expected exactly one staging record, got {len(records)}; "
                "select one with PATH#ROW_ID"
            )
        return records[0]
    matched = [record for record in records if record.get("id") == row_id]
    if len(matched) != 1:
        raise ValueError(f"{path}: expected exactly one record with id={row_id}")
    return matched[0]


def _parse_source(source: str | Path) -> tuple[Path, int | None]:
    raw = str(source)
    path_text, separator, row_text = raw.rpartition("#")
    if not separator:
        return Path(raw), None
    if not row_text.isdigit():
        raise ValueError(f"invalid source selector {raw!r}; expected PATH#ROW_ID")
    return Path(path_text), int(row_text)


def _only_key(mapping: object, *, field: str, path: Path) -> str:
    if not isinstance(mapping, dict) or len(mapping) != 1:
        raise ValueError(f"{path}: {field} must contain exactly one value")
    return str(next(iter(mapping)))


def _minimum_numeric_key(mapping: object, *, field: str, path: Path) -> int:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{path}: missing {field} distribution")
    try:
        return min(int(key) for key in mapping)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field} distribution") from exc


def _normalized_question(question: str) -> str:
    folded = unicodedata.normalize("NFKC", question).casefold()
    folded = re.sub(r"[^\w%]+", " ", folded, flags=re.UNICODE)
    return " ".join(folded.split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _query_fingerprint(query: str) -> str:
    try:
        canonical = ast.dump(
            ast.parse(query), annotate_fields=True, include_attributes=False
        )
    except SyntaxError as exc:
        raise ValueError(f"pandas_query is not valid Python: {exc}") from exc
    return _sha256(canonical)


def _candidate_from_source(
    source: Path | str, *, promoted_id: int
) -> tuple[dict[str, object], dict[str, object]]:
    input_path, selected_row_id = _parse_source(source)
    record = _selected_json_record(input_path, selected_row_id)
    sidecar_path = input_path.with_suffix(".distribution.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    deep = sidecar.get("deep_hard")
    if not isinstance(deep, dict):
        raise ValueError(f"{sidecar_path}: missing deep_hard proof")
    source_count = sidecar.get("count")
    if not isinstance(source_count, int):
        source_count = deep.get("verified_count")
    if (
        not isinstance(source_count, int)
        or deep.get("verified_count") != source_count
        or deep.get("pass_count") != source_count
    ):
        raise ValueError(f"{sidecar_path}: not every generated record passed Deep Hard")
    reasoning_depth = _minimum_numeric_key(
        deep.get("reasoning_depth"), field="reasoning_depth", path=sidecar_path
    )
    adaptive_edges = _minimum_numeric_key(
        deep.get("adaptive_edges"), field="adaptive_edges", path=sidecar_path
    )
    if reasoning_depth < MIN_REASONING_DEPTH or adaptive_edges < MIN_ADAPTIVE_EDGES:
        raise ValueError(
            f"{sidecar_path}: insufficient Deep Hard proof "
            f"depth={reasoning_depth}, adaptive_edges={adaptive_edges}"
        )
    template_id = str(record.get("template_id", ""))
    template_distribution = sidecar.get("template_id")
    if (
        not template_id
        or not isinstance(template_distribution, dict)
        or template_id not in template_distribution
    ):
        raise ValueError(f"{input_path}: template_id is absent from sidecar")
    intent = INTENTS_BY_ID.get(template_id)
    if intent is None:
        raise ValueError(f"{input_path}: unknown reviewed template {template_id!r}")
    operation_grammar = intent.operation_grammar
    operation_distribution = sidecar.get("operation_grammar")
    if (
        not isinstance(operation_distribution, dict)
        or operation_grammar not in operation_distribution
    ):
        raise ValueError(
            f"{input_path}: operation grammar for {template_id} is absent from sidecar"
        )
    question = str(record.get("question", ""))
    query = str(record.get("pandas_query", ""))
    promoted = dict(record)
    promoted["id"] = promoted_id
    proof: dict[str, object] = {
        "id": promoted_id,
        "source": (
            f"{input_path}#{selected_row_id}"
            if selected_row_id is not None
            else str(input_path)
        ),
        "template_id": template_id,
        "operation_grammar": operation_grammar,
        "metric_family": intent.metric_family,
        "reasoning_depth_lower_bound": reasoning_depth,
        "adaptive_edges_lower_bound": adaptive_edges,
        "question_fingerprint": _sha256(_normalized_question(question)),
        "query_fingerprint": _query_fingerprint(query),
    }
    return promoted, proof


def _against_fingerprints(
    against_paths: tuple[Path, ...],
) -> tuple[set[str], set[str]]:
    existing_questions: set[str] = set()
    existing_queries: set[str] = set()
    for against_path in against_paths:
        for record in _records(against_path):
            existing_questions.add(
                _sha256(_normalized_question(str(record.get("question", ""))))
            )
            existing_queries.add(
                _query_fingerprint(str(record.get("pandas_query", "")))
            )
    return existing_questions, existing_queries


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def auto_select_deep_sources(
    input_paths: tuple[Path | str, ...],
    *,
    count: int,
    against_paths: tuple[Path, ...] = (),
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Greedily choose a replay-ready, fingerprint-clean, grammar-diverse shard from pools."""
    if count <= 0:
        raise ValueError("auto-select count must be positive")
    expanded: list[str] = []
    for source in input_paths:
        input_path, selected_row_id = _parse_source(source)
        if selected_row_id is not None:
            expanded.append(str(source))
            continue
        source_records = _records(input_path)
        if len(source_records) == 1:
            expanded.append(str(input_path))
            continue
        for record in source_records:
            row_id = record.get("id")
            if not isinstance(row_id, int):
                raise ValueError(f"{input_path}: every pooled record needs an integer id")
            expanded.append(f"{input_path}#{row_id}")

    existing_questions, existing_queries = _against_fingerprints(against_paths)
    selected: list[str] = []
    selected_templates: set[object] = set()
    selected_operations: set[object] = set()
    selected_questions: set[object] = set()
    selected_queries: set[object] = set()
    decisions: list[dict[str, object]] = []

    for source in expanded:
        _record, proof = _candidate_from_source(source, promoted_id=len(selected) + 1)
        reason = None
        if proof["question_fingerprint"] in existing_questions:
            reason = "question_duplicate_against_ledger"
        elif proof["query_fingerprint"] in existing_queries:
            reason = "query_duplicate_against_ledger"
        elif proof["template_id"] in selected_templates:
            reason = "template_duplicate_within_shard"
        elif proof["operation_grammar"] in selected_operations:
            reason = "operation_grammar_duplicate_within_shard"
        elif proof["question_fingerprint"] in selected_questions:
            reason = "question_duplicate_within_shard"
        elif proof["query_fingerprint"] in selected_queries:
            reason = "query_duplicate_within_shard"

        if reason is not None:
            decisions.append({"source": source, "selected": False, "reason": reason})
            continue
        selected.append(source)
        selected_templates.add(proof["template_id"])
        selected_operations.add(proof["operation_grammar"])
        selected_questions.add(proof["question_fingerprint"])
        selected_queries.add(proof["query_fingerprint"])
        decisions.append({"source": source, "selected": True, "reason": "accepted"})
        if len(selected) == count:
            break

    payload: dict[str, object] = {
        "summary": {
            "requested": count,
            "selected": len(selected),
            "pool_size": len(expanded),
            "capacity_ok": len(selected) == count,
        },
        "selected_sources": selected,
        "decisions": decisions,
    }
    if len(selected) != count:
        raise AutoSelectCapacityError(payload)
    return tuple(selected), payload


def promote_deep_shard(
    input_paths: tuple[Path | str, ...],
    *,
    output_path: Path,
    report_prefix: Path,
    against_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    if not input_paths:
        raise ValueError("at least one staging input is required")

    records: list[dict[str, object]] = []
    proofs: list[dict[str, object]] = []
    for row_id, source in enumerate(input_paths, start=1):
        promoted, proof = _candidate_from_source(source, promoted_id=row_id)
        records.append(promoted)
        proofs.append(proof)

    unique_fields = (
        "template_id",
        "operation_grammar",
        "question_fingerprint",
        "query_fingerprint",
    )
    for field in unique_fields:
        values = [proof[field] for proof in proofs]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field} in proposed shard")

    existing_questions, existing_queries = _against_fingerprints(against_paths)
    repeated_questions = [
        proof["id"]
        for proof in proofs
        if proof["question_fingerprint"] in existing_questions
    ]
    repeated_queries = [
        proof["id"]
        for proof in proofs
        if proof["query_fingerprint"] in existing_queries
    ]
    if repeated_questions:
        raise ValueError(f"question fingerprints already exist for rows {repeated_questions}")
    if repeated_queries:
        raise ValueError(f"query fingerprints already exist for rows {repeated_queries}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    verification = verify_template_batch(temporary_path)
    if not verification.ok:
        raise ValueError("replay/dependency verification failed for proposed shard")
    os.replace(temporary_path, output_path)

    final_verification = verify_template_batch(output_path)
    write_verification_report(
        final_verification,
        json_out=report_prefix.with_suffix(".verification.json"),
        markdown_out=report_prefix.with_suffix(".verification.md"),
    )
    fingerprint_payload: dict[str, object] = {
        "summary": {
            "row_count": len(records),
            "distinct_template_count": len({p["template_id"] for p in proofs}),
            "distinct_operation_grammar_count": len(
                {p["operation_grammar"] for p in proofs}
            ),
            "distinct_question_fingerprint_count": len(
                {p["question_fingerprint"] for p in proofs}
            ),
            "distinct_query_fingerprint_count": len(
                {p["query_fingerprint"] for p in proofs}
            ),
            "all_pass": final_verification.ok,
        },
        "rows": proofs,
    }
    report_prefix.with_suffix(".fingerprints.json").write_text(
        json.dumps(fingerprint_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-prefix", required=True, type=Path)
    parser.add_argument("--against", action="append", default=[], type=Path)
    parser.add_argument("--auto-select", type=int)
    parser.add_argument("--selection-report", type=Path)
    args = parser.parse_args()
    inputs: tuple[str | Path, ...] = tuple(args.inputs)
    if args.auto_select is not None:
        try:
            inputs, selection = auto_select_deep_sources(
                inputs,
                count=args.auto_select,
                against_paths=tuple(args.against),
            )
        except AutoSelectCapacityError as exc:
            if args.selection_report is not None:
                args.selection_report.parent.mkdir(parents=True, exist_ok=True)
                args.selection_report.write_text(
                    json.dumps(exc.payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            raise
        if args.selection_report is not None:
            args.selection_report.parent.mkdir(parents=True, exist_ok=True)
            args.selection_report.write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    payload = promote_deep_shard(
        inputs,
        output_path=args.out,
        report_prefix=args.report_prefix,
        against_paths=tuple(args.against),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
