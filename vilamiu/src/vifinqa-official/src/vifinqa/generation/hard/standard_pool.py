"""Audit and promote bulk Hard-standard pools without imposing Deep-Hard thresholds.

The generator already performs semantic/dependency critics before writing a pool.  This module is
the independent, deterministic publication gate: it replays every query, requires exact declared
table usage, verifies the basic-Hard proof in the distribution sidecar, de-duplicates against an
existing ledger and within the pool, then waits for a complete manual rubric review before
promotion.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata

from pydantic import ValidationError

from vifinqa.generation.hard.recipe.gates import (
    HARD_BASIC_MIN_ADAPTIVE,
    HARD_BASIC_MIN_DEPTH,
)
from vifinqa.generation.hard.template_intents import INTENTS_BY_ID
from vifinqa.generation.manual.schemas import AuditRubric
from vifinqa.generation.validation.pandas_check import check_answer


_PROTECTED_NAMES = frozenset(
    {
        "True",
        "False",
        "None",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "dfs",
        "df",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "pd",
        "range",
        "result",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_question(question: str) -> str:
    folded = unicodedata.normalize("NFKC", question).casefold()
    folded = re.sub(r"[^\w%]+", " ", folded, flags=re.UNICODE)
    return " ".join(folded.split())


def question_fingerprint(question: str) -> str:
    return _sha256(normalized_question(question))


def _parsed_query(query: str) -> ast.Module:
    try:
        return ast.parse(query)
    except SyntaxError as exc:
        raise ValueError(f"pandas_query is not valid Python: {exc}") from exc


def query_fingerprint(query: str) -> str:
    canonical = ast.dump(
        _parsed_query(query), annotate_fields=True, include_attributes=False
    )
    return _sha256(canonical)


class _AlphaRename(ast.NodeTransformer):
    """Canonicalize local names while retaining data literals and public runtime names."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def _name(self, raw: str) -> str:
        if raw in _PROTECTED_NAMES:
            return raw
        if raw not in self.names:
            self.names[raw] = f"v{len(self.names)}"
        return self.names[raw]

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802 - ast visitor contract
        return ast.copy_location(ast.Name(id=self._name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:  # noqa: N802 - ast visitor contract
        return ast.copy_location(
            ast.arg(
                arg=self._name(node.arg),
                annotation=(
                    self.visit(node.annotation) if node.annotation is not None else None
                ),
                type_comment=node.type_comment,
            ),
            node,
        )

    def visit_FunctionDef(  # noqa: N802 - ast visitor contract
        self, node: ast.FunctionDef
    ) -> ast.FunctionDef:
        node.name = self._name(node.name)
        return self.generic_visit(node)


def semantic_query_fingerprint(template_id: str, query: str) -> str:
    tree = _AlphaRename().visit(_parsed_query(query))
    ast.fix_missing_locations(tree)
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return _sha256(f"{template_id}\n{canonical}")


@dataclass(frozen=True, slots=True)
class PoolDecision:
    source: str
    source_id: int | None
    auto_status: str
    final_status: str
    reasons: tuple[str, ...]
    question_fingerprint: str = ""
    query_fingerprint: str = ""
    semantic_query_fingerprint: str = ""
    actual: object = None
    expected: object = None
    accessed_refs: tuple[str, ...] = ()
    declared_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HardStandardPoolAudit:
    inputs: tuple[str, ...]
    against: tuple[str, ...]
    decisions: tuple[PoolDecision, ...]
    accepted_records: tuple[dict[str, object], ...]

    @property
    def summary(self) -> dict[str, object]:
        final = Counter(decision.final_status for decision in self.decisions)
        auto = Counter(decision.auto_status for decision in self.decisions)
        return {
            "input_record_count": len(self.decisions),
            "auto_clean_count": auto["clean"],
            "auto_reject_count": auto["reject"],
            "pending_manual_count": final["pending_manual_review"],
            "manual_reject_count": final["manual_reject"],
            "accepted_count": final["accepted"],
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "inputs": list(self.inputs),
            "against": list(self.against),
            "decisions": [asdict(decision) for decision in self.decisions],
        }


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _minimum_distribution_key(value: object, *, field: str) -> int:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"missing {field} distribution")
    try:
        return min(int(key) for key in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} distribution") from exc


def _basic_hard_fingerprint_proof(path: Path, *, expected_count: int) -> str:
    fingerprint_path = path.with_suffix(".fingerprints.json")
    verification_path = path.with_suffix(".verification.json")
    try:
        fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"published_proof_unreadable:{exc}"
    fingerprint_summary = fingerprints.get("summary")
    verification_summary = verification.get("summary")
    rows = fingerprints.get("rows")
    if (
        not isinstance(fingerprint_summary, dict)
        or not isinstance(verification_summary, dict)
        or not isinstance(rows, list)
        or fingerprint_summary.get("all_pass") is not True
        or verification_summary.get("all_pass") is not True
        or fingerprint_summary.get("row_count") != expected_count
        or verification_summary.get("row_count") != expected_count
        or len(rows) != expected_count
    ):
        return "published_proof_count_or_status_mismatch"
    try:
        min_depth = min(
            int(
                row["reasoning_depth_lower_bound"]
                if "reasoning_depth_lower_bound" in row
                else row["reasoning_depth"]
            )
            for row in rows
        )
        min_adaptive = min(
            int(
                row["adaptive_edges_lower_bound"]
                if "adaptive_edges_lower_bound" in row
                else row["adaptive_edges"]
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        return f"published_proof_invalid_hardness:{exc}"
    if min_depth < HARD_BASIC_MIN_DEPTH:
        return f"hardness_depth_below_basic:{min_depth}"
    if min_adaptive < HARD_BASIC_MIN_ADAPTIVE:
        return f"hardness_adaptive_below_basic:{min_adaptive}"
    return ""


def _published_checkpoint_proof(path: Path, *, expected_count: int) -> str:
    manifest_path = path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"checkpoint_manifest_unreadable:{exc}"
    summary = manifest.get("summary")
    sources = manifest.get("sources")
    if (
        not isinstance(summary, dict)
        or summary.get("all_pass") is not True
        or summary.get("row_count") != expected_count
        or not isinstance(sources, list)
        or not sources
    ):
        return "checkpoint_manifest_count_or_status_mismatch"
    source_count = 0
    for source in sources:
        if (
            not isinstance(source, dict)
            or source.get("replay_dependency_pass") is not True
            or not source.get("manual_review")
        ):
            return "checkpoint_source_not_reviewed_or_verified"
        source_path = Path(str(source.get("path", "")))
        row_count = source.get("row_count")
        if not isinstance(row_count, int):
            return "checkpoint_source_count_invalid"
        proof_error = _basic_hard_fingerprint_proof(
            source_path, expected_count=row_count
        )
        if proof_error:
            return f"checkpoint_source:{proof_error}"
        if not Path(str(source["manual_review"])).is_file():
            return "checkpoint_manual_review_missing"
        source_count += row_count
    return "" if source_count == expected_count else "checkpoint_source_count_mismatch"


def _is_published_checkpoint(path: Path) -> bool:
    manifest_path = path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    summary = manifest.get("summary")
    return (
        isinstance(summary, dict)
        and summary.get("all_pass") is True
        and isinstance(manifest.get("sources"), list)
        and bool(manifest["sources"])
    )


def _verify_basic_hard_sidecar(path: Path, records: list[dict[str, object]]) -> str:
    sidecar_path = path.with_suffix(".distribution.json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fingerprint_path = path.with_suffix(".fingerprints.json")
        if fingerprint_path.is_file():
            return _basic_hard_fingerprint_proof(path, expected_count=len(records))
        manifest_path = path.with_suffix(".json")
        if manifest_path.is_file():
            return _published_checkpoint_proof(path, expected_count=len(records))
        return f"hardness_sidecar_unreadable:{exc}"
    evidence = sidecar.get("hardness", sidecar.get("deep_hard"))
    if not isinstance(evidence, dict):
        return "hardness_sidecar_missing_evidence"
    count = len(records)
    if sidecar.get("count") != count or evidence.get("verified_count") != count:
        return "hardness_sidecar_count_mismatch"
    try:
        depth = _minimum_distribution_key(
            evidence.get("reasoning_depth"), field="reasoning_depth"
        )
        adaptive = _minimum_distribution_key(
            evidence.get("adaptive_edges"), field="adaptive_edges"
        )
    except ValueError as exc:
        return f"hardness_sidecar_invalid:{exc}"
    if depth < HARD_BASIC_MIN_DEPTH:
        return f"hardness_depth_below_basic:{depth}"
    if adaptive < HARD_BASIC_MIN_ADAPTIVE:
        return f"hardness_adaptive_below_basic:{adaptive}"
    return ""


def _review_map(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    reviews: dict[str, dict[str, object]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        raw = json.loads(raw_line)
        source = str(raw.get("source", ""))
        if not source or source in reviews:
            raise ValueError(f"{path}:{line_number}: missing or duplicate source")
        try:
            rubric = AuditRubric.model_validate(raw.get("rubric"))
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid rubric: {exc}") from exc
        verdict = raw.get("verdict")
        if verdict not in {"accept", "reject"}:
            raise ValueError(f"{path}:{line_number}: verdict must be accept or reject")
        issues = raw.get("issues", [])
        if verdict == "accept" and rubric.failed_items():
            raise ValueError(
                f"{path}:{line_number}: accept has failed rubric {rubric.failed_items()}"
            )
        if verdict == "reject" and not issues:
            raise ValueError(f"{path}:{line_number}: reject must list issues")
        reviews[source] = {
            "verdict": verdict,
            "rubric": rubric,
            "issues": issues,
        }
    return reviews


def _intent_alias_key(template_id: str) -> tuple[str, ...] | None:
    intent = INTENTS_BY_ID.get(template_id)
    if intent is None:
        return None
    return (
        intent.operation_grammar,
        intent.metric_family,
        intent.universe_kind.value,
        intent.terminal_operation.value,
        intent.terminal_unit,
    )


def _existing_fingerprints(
    paths: tuple[Path, ...],
) -> tuple[set[str], set[str], set[str], dict[tuple[str, ...], set[str]]]:
    questions: set[str] = set()
    queries: set[str] = set()
    semantics: set[str] = set()
    alias_owners: dict[tuple[str, ...], set[str]] = {}
    for path in paths:
        for record in _records(path):
            template_id = str(record.get("template_id", ""))
            question = str(record.get("question", ""))
            query = str(record.get("pandas_query", ""))
            questions.add(question_fingerprint(question))
            queries.add(query_fingerprint(query))
            semantics.add(semantic_query_fingerprint(template_id, query))
            alias_key = _intent_alias_key(template_id)
            if alias_key is not None:
                alias_owners.setdefault(alias_key, set()).add(template_id)
    return questions, queries, semantics, alias_owners


def audit_standard_pool(
    input_paths: tuple[Path, ...],
    *,
    against_paths: tuple[Path, ...] = (),
    reviews_path: Path | None = None,
) -> HardStandardPoolAudit:
    if not input_paths:
        raise ValueError("at least one input pool is required")
    reviews = _review_map(reviews_path)
    (
        seen_questions,
        seen_queries,
        seen_semantics,
        seen_alias_owners,
    ) = _existing_fingerprints(against_paths)
    decisions: list[PoolDecision] = []
    accepted_records: list[dict[str, object]] = []
    encountered_sources: set[str] = set()

    for input_path in input_paths:
        records = _records(input_path)
        proof_error = _verify_basic_hard_sidecar(input_path, records)
        # A published checkpoint may contain legacy aliases that were independently reviewed.
        # Grandfather those owners while still replaying every checkpoint row.  New pools are not
        # pre-registered, so a fresh alias of an existing reviewed template is rejected.
        if not proof_error and _is_published_checkpoint(input_path):
            for checkpoint_record in records:
                checkpoint_template = str(checkpoint_record.get("template_id", ""))
                checkpoint_alias = _intent_alias_key(checkpoint_template)
                if checkpoint_alias is not None:
                    seen_alias_owners.setdefault(checkpoint_alias, set()).add(
                        checkpoint_template
                    )
        source_ids = [record.get("id") for record in records]
        duplicate_source_ids = {
            value for value, count in Counter(source_ids).items() if count > 1
        }
        for record in records:
            source_id = record.get("id") if isinstance(record.get("id"), int) else None
            source = f"{input_path}#{source_id}"
            reasons: list[str] = []
            if source in encountered_sources:
                reasons.append("duplicate_source")
            encountered_sources.add(source)
            if proof_error:
                reasons.append(proof_error)
            if source_id is None or source_id in duplicate_source_ids:
                reasons.append("invalid_or_duplicate_source_id")

            question = str(record.get("question", ""))
            query = str(record.get("pandas_query", ""))
            template_id = str(record.get("template_id", ""))
            answer = record.get("answer")
            relevant_tables = tuple(map(str, record.get("relevant_tables", ())))
            raw_paths = record.get("csv_path", ())
            csv_paths = (raw_paths,) if isinstance(raw_paths, str) else tuple(raw_paths)
            relevant_docs = tuple(map(str, record.get("relevant_docs", ())))
            expected_docs = tuple(
                dict.fromkeys(ref.split("|", 1)[0] for ref in relevant_tables)
            )

            if not question.strip():
                reasons.append("question_empty")
            if record.get("difficulty") != "hard":
                reasons.append("difficulty_not_hard")
            if template_id not in INTENTS_BY_ID:
                reasons.append("unknown_template_id")
            alias_key = _intent_alias_key(template_id)
            if (
                alias_key is not None
                and (owners := seen_alias_owners.get(alias_key))
                and template_id not in owners
            ):
                reasons.append("template_semantic_alias:" + ",".join(sorted(owners)))
            if (
                isinstance(answer, bool)
                or not isinstance(answer, (int, float))
                or not math.isfinite(float(answer))
            ):
                reasons.append("answer_not_finite_numeric")
            if not relevant_tables or len(set(relevant_tables)) != len(relevant_tables):
                reasons.append("relevant_tables_empty_or_duplicate")
            if len(relevant_tables) != len(csv_paths):
                reasons.append("csv_path_length_mismatch")
            if set(relevant_docs) != set(expected_docs) or len(relevant_docs) != len(
                set(relevant_docs)
            ):
                reasons.append("relevant_docs_mismatch")

            qfp = question_fingerprint(question)
            query_fp = ""
            semantic_fp = ""
            try:
                query_fp = query_fingerprint(query)
                semantic_fp = semantic_query_fingerprint(template_id, query)
            except ValueError as exc:
                reasons.append(f"query_syntax:{exc}")

            if qfp in seen_questions:
                reasons.append("question_duplicate")
            if query_fp and query_fp in seen_queries:
                reasons.append("query_duplicate")
            if semantic_fp and semantic_fp in seen_semantics:
                reasons.append("semantic_query_duplicate")

            actual: object = None
            accessed: tuple[str, ...] = ()
            if len(relevant_tables) == len(csv_paths) and relevant_tables:
                csv_map = {
                    ref: Path(str(csv_path))
                    for ref, csv_path in zip(relevant_tables, csv_paths, strict=True)
                }
                missing_paths = [
                    str(path) for path in csv_map.values() if not path.is_file()
                ]
                if missing_paths:
                    reasons.append(f"csv_path_missing:{missing_paths[0]}")
                else:
                    replay = check_answer(query, csv_map, answer)
                    actual = replay.actual
                    accessed = tuple(sorted(replay.accessed_refs))
                    if not replay.ok:
                        reasons.append(f"answer_replay_failed:{replay.detail}")
                    if replay.accessed_refs != frozenset(relevant_tables):
                        reasons.append("accessed_refs_mismatch")

            auto_status = "reject" if reasons else "clean"
            if auto_status == "clean":
                seen_questions.add(qfp)
                seen_queries.add(query_fp)
                seen_semantics.add(semantic_fp)
                if alias_key is not None:
                    seen_alias_owners.setdefault(alias_key, set()).add(template_id)

            review = reviews.get(source)
            if auto_status == "reject":
                final_status = "auto_reject"
            elif review is None:
                final_status = "pending_manual_review"
            elif review["verdict"] == "reject":
                final_status = "manual_reject"
                reasons.extend(f"manual:{issue}" for issue in review["issues"])
            else:
                final_status = "accepted"
                promoted = dict(record)
                promoted["id"] = len(accepted_records) + 1
                accepted_records.append(promoted)

            decisions.append(
                PoolDecision(
                    source=source,
                    source_id=source_id,
                    auto_status=auto_status,
                    final_status=final_status,
                    reasons=tuple(reasons),
                    question_fingerprint=qfp,
                    query_fingerprint=query_fp,
                    semantic_query_fingerprint=semantic_fp,
                    actual=actual,
                    expected=answer,
                    accessed_refs=accessed,
                    declared_refs=tuple(sorted(relevant_tables)),
                )
            )

    unknown_reviews = sorted(set(reviews) - {decision.source for decision in decisions})
    if unknown_reviews:
        raise ValueError(f"reviews reference unknown source: {unknown_reviews[0]}")
    return HardStandardPoolAudit(
        inputs=tuple(map(str, input_paths)),
        against=tuple(map(str, against_paths)),
        decisions=tuple(decisions),
        accepted_records=tuple(accepted_records),
    )


def render_markdown(report: HardStandardPoolAudit) -> str:
    summary = report.summary
    lines = [
        "# Hard-standard pool audit",
        "",
        f"- Input records: {summary['input_record_count']}",
        f"- Auto clean / reject: {summary['auto_clean_count']} / {summary['auto_reject_count']}",
        f"- Pending manual rubric: {summary['pending_manual_count']}",
        f"- Manual reject / accepted: {summary['manual_reject_count']} / {summary['accepted_count']}",
        "",
        "| source | auto | final | replay | dependencies | reasons |",
        "|---|---|---|---|---|---|",
    ]
    for decision in report.decisions:
        replay = (
            "PASS"
            if decision.actual is not None
            and not any(
                reason.startswith("answer_replay_failed") for reason in decision.reasons
            )
            else "FAIL"
        )
        deps = "PASS" if decision.accessed_refs == decision.declared_refs else "FAIL"
        lines.append(
            f"| {decision.source} | {decision.auto_status} | {decision.final_status} | "
            f"{replay} | {deps} | {'; '.join(decision.reasons)} |"
        )
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_standard_publication_evidence(
    path: Path,
    report: HardStandardPoolAudit,
    *,
    reviews_path: Path,
) -> None:
    """Persist replay, dependency, hardness, and review provenance for a promoted pool."""

    accepted_decisions = [
        decision for decision in report.decisions if decision.final_status == "accepted"
    ]
    if len(accepted_decisions) != len(report.accepted_records):
        raise ValueError("accepted decision/record count mismatch")
    if not reviews_path.is_file():
        raise ValueError(f"manual review file missing: {reviews_path}")

    fingerprint_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []
    template_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for promoted, decision in zip(
        report.accepted_records, accepted_decisions, strict=True
    ):
        row_id = int(promoted["id"])
        template_id = str(promoted["template_id"])
        intent = INTENTS_BY_ID[template_id]
        template_counts[template_id] += 1
        operation_counts[intent.operation_grammar] += 1
        fingerprint_rows.append(
            {
                "id": row_id,
                "source": decision.source,
                "template_id": template_id,
                "operation_grammar": intent.operation_grammar,
                "reasoning_depth_lower_bound": HARD_BASIC_MIN_DEPTH,
                "adaptive_edges_lower_bound": HARD_BASIC_MIN_ADAPTIVE,
                "question_fingerprint": decision.question_fingerprint,
                "query_fingerprint": decision.query_fingerprint,
                "semantic_query_fingerprint": decision.semantic_query_fingerprint,
            }
        )
        verification_rows.append(
            {
                "id": row_id,
                "source": decision.source,
                "replay_pass": decision.actual is not None,
                "dependency_pass": decision.accessed_refs == decision.declared_refs,
                "actual": decision.actual,
                "expected": decision.expected,
                "accessed_refs": list(decision.accessed_refs),
                "declared_refs": list(decision.declared_refs),
            }
        )

    row_count = len(fingerprint_rows)
    all_pass = all(
        row["replay_pass"] and row["dependency_pass"] for row in verification_rows
    )
    if not all_pass:
        raise ValueError("cannot publish a pool with failed replay/dependency evidence")
    fingerprint_payload = {
        "summary": {"row_count": row_count, "all_pass": True},
        "manual_review": str(reviews_path),
        "rows": fingerprint_rows,
    }
    verification_payload = {
        "summary": {"row_count": row_count, "all_pass": True},
        "rows": verification_rows,
    }
    distribution_payload = {
        "count": row_count,
        "hardness": {
            "required_reasoning_depth": HARD_BASIC_MIN_DEPTH,
            "required_adaptive_edges": HARD_BASIC_MIN_ADAPTIVE,
            "verified_count": row_count,
            "pass_count": row_count,
            "reasoning_depth": {str(HARD_BASIC_MIN_DEPTH): row_count},
            "adaptive_edges": {str(HARD_BASIC_MIN_ADAPTIVE): row_count},
        },
        "template_counts": dict(sorted(template_counts.items())),
        "operation_grammar_counts": dict(sorted(operation_counts.items())),
    }
    path.with_suffix(".fingerprints.json").write_text(
        json.dumps(fingerprint_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".verification.json").write_text(
        json.dumps(verification_payload, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".distribution.json").write_text(
        json.dumps(distribution_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--against", action="append", default=[], type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--report-prefix", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit_standard_pool(
        tuple(args.inputs),
        against_paths=tuple(args.against),
        reviews_path=args.reviews,
    )
    args.report_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.report_prefix.with_suffix(".json").write_text(
        json.dumps(report.to_payload(), ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    args.report_prefix.with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    if args.out is not None:
        if args.reviews is None:
            raise ValueError(
                "--out requires --reviews with the complete normal-Hard rubric"
            )
        _write_jsonl(args.out, report.accepted_records)
        write_standard_publication_evidence(args.out, report, reviews_path=args.reviews)
    print(json.dumps(report.summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
