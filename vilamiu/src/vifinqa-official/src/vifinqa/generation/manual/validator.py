"""Validation for author records and audit re-execution in the Hard manual lane."""

from __future__ import annotations

import ast
import csv
import hashlib
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from vifinqa.generation.schemas import QARecord
from vifinqa.generation.manual.packs import stable_id
from vifinqa.generation.manual.schemas import ManualRecordDraft, MetamorphicSpec
from vifinqa.generation.validation.pandas_check import check_answer, execute_query, values_match

_DYNAMIC_NAME_RE = re.compile(
    r"(?:winner|survivor|selected_(?:entity|period|key|set)|filtered_(?:entities|periods))",
    re.IGNORECASE,
)


def record_id(batch_id: str, position: int) -> int:
    digest = stable_id("record", batch_id, position).split("-", 1)[1]
    return int(digest[:15], 16)


def _finite_numeric(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_embedding_errors(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"pandas_query_syntax_error:{exc.msg}"]

    constant_vars: set[str] = set()
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = [target.id for target in targets if isinstance(target, ast.Name)]
        if isinstance(value, ast.Constant):
            constant_vars.update(target_names)
        for name in target_names:
            if _DYNAMIC_NAME_RE.search(name) and isinstance(
                value, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)
            ):
                errors.append(f"literal_dynamic_output:{name}")

        if "result" not in target_names:
            continue
        if isinstance(value, ast.Constant):
            errors.append("literal_terminal_answer")
        if isinstance(value, ast.Name) and value.id in constant_vars:
            errors.append("literal_terminal_answer_via_constant")
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant):
            errors.append("literal_terminal_winner_or_period")
    return sorted(set(errors))


@dataclass(slots=True)
class MetamorphicResult:
    name: str
    ok: bool
    actual: object = None
    changed: bool = False
    accessed_refs: list[str] = field(default_factory=list)
    detail: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "actual": self.actual,
            "changed": self.changed,
            "accessed_refs": self.accessed_refs,
            "detail": self.detail,
        }


@dataclass(slots=True)
class RecordValidation:
    record_id: int
    accepted: bool
    qa_record: QARecord | None
    execution_result: object = None
    accessed_refs: list[str] = field(default_factory=list)
    check_answer_ok: bool = False
    reject_reasons: list[str] = field(default_factory=list)
    metamorphic_results: list[MetamorphicResult] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "accepted": self.accepted,
            "qa_record": self.qa_record.model_dump(mode="json") if self.qa_record else None,
            "execution_result": self.execution_result,
            "accessed_refs": self.accessed_refs,
            "check_answer_ok": self.check_answer_ok,
            "reject_reasons": self.reject_reasons,
            "metamorphic_results": [result.to_payload() for result in self.metamorphic_results],
        }


class ManualRecordValidator:
    def validate(
        self,
        *,
        draft: ManualRecordDraft,
        pack: dict[str, object],
        batch_id: str,
        position: int,
    ) -> RecordValidation:
        assigned_id = record_id(batch_id, position)
        reasons: list[str] = []
        if not _finite_numeric(draft.answer):
            reasons.append("answer_not_finite_numeric_scalar")

        failed_review = draft.self_review.failed_items()
        reasons.extend(f"self_review_failed:{name}" for name in failed_review)
        reasons.extend(_literal_embedding_errors(draft.pandas_query))

        expected_paths = {
            str(table["table_ref"]): Path(str(table["csv_path"]))
            for table in pack.get("tables", [])
            if isinstance(table, dict) and "table_ref" in table and "csv_path" in table
        }
        csv_values = [draft.csv_path] if isinstance(draft.csv_path, str) else list(draft.csv_path)
        if len(csv_values) != len(draft.relevant_tables):
            reasons.append("csv_path_length_mismatch")

        csv_map: dict[str, Path] = {}
        for index, table_ref in enumerate(draft.relevant_tables):
            expected = expected_paths.get(table_ref)
            if expected is None:
                reasons.append(f"table_not_in_pack:{table_ref}")
                continue
            if index >= len(csv_values):
                continue
            supplied = Path(csv_values[index])
            if not supplied.exists():
                reasons.append(f"csv_path_missing:{supplied}")
            try:
                if supplied.resolve() != expected.resolve():
                    reasons.append(f"csv_path_order_or_provenance_mismatch:{table_ref}")
            except OSError:
                reasons.append(f"csv_path_unresolvable:{supplied}")
            csv_map[table_ref] = expected

        expected_docs = list(dict.fromkeys(ref.split("|table_", 1)[0] for ref in draft.relevant_tables))
        if draft.relevant_docs != expected_docs:
            reasons.append("relevant_docs_mismatch")

        execution_result: object = None
        accessed_refs: list[str] = []
        check_ok = False
        if len(csv_map) == len(draft.relevant_tables) and all(path.exists() for path in csv_map.values()):
            execution = execute_query(draft.pandas_query, csv_map)
            execution_result = execution.actual
            accessed_refs = sorted(execution.accessed_refs)
            if not execution.ok:
                reasons.append(f"execution_failed:{execution.detail}")
            else:
                if not _finite_numeric(execution.actual):
                    reasons.append("execution_not_finite_numeric_scalar")
                answer_check = check_answer(draft.pandas_query, csv_map, draft.answer)
                check_ok = answer_check.ok
                if not answer_check.ok:
                    reasons.append(f"check_answer_failed:{answer_check.detail}")
                if answer_check.accessed_refs != frozenset(draft.relevant_tables):
                    reasons.append("accessed_refs_mismatch")

        qa_record: QARecord | None = None
        if _finite_numeric(draft.answer):
            qa_record = QARecord(
                id=assigned_id,
                question=draft.question,
                answer=draft.answer,  # type: ignore[arg-type]
                relevant_docs=draft.relevant_docs,
                relevant_tables=draft.relevant_tables,
                pandas_query=draft.pandas_query,
                csv_path=draft.csv_path,
                difficulty="hard",
            )

        metamorphic_results: list[MetamorphicResult] = []
        if not reasons and execution_result is not None:
            for spec in draft.metamorphic_specs:
                result = self._run_metamorphic(
                    spec=spec,
                    code=draft.pandas_query,
                    csv_map=csv_map,
                    original_result=execution_result,
                    expected_refs=frozenset(draft.relevant_tables),
                )
                metamorphic_results.append(result)
                if not result.ok:
                    reasons.append(f"metamorphic_failed:{spec.name}:{result.detail}")

        return RecordValidation(
            record_id=assigned_id,
            accepted=not reasons,
            qa_record=qa_record,
            execution_result=execution_result,
            accessed_refs=accessed_refs,
            check_answer_ok=check_ok,
            reject_reasons=reasons,
            metamorphic_results=metamorphic_results,
        )

    def _run_metamorphic(
        self,
        *,
        spec: MetamorphicSpec,
        code: str,
        csv_map: dict[str, Path],
        original_result: object,
        expected_refs: frozenset[str],
    ) -> MetamorphicResult:
        original_hashes = {ref: _file_hash(path) for ref, path in csv_map.items()}
        with tempfile.TemporaryDirectory(prefix="qagen-hard-manual-") as temp_dir:
            temp_root = Path(temp_dir)
            copies: dict[str, Path] = {}
            for index, (ref, source) in enumerate(csv_map.items()):
                copy_path = temp_root / f"{index:03d}-{source.name}"
                shutil.copy2(source, copy_path)
                copies[ref] = copy_path

            try:
                for mutation in spec.mutations:
                    copy_path = copies.get(mutation.table_ref)
                    if copy_path is None:
                        raise ValueError(f"mutation table_ref is outside relevant_tables: {mutation.table_ref}")
                    rows = list(csv.reader(copy_path.open(encoding="utf-8-sig", newline="")))
                    physical_row = mutation.row_idx + 1
                    if physical_row >= len(rows) or mutation.col_idx >= len(rows[physical_row]):
                        raise ValueError(
                            f"mutation coordinate is outside the CSV: {mutation.table_ref} "
                            f"({mutation.row_idx}, {mutation.col_idx})"
                        )
                    rows[physical_row][mutation.col_idx] = mutation.replacement
                    with copy_path.open("w", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerows(rows)
            except (OSError, ValueError) as exc:
                return MetamorphicResult(name=spec.name, ok=False, detail=str(exc))

            execution = execute_query(code, copies)
            if not execution.ok:
                return MetamorphicResult(
                    name=spec.name,
                    ok=False,
                    detail=execution.detail,
                    accessed_refs=sorted(execution.accessed_refs),
                )
            if not _finite_numeric(execution.actual):
                return MetamorphicResult(
                    name=spec.name,
                    ok=False,
                    actual=execution.actual,
                    detail="metamorphic result is not a finite numeric scalar",
                    accessed_refs=sorted(execution.accessed_refs),
                )
            changed = not values_match(original_result, execution.actual)
            refs_ok = execution.accessed_refs == expected_refs
            change_ok = changed if spec.expect_result_change else not changed
            corpus_unchanged = all(
                _file_hash(csv_map[ref]) == digest for ref, digest in original_hashes.items()
            )
            details: list[str] = []
            if not refs_ok:
                details.append("accessed_refs changed")
            if not change_ok:
                details.append(
                    "result did not change" if spec.expect_result_change else "result unexpectedly changed"
                )
            if not corpus_unchanged:
                details.append("original corpus changed")
            return MetamorphicResult(
                name=spec.name,
                ok=not details,
                actual=execution.actual,
                changed=changed,
                accessed_refs=sorted(execution.accessed_refs),
                detail="; ".join(details),
            )
