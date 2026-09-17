"""Bind retrieval-grounded FinancialPlans to corpus cells and Pandas.

The LLM-facing plan contains row references but never source numbers or code.
This module is the benchmark_locked boundary: it validates document dimensions,
chooses the requested period column, converts units, evaluates the plan, and
emits replayable evidence/query artifacts.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .experiments import SubmissionArchive
from .financial_ir import FactRequest, FinancialPlan, FinancialPlanV2, compile_pandas, evaluate_plan
from .l1_fact_layer import (
    SOURCE_UNIT_SCALE,
    TARGET_UNIT_SCALE,
    ParsedQuestion,
    column_header,
    column_role_score,
    contextual_row_label,
    detect_target_unit,
    infer_cell_source_unit,
    normalize,
    pandas_query_for,
    parse_number,
    period_score,
    schema_link_score,
    units_compatible,
)


ROW_REF_RE = re.compile(r"^t(?P<table_id>\d+)r(?P<row_index>\d+)$")


@dataclass(frozen=True)
class GroundedCell:
    fact_id: str
    row_ref: str
    ticker: str
    year: int
    requested_scope: str
    document_scope: str
    document_id: str
    evidence_key: str
    table_id: int
    source_line_0: int
    source_line_1: int
    row_index: int
    column_index: int
    row_text: str
    column_text: str
    raw_value: str
    parsed_value: float
    source_unit: str
    source_scale: float
    target_unit: str
    target_scale: float
    value: float
    transform: str
    column_score: float
    runner_up_gap: float
    ambiguity: str


@dataclass(frozen=True)
class GroundedPlanResult:
    plan: FinancialPlan | FinancialPlanV2
    bindings: tuple[GroundedCell, ...]
    answer: float
    pandas_query: str

    def audit_record(self) -> dict[str, Any]:
        return {
            "question_id": getattr(self.plan, "question_id", -1),
            "status": "BOUND",
            "answer": self.answer,
            "output_unit": self.plan.output_unit,
            "pandas_query": self.pandas_query,
            "bindings": [asdict(binding) for binding in self.bindings],
        }


def _unit_scale(unit: str) -> float:
    if unit in TARGET_UNIT_SCALE:
        return TARGET_UNIT_SCALE[unit]
    if unit in {"number", "ratio", "count", "year"}:
        return 1.0
    raise ValueError(f"Unsupported grounded fact unit {unit!r}")


def validate_requested_output_unit(plan: FinancialPlan | FinancialPlanV2) -> None:
    """Reject a plan whose arithmetic result is not in the requested unit."""

    try:
        requested = detect_target_unit(plan.question)
    except ValueError:
        question = normalize(plan.question)
        requested = (
            "ratio"
            if any(
                phrase in question
                for phrase in ("ty le", "ty so", "bao nhieu lan", "gap bao nhieu")
            )
            else None
        )
    if requested is None:
        return
    if plan.output_unit != requested:
        raise ValueError(
            f"q{getattr(plan, 'question_id', -1)}: output unit {plan.output_unit!r} does not match "
            f"requested unit {requested!r}"
        )


def validate_computational_units(plan: FinancialPlan | FinancialPlanV2) -> None:
    """Catch missing scale/ratio operations without pretending full unit algebra."""

    units = {fact.id: fact.unit for fact in plan.facts}
    for node in plan.nodes:
        inputs = [units[value] for value in node.inputs]
        if node.op == "literal":
            unit = "number"
        elif node.op in {"identity", "negate", "abs", "round"}:
            unit = inputs[0]
        elif node.op in {"add", "subtract"}:
            unit = inputs[0] if inputs[0] == inputs[1] else "unknown"
        elif node.op == "ratio_percent" or node.op == "percent_change":
            unit = "percent"
        elif node.op == "divide":
            unit = "ratio"
        elif node.op in {"count", "count_if"}:
            unit = "count"
        elif node.op in {"mean", "median", "min", "max", "sum"}:
            unit = "unknown"
        else:
            # A scale/multiply/vector/select operation may intentionally
            # change representation, so leave it to the explicit output-unit
            # and execution checks instead of guessing.
            unit = "unknown"
        units[node.id] = unit
    inferred = units[plan.output]
    if inferred != "unknown" and inferred != plan.output_unit:
        raise ValueError(
            f"q{getattr(plan, 'question_id', -1)}: DAG output unit {inferred!r} does not match "
            f"declared output unit {plan.output_unit!r}"
        )


def _scope_matches(requested: str, actual: str) -> bool:
    return (
        requested == "any"
        or requested == actual
        or actual in {"unspecified", "aggregated"}
    )


def _fact_transform(plan: FinancialPlan | FinancialPlanV2, fact: FactRequest, value: float) -> str:
    metric = normalize(fact.metric)
    if (
        plan.conventions.expense_sign == "absolute"
        and value < 0
        and any(phrase in metric for phrase in ("chi phi", "du phong", "hao mon"))
    ):
        return "absolute_expense"
    return "reported"


class GroundedBinder:
    """Resolve trusted ``row_ref`` values to an auditable source cell."""

    def __init__(
        self,
        database: Path,
        *,
        bank_tickers: Optional[set[str]] = None,
        minimum_distinct_gap: float = 2.0,
    ) -> None:
        self.connection = sqlite3.connect(str(database))
        self.connection.row_factory = sqlite3.Row
        self.bank_tickers = bank_tickers or set()
        self.minimum_distinct_gap = minimum_distinct_gap

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "GroundedBinder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _table(self, table_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            """
            SELECT t.*, d.ticker, d.year, d.report_scope
            FROM tables t JOIN documents d ON d.document_id = t.document_id
            WHERE t.table_id = ?
            """,
            (table_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown grounded table t{table_id}")
        return row

    def bind_fact(self, plan: FinancialPlan | FinancialPlanV2, fact: FactRequest) -> GroundedCell:
        if fact.row_ref is None:
            raise ValueError(f"fact {fact.id}: grounded plan is missing row_ref")
        match = ROW_REF_RE.fullmatch(fact.row_ref)
        if match is None:  # FactRequest also checks this; keep boundary explicit.
            raise ValueError(f"fact {fact.id}: invalid row_ref {fact.row_ref!r}")
        table_id = int(match.group("table_id"))
        row_index = int(match.group("row_index"))
        table = self._table(table_id)
        if table["ticker"] != fact.ticker or int(table["year"]) != fact.year:
            raise ValueError(
                f"fact {fact.id}: {fact.row_ref} belongs to "
                f"{table['ticker']} {table['year']}, not {fact.ticker} {fact.year}"
            )
        if not _scope_matches(fact.scope, table["report_scope"]):
            raise ValueError(
                f"fact {fact.id}: {fact.row_ref} scope {table['report_scope']!r} "
                f"does not match {fact.scope!r}"
            )

        grid = json.loads(table["grid_json"])
        if not 0 <= row_index < len(grid):
            raise ValueError(f"fact {fact.id}: {fact.row_ref} is outside the table")
        row = grid[row_index]
        numeric_cells = [
            (column_index, str(raw), number)
            for column_index, raw in enumerate(row)
            if (number := parse_number(str(raw))) is not None
        ]
        if not numeric_cells:
            raise ValueError(f"fact {fact.id}: {fact.row_ref} has no numeric cell")

        parsed = ParsedQuestion(
            id=getattr(plan, "question_id", -1),
            question=plan.question,
            ticker=fact.ticker,
            matched_company_alias=fact.ticker.lower(),
            year=fact.year,
            scope=fact.scope,
            period_kind="start" if fact.period == "start" else "end_or_flow",
            target_unit=fact.unit,
            metric_text=normalize(fact.metric),
            metric_tokens=tuple(normalize(fact.metric).split()),
        )
        label = contextual_row_label(grid, row_index) or fact.metric
        scored: list[dict[str, Any]] = []
        for value_rank, (column_index, raw, number) in enumerate(numeric_cells):
            header = column_header(
                grid,
                row_index,
                column_index,
                int(table["header_row_count"]),
            )
            temporal = period_score(parsed, header, value_rank, label)
            role = column_role_score(
                parsed, header, raw, number, column_index, numeric_cells
            )
            source_unit = infer_cell_source_unit(
                table["unit_code"],
                table["unit_text"],
                header,
                raw,
                label,
                fact.unit,
                "VND_1e6" if fact.ticker in self.bank_tickers else "VND_1",
            )
            if source_unit not in SOURCE_UNIT_SCALE or not units_compatible(
                source_unit, fact.unit
            ):
                continue
            preference = 0.0
            if fact.source_preference == "primary_statement":
                preference = 2.0 if table["table_kind"].startswith("primary_") else -2.0
            elif fact.source_preference == "note_table":
                preference = 2.0 if table["table_kind"] == "financial_notes" else -2.0
            link = schema_link_score(parsed, label, header)
            scored.append(
                {
                    "column_index": column_index,
                    "raw": raw,
                    "number": number,
                    "header": header,
                    "source_unit": source_unit,
                    "score": temporal + role + preference + link,
                }
            )
        if not scored:
            raise ValueError(f"fact {fact.id}: no unit-compatible numeric cell")
        scored.sort(key=lambda item: (-item["score"], item["column_index"]))
        selected = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        gap = selected["score"] - runner_up["score"] if runner_up else math.inf
        equivalent = bool(
            runner_up
            and selected["number"] == runner_up["number"]
            and gap < self.minimum_distinct_gap
        )
        if runner_up and gap < self.minimum_distinct_gap and not equivalent:
            raise ValueError(
                f"fact {fact.id}: ambiguous period column at {fact.row_ref}; "
                f"best gap={gap:.2f}"
            )

        source_scale = SOURCE_UNIT_SCALE[selected["source_unit"]]
        target_scale = _unit_scale(fact.unit)
        value = float(selected["number"]) * source_scale / target_scale
        transform = _fact_transform(plan, fact, value)
        if transform == "absolute_expense":
            value = abs(value)
        return GroundedCell(
            fact_id=fact.id,
            row_ref=fact.row_ref,
            ticker=fact.ticker,
            year=fact.year,
            requested_scope=fact.scope,
            document_scope=table["report_scope"],
            document_id=table["document_id"],
            evidence_key=table["evidence_key"],
            table_id=table_id,
            source_line_0=int(table["source_line_0"]),
            source_line_1=int(table["source_line_1"]),
            row_index=row_index,
            column_index=int(selected["column_index"]),
            row_text=label,
            column_text=selected["header"],
            raw_value=selected["raw"],
            parsed_value=float(selected["number"]),
            source_unit=selected["source_unit"],
            source_scale=source_scale,
            target_unit=fact.unit,
            target_scale=target_scale,
            value=value,
            transform=transform,
            column_score=round(float(selected["score"]), 4),
            runner_up_gap=round(float(gap), 4) if math.isfinite(gap) else math.inf,
            ambiguity="equivalent_duplicate" if equivalent else "none",
        )

    def bind_plan(self, plan: FinancialPlan | FinancialPlanV2) -> GroundedPlanResult:
        validate_requested_output_unit(plan)
        validate_computational_units(plan)
        bindings = tuple(self.bind_fact(plan, fact) for fact in plan.facts)
        source_cells = [
            (binding.table_id, binding.row_index, binding.column_index)
            for binding in bindings
        ]
        if len(source_cells) != len(set(source_cells)):
            raise ValueError(
                f"q{getattr(plan, 'question_id', -1)}: distinct facts resolve to the same source cell"
            )
        by_fact = {binding.fact_id: binding for binding in bindings}
        table_variables: dict[int, str] = {}
        expressions: dict[str, str] = {}
        for binding in bindings:
            variable = table_variables.setdefault(
                binding.table_id, f"df{len(table_variables) + 1}"
            )
            expression = pandas_query_for(
                binding.raw_value,
                binding.row_index,
                binding.column_index,
                binding.source_scale,
                binding.target_scale,
            ).replace("df1", variable, 1)
            if binding.transform == "absolute_expense":
                expression = f"abs({expression})"
            expressions[binding.fact_id] = expression
        values = {binding.fact_id: binding.value for binding in bindings}
        answer = evaluate_plan(plan, values)
        query = compile_pandas(plan, expressions)
        return GroundedPlanResult(plan, bindings, answer, query)

    def grid_for(self, table_id: int) -> list[list[str]]:
        return json.loads(self._table(table_id)["grid_json"])


def load_grounded_plans(path: Path) -> list[FinancialPlan | FinancialPlanV2]:
    plans: list[FinancialPlan | FinancialPlanV2] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("candidate", row)
            if row.get("status", "VALID") != "VALID":
                continue
            plan = FinancialPlan.from_dict(payload)
            if getattr(plan, "question_id", -1) in seen and getattr(plan, "question_id", -1) != -1:
                raise ValueError(f"{path}:{line_number}: duplicate q{plan.question_id}")
            seen.add(getattr(plan, "question_id", -1))
            plans.append(plan)
    return plans


def bind_plans(
    plans: Iterable[FinancialPlan | FinancialPlanV2], database: Path
) -> tuple[list[GroundedPlanResult], list[dict[str, Any]]]:
    results: list[GroundedPlanResult] = []
    audit: list[dict[str, Any]] = []
    with GroundedBinder(database) as binder:
        for plan in plans:
            try:
                result = binder.bind_plan(plan)
            except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                audit.append(
                    {
                        "question_id": getattr(plan, "question_id", -1),
                        "status": "REJECTED",
                        "reason": str(error),
                    }
                )
                continue
            results.append(result)
            audit.append(result.audit_record())
    return results, audit


def _csv_bytes(grid: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    width = max((len(row) for row in grid), default=0)
    if width == 0:
        raise ValueError("Cannot export an empty evidence table")
    writer.writerow([f"col_{index}" for index in range(width)])
    for row in grid:
        writer.writerow([*row, *("" for _ in range(width - len(row)))])
    return output.getvalue().encode("utf-8")


def _submission_item(
    result: GroundedPlanResult,
    grids: Mapping[int, list[list[str]]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    table_variables: dict[int, str] = {}
    evidence: list[dict[str, str]] = []
    evidence_bytes: dict[str, bytes] = {}
    documents: list[str] = []
    tables: list[str] = []
    for binding in result.bindings:
        if binding.document_id not in documents:
            documents.append(binding.document_id)
        table_key = f"{binding.document_id}|{binding.source_line_1}"
        if table_key not in tables:
            tables.append(table_key)
        if binding.table_id in table_variables:
            continue
        variable = f"df{len(table_variables) + 1}"
        table_variables[binding.table_id] = variable
        path = (
            f"data/grounded_q{getattr(result.plan, 'question_id', 0):04d}_"
            f"t{binding.table_id}_evidence.csv"
        )
        evidence.append({"variable": variable, "csv_path": path})
        evidence_bytes[path] = _csv_bytes(grids[binding.table_id])
    item = {
        "id": getattr(result.plan, "question_id", -1),
        "question": result.plan.question,
        "answer": float(result.answer),
        "relevant_docs": documents,
        "relevant_tables": tables,
        "evidence": evidence,
        "pandas_query": result.pandas_query,
    }
    return item, evidence_bytes


def build_candidate_archive(
    *,
    baseline_zip: Path,
    plans: Iterable[FinancialPlan | FinancialPlanV2],
    database: Path,
    output_zip: Path,
) -> dict[str, Any]:
    """Replace only successfully bound questions in a complete baseline ZIP."""

    if output_zip.exists():
        raise FileExistsError(output_zip)
    baseline = SubmissionArchive.load(baseline_zip)
    results, audit = bind_plans(plans, database)
    if not results:
        raise ValueError("No grounded plan passed benchmark_locked binding")
    replacements: dict[int, dict[str, Any]] = {}
    new_evidence: dict[str, bytes] = {}
    with GroundedBinder(database) as binder:
        for result in results:
            grids = {
                binding.table_id: binder.grid_for(binding.table_id)
                for binding in result.bindings
            }
            item, evidence = _submission_item(result, grids)
            replacements[getattr(result.plan, "question_id", -1)] = item
            new_evidence.update(evidence)
    unknown = set(replacements) - set(baseline.by_id)
    if unknown:
        raise ValueError(f"Baseline is missing grounded IDs {sorted(unknown)}")
    items = [replacements.get(item["id"], item) for item in baseline.items]
    required_paths = {
        evidence["csv_path"] for item in items for evidence in item["evidence"]
    }
    evidence_bytes = {
        path: new_evidence.get(path, baseline.members.get(path, b""))
        for path in required_paths
    }
    missing = [path for path, content in evidence_bytes.items() if not content]
    if missing:
        raise ValueError(f"Missing evidence bytes for {missing[0]}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    submission = (json.dumps(items, ensure_ascii=False, indent=2) + "\n").encode()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", submission)
        for path in sorted(evidence_bytes):
            archive.writestr(path, evidence_bytes[path])
    SubmissionArchive.load(output_zip)
    return {
        "candidate_zip": str(output_zip),
        "replaced_ids": sorted(replacements),
        "rejected_ids": sorted(
            row["question_id"] for row in audit if row["status"] == "REJECTED"
        ),
        "audit": audit,
    }


def build_retrieval_union_archive(
    *,
    baseline_zip: Path,
    plans: Iterable[FinancialPlan | FinancialPlanV2],
    database: Path,
    output_zip: Path,
) -> dict[str, Any]:
    """Add same-answer grounded provenance without changing execution fields."""

    if output_zip.exists():
        raise FileExistsError(output_zip)
    baseline = SubmissionArchive.load(baseline_zip)
    results, audit = bind_plans(plans, database)
    replacements: dict[int, dict[str, Any]] = {}
    skipped_changed_answer: list[int] = []
    for result in results:
        question_id = getattr(result.plan, "question_id", -1)
        original = baseline.by_id.get(question_id)
        if original is None:
            raise ValueError(f"Baseline is missing q{question_id}")
        if not math.isclose(
            float(result.answer),
            float(original["answer"]),
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            skipped_changed_answer.append(question_id)
            continue
        documents = list(original["relevant_docs"])
        tables = list(original["relevant_tables"])
        for binding in result.bindings:
            if binding.document_id not in documents:
                documents.append(binding.document_id)
            table_key = f"{binding.document_id}|{binding.source_line_1}"
            if table_key not in tables:
                tables.append(table_key)
        if (
            documents != original["relevant_docs"]
            or tables != original["relevant_tables"]
        ):
            replacements[question_id] = {
                **original,
                "relevant_docs": documents,
                "relevant_tables": tables,
            }
    if not replacements:
        raise ValueError("No same-answer plan adds retrieval provenance")
    items = [replacements.get(item["id"], item) for item in baseline.items]
    required_paths = {
        evidence["csv_path"] for item in items for evidence in item["evidence"]
    }
    evidence_bytes = {path: baseline.members[path] for path in required_paths}
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    submission = (json.dumps(items, ensure_ascii=False, indent=2) + "\n").encode()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", submission)
        for path in sorted(evidence_bytes):
            archive.writestr(path, evidence_bytes[path])
    SubmissionArchive.load(output_zip)
    return {
        "candidate_zip": str(output_zip),
        "retrieval_union_ids": sorted(replacements),
        "skipped_changed_answer_ids": sorted(skipped_changed_answer),
        "rejected_ids": sorted(
            row["question_id"] for row in audit if row["status"] == "REJECTED"
        ),
        "added_docs": sum(
            len(replacements[question_id]["relevant_docs"])
            - len(baseline.by_id[question_id]["relevant_docs"])
            for question_id in replacements
        ),
        "added_tables": sum(
            len(replacements[question_id]["relevant_tables"])
            - len(baseline.by_id[question_id]["relevant_tables"])
            for question_id in replacements
        ),
        "audit": audit,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--candidate-zip", type=Path)
    parser.add_argument("--retrieval-union-only", action="store_true")
    args = parser.parse_args()
    plans = load_grounded_plans(args.plans)
    if bool(args.baseline_zip) != bool(args.candidate_zip):
        parser.error("--baseline-zip and --candidate-zip must be provided together")
    if args.baseline_zip:
        if args.retrieval_union_only:
            summary = build_retrieval_union_archive(
                baseline_zip=args.baseline_zip,
                plans=plans,
                database=args.database,
                output_zip=args.candidate_zip,
            )
        else:
            summary = build_candidate_archive(
                baseline_zip=args.baseline_zip,
                plans=plans,
                database=args.database,
                output_zip=args.candidate_zip,
            )
        _write_jsonl(args.audit, summary.pop("audit"))
    else:
        _results, audit = bind_plans(plans, args.database)
        _write_jsonl(args.audit, audit)
        summary = {
            "bound": sum(row["status"] == "BOUND" for row in audit),
            "rejected": sum(row["status"] == "REJECTED" for row in audit),
        }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
