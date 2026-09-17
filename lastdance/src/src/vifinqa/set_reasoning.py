"""Generic, auditable execution engine for cross-entity and set questions.

The question-specific modules only describe facts and operators.  This module
retrieves every fact through the existing L1 layer, keeps its source cell, and
turns the resulting plan into a replayable pandas expression and evidence CSVs.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from .l1_fact_layer import (
    FactCandidate,
    FactRetriever,
    ManualOverride,
    ParsedQuestion,
    load_companies,
    normalize,
    pandas_query_for,
    write_source_table_csv,
)


@dataclass(frozen=True)
class FactSpec:
    name: str
    ticker: str
    year: int
    metric: str
    target_unit: str = "VND_1"
    scope: str = "consolidated"
    period_kind: str = "end_or_flow"


@dataclass(frozen=True)
class ItemSpec:
    key: str
    expression: str
    facts: tuple[FactSpec, ...]
    selector_expression: Optional[str] = None


@dataclass(frozen=True)
class SetSpec:
    question_id: int
    operation: str
    items: tuple[ItemSpec, ...]
    threshold: Optional[float] = None


@dataclass(frozen=True)
class SetOverride:
    question_id: int
    item: int
    component: str
    year: int
    document_id: str
    source_line_1: int
    row_index: int
    column_index: int
    raw_value: str
    review_note: str


def fact(name: str, ticker: str, year: int, metric: str,
         unit: str = "VND_1", scope: str = "consolidated",
         period: str = "end_or_flow") -> FactSpec:
    return FactSpec(name, ticker, year, metric, unit, scope, period)


def direct_item(key: object, ticker: str, year: int, metric: str,
                unit: str = "VND_1", scope: str = "consolidated",
                period: str = "end_or_flow") -> ItemSpec:
    return ItemSpec(str(key), "a", (fact("a", ticker, year, metric, unit, scope, period),))


def ratio_item(key: object, ticker: str, year: int, numerator: str,
               denominator: str, scope: str = "consolidated",
               period: str = "end_or_flow", multiplier: float = 100.0) -> ItemSpec:
    return ItemSpec(
        str(key),
        f"a / b * {multiplier:.1f}",
        (
            fact("a", ticker, year, numerator, "VND_1", scope, period),
            fact("b", ticker, year, denominator, "VND_1", scope, period),
        ),
    )


def absolute_ratio_item(key: object, ticker: str, year: int, numerator: str,
                        denominator: str, scope: str = "consolidated",
                        period: str = "end_or_flow",
                        multiplier: float = 100.0) -> ItemSpec:
    """Ratio of magnitudes for accounting rows whose display sign may vary."""
    return ItemSpec(
        str(key),
        f"abs(a) / abs(b) * {multiplier:.1f}",
        (
            fact("a", ticker, year, numerator, "VND_1", scope, period),
            fact("b", ticker, year, denominator, "VND_1", scope, period),
        ),
    )


def load_questions(path: Path, first_id: int, last_id: int) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = int(row["id"])
            if first_id <= question_id <= last_id:
                rows[question_id] = row
    if sorted(rows) != list(range(first_id, last_id + 1)):
        raise ValueError(f"Expected complete question range {first_id}..{last_id}")
    return rows


def validate_specs(specs: dict[int, SetSpec], first_id: int, last_id: int) -> None:
    if sorted(specs) != list(range(first_id, last_id + 1)):
        missing = sorted(set(range(first_id, last_id + 1)) - set(specs))
        extra = sorted(set(specs) - set(range(first_id, last_id + 1)))
        raise ValueError(f"Invalid spec coverage; missing={missing}, extra={extra}")
    allowed = {
        "difference", "absolute_difference", "sum", "mean", "max_value",
        "argmax_key", "count_gt", "count_positive", "count_nonzero",
        "aggregate_ratio", "count_negative", "select_argmax_answer",
        "select_argmin_answer", "select_filter_gt_answer",
        "select_filter_ge_answer_sum", "select_max_then_argmax_key",
        "select_filter_positive_argmax_key",
    }
    for question_id, spec in specs.items():
        if spec.question_id != question_id:
            raise ValueError(f"q{question_id}: mismatched spec id")
        if spec.operation not in allowed:
            raise ValueError(f"q{question_id}: unsupported operation {spec.operation}")
        if not spec.items or any(not item.facts for item in spec.items):
            raise ValueError(f"q{question_id}: empty item/fact list")
        if spec.operation == "difference" and len(spec.items) != 2:
            raise ValueError(f"q{question_id}: difference needs two items")
        if spec.operation == "absolute_difference" and len(spec.items) != 2:
            raise ValueError(f"q{question_id}: absolute difference needs two items")
        if spec.operation == "aggregate_ratio" and any(
            set(f.name for f in item.facts) != {"a", "b"} for item in spec.items
        ):
            raise ValueError(f"q{question_id}: aggregate ratio needs a and b per item")
        if spec.operation == "count_gt" and spec.threshold is None:
            raise ValueError(f"q{question_id}: count_gt needs a threshold")
        if spec.operation.startswith("select_") and any(
            item.selector_expression is None for item in spec.items
        ):
            raise ValueError(f"q{question_id}: selector operation needs selector expressions")
        if spec.operation in {"select_filter_gt_answer", "select_filter_ge_answer_sum"} \
                and spec.threshold is None:
            raise ValueError(f"q{question_id}: filtered selector needs a threshold")


def load_overrides(path: Optional[Path]) -> dict[tuple[int, int, str], SetOverride]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "question_id", "item", "component", "year", "document_id",
        "source_line_1", "row_index", "column_index", "raw_value", "review_note",
    }
    if rows and set(rows[0]) != expected:
        raise ValueError(f"Override columns must be {sorted(expected)}")
    result = {}
    for row in rows:
        override = SetOverride(
            question_id=int(row["question_id"]), item=int(row["item"]),
            component=row["component"], year=int(row["year"]),
            document_id=row["document_id"], source_line_1=int(row["source_line_1"]),
            row_index=int(row["row_index"]), column_index=int(row["column_index"]),
            raw_value=row["raw_value"], review_note=row["review_note"],
        )
        key = (override.question_id, override.item, override.component)
        if key in result:
            raise ValueError(f"Duplicate override {key}")
        result[key] = override
    return result


def _parsed(question: dict, spec: FactSpec) -> ParsedQuestion:
    tokens = tuple(dict.fromkeys(normalize(spec.metric).split()))
    return ParsedQuestion(
        id=int(question["id"]), question=question["question"], ticker=spec.ticker,
        matched_company_alias=spec.ticker.lower(), year=spec.year, scope=spec.scope,
        period_kind=spec.period_kind, target_unit=spec.target_unit,
        metric_text=" ".join(tokens), metric_tokens=tokens,
    )


def _manual_override(override: SetOverride) -> ManualOverride:
    return ManualOverride(
        question_id=override.question_id, document_id=override.document_id,
        source_line_1=override.source_line_1, row_index=override.row_index,
        column_index=override.column_index, raw_value=override.raw_value,
        review_note=override.review_note,
    )


def _record(candidate: FactCandidate, item_index: int, name: str, rank: int) -> dict:
    return {**asdict(candidate), "item": item_index, "component": name,
            "candidate_rank": rank}


def run_candidates(specs: dict[int, SetSpec], questions_path: Path, database: Path,
                   output_dir: Path, overrides_path: Optional[Path]) -> None:
    first_id, last_id = min(specs), max(specs)
    validate_specs(specs, first_id, last_id)
    questions = load_questions(questions_path, first_id, last_id)
    companies = load_companies(Path("ViFinQA/code_stock.csv"))
    bank_tickers = {
        company.ticker for company in companies if "ngan hang" in normalize(company.name)
    }
    overrides = load_overrides(overrides_path)
    retriever = FactRetriever(database, bank_tickers=bank_tickers)
    candidates_output, plans = [], []
    try:
        for offset, question_id in enumerate(sorted(specs), 1):
            spec = specs[question_id]
            plan = {
                "question": questions[question_id], "operation": spec.operation,
                "threshold": spec.threshold, "items": [],
            }
            for item_index, item_spec in enumerate(spec.items, 1):
                item_record = {
                    "key": item_spec.key,
                    "expression": item_spec.expression,
                    "selector_expression": item_spec.selector_expression,
                    "facts": {},
                }
                for component in item_spec.facts:
                    parsed = _parsed(questions[question_id], component)
                    candidates = retriever.retrieve(parsed, limit=5)
                    override = overrides.get((question_id, item_index, component.name))
                    if override is not None:
                        if override.year != component.year:
                            raise ValueError(f"q{question_id} item {item_index}: wrong year")
                        reviewed = retriever.retrieve_reviewed(parsed, _manual_override(override))
                        candidates = [reviewed, *[
                            candidate for candidate in candidates
                            if (candidate.document_id, candidate.source_line_1,
                                candidate.row_index, candidate.column_index)
                            != (reviewed.document_id, reviewed.source_line_1,
                                reviewed.row_index, reviewed.column_index)
                        ][:4]]
                    if not candidates:
                        raise ValueError(
                            f"q{question_id} item {item_index} {component.name}: no candidate"
                        )
                    records = [
                        _record(candidate, item_index, component.name, rank)
                        for rank, candidate in enumerate(candidates, 1)
                    ]
                    candidates_output.extend({
                        **record, "question": questions[question_id]["question"],
                        "component_metric": component.metric,
                    } for record in records)
                    item_record["facts"][component.name] = records[0]
                plan["items"].append(item_record)
            plans.append(plan)
            if offset % 20 == 0:
                print(f"retrieved {offset}/{len(specs)} set questions", flush=True)
    finally:
        retriever.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates_output),
        encoding="utf-8",
    )
    (output_dir / "plans.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans),
        encoding="utf-8",
    )
    summary = {
        "questions": len(plans),
        "items": sum(len(plan["items"]) for plan in plans),
        "components": sum(len(item["facts"]) for plan in plans for item in plan["items"]),
        "candidate_facts": len(candidates_output),
        "manual_components": sum(
            fact["selection_source"] == "manual" for plan in plans
            for item in plan["items"] for fact in item["facts"].values()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def load_plans(path: Path, specs: dict[int, SetSpec]) -> list[dict]:
    plans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [int(plan["question"]["id"]) for plan in plans] != sorted(specs):
        raise ValueError("Plan file does not cover the complete spec registry")
    return plans


def _cell_query(fact_record: dict, variable: str) -> str:
    return pandas_query_for(
        fact_record["raw_value"], int(fact_record["row_index"]),
        int(fact_record["column_index"]), float(fact_record["source_scale"]),
        float(fact_record["target_scale"]),
    ).replace("df1", variable, 1)


def _column_sum_query(fact_record: dict, variable: str) -> str:
    """Sum the selected source column without storing a precomputed total.

    This operator is reserved for source notes that list transactions without
    a printed grand-total row.  Non-numeric labels and dashes coerce to NaN and
    then zero; every numeric cell in the cited column contributes at replay.
    """

    column = int(fact_record["column_index"])
    return (
        f"pd.to_numeric({variable}.iloc[:, {column}].astype(str)"
        '.str.strip().str.replace("(", "-", regex=False)'
        '.str.replace(")", "", regex=False)'
        '.str.replace(".", "", regex=False), '
        'errors="coerce").fillna(0.0).sum()'
    )


def _expression_query(item: dict, variables: dict[str, str], expression: str) -> str:
    for name in sorted(item["facts"], key=len, reverse=True):
        expression = re.sub(
            rf"\bsumcol\({re.escape(name)}\)",
            lambda _match: f"({_column_sum_query(item['facts'][name], variables[name])})",
            expression,
        )
    for name in sorted(item["facts"], key=len, reverse=True):
        expression = re.sub(
            rf"\b{re.escape(name)}\b",
            lambda _match: f"({_cell_query(item['facts'][name], variables[name])})",
            expression,
        )
    return expression


def _item_query(item: dict, variables: dict[str, str]) -> str:
    return _expression_query(item, variables, item["expression"])


def plan_query(plan: dict) -> str:
    item_queries = []
    selector_queries = []
    variable_number = 0
    for item in plan["items"]:
        variables = {}
        for name in item["facts"]:
            variable_number += 1
            variables[name] = f"df{variable_number}"
        item_queries.append(_item_query(item, variables))
        if item.get("selector_expression") is not None:
            selector_queries.append(
                _expression_query(item, variables, item["selector_expression"])
            )
    operation = plan["operation"]
    if operation == "difference":
        return f"({item_queries[0]}) - ({item_queries[1]})"
    if operation == "absolute_difference":
        return f"abs(({item_queries[0]}) - ({item_queries[1]}))"
    series = f"pd.Series([{', '.join(f'({query})' for query in item_queries)}])"
    selector_series = (
        f"pd.Series([{', '.join(f'({query})' for query in selector_queries)}])"
        if selector_queries else None
    )
    if operation == "select_argmax_answer":
        return f"float({series}.iloc[int({selector_series}.idxmax())])"
    if operation == "select_argmin_answer":
        return f"float({series}.iloc[int({selector_series}.idxmin())])"
    if operation == "select_filter_gt_answer":
        return f"float({series}[{selector_series} > {float(plan['threshold']):.12g}].iloc[0])"
    if operation == "select_filter_ge_answer_sum":
        return f"float({series}[{selector_series} >= {float(plan['threshold']):.12g}].sum())"
    if operation == "select_max_then_argmax_key":
        keys = ", ".join(repr(item["key"]) for item in plan["items"])
        return (
            f"int(pd.Series([{keys}]).iloc[int("
            f"{series}.where({selector_series} == {selector_series}.max()).idxmax())])"
        )
    if operation == "select_filter_positive_argmax_key":
        keys = ", ".join(repr(item["key"]) for item in plan["items"])
        return (
            f"int(pd.Series([{keys}]).iloc[int("
            f"{series}.where({selector_series} > 0.0).idxmax())])"
        )
    if operation == "sum":
        return f"float({series}.sum())"
    if operation == "mean":
        return f"float({series}.mean())"
    if operation == "max_value":
        return f"float({series}.max())"
    if operation == "argmax_key":
        keys = ", ".join(repr(item["key"]) for item in plan["items"])
        return f"int(pd.Series([{keys}]).iloc[int({series}.idxmax())])"
    if operation == "count_gt":
        return f"int(({series} > {float(plan['threshold']):.12g}).sum())"
    if operation == "count_positive":
        return f"int(({series} > 0.0).sum())"
    if operation == "count_nonzero":
        return f"int(({series} != 0.0).sum())"
    if operation == "count_negative":
        return f"int(({series} < 0.0).sum())"
    if operation == "aggregate_ratio":
        numerator_queries, denominator_queries = [], []
        variable_number = 0
        for item in plan["items"]:
            variables = {}
            for name in item["facts"]:
                variable_number += 1
                variables[name] = f"df{variable_number}"
            numerator_queries.append(
                f"abs({_cell_query(item['facts']['a'], variables['a'])})"
            )
            denominator_queries.append(
                f"abs({_cell_query(item['facts']['b'], variables['b'])})"
            )
        return (
            f"(pd.Series([{', '.join(f'({q})' for q in numerator_queries)}]).sum() / "
            f"pd.Series([{', '.join(f'({q})' for q in denominator_queries)}]).sum()) * 100.0"
        )
    raise ValueError(f"Unsupported operation {operation}")


def replay_query(query: str, frames: dict[str, object]):
    import pandas as pd
    return eval(  # noqa: S307 - query is regenerated and compared byte-for-byte
        query, {"__builtins__": {}, "pd": pd, "float": float, "int": int, "str": str,
                "abs": abs}, frames,
    )


def copy_base_submission(base_submission: Path, output_dir: Path) -> list[dict]:
    items = json.loads(base_submission.read_text(encoding="utf-8"))
    for item in items:
        for evidence in item["evidence"]:
            source = base_submission.parent / evidence["csv_path"]
            target = output_dir / evidence["csv_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return items


def build_submission(plans_path: Path, specs: dict[int, SetSpec], database: Path,
                     base_submission: Path, output_dir: Path, line_base: int = 1) -> Path:
    plans = load_plans(plans_path, specs)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    submission = copy_base_submission(base_submission, output_dir)
    connection = sqlite3.connect(str(database))
    try:
        for plan in plans:
            question_id = int(plan["question"]["id"])
            evidence, docs, tables = [], [], []
            variable_number = 0
            for item_index, item in enumerate(plan["items"], 1):
                for name, selected in item["facts"].items():
                    variable_number += 1
                    variable = f"df{variable_number}"
                    row = connection.execute(
                        "SELECT grid_json FROM tables WHERE table_id = ?",
                        (int(selected["table_id"]),),
                    ).fetchone()
                    if row is None:
                        raise ValueError(f"q{question_id}: missing table {selected['table_id']}")
                    relative = f"data/q{question_id:04d}_i{item_index}_{name}.csv"
                    write_source_table_csv(output_dir / relative, json.loads(row[0]))
                    evidence.append({"variable": variable, "csv_path": relative})
                    if selected["document_id"] not in docs:
                        docs.append(selected["document_id"])
                    line = selected["source_line_0"] if line_base == 0 else selected["source_line_1"]
                    table_key = f"{selected['document_id']}|{line}"
                    if table_key not in tables:
                        tables.append(table_key)
            query = plan_query(plan)
            frames = {}
            import pandas as pd
            for evidence_item in evidence:
                frames[evidence_item["variable"]] = pd.read_csv(output_dir / evidence_item["csv_path"])
            answer = replay_query(query, frames)
            if isinstance(answer, float) and not math.isfinite(answer):
                raise ValueError(f"q{question_id}: non-finite answer")
            submission.append({
                "id": question_id, "question": plan["question"]["question"],
                "answer": int(answer) if plan["operation"] in {
                    "argmax_key", "count_gt", "count_positive", "count_nonzero",
                    "count_negative", "select_max_then_argmax_key",
                    "select_filter_positive_argmax_key",
                } else float(answer),
                "relevant_docs": docs, "relevant_tables": tables,
                "evidence": evidence, "pandas_query": query,
            })
    finally:
        connection.close()
    submission.sort(key=lambda row: int(row["id"]))
    submission_path = output_dir / "submission.json"
    submission_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_submission(submission_path, plans_path, specs, base_submission)
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for path in sorted((output_dir / "data").glob("*.csv")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    validation = validate_submission(
        submission_path, plans_path, specs, base_submission, zip_path
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"submission": str(submission_path), "zip": str(zip_path),
                      **validation}, ensure_ascii=False))
    return zip_path


def validate_submission(submission_path: Path, plans_path: Path,
                        specs: dict[int, SetSpec], base_submission: Path,
                        zip_path: Optional[Path] = None) -> dict:
    import pandas as pd
    items = json.loads(submission_path.read_text(encoding="utf-8"))
    plans = {int(plan["question"]["id"]): plan for plan in load_plans(plans_path, specs)}
    base_items = {int(row["id"]): row for row in json.loads(base_submission.read_text(encoding="utf-8"))}
    errors, ids, evidence_paths = [], set(), set()
    replayed = 0
    for item in items:
        question_id = int(item["id"])
        if question_id in ids:
            errors.append(f"q{question_id}: duplicate id")
        ids.add(question_id)
        if question_id in base_items and item != base_items[question_id]:
            errors.append(f"q{question_id}: base item changed")
        frames = {}
        for evidence in item["evidence"]:
            path = submission_path.parent / evidence["csv_path"]
            evidence_paths.add(evidence["csv_path"])
            if not path.is_file():
                errors.append(f"q{question_id}: missing {evidence['csv_path']}")
                continue
            frames[evidence["variable"]] = pd.read_csv(path)
        if question_id in plans:
            expected_query = plan_query(plans[question_id])
            if item["pandas_query"] != expected_query:
                errors.append(f"q{question_id}: query differs from canonical plan")
                continue
            try:
                actual = replay_query(expected_query, frames)
                if not math.isclose(float(actual), float(item["answer"]), rel_tol=1e-12, abs_tol=1e-9):
                    errors.append(f"q{question_id}: replay mismatch")
                else:
                    replayed += 1
            except Exception as error:  # noqa: BLE001 - collect all validation failures
                errors.append(f"q{question_id}: replay failed: {error}")
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        expected = {"submission.json", *evidence_paths}
        if names != expected:
            errors.append("zip members do not exactly match submission evidence")
    result = {
        "valid": not errors, "items": len(items), "base_items": len(base_items),
        "new_items": len(plans), "replayed": replayed,
        "evidence_files": len(evidence_paths), "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result
