"""Build audited two-period facts and a combined executable L1+L2 submission."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sqlite3
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .l1_fact_layer import (
    FactCandidate,
    FactRetriever,
    ManualOverride,
    ParsedQuestion,
    TARGET_UNIT_SCALE,
    detect_target_unit,
    load_companies,
    metric_surface,
    normalize,
    pandas_query_for,
    parse_number,
    replay_generated_query,
    resolve_company,
    write_source_table_csv,
)


L2_FIRST_ID = 578
L2_LAST_ID = 655
YEAR_RE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)")

GROWTH_IDS = {
    579, 582, 583, 585, 586, 587, 596, 597, 598, 605, 606, 609, 610,
    612, 614, 615, 617, 620, 626, 629, 631, 632, 633, 635, 637, 638,
    639, 640, 644, 645, 647, 648, 650, 651, 655,
}

# These questions explicitly ask how much the later value is smaller than the
# earlier value. Their reported result is therefore earlier minus later.
REVERSE_DIFFERENCE_IDS = {599, 603, 634, 641}
REVERSE_GROWTH_IDS = {651}

L2_OPERATION_STOPWORDS = {
    "bao", "be", "bien", "chenh", "cung", "di", "do", "gap", "giua",
    "hieu", "hon", "ket", "lech", "lon", "muc", "nhieu", "phan", "qua",
    "sang", "so", "tang", "thap", "thay", "the", "tinh", "toc", "tram",
    "tru", "truong", "tu", "voi",
}


@dataclass(frozen=True)
class TemporalQuestion:
    id: int
    question: str
    ticker: str
    matched_company_alias: str
    earlier_year: int
    later_year: int
    scope: str
    period_kind: str
    operation: str
    reverse_result: bool
    output_unit: str
    component_target_unit: str
    metric_text: str
    metric_tokens: tuple[str, ...]


@dataclass(frozen=True)
class TemporalOverride:
    question_id: int
    component: str
    part: int
    year: int
    document_id: str
    source_line_1: int
    row_index: int
    column_index: int
    raw_value: str
    review_note: str


def load_l2_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if L2_FIRST_ID <= int(row["id"]) <= L2_LAST_ID:
                rows.append(row)
    expected = list(range(L2_FIRST_ID, L2_LAST_ID + 1))
    if [int(row["id"]) for row in rows] != expected:
        raise ValueError(f"Expected the complete L2 ID range {L2_FIRST_ID}..{L2_LAST_ID}")
    return rows


def temporal_metric_surface(
    question: str,
    company,
    matched_alias: str,
    later_year: int,
) -> tuple[str, tuple[str, ...]]:
    """Reuse the L1 owner cleanup, then remove two-period operator language."""

    _surface, tokens = metric_surface(question, company, matched_alias, later_year)
    cleaned = []
    for token in tokens:
        if token.isdigit() or token in L2_OPERATION_STOPWORDS:
            continue
        cleaned.append(token)
    deduplicated = tuple(dict.fromkeys(cleaned))
    if not deduplicated:
        raise ValueError("No metric tokens remain after temporal cleanup")
    return " ".join(deduplicated), deduplicated


def parse_temporal_question(row: dict, companies) -> TemporalQuestion:
    question = row["question"]
    question_id = int(row["id"])
    company, alias = resolve_company(question, companies)
    years = sorted({int(value) for value in YEAR_RE.findall(question)})
    if len(years) != 2:
        raise ValueError(f"Expected two years, found {years}")
    normalized = normalize(question)
    scope = (
        "separate"
        if "cong ty me" in normalized or "bao cao rieng" in normalized
        else "consolidated"
    )
    period_kind = "start" if "dau nam" in normalized or "dau ky" in normalized else "end_or_flow"
    output_unit = detect_target_unit(question)
    operation = "growth" if question_id in GROWTH_IDS else "difference"
    if operation == "growth" and output_unit != "percent":
        raise ValueError(f"q{question_id}: growth question must request percent")
    component_target_unit = "VND_1" if operation == "growth" else output_unit
    metric_text, metric_tokens = temporal_metric_surface(
        question, company, alias, years[1]
    )
    reverse_result = (
        question_id in REVERSE_GROWTH_IDS
        if operation == "growth"
        else question_id in REVERSE_DIFFERENCE_IDS
    )
    return TemporalQuestion(
        id=question_id,
        question=question,
        ticker=company.ticker,
        matched_company_alias=alias,
        earlier_year=years[0],
        later_year=years[1],
        scope=scope,
        period_kind=period_kind,
        operation=operation,
        reverse_result=reverse_result,
        output_unit=output_unit,
        component_target_unit=component_target_unit,
        metric_text=metric_text,
        metric_tokens=metric_tokens,
    )


def component_question(parsed: TemporalQuestion, component: str) -> ParsedQuestion:
    if component not in {"earlier", "later"}:
        raise ValueError(f"Unknown temporal component {component!r}")
    year = parsed.earlier_year if component == "earlier" else parsed.later_year
    return ParsedQuestion(
        id=parsed.id,
        question=parsed.question,
        ticker=parsed.ticker,
        matched_company_alias=parsed.matched_company_alias,
        year=year,
        scope=parsed.scope,
        period_kind=parsed.period_kind,
        target_unit=parsed.component_target_unit,
        metric_text=parsed.metric_text,
        metric_tokens=parsed.metric_tokens,
    )


def load_temporal_overrides(path: Optional[Path]) -> dict[tuple[int, str], list[TemporalOverride]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "question_id", "component", "part", "year", "document_id", "source_line_1",
        "row_index", "column_index", "raw_value", "review_note",
    }
    if rows and set(rows[0]) != required:
        raise ValueError(f"Temporal override columns must be {sorted(required)}")
    result = {}
    for row in rows:
        override = TemporalOverride(
            question_id=int(row["question_id"]),
            component=row["component"],
            part=int(row["part"]),
            year=int(row["year"]),
            document_id=row["document_id"],
            source_line_1=int(row["source_line_1"]),
            row_index=int(row["row_index"]),
            column_index=int(row["column_index"]),
            raw_value=row["raw_value"],
            review_note=row["review_note"],
        )
        key = (override.question_id, override.component)
        if override.component not in {"earlier", "later"}:
            raise ValueError(f"Invalid component {override.component!r}")
        result.setdefault(key, []).append(override)
    for key, overrides in result.items():
        overrides.sort(key=lambda item: item.part)
        parts = [item.part for item in overrides]
        if parts != list(range(1, len(parts) + 1)):
            raise ValueError(f"Temporal override {key} has invalid parts {parts}")
    return result


def as_l1_override(override: TemporalOverride) -> ManualOverride:
    return ManualOverride(
        question_id=override.question_id,
        document_id=override.document_id,
        source_line_1=override.source_line_1,
        row_index=override.row_index,
        column_index=override.column_index,
        raw_value=override.raw_value,
        review_note=override.review_note,
    )


def fact_record(candidate: FactCandidate, component: str, rank: int) -> dict:
    record = asdict(candidate)
    record["component"] = component
    record["candidate_rank"] = rank
    return record


def run_candidates(
    questions_path: Path,
    companies_path: Path,
    database: Path,
    output_dir: Path,
    overrides_path: Optional[Path],
) -> None:
    companies = load_companies(companies_path)
    parsed_rows = []
    errors = []
    for row in load_l2_questions(questions_path):
        try:
            parsed_rows.append(parse_temporal_question(row, companies))
        except ValueError as error:
            errors.append({"id": row["id"], "question": row["question"], "error": str(error)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parse_errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
        encoding="utf-8",
    )
    (output_dir / "parsed_questions.jsonl").write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in parsed_rows),
        encoding="utf-8",
    )
    bank_tickers = {
        company.ticker for company in companies if "ngan hang" in normalize(company.name)
    }
    overrides = load_temporal_overrides(overrides_path)
    retriever = FactRetriever(database, bank_tickers=bank_tickers)
    candidates_output = []
    pairs_output = []
    try:
        for index, parsed in enumerate(parsed_rows, 1):
            pair = {"question": asdict(parsed)}
            complete = True
            for component in ("earlier", "later"):
                component_parsed = component_question(parsed, component)
                candidates = retriever.retrieve(component_parsed, limit=5)
                component_overrides = overrides.get((parsed.id, component), [])
                aggregate_records = []
                if component_overrides:
                    reviewed_parts = []
                    for override in component_overrides:
                        if override.year != component_parsed.year:
                            raise ValueError(
                                f"q{parsed.id} {component}: override year {override.year} "
                                f"!= {component_parsed.year}"
                            )
                        reviewed_parts.append(
                            retriever.retrieve_reviewed(
                                component_parsed, as_l1_override(override)
                            )
                        )
                    if len(reviewed_parts) == 1:
                        reviewed = reviewed_parts[0]
                        candidates = [
                            reviewed,
                            *[
                                candidate for candidate in candidates
                                if not (
                                    candidate.document_id == reviewed.document_id
                                    and candidate.source_line_1 == reviewed.source_line_1
                                    and candidate.row_index == reviewed.row_index
                                    and candidate.column_index == reviewed.column_index
                                )
                            ][:4],
                        ]
                    else:
                        aggregate_records = [
                            fact_record(candidate, component, part)
                            for part, candidate in enumerate(reviewed_parts, 1)
                        ]
                records = [
                    fact_record(candidate, component, rank)
                    for rank, candidate in enumerate(candidates, 1)
                ]
                candidates_output.extend(
                    {**record, "question": parsed.question, "metric_text": parsed.metric_text}
                    for record in records
                )
                if aggregate_records:
                    pair[component] = {"aggregation": "sum", "parts": aggregate_records}
                elif records:
                    pair[component] = records[0]
                else:
                    complete = False
            if complete:
                pairs_output.append(pair)
            if index % 10 == 0:
                print(f"retrieved {index}/{len(parsed_rows)} temporal questions", flush=True)
    finally:
        retriever.close()
    (output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates_output),
        encoding="utf-8",
    )
    (output_dir / "top_pairs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs_output),
        encoding="utf-8",
    )
    summary = {
        "questions": L2_LAST_ID - L2_FIRST_ID + 1,
        "parsed": len(parsed_rows),
        "parse_errors": len(errors),
        "complete_pairs": len(pairs_output),
        "candidate_facts": len(candidates_output),
        "manual_components": sum(
            all(fact["selection_source"] == "manual" for fact in component_facts(pair[component]))
            for pair in pairs_output
            for component in ("earlier", "later")
        ),
        "manual_cells": sum(
            fact["selection_source"] == "manual"
            for pair in pairs_output
            for component in ("earlier", "later")
            for fact in component_facts(pair[component])
        ),
        "operations": dict(Counter(row.operation for row in parsed_rows)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def cell_query(fact: dict, variable: str) -> str:
    query = pandas_query_for(
        fact["raw_value"],
        int(fact["row_index"]),
        int(fact["column_index"]),
        float(fact["source_scale"]),
        float(fact["target_scale"]),
    )
    return query.replace("df1", variable, 1)


def component_facts(component: dict) -> list[dict]:
    return component["parts"] if component.get("aggregation") == "sum" else [component]


def component_query_and_value(component: dict, variable: str) -> tuple[str, float]:
    facts = component_facts(component)
    queries = [f"({cell_query(fact, variable)})" for fact in facts]
    values = [float(fact["answer"]) for fact in facts]
    if len(queries) == 1:
        return queries[0][1:-1], values[0]
    return " + ".join(queries), sum(values)


def temporal_query_and_answer(pair: dict) -> tuple[str, float]:
    question = pair["question"]
    earlier, later = pair["earlier"], pair["later"]
    earlier_query, earlier_value = component_query_and_value(earlier, "df1")
    later_query, later_value = component_query_and_value(later, "df2")
    if question["reverse_result"]:
        first_query, second_query = earlier_query, later_query
        first_value, second_value = earlier_value, later_value
    else:
        first_query, second_query = later_query, earlier_query
        first_value, second_value = later_value, earlier_value
    if question["operation"] == "difference":
        return f"({first_query}) - ({second_query})", first_value - second_value
    if earlier_value == 0:
        raise ValueError(f"q{question['id']}: growth denominator is zero")
    query = f"(({first_query}) - ({second_query})) / ({earlier_query}) * 100.0"
    return query, (first_value - second_value) / earlier_value * 100.0


def _copy_l1_submission(l1_submission: Path, output_dir: Path) -> list[dict]:
    items = json.loads(l1_submission.read_text(encoding="utf-8"))
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        for evidence in item["evidence"]:
            source = l1_submission.parent / evidence["csv_path"]
            target = output_dir / evidence["csv_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return items


def build_submission(
    top_pairs: Path,
    database: Path,
    l1_submission: Path,
    output_dir: Path,
    line_base: int,
) -> Path:
    with top_pairs.open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    if len(pairs) != L2_LAST_ID - L2_FIRST_ID + 1:
        raise ValueError("A full L2 build requires all 78 temporal pairs")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    submission = _copy_l1_submission(l1_submission, output_dir)
    connection = sqlite3.connect(str(database))
    try:
        for pair in pairs:
            question = pair["question"]
            question_id = int(question["id"])
            evidence = []
            relevant_docs = []
            relevant_tables = []
            for component, variable, suffix in (
                ("earlier", "df1", "a"), ("later", "df2", "b")
            ):
                facts = component_facts(pair[component])
                table_ids = {int(fact["table_id"]) for fact in facts}
                if len(table_ids) != 1:
                    raise ValueError(
                        f"q{question_id} {component}: aggregate parts must share one table"
                    )
                fact = facts[0]
                table = connection.execute(
                    "SELECT grid_json FROM tables WHERE table_id = ?",
                    (int(fact["table_id"]),),
                ).fetchone()
                if table is None:
                    raise ValueError(f"q{question_id}: missing table {fact['table_id']}")
                relative = f"data/q{question_id:04d}_{suffix}_evidence.csv"
                write_source_table_csv(output_dir / relative, json.loads(table[0]))
                evidence.append({"variable": variable, "csv_path": relative})
                if fact["document_id"] not in relevant_docs:
                    relevant_docs.append(fact["document_id"])
                source_line = int(
                    fact["source_line_0"] if line_base == 0 else fact["source_line_1"]
                )
                table_key = f"{fact['document_id']}|{source_line}"
                if table_key not in relevant_tables:
                    relevant_tables.append(table_key)
            query, answer = temporal_query_and_answer(pair)
            submission.append(
                {
                    "id": question_id,
                    "question": question["question"],
                    "answer": float(answer),
                    "relevant_docs": relevant_docs,
                    "relevant_tables": relevant_tables,
                    "evidence": evidence,
                    "pandas_query": query,
                }
            )
    finally:
        connection.close()
    submission.sort(key=lambda item: int(item["id"]))
    submission_path = output_dir / "submission.json"
    submission_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_combined_submission(submission_path, top_pairs)
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for path in sorted((output_dir / "data").glob("*.csv")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    validation = validate_combined_submission(submission_path, top_pairs, zip_path)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"submission": str(submission_path), "zip": str(zip_path), **validation}, ensure_ascii=False))
    return zip_path


def validate_combined_submission(
    submission_path: Path,
    top_pairs: Path,
    zip_path: Optional[Path] = None,
) -> dict:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("pandas is required for validation") from error
    items = json.loads(submission_path.read_text(encoding="utf-8"))
    pairs = {
        int(row["question"]["id"]): row
        for row in (json.loads(line) for line in top_pairs.read_text(encoding="utf-8").splitlines())
        if row
    }
    errors = []
    replayed = 0
    paths = set()
    ids = set()
    for item in items:
        question_id = int(item["id"])
        if question_id in ids:
            errors.append(f"q{question_id}: duplicate id")
        ids.add(question_id)
        frames = {}
        for evidence in item["evidence"]:
            relative = evidence["csv_path"]
            path = submission_path.parent / relative
            paths.add(relative)
            if not path.is_file():
                errors.append(f"q{question_id}: missing {relative}")
                continue
            frames[evidence["variable"]] = pd.read_csv(path)
        try:
            if question_id <= 361:
                actual = replay_generated_query(item["pandas_query"], frames)
            else:
                expected_query, actual = temporal_query_and_answer(pairs[question_id])
                if item["pandas_query"] != expected_query:
                    raise ValueError("temporal query differs from audited canonical query")
                # Re-evaluate the two canonical cells from the exported frames.
                earlier = pairs[question_id]["earlier"]
                later = pairs[question_id]["later"]
                earlier_actuals = [
                    replay_generated_query(
                        cell_query(fact, "df1"), {"df1": frames["df1"]}
                    )
                    for fact in component_facts(earlier)
                ]
                later_actuals = [
                    replay_generated_query(
                        cell_query(fact, "df2").replace("df2", "df1", 1),
                        {"df1": frames["df2"]},
                    )
                    for fact in component_facts(later)
                ]
                replay_pair = {**pairs[question_id]}
                if earlier.get("aggregation") == "sum":
                    replay_pair["earlier"] = {
                        **earlier,
                        "parts": [
                            {**fact, "answer": value}
                            for fact, value in zip(component_facts(earlier), earlier_actuals)
                        ],
                    }
                else:
                    replay_pair["earlier"] = {**earlier, "answer": earlier_actuals[0]}
                if later.get("aggregation") == "sum":
                    replay_pair["later"] = {
                        **later,
                        "parts": [
                            {**fact, "answer": value}
                            for fact, value in zip(component_facts(later), later_actuals)
                        ],
                    }
                else:
                    replay_pair["later"] = {**later, "answer": later_actuals[0]}
                _query, actual = temporal_query_and_answer(replay_pair)
            if not math.isclose(float(actual), float(item["answer"]), rel_tol=1e-12, abs_tol=1e-8):
                errors.append(f"q{question_id}: replay {actual} != {item['answer']}")
            else:
                replayed += 1
        except (KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(f"q{question_id}: replay failed: {error}")
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
        if members != {"submission.json", *paths}:
            errors.append("ZIP members differ from evidence references")
    result = {
        "valid": not errors,
        "items": len(items),
        "unique_ids": len(ids),
        "l2_items": sum(L2_FIRST_ID <= value <= L2_LAST_ID for value in ids),
        "replayed": replayed,
        "evidence_files": len(paths),
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--companies", type=Path, default=Path("ViFinQA/code_stock.csv"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l2-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l2_manual_overrides.csv"))
    submission = subparsers.add_parser("submission")
    submission.add_argument("--top-pairs", type=Path, default=Path("outputs/l2-facts/top_pairs.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--l1-submission", type=Path, default=Path("outputs/l1-submission/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-submission"))
    submission.add_argument("--line-base", choices=(0, 1), type=int, default=1)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--submission", type=Path, default=Path("outputs/l1-l2-submission/submission.json"))
    validate.add_argument("--top-pairs", type=Path, default=Path("outputs/l2-facts/top_pairs.jsonl"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(args.questions, args.companies, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        build_submission(args.top_pairs, args.database, args.l1_submission, args.output_dir, args.line_base)
    else:
        print(json.dumps(validate_combined_submission(args.submission, args.top_pairs, args.zip_path), ensure_ascii=False))


if __name__ == "__main__":
    main()
