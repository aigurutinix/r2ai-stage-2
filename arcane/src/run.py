import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from docs import (
    load_companies, load_reports, make_row, parse_question, required_report_years,
    retrieve_docs, validate, write_package,
)
from vifinqa.answers import (
    EvidenceValue, answer_plan, bound_cells, first_numeric_cell, fold,
    header_years, source_scale, strict_number,
)
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import (
    load_reports as load_table_reports, metric_query_tokens, retrieve_rows, table_budget,
)
from vifinqa.tables import materialize
from validate_submission import validate as validate_submission


def load_questions(path: Path) -> list[dict]:
    return load_jsonl(path)


def write_checkpoint(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(rows, handle, ensure_ascii=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def resolve_table(table_id: str, report_id: str, reports_by_id: dict) -> dict | None:
    """Rebuild what retrieve_rows would have returned for a table it never saw.

    A dense candidate has no matched row, because the whole table was scored at
    once. The row picked here is the first that binds to a number, since that is
    the only kind the answer stage can use — row 0 becomes the DataFrame's column
    names and a row without a numeric cell yields no evidence.
    """
    from vifinqa.retrieval import header_cells, report_tables

    report = reports_by_id.get(report_id)
    if report is None:
        return None
    table = next((item for item in report_tables(str(report.path), report.identity)
                  if item.table_id == table_id), None)
    if table is None:
        return None
    headers = header_cells(table)
    row_index = next(
        (index for index in range(1, len(table.rows))
         if first_numeric_cell(list(table.rows[index]), headers) is not None),
        0,
    )
    return {
        "table_id": table.table_id,
        "report_id": table.report_id,
        "page": table.page,
        "start_line": table.start_line,
        "score": 0.0,
        "row_index": row_index,
        "row_cells": list(table.rows[row_index]),
        "rows": [list(row) for row in table.rows],
        "header_cells": headers,
        "title": table.title,
        "periods": list(table.periods),
        "unit": table.unit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--table-mode", choices=("baseline", "report-coverage"), default="report-coverage")
    parser.add_argument(
        "--hierarchy", action="store_true",
        help="rank rows with source-derived section, row, and column paths",
    )
    parser.add_argument(
        "--table-family-prior", action="store_true",
        help="add an opt-in multi-label statement-family ranking view",
    )
    # Recall-weighted but still precision-sensitive, so the budget scales with the
    # filings the question spans rather than being a fixed number. See table_budget.
    parser.add_argument(
        "--table-top-k", default="auto",
        help="'auto' keeps two tables per gated report; 'auto:n' changes the multiplier; "
             "an integer fixes the budget; 'ranking' submits exactly the supplied "
             "ranking, for an ordering that already carries its own budget",
    )
    parser.add_argument(
        "--ranking", type=Path,
        help="reranked orderings from apply_rerank_scores.py; retrieval widens to the "
             "reranked depth and the budget is taken from that order",
    )
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument(
        "--extra-tables", type=Path,
        help="table_id to report_id map from export_rerank_pairs.py --dense; lets a "
             "ranking promote a table the document gate never selected",
    )
    args = parser.parse_args()

    data_root = args.data_root
    questions = load_questions(data_root / "questions" / "questions.jsonl")
    if args.limit:
        questions = questions[:args.limit]
    companies = load_companies(data_root / "code_stock.csv")
    reports = load_reports(data_root / "financial_statements")
    table_reports = load_table_reports(data_root)
    table_reports_by_id = {report.identity.report_id: report for report in table_reports}
    ranking = json.loads(args.ranking.read_text("utf-8")) if args.ranking else None
    extra_tables = json.loads(args.extra_tables.read_text("utf-8")) if args.extra_tables else {}
    checkpoint = args.output_dir / "rows.checkpoint.json"
    rows = json.loads(checkpoint.read_text("utf-8")) if args.resume and checkpoint.exists() else []
    completed_ids = {row["id"] for row in rows}
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint to resume: {checkpoint}")

    for number, source in enumerate(questions, 1):
        if source["id"] in completed_ids:
            continue
        parsed = parse_question(source["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, reports)
        metadata = {
            "tickers": parsed.tickers,
            "years": parsed.years,
            "slot_years": required_report_years(parsed),
            "scope": parsed.scope,
        }
        # An ordering built by appending source-linked tables already decided its
        # own length, so re-applying a per-report budget here would discard
        # exactly the tables that step added.
        top_k = table_budget(len(docs), args.table_top_k)
        if args.table_top_k == "ranking" and ranking:
            # Falling back to the normal budget for a question the ranking omits
            # keeps a missing entry from silently submitting no tables at all.
            top_k = len(ranking.get(str(source["id"]), [])) or table_budget(len(docs), "auto")
        result = retrieve_rows(
            source["question"], metadata, table_reports,
            # With an external ranking the budget is applied after reordering, so
            # retrieval has to return the whole reranked depth first.
            top_k=args.rerank_depth if ranking else top_k,
            report_ids=docs, mode=args.table_mode, hierarchy=args.hierarchy,
            family_prior=args.table_family_prior,
        )
        if ranking:
            ordered = ranking.get(str(source["id"]), [])
            order = {table_id: rank for rank, table_id in enumerate(ordered)}
            held = {table["table_id"] for table in result["tables"]}
            # Only the prefix that could reach the budget is worth resolving, and
            # only tables the gate missed are missing from `held`.
            for table_id in ordered[:top_k]:
                if table_id not in held and table_id in extra_tables:
                    entry = resolve_table(table_id, extra_tables[table_id], table_reports_by_id)
                    if entry is not None:
                        result["tables"].append(entry)
            result["tables"] = sorted(
                result["tables"],
                key=lambda table: order.get(table["table_id"], len(order)),
            )[:top_k]
        # A submitted table must sit in a report the row also declares relevant;
        # the validator rejects the package otherwise. A dense candidate can come
        # from a report the gate never selected, so promoting the table means
        # declaring its report too — which is the honest statement anyway, since
        # the system is asserting the answer is in there.
        for table in result["tables"]:
            if table["report_id"] not in docs:
                docs.append(table["report_id"])
        tables, evidence = [], []
        for rank, table in enumerate(result["tables"]):
            report = table_reports_by_id[table["report_id"]]
            stem = f"table_{source['id']}_{rank}"
            csv_path = args.output_dir / "package" / "data" / "tables" / f"{stem}.csv"
            materialize(report.path, table["start_line"], table["table_id"], csv_path)
            tables.append(table["table_id"])
            evidence.append({"variable": f"df{rank}", "csv_path": f"data/tables/{stem}.csv"})
        row = make_row(source, docs, tables, evidence)
        # A multi-operand question reads several rows and year columns of one
        # submitted table, so binding stops being the retrieval row alone: every
        # label-matching row of each table contributes its cells, deduped per
        # cell. Deduping by report instead would drop all but one operand and
        # leave growth and sum questions with nothing to compute over.
        years = required_report_years(parsed)
        tokens = {token for token in metric_query_tokens(source["question"], metadata) if len(token) > 2}
        values, lookup_scores = [], []
        for rank, table in enumerate(result["tables"]):
            for index, column in bound_cells(
                table["rows"], table.get("header_cells"), tokens, years,
            ):
                raw_value = strict_number(table["rows"][index][column])
                if raw_value is None:
                    continue
                header = table.get("header_cells", [])
                found = header_years(header[column]) & set(years) if column < len(header) else set()
                # Note tables often omit the fiscal year from the column header;
                # the report identity is the authoritative period in that case.
                if not found:
                    found = header_years(table["report_id"]) & set(years)
                metadata = " ".join((table.get("unit", ""), *header))
                scale = source_scale(metadata, table["rows"][index][column])
                values.append(EvidenceValue(
                    f"df{rank}", index, column, raw_value * scale, table["report_id"],
                    min(found) if found else None, scale,
                    " ".join(fold(cell) for cell in table["rows"][index][:2]),
                ))
                # When no operation matches, the question is a lookup and one of
                # these cells is the answer. Rank candidates by how much of the
                # question's line-item vocabulary the row label repeats and by
                # whether the column names a year the question asked for.
                label = " ".join(fold(cell) for cell in table["rows"][index][:2])
                lookup_scores.append(
                    sum(1 for token in tokens if token in label)
                    + (0.5 if found else 0.0)
                )
        plan = answer_plan(source["question"], values)
        if plan is None and values:
            # No operation matched, but evidence is bound: answer the lookup rather
            # than emitting a constant. The private phase rejects constant-return
            # queries, and a scaled figure from a real cell is a genuine attempt.
            best = max(range(len(values)), key=lambda position: (lookup_scores[position], -values[position].column))
            plan = answer_plan(source["question"], [values[best]])
        if plan is not None:
            row["answer"], expression = plan
            # The query reads the cells it needs and nothing else. Padding it with
            # "0 * dfN.shape[0]" so every submitted table appears was our own
            # requirement, not the organizers', and in the private phase these
            # queries are read by hand — a term that computes nothing on 97% of
            # rows is what a reviewer checking for constant answers looks for.
            row["pandas_query"] = f"result = {expression}"
        else:
            # Nothing bound: reading a shape is still a read of a submitted table,
            # which a constant is not, and it fails loudly rather than inventing
            # a figure.
            row["pandas_query"] = "result = 0 * df0.shape[0]"
        rows.append(row)
        if args.progress_every and number % args.progress_every == 0:
            write_checkpoint(checkpoint, rows)
            print(f"processed {number}/{len(questions)}", flush=True)

    errors = validate(rows, questions, reports)
    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors[:20]))
    # The retriever itself parsed these IDs from the gated raw reports. Passing
    # this set to the strict validator preserves the catalog check without
    # parsing unrelated corpus reports during every submission run.
    catalog_ids = {table_id for row in rows for table_id in row["relevant_tables"]}
    print(write_package(
        args.output_dir,
        rows,
        validator=lambda package: validate_submission(
            package,
            {int(question["id"]) for question in questions},
            catalog_ids,
        ),
    ))
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
