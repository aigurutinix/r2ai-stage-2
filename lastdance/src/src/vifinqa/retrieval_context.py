"""Build number-masked table/row context for FinancialPlan LLM proposals."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from rapidfuzz.fuzz import token_set_ratio

from .financial_ir import FinancialPlan
from .l1_fact_layer import load_companies, normalize, parse_number
from .semantic_parser import QuestionSpec


STOPWORDS = {
    "bao",
    "nhieu",
    "la",
    "cua",
    "cong",
    "ty",
    "co",
    "phan",
    "ngan",
    "hang",
    "tmcp",
    "nam",
    "trong",
    "vao",
    "theo",
    "don",
    "vi",
    "tinh",
    "den",
    "ngay",
    "bao",
    "cao",
    "duoc",
    "cho",
    "voi",
    "bao",
    "nhiêu",
    "mot",
    "so",
    "me",
    "tu",
    "tren",
    "viet",
}
PRIMARY_KINDS = {
    "primary_balance_sheet",
    "primary_income_statement",
    "primary_cash_flow",
}


def question_years(question: str) -> list[int]:
    years = {int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)}
    for start_text, end_text in re.findall(
        r"(?<!\d)(20\d{2})\s*[-–—]\s*(20\d{2})(?!\d)", question
    ):
        start, end = int(start_text), int(end_text)
        if start <= end and end - start <= 15:
            years.update(range(start, end + 1))
    if not years:
        raise ValueError("Question has no explicit financial-statement year")
    return sorted(years)


def question_scope(question: str) -> str:
    value = normalize(question)
    if "cong ty me" in value or "bao cao rieng" in value or "bctc rieng" in value:
        return "separate"
    if "hop nhat" in value:
        return "consolidated"
    return "consolidated"


def query_tokens(
    question: str,
    tickers: set[str],
    limit: int = 18,
    entity_aliases: Optional[set[str]] = None,
) -> list[str]:
    ticker_tokens = {normalize(ticker) for ticker in tickers}
    normalized_question = normalize(question)
    for alias in sorted(entity_aliases or set(), key=len, reverse=True):
        normalized_alias = normalize(alias)
        if len(normalized_alias) >= 4:
            normalized_question = re.sub(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", " ", normalized_question
            )
    result = []
    for token in normalized_question.split():
        if (
            token in STOPWORDS
            or token in ticker_tokens
            or token.isdigit()
            or len(token) < 2
            or token in result
        ):
            continue
        result.append(token)
    # Longer metric-bearing tokens are more discriminative in FTS BM25.
    return sorted(result, key=lambda token: (-len(token), result.index(token)))[:limit]


def fts_query(tokens: list[str] | tuple[str, ...], operator: str = "OR") -> str:
    if not tokens:
        raise ValueError("No usable FTS tokens")
    if operator not in {"OR", "AND"}:
        raise ValueError(f"Unsupported FTS operator {operator!r}")
    return f" {operator} ".join(
        f'"{token.replace(chr(34), "")}"' for token in tokens
    )


def mask_cell(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"20\d{2}", text):
        return f"<YEAR:{text}>"
    if parse_number(text) is not None:
        return "<NUM>"
    # Mask embedded long numeric values while preserving note labels such as
    # "Khoản mục 12" and period words.
    text = re.sub(r"(?<!\w)[(+-]?\d[\d., ]{3,}\)?%?(?!\w)", "<NUM>", text)
    return re.sub(r"\s+", " ", text).strip()


def row_label(row: list[Any]) -> str:
    labels = []
    for value in row:
        masked = mask_cell(value)
        if (
            masked
            and masked != "<NUM>"
            and not masked.startswith("<YEAR:")
            and re.search(r"[A-Za-zÀ-ỹĐđ]", masked)
        ):
            labels.append(masked)
    return " | ".join(dict.fromkeys(labels))


def row_score(question: str, tokens: list[str], label: str) -> float:
    normalized_label = normalize(label)
    label_tokens = set(normalized_label.split())
    overlap = sum(1.0 + math.log1p(len(token)) for token in tokens if token in label_tokens)
    fuzzy = token_set_ratio(normalize(question), normalized_label) / 100.0
    return overlap * 3.0 + fuzzy


def select_rows(
    grid: list[list[Any]],
    question: str,
    tokens: list[str],
    header_rows: int,
    limit: int,
    *,
    table_title: str = "",
    hint_groups: tuple[tuple[str, tuple[str, ...]], ...] = (),
    required_row_indices: frozenset[int] = frozenset(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    header_count = max(0, min(header_rows, len(grid), 3))
    headers = [
        {"row_index": index, "cells": [mask_cell(value) for value in grid[index]]}
        for index in range(header_count)
    ]
    candidates = []
    active_section: Optional[str] = None
    for index, row in enumerate(grid):
        if index < header_count:
            continue
        masked_cells = [mask_cell(value) for value in row]
        textual_cells = [
            value
            for value in masked_cells
            if value
            and value != "<NUM>"
            and not value.startswith("<YEAR:")
            and re.search(r"[A-Za-zÀ-ỹĐđ]", value)
        ]
        normalized_texts = {normalize(value) for value in textual_cells}
        hierarchy_role = "row"
        if len(textual_cells) >= 2 and len(normalized_texts) == 1:
            active_section = textual_cells[0]
            hierarchy_role = "section_header"
        label = row_label(row)
        if (
            label
            and active_section
            and hierarchy_role == "row"
            and normalize(active_section) not in normalize(label)
        ):
            label = f"{active_section} — {label}"
        if not label and "<NUM>" in masked_cells and active_section:
            label = f"TỔNG — {active_section}"
            hierarchy_role = "section_total"
        elif (
            not label
            and "<NUM>" in masked_cells
            and table_title
            and index == len(grid) - 1
        ):
            # Many note tables end with an unlabeled numeric grand total but
            # have no repeated section-header row.  The table title is the
            # only semantic label available for retrieval.
            label = f"TỔNG — {table_title}"
            hierarchy_role = "table_total"
        elif not label and "<NUM>" in masked_cells and index in required_row_indices:
            label = f"BASELINE ROW — {table_title or 'source table'}"
            hierarchy_role = "baseline_seed"
        if not label:
            continue
        candidates.append(
            {
                "row_index": index,
                "label": label,
                "masked_cells": masked_cells,
                "row_score": round(row_score(question, tokens, label), 4),
                "hierarchy_role": hierarchy_role,
                "baseline_seed": index in required_row_indices,
            }
        )
    if len(candidates) <= limit:
        selected = candidates
    else:
        mandatory_by_index = {
            row["row_index"]: row
            for row in candidates
            if row["hierarchy_role"] in {"section_total", "table_total"}
            or row["row_index"] in required_row_indices
        }
        # Aggregate question wording can hide the exact component labels.
        # Preserve the strongest rows for each decomposition hint before
        # filling the remaining budget using the combined question score.
        for hint, hint_tokens in hint_groups:
            ranked_for_hint = sorted(
                candidates,
                key=lambda row: (
                    -row_score(hint, list(hint_tokens), row["label"]),
                    row["row_index"],
                ),
            )
            for row in ranked_for_hint[:2]:
                mandatory_by_index[row["row_index"]] = row
        mandatory = list(mandatory_by_index.values())
        optional = [
            row for row in candidates if row["row_index"] not in mandatory_by_index
        ]
        selected = [
            *mandatory,
            *sorted(
                optional, key=lambda row: (-row["row_score"], row["row_index"])
            )[: max(0, limit - len(mandatory))],
        ]
        selected.sort(key=lambda row: row["row_index"])
    return headers, selected


class RetrievalContextBuilder:
    def __init__(self, database: Path):
        self.connection = sqlite3.connect(str(database))
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "RetrievalContextBuilder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _table_ids(
        self,
        *,
        ticker: str,
        year: int,
        scope: str,
        query: str,
        table_limit: int,
    ) -> list[int]:
        fts_rows = self.connection.execute(
            """
            SELECT t.table_id, bm25(table_fts) AS rank
            FROM table_fts
            JOIN tables t ON t.table_id = table_fts.table_id
            JOIN documents d ON d.document_id = t.document_id
            WHERE table_fts MATCH ? AND d.ticker = ? AND d.year = ?
              AND d.report_scope = ?
            ORDER BY bm25(table_fts), t.table_id
            LIMIT ?
            """,
            (query, ticker, year, scope, table_limit * 6),
        ).fetchall()
        primary_rows = self.connection.execute(
            """
            SELECT t.table_id
            FROM tables t JOIN documents d ON d.document_id = t.document_id
            WHERE d.ticker = ? AND d.year = ? AND d.report_scope = ?
              AND t.table_kind IN (
                'primary_balance_sheet', 'primary_income_statement',
                'primary_cash_flow'
              )
            ORDER BY t.table_id
            """,
            (ticker, year, scope),
        ).fetchall()
        return list(
            dict.fromkeys(
                [int(row["table_id"]) for row in fts_rows]
                + [int(row["table_id"]) for row in primary_rows]
            )
        )

    def _search_scopes(self, ticker: str, year: int, requested: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT report_scope FROM documents WHERE ticker = ? AND year = ?",
            (ticker, year),
        ).fetchall()
        available = sorted(str(row["report_scope"]) for row in rows)
        if requested in available:
            return [requested]
        if "unspecified" in available:
            return ["unspecified"]
        return available

    def build(
        self,
        *,
        question_id: int,
        question: str,
        tickers: set[str],
        table_limit: int = 10,
        rows_per_table: int = 16,
        entity_aliases: Optional[set[str]] = None,
        retrieval_hints: tuple[str, ...] = (),
        dimensions: Optional[set[tuple[str, int, str]]] = None,
        baseline_seed_refs: frozenset[tuple[int, int]] = frozenset(),
        question_spec: Optional[QuestionSpec] = None,
    ) -> dict[str, Any]:
        if question_spec is not None:
            question_year_values = [p.year for p in question_spec.periods if p.year is not None]
            scope = question_spec.scope
        else:
            question_year_values = question_years(question)
            scope = question_scope(question)
        if dimensions:
            search_dimensions = sorted(dimensions)
            years = sorted({year for _ticker, year, _scope in search_dimensions})
        else:
            years = question_year_values
            search_dimensions = [
                (ticker, year, scope)
                for ticker in sorted(tickers)
                for year in years
            ]
        question_tokens = query_tokens(
            question, tickers, entity_aliases=entity_aliases
        )
        hint_groups = tuple(
            (hint, tuple(tokens))
            for hint in retrieval_hints
            if (
                tokens := query_tokens(
                    hint, tickers, entity_aliases=entity_aliases
                )
            )
        )
        token_groups = [question_tokens, *(list(tokens) for _hint, tokens in hint_groups)]
        tokens = list(dict.fromkeys(token for group in token_groups for token in group))
        retrieval_text = " ".join((question, *retrieval_hints))
        table_ids = []
        mandatory_table_ids: set[int] = {
            table_id for table_id, _row_index in baseline_seed_refs
        }
        table_ids.extend(sorted(mandatory_table_ids))
        scope_resolution = []
        for ticker, year, requested_scope in search_dimensions:
            scopes = self._search_scopes(ticker, year, requested_scope)
            scope_resolution.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "requested": requested_scope,
                    "searched": scopes,
                }
            )
            for resolved_scope in scopes:
                for group_index, group in enumerate(token_groups):
                    group_table_ids = self._table_ids(
                            ticker=ticker,
                            year=year,
                            scope=resolved_scope,
                            query=fts_query(group),
                            table_limit=table_limit,
                    )
                    table_ids.extend(group_table_ids)
                    if group_index > 0:
                        precise_table_ids = self._table_ids(
                                ticker=ticker,
                                year=year,
                                scope=resolved_scope,
                                query=fts_query(group, "AND"),
                                table_limit=table_limit,
                        )
                        table_ids.extend(precise_table_ids)
                        mandatory_table_ids.update(precise_table_ids[:2])
        table_ids = list(dict.fromkeys(table_ids))
        tables = []
        for table_id in table_ids:
            row = self.connection.execute(
                """
                SELECT t.table_id, t.evidence_key, t.document_id, t.page_number,
                       t.source_line_1, t.header_row_count, t.unit_code,
                       t.unit_text, t.table_kind, t.title_hint, t.grid_json,
                       d.ticker, d.year, d.report_scope
                FROM tables t JOIN documents d ON d.document_id = t.document_id
                WHERE t.table_id = ?
                """,
                (table_id,),
            ).fetchone()
            if row is None:
                continue
            grid = json.loads(row["grid_json"])
            headers, candidate_rows = select_rows(
                grid,
                retrieval_text,
                tokens,
                int(row["header_row_count"]),
                rows_per_table,
                table_title=str(row["title_hint"]),
                hint_groups=hint_groups,
                required_row_indices=frozenset(
                    row_index
                    for seeded_table_id, row_index in baseline_seed_refs
                    if seeded_table_id == table_id
                ),
            )
            if not candidate_rows:
                continue
            title_score = row_score(retrieval_text, tokens, str(row["title_hint"]))
            best_row_score = max(
                (candidate["row_score"] for candidate in candidate_rows), default=0.0
            )
            hint_scores = [
                max(
                    (
                        row_score(hint, list(hint_tokens), candidate["label"])
                        for candidate in candidate_rows
                    ),
                    default=0.0,
                )
                + 0.5 * row_score(
                    hint, list(hint_tokens), str(row["title_hint"])
                )
                for hint, hint_tokens in hint_groups
            ]
            tables.append(
                {
                    "table_ref": f"t{table_id}",
                    "table_id": table_id,
                    "evidence_key": row["evidence_key"],
                    "document_id": row["document_id"],
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "scope": row["report_scope"],
                    "page_number": row["page_number"],
                    "source_line_1": row["source_line_1"],
                    "table_kind": row["table_kind"],
                    "unit": row["unit_code"],
                    "unit_text": mask_cell(row["unit_text"]),
                    "title": mask_cell(row["title_hint"]),
                    "retrieval_score": round(best_row_score + 0.5 * title_score, 4),
                    "hint_scores": [round(score, 4) for score in hint_scores],
                    "headers": headers,
                    "candidate_rows": [
                        {**candidate, "row_ref": f"t{table_id}r{candidate['row_index']}"}
                        for candidate in candidate_rows
                    ],
                }
            )
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for table in tables:
            grouped.setdefault((table["ticker"], int(table["year"])), []).append(table)
        tables = []
        for key in sorted(grouped):
            primary = [
                table for table in grouped[key] if table["table_kind"] in PRIMARY_KINDS
            ]
            other = [
                table for table in grouped[key] if table["table_kind"] not in PRIMARY_KINDS
            ]
            mandatory_other = [
                table for table in other if table["table_id"] in mandatory_table_ids
            ]
            mandatory_other_ids = {table["table_id"] for table in mandatory_other}
            for hint_index in range(len(hint_groups)):
                for table in sorted(
                    other,
                    key=lambda table: (
                        -table["hint_scores"][hint_index],
                        table["table_id"],
                    ),
                )[:5]:
                    if table["table_id"] not in mandatory_other_ids:
                        mandatory_other.append(table)
                        mandatory_other_ids.add(table["table_id"])
            optional_other = [
                table for table in other if table["table_id"] not in mandatory_other_ids
            ]
            selected_other = [
                *sorted(
                    mandatory_other,
                    key=lambda table: (-table["retrieval_score"], table["table_id"]),
                ),
                *sorted(
                    optional_other,
                key=lambda table: (-table["retrieval_score"], table["table_id"]),
                )[: max(0, table_limit - len(mandatory_other))],
            ]
            tables.extend(
                sorted(
                    [*primary, *selected_other],
                    key=lambda table: (-table["retrieval_score"], table["table_id"]),
                )
            )
        return {
            "question_id": question_id,
            "question": question,
            "tickers": sorted(tickers),
            "years": years,
            "question_years": question_year_values,
            "scope": scope,
            "scope_resolution": scope_resolution,
            "query_tokens": tokens,
            "retrieval_hints": list(retrieval_hints),
            "baseline_seed_refs": [
                f"t{table_id}r{row_index}"
                for table_id, row_index in sorted(baseline_seed_refs)
            ],
            "numeric_cells_masked": True,
            "tables": tables,
        }


def seed_refs_from_plan(seed_plan: dict[str, Any]) -> frozenset[tuple[int, int]]:
    """Read grounded rows from either scalar-formula or set-plan archives."""

    question_id = int(seed_plan["question"]["id"])
    if "components" in seed_plan:
        components = seed_plan["components"].values()
    elif "items" in seed_plan:
        components = (
            component
            for item in seed_plan["items"]
            for component in item["facts"].values()
        )
    else:
        raise ValueError(f"q{question_id}: seed plan needs components or items")
    return frozenset(
        (int(component["table_id"]), int(component["row_index"]))
        for component in components
    )


def export_contexts(
    plans_path: Path,
    database: Path,
    output_path: Path,
    selected_ids: Optional[set[int]] = None,
    companies_path: Path = Path("ViFinQA/code_stock.csv"),
    base_contexts_path: Optional[Path] = None,
    seed_plans_path: Optional[Path] = None,
) -> dict[str, Any]:
    teachers = [
        FinancialPlan.from_dict(json.loads(line))
        for line in plans_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if selected_ids is not None:
        teachers = [plan for plan in teachers if plan.question_id in selected_ids]
        if {plan.question_id for plan in teachers} != selected_ids:
            raise ValueError("Selected IDs are missing from plan file")
    company_aliases = {
        company.ticker: set(company.aliases) for company in load_companies(companies_path)
    }
    seed_refs_by_id: dict[int, frozenset[tuple[int, int]]] = {}
    if seed_plans_path is not None:
        for raw in seed_plans_path.read_text(encoding="utf-8").splitlines():
            if not raw:
                continue
            seed_plan = json.loads(raw)
            question_id = int(seed_plan["question"]["id"])
            seed_refs_by_id[question_id] = seed_refs_from_plan(seed_plan)
    with RetrievalContextBuilder(database) as builder:
        contexts = [
            builder.build(
                question_id=plan.question_id,
                question=plan.question,
                tickers={fact.ticker for fact in plan.facts},
                entity_aliases={
                    alias
                    for ticker in {fact.ticker for fact in plan.facts}
                    for alias in company_aliases.get(ticker, set())
                },
                retrieval_hints=tuple(dict.fromkeys(fact.metric for fact in plan.facts)),
                dimensions={
                    (fact.ticker, fact.year, fact.scope) for fact in plan.facts
                },
                baseline_seed_refs=seed_refs_by_id.get(plan.question_id, frozenset()),
            )
            for plan in teachers
        ]
    replaced = 0
    if base_contexts_path is not None:
        base_contexts = [
            json.loads(line)
            for line in base_contexts_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        by_id = {int(row["question_id"]): row for row in base_contexts}
        if len(by_id) != len(base_contexts):
            raise ValueError("Base retrieval contexts contain duplicate IDs")
        for context in contexts:
            question_id = int(context["question_id"])
            replaced += question_id in by_id
            by_id[question_id] = context
        contexts = [by_id[question_id] for question_id in sorted(by_id)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in contexts),
        encoding="utf-8",
    )
    return {
        "questions": len(contexts),
        "tables": sum(len(context["tables"]) for context in contexts),
        "rows": sum(
            len(table["candidate_rows"])
            for context in contexts
            for table in context["tables"]
        ),
        "replaced": replaced,
        "output": str(output_path),
    }


def proxy_recall(contexts_path: Path, legacy_plans_path: Path) -> dict[str, Any]:
    """Measure whether reviewed legacy source rows survive masked retrieval."""

    contexts = {
        int(row["question_id"]): row
        for row in (
            json.loads(line)
            for line in contexts_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    plans = [
        json.loads(line)
        for line in legacy_plans_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    components = []
    for plan in plans:
        question_id = int(plan["question"]["id"])
        if question_id not in contexts:
            continue
        if "components" in plan:
            named_components = plan["components"].items()
        elif "items" in plan:
            named_components = (
                (f"i{item_index:02d}_{name}", selected)
                for item_index, item in enumerate(plan["items"], 1)
                for name, selected in item["facts"].items()
            )
        else:
            raise ValueError(f"q{question_id}: legacy plan needs components or items")
        for name, selected in named_components:
            components.append((question_id, name, selected))
    table_hits = 0
    row_hits = 0
    misses = []
    for question_id, name, selected in components:
        context = contexts[question_id]
        table_id = int(selected["table_id"])
        row_index = int(selected["row_index"])
        table = next(
            (table for table in context["tables"] if int(table["table_id"]) == table_id),
            None,
        )
        if table is not None:
            table_hits += 1
        row_refs = (
            {row["row_ref"] for row in table["candidate_rows"]}
            if table is not None
            else set()
        )
        expected_ref = f"t{table_id}r{row_index}"
        if expected_ref in row_refs:
            row_hits += 1
        else:
            misses.append(
                {
                    "question_id": question_id,
                    "component": name,
                    "table_id": table_id,
                    "row_index": row_index,
                    "table_present": table is not None,
                }
            )
    total = len(components)
    return {
        "components": total,
        "table_hits": table_hits,
        "table_recall": table_hits / total if total else 0.0,
        "row_hits": row_hits,
        "row_recall": row_hits / total if total else 0.0,
        "misses": misses,
    }


def compress_with_reranker(
    contexts_path: Path, rerank_path: Path, output_path: Path
) -> dict[str, Any]:
    contexts = {
        int(row["question_id"]): row
        for row in (
            json.loads(line)
            for line in contexts_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    reranks = [
        json.loads(line)
        for line in rerank_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    compact = []
    for rerank in reranks:
        if rerank.get("status") != "VALID":
            continue
        question_id = int(rerank["question_id"])
        context = contexts[question_id]
        selected_refs = {
            row["row_ref"] for row in rerank["selection"]["selected"]
        }
        selected_tables = []
        for table in context["tables"]:
            selected_rows = [
                row for row in table["candidate_rows"] if row["row_ref"] in selected_refs
            ]
            if not selected_rows:
                continue
            section_headers = []
            for selected in selected_rows:
                preceding = [
                    row
                    for row in table["candidate_rows"]
                    if row["row_index"] < selected["row_index"]
                    and row.get("hierarchy_role") == "section_header"
                ]
                if preceding:
                    section_headers.append(preceding[-1])
            keep = {
                row["row_ref"]: row for row in [*section_headers, *selected_rows]
            }
            selected_tables.append(
                {**table, "candidate_rows": sorted(keep.values(), key=lambda row: row["row_index"])}
            )
        compact.append({**context, "tables": selected_tables})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compact),
        encoding="utf-8",
    )
    return {
        "questions": len(compact),
        "tables": sum(len(row["tables"]) for row in compact),
        "rows": sum(
            len(table["candidate_rows"])
            for row in compact
            for table in row["tables"]
        ),
        "output": str(output_path),
    }


def prune_for_reranker(
    contexts_path: Path,
    output_path: Path,
    *,
    tables_per_dimension: int = 4,
    rows_per_table: int = 8,
) -> dict[str, Any]:
    """Shrink masked contexts before LLM reranking without dropping seeds."""

    if tables_per_dimension < 1 or rows_per_table < 1:
        raise ValueError("reranker context limits must be positive")
    contexts = [
        json.loads(line)
        for line in contexts_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pruned = []
    for context in contexts:
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for table in context["tables"]:
            grouped.setdefault(
                (table["ticker"], int(table["year"]), table["scope"]), []
            ).append(table)
        selected_tables: list[dict[str, Any]] = []
        for dimension in sorted(grouped):
            tables = grouped[dimension]
            selected_by_id = {
                int(table["table_id"]): table
                for table in tables
                if any(row.get("baseline_seed") for row in table["candidate_rows"])
            }
            ranked = sorted(
                tables,
                key=lambda table: (-float(table["retrieval_score"]), table["table_id"]),
            )
            for table in ranked[:tables_per_dimension]:
                selected_by_id.setdefault(int(table["table_id"]), table)
            primary = [table for table in ranked if table["table_kind"] in PRIMARY_KINDS]
            if primary:
                selected_by_id.setdefault(int(primary[0]["table_id"]), primary[0])
            hint_count = max((len(table.get("hint_scores", [])) for table in tables), default=0)
            for hint_index in range(hint_count):
                best = max(
                    tables,
                    key=lambda table: (
                        float(table["hint_scores"][hint_index]),
                        -int(table["table_id"]),
                    ),
                )
                selected_by_id.setdefault(int(best["table_id"]), best)

            for table in selected_by_id.values():
                rows = table["candidate_rows"]
                mandatory = {
                    row["row_ref"]: row
                    for row in rows
                    if row.get("baseline_seed")
                    or row.get("hierarchy_role") in {"section_total", "table_total"}
                }
                optional = [row for row in rows if row["row_ref"] not in mandatory]
                for row in sorted(
                    optional,
                    key=lambda row: (-float(row["row_score"]), row["row_index"]),
                )[: max(0, rows_per_table - len(mandatory))]:
                    mandatory[row["row_ref"]] = row
                selected_tables.append(
                    {
                        **table,
                        "candidate_rows": sorted(
                            mandatory.values(), key=lambda row: row["row_index"]
                        ),
                    }
                )
        pruned.append(
            {
                **context,
                "tables": sorted(
                    selected_tables,
                    key=lambda table: (
                        table["ticker"],
                        int(table["year"]),
                        table["scope"],
                        -float(table["retrieval_score"]),
                        int(table["table_id"]),
                    ),
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pruned),
        encoding="utf-8",
    )
    return {
        "questions": len(pruned),
        "tables": sum(len(row["tables"]) for row in pruned),
        "rows": sum(
            len(table["candidate_rows"])
            for row in pruned
            for table in row["tables"]
        ),
        "output": str(output_path),
    }


def _parse_ids(value: str) -> set[int]:
    return {
        int(token.strip().lower().removeprefix("q"))
        for token in value.split(",")
        if token.strip()
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans", type=Path, default=Path("outputs/financial-ir/l2-formulas.jsonl")
    )
    parser.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/financial-ir/retrieval-context.jsonl")
    )
    parser.add_argument("--ids")
    parser.add_argument("--legacy-plans", type=Path)
    parser.add_argument("--base-contexts", type=Path)
    parser.add_argument("--seed-plans", type=Path)
    parser.add_argument(
        "--companies", type=Path, default=Path("ViFinQA/code_stock.csv")
    )
    parser.add_argument("--compress-contexts", type=Path)
    parser.add_argument("--rerank-results", type=Path)
    parser.add_argument("--prune-contexts", type=Path)
    parser.add_argument("--tables-per-dimension", type=int, default=4)
    parser.add_argument("--rows-per-table", type=int, default=8)
    args = parser.parse_args(argv)
    if args.prune_contexts:
        result = prune_for_reranker(
            args.prune_contexts,
            args.output,
            tables_per_dimension=args.tables_per_dimension,
            rows_per_table=args.rows_per_table,
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.compress_contexts or args.rerank_results:
        if args.compress_contexts is None or args.rerank_results is None:
            parser.error("compression requires --compress-contexts and --rerank-results")
        result = compress_with_reranker(
            args.compress_contexts, args.rerank_results, args.output
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    result = export_contexts(
        args.plans,
        args.database,
        args.output,
        _parse_ids(args.ids) if args.ids else None,
        args.companies,
        args.base_contexts,
        args.seed_plans,
    )
    if args.legacy_plans:
        result["proxy_recall"] = proxy_recall(args.output, args.legacy_plans)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
