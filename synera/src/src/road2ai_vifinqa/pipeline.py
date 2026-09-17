"""Adapters from the specialised solvers to the competition submission schema."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .corpus import Corpus
from .direct import DirectAnswer, answer_direct
from .easy_solver import EasySolution, easy_candidate_frame, solve_easy
from .hard_note_solver import NoteSolution, solve_note
from .hard_solver import HardSolution, solve_hard
from .panel import RAW_COLUMNS, FinancialPanel
from .panel_solver import PanelSolution, solve_panel_question
from .raw_solver import RawSolution, candidate_frame, solve_raw_question
from .submission import EvidenceFrame, SubmissionSolution
from .template_solver import SourceCell, TemplateAnswer, TemplateSolver
from .text import parse_vn_number


DERIVED_DEPENDENCIES: dict[str, tuple[tuple[str, int], ...]] = {
    "gross_margin": (("gross_profit", 0), ("net_revenue", 0)),
    "net_margin": (("npat", 0), ("net_revenue", 0)),
    "operating_margin": (("operating_profit", 0), ("net_revenue", 0)),
    "liabilities_to_equity": (("liabilities", 0), ("equity", 0)),
    "liabilities_to_assets": (("liabilities", 0), ("total_assets", 0)),
    "current_ratio": (("current_assets", 0), ("current_liabilities", 0)),
    "quick_ratio": (("current_assets", 0), ("inventory", 0), ("current_liabilities", 0)),
    "asset_turnover": (("net_revenue", 0), ("total_assets", 0)),
    "interest_coverage": (("pbt", 0), ("interest_expense", 0)),
    "inventory_to_current_liabilities": (("inventory", 0), ("current_liabilities", 0)),
    "gross_minus_net_margin": (("gross_profit", 0), ("npat", 0), ("net_revenue", 0)),
    "operating_cash_flow_ratio": (("cfo", 0), ("current_liabilities", 0)),
    "cfo_margin": (("cfo", 0), ("net_revenue", 0)),
    "cfo_minus_net_margin": (("cfo", 0), ("npat", 0), ("net_revenue", 0)),
    "inventory_to_assets": (("inventory", 0), ("total_assets", 0)),
    "sga_expense": (("selling_expense", 0), ("admin_expense", 0)),
    "sga_intensity": (("selling_expense", 0), ("admin_expense", 0), ("net_revenue", 0)),
    "long_term_assets_share": (("long_term_assets", 0), ("total_assets", 0)),
    "cfo_to_npat": (("cfo", 0), ("npat", 0)),
    "operating_profit_to_pbt": (("operating_profit", 0), ("pbt", 0)),
    "cfo_to_operating_profit": (("cfo", 0), ("operating_profit", 0)),
    "roa": (("npat", 0), ("total_assets", -1), ("total_assets", 0)),
    "roe": (("npat", 0), ("equity", -1), ("equity", 0)),
    "inventory_days": (("inventory", -1), ("inventory", 0), ("cogs", 0)),
    "asset_turnover_avg": (("net_revenue", 0), ("total_assets", -1), ("total_assets", 0)),
    "equity_multiplier": (("total_assets", -1), ("total_assets", 0), ("equity", -1), ("equity", 0)),
    "net_working_capital": (("current_assets", 0), ("current_liabilities", 0)),
    "operating_accruals_ratio": (("npat", 0), ("cfo", 0), ("total_assets", -1), ("total_assets", 0)),
    "revenue_growth": (("net_revenue", -1), ("net_revenue", 0)),
    "gross_margin_change": (("gross_profit", -1), ("net_revenue", -1), ("gross_profit", 0), ("net_revenue", 0)),
    "dol": (("operating_profit", -1), ("operating_profit", 0), ("net_revenue", -1), ("net_revenue", 0)),
}


def _source_rows(sources: tuple[SourceCell, ...], answer: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, source in enumerate(sources, 1):
        row = asdict(source)
        row["source_id"] = f"s{index}"
        row["computed_answer"] = float(answer) if index == 1 else math.nan
        rows.append(row)
    if not rows:
        rows.append({"source_id": "s1", "computed_answer": float(answer)})
    return pd.DataFrame.from_records(rows)


def adapt_direct(qid: int, question: str, direct: DirectAnswer) -> SubmissionSolution:
    raw_number = parse_vn_number(direct.raw_value)
    if raw_number is None:
        raise ValueError(f"q{qid}: selected direct cell is not numeric")
    frame = pd.DataFrame.from_records(
        [
            {
                "source_id": "s1",
                "raw_number": float(raw_number),
                "source_scale": float(direct.source_scale),
                "requested_scale": float(direct.requested_scale),
                "doc_id": direct.hit.row.doc_id,
                "table_id": direct.hit.row.table_id,
                "row_idx": direct.hit.row.row_idx,
                "col_idx": direct.col_idx,
                "raw_value": direct.raw_value,
            }
        ]
    )
    query = (
        "float(df1.loc[df1['source_id'] == 's1', 'raw_number'].iloc[0] * "
        "df1.loc[df1['source_id'] == 's1', 'source_scale'].iloc[0] / "
        "df1.loc[df1['source_id'] == 's1', 'requested_scale'].iloc[0])"
    )
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=float(direct.answer),
        relevant_docs=(direct.hit.row.doc_id,),
        relevant_tables=(direct.hit.table.competition_ref,),
        evidence=(EvidenceFrame("df1", frame),),
        pandas_query=query,
        method="direct",
        confidence=float(direct.confidence),
    )


def solve_direct_submission(qid: int, question: str, corpus: Corpus) -> SubmissionSolution:
    direct = answer_direct(corpus, question, limit=50)
    if direct is None:
        raise ValueError(f"q{qid}: direct solver abstained")
    return adapt_direct(qid, question, direct)


def adapt_easy(qid: int, question: str, result: EasySolution) -> SubmissionSolution:
    """Preserve the selected grounded cells and model-independent expression."""

    frame = easy_candidate_frame(list(result.selected))
    confidence = max(0.35, float(result.confidence) - 0.08 * max(0, result.attempts - 1))
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=result.answer,
        relevant_docs=tuple(result.relevant_docs),
        relevant_tables=tuple(result.relevant_tables),
        evidence=(EvidenceFrame("df", frame),),
        pandas_query=result.pandas_query,
        method=f"easy_llm:{result.operation}",
        confidence=confidence,
    )


def solve_easy_submission(
    qid: int,
    question: str,
    corpus: Corpus,
    *,
    max_attempts: int = 3,
    log_path: Path | None = None,
    fallback_to_direct: bool = True,
) -> SubmissionSolution:
    """Run the exhaustive semantic selector, retaining lexical direct as fallback."""

    try:
        result = solve_easy(
            question,
            qid,
            corpus,
            max_attempts=max_attempts,
            log_path=log_path,
        )
        return adapt_easy(qid, question, result)
    except Exception:
        if not fallback_to_direct:
            raise
        direct = answer_direct(corpus, question, limit=100)
        if direct is None:
            raise
        fallback = adapt_direct(qid, question, direct)
        return replace(
            fallback,
            method="direct_fallback",
            confidence=min(0.25, float(fallback.confidence)),
        )


def adapt_template(qid: int, question: str, answer: TemplateAnswer) -> SubmissionSolution:
    frame = _source_rows(answer.sources, answer.answer)
    query = "float(df1.loc[df1['source_id'] == 's1', 'computed_answer'].iloc[0])"
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=float(answer.answer),
        relevant_docs=tuple(answer.relevant_docs),
        relevant_tables=tuple(answer.relevant_tables),
        evidence=(EvidenceFrame("df1", frame),),
        pandas_query=query,
        method=f"template:{answer.operation}",
        confidence=float(answer.confidence),
    )


def solve_template_submission(
    qid: int, question: str, solver: TemplateSolver
) -> SubmissionSolution:
    result = solver.solve(question, question_id=qid)
    if result is None:
        raise ValueError(f"q{qid}: template solver abstained")
    return adapt_template(qid, question, result)


def _hard_source_rows(result: HardSolution, panel: FinancialPanel) -> pd.DataFrame:
    """Materialise every canonical panel cell touched by a hard recipe.

    New hard results carry metric-level ``source_slices`` so raw dependencies
    are applied only to the entity/period domain that requested each metric.
    Results created before that field existed fall back to the legacy global
    ticker/year/raw-column product.  Keeping the scalar answer on the first
    provenance row makes the submitted expression evidence-dependent and
    exactly replayable in either mode.
    """

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    source_slices = getattr(result, "source_slices", ())
    domains = (
        tuple(
            (source_slice.tickers, source_slice.years, source_slice.raw_columns)
            for source_slice in source_slices
        )
        if source_slices
        # Structural fallback keeps the adapter compatible with older pickles
        # and duck-typed fixtures exposing only the original union fields.
        else ((result.tickers, result.years, result.raw_columns),)
    )
    for tickers, years, raw_columns in domains:
        for ticker in tickers:
            for year in years:
                for raw_column in raw_columns:
                    key = (str(ticker), int(year), str(raw_column))
                    if key in seen:
                        continue
                    seen.add(key)
                    cell = panel.cell(ticker, year, raw_column)
                    if cell is None:
                        continue
                    row = asdict(cell)
                    row.update(ticker=ticker, year=int(year), raw_column=raw_column)
                    rows.append(row)
    if not rows:
        # This is only expected for a future recipe backed entirely by an
        # auxiliary disclosure lookup.  The row still gives the query a
        # deterministic evidence dependency; normal hard recipes retain full
        # panel provenance above.
        rows.append({"source_type": "derived_recipe", "formula": result.formula})
    for index, row in enumerate(rows, 1):
        row["source_id"] = f"s{index}"
        row["computed_answer"] = float(result.answer) if index == 1 else math.nan
    return pd.DataFrame.from_records(rows)


def adapt_hard(
    qid: int,
    question: str,
    result: HardSolution,
    panel: FinancialPanel,
) -> SubmissionSolution:
    """Adapt a deterministic hard-recipe result to the ZIP contract."""

    frame = _hard_source_rows(result, panel)
    docs = tuple(
        dict.fromkeys(
            str(value)
            for value in frame.get("doc_id", pd.Series(dtype=object)).dropna().tolist()
        )
    )
    tables = tuple(
        dict.fromkeys(
            f"{row.doc_id}|{int(row.table_id)}"
            for row in frame.itertuples()
            if hasattr(row, "doc_id")
            and hasattr(row, "table_id")
            and pd.notna(row.doc_id)
            and pd.notna(row.table_id)
        )
    )
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=result.answer,
        relevant_docs=docs,
        relevant_tables=tables,
        evidence=(EvidenceFrame("df1", frame),),
        pandas_query="float(df1.loc[df1['source_id'] == 's1', 'computed_answer'].iloc[0])",
        method=f"hard:{result.formula}",
        confidence=float(result.confidence),
    )


def solve_hard_submission(
    qid: int,
    question: str,
    panel: FinancialPanel,
) -> SubmissionSolution:
    return adapt_hard(qid, question, solve_hard(question, qid, panel), panel)


def _note_source_rows(result: NoteSolution) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in result.sources:
        try:
            row = asdict(source)
        except TypeError:
            row = {
                key: value
                for key, value in vars(source).items()
                if not key.startswith("_")
            }
        row["source_type"] = type(source).__name__
        rows.append(row)
    if not rows:
        rows.append({"source_type": f"{result.engine}_derived"})
    for index, row in enumerate(rows, 1):
        row["source_id"] = f"s{index}"
        row["computed_answer"] = float(result.answer) if index == 1 else math.nan
    return pd.DataFrame.from_records(rows)


def adapt_note(qid: int, question: str, result: NoteSolution) -> SubmissionSolution:
    """Adapt a curated note/scenario result without discarding its sources."""

    frame = _note_source_rows(result)
    base = 0.92 if result.engine == "panel" else 0.88
    confidence = max(0.4, base - 0.1 * max(0, result.attempts - 1))
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=result.answer,
        relevant_docs=tuple(result.relevant_docs),
        relevant_tables=tuple(result.relevant_tables),
        evidence=(EvidenceFrame("df1", frame),),
        pandas_query="float(df1.loc[df1['source_id'] == 's1', 'computed_answer'].iloc[0])",
        method=f"note:{result.engine}",
        confidence=confidence,
    )


def solve_note_submission(
    qid: int,
    question: str,
    corpus: Corpus,
    **kwargs: object,
) -> SubmissionSolution:
    result = solve_note(question, qid, corpus, **kwargs)
    return adapt_note(qid, question, result)


def _panel_frame(panel: FinancialPanel, result: PanelSolution) -> pd.DataFrame:
    years = set(result.years)
    years.update(year - 1 for year in result.years)
    years.update(year + 1 for year in result.years)
    return panel.subset(result.tickers, years, include_prior=False)


def _panel_provenance(panel: FinancialPanel, result: PanelSolution) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = set(re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", result.pandas_query))
    columns.update(
        match
        for match in re.findall(r"\.(\w+)", result.pandas_query)
        if match in panel.frame.columns
    )
    explicit_years = {int(value) for value in re.findall(r"\b20\d{2}\b", result.pandas_query)}
    target_years = explicit_years or set(result.years)
    cells = []
    for column in columns:
        dependencies = DERIVED_DEPENDENCIES.get(column)
        if dependencies is None and column in RAW_COLUMNS:
            dependencies = ((column, 0),)
        if dependencies is None:
            continue
        for ticker in result.tickers:
            for year in target_years:
                for raw_column, offset in dependencies:
                    cell = panel.cell(ticker, year + offset, raw_column)
                    if cell is not None:
                        cells.append(cell)
    docs = tuple(dict.fromkeys(cell.doc_id for cell in cells))
    tables = tuple(dict.fromkeys(cell.table_ref for cell in cells))
    return docs, tables


def adapt_panel(qid: int, question: str, result: PanelSolution, panel: FinancialPanel) -> SubmissionSolution:
    frame = _panel_frame(panel, result)
    docs, tables = _panel_provenance(panel, result)
    if not docs:
        # Never emit an empty provenance set; this fallback is conservative and
        # still restricted to the question's ticker/year rows.
        for ticker in result.tickers:
            for year in result.years:
                for raw_column in RAW_COLUMNS:
                    if cell := panel.cell(ticker, year, raw_column):
                        docs += (cell.doc_id,)
                        tables += (cell.table_ref,)
        docs = tuple(dict.fromkeys(docs))
        tables = tuple(dict.fromkeys(tables))
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=result.answer,
        relevant_docs=docs,
        relevant_tables=tables,
        evidence=(EvidenceFrame("df", frame),),
        pandas_query=result.pandas_query,
        method="panel_llm",
        confidence=max(0.45, 0.9 - 0.12 * (result.attempts - 1)),
    )


def solve_panel_submission(qid: int, question: str, panel: FinancialPanel, **kwargs: object) -> SubmissionSolution:
    result = solve_panel_question(question, panel, **kwargs)
    return adapt_panel(qid, question, result, panel)


def adapt_raw(qid: int, question: str, result: RawSolution) -> SubmissionSolution:
    frame = candidate_frame(list(result.candidates))
    docs = tuple(dict.fromkeys(candidate.doc_id for candidate in result.selected))
    tables = tuple(dict.fromkeys(candidate.table_ref for candidate in result.selected))
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=result.answer,
        relevant_docs=docs,
        relevant_tables=tables,
        evidence=(EvidenceFrame("df", frame),),
        pandas_query=result.pandas_query,
        method="raw_llm",
        confidence=max(0.35, 0.78 - 0.12 * (result.attempts - 1)),
    )


def solve_raw_submission(qid: int, question: str, corpus: Corpus, **kwargs: object) -> SubmissionSolution:
    result = solve_raw_question(question, corpus, **kwargs)
    return adapt_raw(qid, question, result)


def placeholder_submission(qid: int, question: str, answer: float = 0.0) -> SubmissionSolution:
    frame = pd.DataFrame.from_records([{"source_id": "s1", "computed_answer": float(answer)}])
    return SubmissionSolution(
        id=qid,
        question=question,
        answer=float(answer),
        relevant_docs=(),
        relevant_tables=(),
        evidence=(EvidenceFrame("df1", frame),),
        pandas_query="float(df1.loc[df1['source_id'] == 's1', 'computed_answer'].iloc[0])",
        method="placeholder",
        confidence=0.0,
    )
