
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from vifinqa.answering.base import AnswerRequest, AnswerResult, AnswerStrategy
from vifinqa.answering.table_metadata import build_table_metadata_lookup
from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import get_company_meta
from vifinqa.constants import ANSWER_ABS_TOL
from vifinqa.evaluation.llm_runner import QuestionAnswerResult, answer_request, question_result_from_answer
from vifinqa.llm.base import ChatLLM
from vifinqa.common.schemas.schema import Question
from vifinqa.retrieval.base import RetrievalIndex

logger = logging.getLogger(__name__)

DocLookup = dict[tuple[str, str, str], DocumentRef]


def build_doc_lookup(data_root: Path) -> DocLookup:
    return {(d.ticker, d.year, d.doc_name): d for d in scan_catalog(data_root)}


def _resolve_csv_path(doc_lookup: DocLookup, *, ticker: str, year: str, doc_name: str, table_id: int) -> Path | None:
    doc = doc_lookup.get((ticker, year, doc_name))
    if doc is None or doc.tables_dir is None:
        return None
    return doc.table_csv_path(table_id)


def evaluate_question(
    index: RetrievalIndex,
    question: Question,
    *,
    ks: list[int],
    max_context_tables: int,
    strategy: AnswerStrategy,
    llm: ChatLLM,
    doc_lookup: DocLookup,
    company_names: dict[str, str] | None = None,
    abs_tol: float = ANSWER_ABS_TOL,
) -> list[QuestionAnswerResult]:
    max_k = max(ks)
    hits = index.search(question.question, top_k=max_k)
    metadata_lookup = build_table_metadata_lookup(
        {hit.table.table_ref for hit in hits[: min(max_k, max_context_tables)]},
        doc_name_lookup={doc.doc_name: doc for doc in doc_lookup.values()},
        company_names=company_names or {},
    )

    results: list[QuestionAnswerResult] = []
    answer_cache: dict[tuple[str, ...], AnswerResult] = {}
    for k in ks:
        n_context = min(k, max_context_tables)
        if k > max_context_tables:
            logger.warning(
                "question %s: k=%d > max_context_tables=%d; keeping only the top %d tables in context",
                question.original,
                k,
                max_context_tables,
                max_context_tables,
            )

        csv_paths: dict[str, Path] = {}
        for hit in hits[:n_context]:
            t = hit.table
            path = _resolve_csv_path(doc_lookup, ticker=t.ticker, year=t.year, doc_name=t.doc_name, table_id=t.table_id)
            if path is not None:
                csv_paths[t.table_ref] = path

        # Keep LLM behavior within the declared contract.
        context_key = tuple(csv_paths)
        cached = answer_cache.get(context_key)
        if cached is None:
            req = AnswerRequest(
                question=question.question,
                csv_paths=csv_paths,
                table_metadata={ref: metadata_lookup[ref] for ref in csv_paths if ref in metadata_lookup},
            )
            answer_result = answer_request(req=req, strategy=strategy, llm=llm)
            answer_cache[context_key] = answer_result
        else:
            answer_result = cached
        results.append(
            question_result_from_answer(question, result=answer_result, k=k, abs_tol=abs_tol)
        )
    return results


def evaluate_questions(
    index: RetrievalIndex,
    questions: list[Question],
    *,
    ks: list[int],
    max_context_tables: int,
    strategy: AnswerStrategy,
    llm: ChatLLM,
    data_root: Path,
    company_meta_path: Path | None = None,
    abs_tol: float = ANSWER_ABS_TOL,
    show_progress: bool = True,
    on_result: Callable[[QuestionAnswerResult], None] | None = None,
) -> list[QuestionAnswerResult]:
    doc_lookup = build_doc_lookup(data_root)
    company_names = (
        {ticker: info.name for ticker, info in get_company_meta(company_meta_path).items()}
        if company_meta_path is not None
        else {}
    )
    results: list[QuestionAnswerResult] = []
    for question in tqdm(questions, desc="eval-e2e", disable=not show_progress):
        question_results = evaluate_question(
            index,
            question,
            ks=ks,
            max_context_tables=max_context_tables,
            strategy=strategy,
            llm=llm,
            doc_lookup=doc_lookup,
            company_names=company_names,
            abs_tol=abs_tol,
        )
        results.extend(question_results)
        if on_result is not None:
            for result in question_results:
                on_result(result)
    return results
