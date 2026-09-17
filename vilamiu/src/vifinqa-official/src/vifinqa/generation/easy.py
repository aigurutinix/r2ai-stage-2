
from __future__ import annotations

import logging
import random
from pathlib import Path

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import (
    CandidateTable,
    answer_shape_error,
    build_candidate_table,
    call_structured,
    judge_and_maybe_rewrite,
    question_style_error,
    report_scope,
)
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS, run_parallel_generation
from vifinqa.generation.prompts.easy import build_easy_fact_prompt, build_easy_question_prompt
from vifinqa.generation.schemas import EasyFactQuery, QARecord, QuestionDraft
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.validation.pandas_check import execute_query

logger = logging.getLogger(__name__)

MAX_ATTEMPTS_PER_TABLE = 2
MAX_QUESTION_ATTEMPTS = 2


def _candidate_tables(docs: list[DocumentRef]) -> list[tuple[DocumentRef, int]]:
    return [(doc, table_id) for doc in docs if doc.tables_dir is not None and doc.text_path is not None for table_id in doc.table_ids]


def _try_generate(
    *,
    llm: ChatLLM,
    candidate: CandidateTable,
    record_id: int,
) -> QARecord | None:
    doc_name = candidate.doc_name
    table_ref = candidate.table_ref
    csv_path = candidate.csv_path
    for _attempt in range(MAX_ATTEMPTS_PER_TABLE):
        # Hand the model the scope this loop is about to check it against, six
        # lines below. LOCAL ADDITION; see build_easy_fact_prompt.
        system, user = build_easy_fact_prompt(
            candidate, known_scope=report_scope(doc_name))
        fact, err = call_structured(llm, system=system, user=user, schema=EasyFactQuery)
        if fact is None:
            logger.debug("Could not create a fact for %s: %s", table_ref, err)
            continue

        known_scope = report_scope(doc_name)
        if known_scope != "unknown" and fact.report_scope != known_scope:
            logger.debug("LLM report scope does not match document name %s", doc_name)
            continue

        execution = execute_query(fact.pandas_query, csv_path)
        if not execution.ok:
            logger.debug("Validation failed for %s: %s", table_ref, execution.detail)
            continue

        shape_error = answer_shape_error(execution.actual)
        if shape_error:
            logger.debug("Result has the wrong format for %s: %r", table_ref, execution.actual)
            continue
        fact = fact.model_copy(update={"answer": execution.actual})

        question_feedback = ""
        for _question_attempt in range(MAX_QUESTION_ATTEMPTS):
            system, user = build_easy_question_prompt(candidate, fact)
            if question_feedback:
                user = f"{user}\n\nPrevious error: {question_feedback}\nRewrite it and return the required JSON."
            draft, err = call_structured(
                llm, system=system, user=user, schema=QuestionDraft, max_attempts=1
            )
            if draft is None:
                question_feedback = err
                continue
            style_error = question_style_error(draft.question)
            if style_error:
                question_feedback = style_error
                continue
            question, natural_ok, judge_detail = judge_and_maybe_rewrite(
                llm=llm, question=draft.question, table_labels=candidate.table_labels
            )
            if not natural_ok:
                question_feedback = f"Question is unnatural: {judge_detail}"
                continue

            return QARecord(
                id=record_id,
                question=question,
                answer=fact.answer,
                relevant_docs=[doc_name],
                relevant_tables=[table_ref],
                pandas_query=fact.pandas_query,
                csv_path=str(csv_path),
                difficulty="easy",
            )

        logger.debug("Could not write an acceptable question for %s: %s", table_ref, question_feedback)

    logger.info("Skipping %s after %d unsuccessful attempts.", table_ref, MAX_ATTEMPTS_PER_TABLE)
    return None


def generate_easy(
    *,
    settings: Settings,
    llm: ChatLLM,
    count: int,
    out_path: Path,
    seed: int | None = None,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    rng = random.Random(seed)
    docs = scan_catalog(settings.data_root)
    candidates = _candidate_tables(docs)
    rng.shuffle(candidates)
    companies = load_company_meta(settings.company_meta_path)

    writer = JsonlWriter(out_path)

    def _build_record(item: tuple[DocumentRef, int]) -> QARecord | None:
        doc, table_id = item
        table = load_table(
            doc.table_csv_path(table_id),
            ticker=doc.ticker,
            year=doc.year,
            doc_name=doc.doc_name,
            table_id=table_id,
        )
        if not is_table_eligible(table):
            return None

        document = parse_document(doc.text_path)  # type: ignore[arg-type]
        company = companies.get(doc.ticker)
        candidate = build_candidate_table(
            doc=doc,
            table=table,
            document=document,
            company_name=company.name if company else doc.ticker,
            context_before=context_pages_before,
            context_after=context_pages_after,
        )
        return _try_generate(llm=llm, candidate=candidate, record_id=0)

    generated = run_parallel_generation(
        work_items=candidates,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )

    if generated < count:
        logger.warning("Generated only %d/%d questions because candidates were exhausted or validation kept failing.", generated, count)

    return generated
