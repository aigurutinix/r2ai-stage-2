
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from vifinqa.answering.base import AnswerRequest, AnswerResult, AnswerStrategy, TablePromptMetadata
from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.constants import ANSWER_ABS_TOL
from vifinqa.evaluation.answer_match import coerce_number, is_correct
from vifinqa.llm.base import ChatLLM, LLMError
from vifinqa.common.schemas.schema import Question
from vifinqa.retrieval.base import parse_table_ref

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QuestionAnswerResult:
    question_id: int
    original: str
    difficulty: str
    accuracy: float  # 1.0 | 0.0
    expected: object
    actual: object
    k: int | None = None  # None cho eval-llm; set cho eval-e2e
    error: str = ""
    raw_output: str = ""
    # Paper Table 13 "Crash": the strategy raised, or produced no parseable numeric scalar.
    # For direct answering the reporter folds this back into Fail.
    crash: bool = False


def _gold_csv_paths(question: Question) -> dict[str, Path]:
    return dict(zip(question.relevant_tables, question.csv_paths, strict=True))


def _retrieved_csv_paths(
    question: Question, *, retrieved_tables: dict[int, list[str]], doc_name_lookup: dict[str, DocumentRef]
) -> dict[str, Path]:
    csv_paths: dict[str, Path] = {}
    for ref in retrieved_tables.get(question.id, []):
        doc_name, table_id = parse_table_ref(ref)
        doc = doc_name_lookup.get(doc_name)
        if doc is None or doc.tables_dir is None:
            logger.warning(
                "context=retrieved: CSV not found for table_ref=%s (question %s)", ref, question.original
            )
            continue
        csv_paths[ref] = doc.table_csv_path(table_id)
    return csv_paths


def _apply_context_table_cutoff(
    csv_paths: dict[str, Path],
    *,
    max_context_tables: int | None,
    question: Question,
    context: str,
) -> dict[str, Path]:
    if max_context_tables is None or len(csv_paths) <= max_context_tables:
        return csv_paths
    logger.warning(
        "question %s: context=%s has %d tables; keeping the first %d for the LLM call",
        question.original,
        context,
        len(csv_paths),
        max_context_tables,
    )
    return dict(islice(csv_paths.items(), max_context_tables))


def evaluate_question(
    question: Question,
    *,
    strategy: AnswerStrategy,
    llm: ChatLLM,
    context: str,
    retrieved_tables: dict[int, list[str]] | None = None,
    doc_name_lookup: dict[str, DocumentRef] | None = None,
    table_metadata_lookup: dict[str, TablePromptMetadata] | None = None,
    max_context_tables: int | None = None,
    abs_tol: float = ANSWER_ABS_TOL,
) -> QuestionAnswerResult:
    if context == "gold":
        csv_paths = _gold_csv_paths(question)
    elif context == "retrieved":
        csv_paths = _retrieved_csv_paths(
            question, retrieved_tables=retrieved_tables or {}, doc_name_lookup=doc_name_lookup or {}
        )
    else:
        csv_paths = {}
    csv_paths = _apply_context_table_cutoff(
        csv_paths,
        max_context_tables=max_context_tables,
        question=question,
        context=context,
    )
    metadata_lookup = table_metadata_lookup or {}
    req = AnswerRequest(
        question=question.question,
        csv_paths=csv_paths,
        table_metadata={ref: metadata_lookup[ref] for ref in csv_paths if ref in metadata_lookup},
    )
    return evaluate_answer_request(question, req=req, strategy=strategy, llm=llm, abs_tol=abs_tol)


def evaluate_answer_request(
    question: Question,
    *,
    req: AnswerRequest,
    strategy: AnswerStrategy,
    llm: ChatLLM,
    k: int | None = None,
    abs_tol: float = ANSWER_ABS_TOL,
) -> QuestionAnswerResult:
    result = answer_request(req=req, strategy=strategy, llm=llm)
    return question_result_from_answer(question, result=result, k=k, abs_tol=abs_tol)


def answer_request(*, req: AnswerRequest, strategy: AnswerStrategy, llm: ChatLLM) -> AnswerResult:
    try:
        return strategy.answer(req, llm)
    except LLMError as exc:
        if exc.fatal:
            raise
        return AnswerResult(actual=None, raw_output="", error=str(exc))
    except OSError as exc:
        return AnswerResult(actual=None, raw_output="", error=f"Failed to read context: {exc}")


def question_result_from_answer(
    question: Question,
    *,
    result: AnswerResult,
    k: int | None = None,
    abs_tol: float = ANSWER_ABS_TOL,
) -> QuestionAnswerResult:
    correct = is_correct(question.answer, result.actual, abs_tol=abs_tol)
    return QuestionAnswerResult(
        question_id=question.id,
        original=question.original,
        difficulty=question.difficulty,
        accuracy=1.0 if correct else 0.0,
        expected=question.answer,
        actual=result.actual,
        k=k,
        error=result.error,
        raw_output=result.raw_output,
        crash=bool(result.error) or coerce_number(result.actual) is None,
    )


def evaluate_questions(
    questions: list[Question],
    *,
    strategy: AnswerStrategy,
    llm: ChatLLM,
    context: str,
    retrieved_tables: dict[int, list[str]] | None = None,
    doc_name_lookup: dict[str, DocumentRef] | None = None,
    table_metadata_lookup: dict[str, TablePromptMetadata] | None = None,
    max_context_tables: int | None = None,
    abs_tol: float = ANSWER_ABS_TOL,
    max_workers: int = 1,
    show_progress: bool = True,
    on_result: Callable[[QuestionAnswerResult], None] | None = None,
) -> list[QuestionAnswerResult]:
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")

    worker = partial(
        evaluate_question,
        strategy=strategy,
        llm=llm,
        context=context,
        retrieved_tables=retrieved_tables,
        doc_name_lookup=doc_name_lookup,
        table_metadata_lookup=table_metadata_lookup,
        max_context_tables=max_context_tables,
        abs_tol=abs_tol,
    )

    if max_workers == 1:
        ordered_results: list[QuestionAnswerResult] = []
        iterator = tqdm(map(worker, questions), total=len(questions), desc="eval-llm", disable=not show_progress)
        for result in iterator:
            ordered_results.append(result)
            if on_result is not None:
                on_result(result)
        return ordered_results

    # ``executor.map()`` yields in input order, so a stalled first item delays checkpoints
    # for later completed items. ``as_completed()`` checkpoints promptly while the indexed
    # result list preserves original report order.
    ordered_results_or_none: list[QuestionAnswerResult | None] = [None] * len(questions)
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vifinqa-llm")
    future_indices: dict[Future[QuestionAnswerResult], int] = {}
    try:
        future_indices = {
            executor.submit(worker, question): index for index, question in enumerate(questions)
        }
        iterator = tqdm(
            as_completed(future_indices),
            total=len(questions),
            desc="eval-llm",
            disable=not show_progress,
        )
        for future in iterator:
            result = future.result()
            ordered_results_or_none[future_indices[future]] = result
            if on_result is not None:
                # The callback runs only on the caller thread; workers never write checkpoints.
                on_result(result)
    except BaseException:
        # Running requests cannot be cancelled, but queued requests must not start after
        # Ctrl-C or a fatal error. Existing checkpoints remain intact.
        for future in future_indices:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    return [result for result in ordered_results_or_none if result is not None]
