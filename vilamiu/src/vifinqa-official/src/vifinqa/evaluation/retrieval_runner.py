
from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from vifinqa.evaluation.retrieval_metrics import (
    RetrievalCounts,
    RetrievalMetrics,
    counts_for_k,
    metrics_from_counts,
    report_counts_for_k,
)
from vifinqa.common.schemas.schema import Question
from vifinqa.retrieval.base import RetrievalIndex


@dataclass(frozen=True, slots=True)
class QuestionRetrievalResult:
    question_id: int
    original: str
    difficulty: str
    k: int
    metrics: RetrievalMetrics
    counts: RetrievalCounts
    retrieved_tables: tuple[str, ...]
    # Report-level scoring of the same ranking; gold evidence is annotated at both granularities.
    report_metrics: RetrievalMetrics | None = None
    report_counts: RetrievalCounts | None = None


def evaluate_question(index: RetrievalIndex, question: Question, *, ks: list[int]) -> list[QuestionRetrievalResult]:
    max_k = max(ks)
    hits = index.search(question.question, top_k=max_k)
    retrieved_refs = [h.table.table_ref for h in hits]
    gold_refs = set(question.relevant_tables)
    gold_docs = set(question.relevant_docs)

    results: list[QuestionRetrievalResult] = []
    for k in ks:
        counts = counts_for_k(retrieved_refs, gold_refs, k)
        doc_counts = report_counts_for_k(retrieved_refs, gold_docs, k)
        results.append(
            QuestionRetrievalResult(
                question_id=question.id,
                original=question.original,
                difficulty=question.difficulty,
                k=k,
                metrics=metrics_from_counts(counts),
                counts=counts,
                retrieved_tables=tuple(retrieved_refs[:k]),
                report_metrics=metrics_from_counts(doc_counts),
                report_counts=doc_counts,
            )
        )
    return results


def evaluate_questions(
    index: RetrievalIndex, questions: list[Question], *, ks: list[int], show_progress: bool = True
) -> list[QuestionRetrievalResult]:
    results: list[QuestionRetrievalResult] = []
    for question in tqdm(questions, desc="eval-retrieval", disable=not show_progress):
        results.extend(evaluate_question(index, question, ks=ks))
    return results
