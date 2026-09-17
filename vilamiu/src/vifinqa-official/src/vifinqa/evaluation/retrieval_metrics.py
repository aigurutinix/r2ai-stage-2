
from __future__ import annotations

from dataclasses import dataclass

from vifinqa.common.schemas.table_ref import parse_table_ref


@dataclass(frozen=True, slots=True)
class RetrievalCounts:

    tp: int
    retrieved: int
    gold: int

    @property
    def precision(self) -> float:
        return self.tp / self.retrieved if self.retrieved else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.gold if self.gold else 0.0

    def f_beta(self, beta: float) -> float:
        p, r = self.precision, self.recall
        if p == 0.0 and r == 0.0:
            return 0.0
        beta2 = beta * beta
        return (1 + beta2) * p * r / (beta2 * p + r)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    f2: float
    recall: float
    precision: float


def counts_for_k(retrieved_refs: list[str], gold_refs: set[str], k: int) -> RetrievalCounts:
    top_k = retrieved_refs[:k]
    tp = len(set(top_k) & gold_refs)
    return RetrievalCounts(tp=tp, retrieved=len(top_k), gold=len(gold_refs))


def report_counts_for_k(retrieved_refs: list[str], gold_docs: set[str], k: int) -> RetrievalCounts:
    """Report-level counts at a table cutoff.

    The paper annotates gold evidence at both report and table granularity. Retrieval ranks
    tables, so the reports surfaced at cutoff ``k`` are the distinct document names carried by
    the top-``k`` tables; several tables from one report count as one retrieved report.
    """

    retrieved_docs = dict.fromkeys(parse_table_ref(ref)[0] for ref in retrieved_refs[:k])
    tp = len(retrieved_docs.keys() & gold_docs)
    return RetrievalCounts(tp=tp, retrieved=len(retrieved_docs), gold=len(gold_docs))


def metrics_from_counts(counts: RetrievalCounts) -> RetrievalMetrics:
    return RetrievalMetrics(f2=counts.f_beta(2.0), recall=counts.recall, precision=counts.precision)


def macro_average(per_question: list[RetrievalMetrics]) -> RetrievalMetrics:
    n = len(per_question)
    if n == 0:
        return RetrievalMetrics(f2=0.0, recall=0.0, precision=0.0)
    return RetrievalMetrics(
        f2=sum(m.f2 for m in per_question) / n,
        recall=sum(m.recall for m in per_question) / n,
        precision=sum(m.precision for m in per_question) / n,
    )


def micro_average(per_question_counts: list[RetrievalCounts]) -> RetrievalMetrics:
    combined = RetrievalCounts(
        tp=sum(c.tp for c in per_question_counts),
        retrieved=sum(c.retrieved for c in per_question_counts),
        gold=sum(c.gold for c in per_question_counts),
    )
    return metrics_from_counts(combined)
