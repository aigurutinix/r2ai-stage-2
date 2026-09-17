
from __future__ import annotations

from dataclasses import dataclass

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.generation.hard.recipe.base import BindingRef, ReasoningGraph


@dataclass(frozen=True, slots=True)
class DependencyRecord:
    binding: BindingRef
    role_id: str
    ticker: str
    period: str

    @property
    def dependency_id(self) -> str:
        b = self.binding
        return f"{self.role_id}:{b.table_ref}:{b.row_idx}:{b.col_idx}"


def collect_dependency_closure(
    graph: ReasoningGraph, *, docs_by_name: dict[str, DocumentRef]
) -> tuple[DependencyRecord, ...]:
    records: dict[str, DependencyRecord] = {}
    for role in graph.metric_roles:
        for terms in role.bindings.values():
            for binding in terms.binding_refs:
                doc_name = binding.table_ref.split("|", 1)[0]
                doc = docs_by_name.get(doc_name)
                ticker = doc.ticker if doc is not None else ""
                period = doc.year if doc is not None else ""
                record = DependencyRecord(binding=binding, role_id=role.role_id, ticker=ticker, period=period)
                records[record.dependency_id] = record
    return tuple(records.values())
