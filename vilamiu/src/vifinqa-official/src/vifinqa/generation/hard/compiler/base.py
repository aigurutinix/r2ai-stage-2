
from __future__ import annotations

from typing import Protocol

from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.hard.schemas import HardP3Plan, HardPlan, MetricBinding


class CompileResult(Protocol):
    pandas_query: str
    expected_answer: object


class Compiler(Protocol):
    def compile(
        self,
        plan: HardPlan | HardP3Plan,
        bindings: list[MetricBinding],
        tables: dict[str, TableAsset],
    ) -> CompileResult: ...
