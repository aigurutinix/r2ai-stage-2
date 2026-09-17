
from __future__ import annotations

from typing import Protocol

from vifinqa.common.corpus.statement import StatementCell


class Cube(Protocol):

    def cell(self, ticker: str, year: str, metric_key: str) -> StatementCell | None: ...

    def tickers(self) -> tuple[str, ...]: ...

    def years(self, ticker: str) -> tuple[str, ...]: ...

    def metric_keys(self, ticker: str, year: str) -> tuple[str, ...]: ...


class CubeStore(Protocol):
    def load_or_build(self) -> Cube: ...
