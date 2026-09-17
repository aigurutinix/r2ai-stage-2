
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.statement import StatementCell, parse_statement_table
from vifinqa.common.corpus.table import load_table
from vifinqa.generation.common import report_scope
from vifinqa.generation.panel.catalog import COST_METRIC_KEYS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DictCube:

    data: dict[str, dict[str, dict[str, StatementCell]]] = field(default_factory=dict)

    def cell(self, ticker: str, year: str, metric_key: str) -> StatementCell | None:
        return self.data.get(ticker, {}).get(year, {}).get(metric_key)

    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self.data))

    def years(self, ticker: str) -> tuple[str, ...]:
        return tuple(sorted(self.data.get(ticker, {})))

    def metric_keys(self, ticker: str, year: str) -> tuple[str, ...]:
        return tuple(sorted(self.data.get(ticker, {}).get(year, {})))


def _normalize_sign(metric_key: str, cell: StatementCell) -> StatementCell:
    if metric_key not in COST_METRIC_KEYS or cell.value >= 0:
        return cell
    return StatementCell(
        ma_so=cell.ma_so,
        label=cell.label,
        value=abs(cell.value),
        raw=cell.raw,
        table_ref=cell.table_ref,
        row_idx=cell.row_idx,
        col_idx=cell.col_idx,
        scale=cell.scale,
    )


def build_cube(docs: list[DocumentRef]) -> DictCube:
    data: dict[str, dict[str, dict[str, StatementCell]]] = {}

    for doc in sorted(docs, key=lambda d: (d.ticker, d.year, d.doc_name)):
        if report_scope(doc.doc_name) != "consolidated":
            continue
        if doc.tables_dir is None or doc.text_path is None:
            continue
        document = parse_document(doc.text_path)
        year_bucket = data.setdefault(doc.ticker, {}).setdefault(doc.year, {})

        for table_id in doc.table_ids:
            table = load_table(
                doc.table_csv_path(table_id),
                ticker=doc.ticker,
                year=doc.year,
                doc_name=doc.doc_name,
                table_id=table_id,
            )
            statement = parse_statement_table(table, document)
            if statement is None:
                continue
            for ma_so, cell in statement.current.items():
                metric_key = f"{statement.kind}:{ma_so}"
                if metric_key in year_bucket:
                    logger.debug(
                        "Duplicate line-item code in one document; keeping the first cell: "
                        "ticker=%s year=%s metric=%s (drop %s, keep %s)",
                        doc.ticker,
                        doc.year,
                        metric_key,
                        cell.table_ref,
                        year_bucket[metric_key].table_ref,
                    )
                    continue
                year_bucket[metric_key] = _normalize_sign(metric_key, cell)

    return DictCube(data=data)
