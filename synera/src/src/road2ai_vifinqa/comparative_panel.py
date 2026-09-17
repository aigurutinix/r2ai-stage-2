"""Fill missing prior-year panel cells from explicit same-scope comparatives.

This is opt-in. It never replaces an existing cell and never uses a separate
statement as a substitute for a consolidated statement. Every new value keeps
the original document/table/row/column and printed unit.
"""
from __future__ import annotations

import copy
import re

from .build_panel import COST_KEYS
from .corpus import Corpus
from .table_semantics import PeriodRole, TableAnalyzer, period_info
from .text import fold_text, parse_vn_number


def is_prior_annual_column(header: str, *, report_year: int, balance: bool) -> bool:
    period = period_info(header, report_year=report_year)
    if period.role == PeriodRole.OPENING:
        # Opening balance at 1 January Y is the closing balance of Y-1.
        # Reject vague 'opening period' and any flow-statement opening column.
        folded = fold_text(header)
        annual_opening = bool(re.search(r"\b0?1 0?1(?: 20\d{2})?\b", folded)) or any(
            marker in folded for marker in ("so dau nam", "dau nam")
        )
        return balance and annual_opening and period.year in (None, report_year)
    if period.year is not None:
        return period.year == report_year - 1
    # These documents are annual statements; 'prior year' is unambiguous.
    # Do not extrapolate from 'prior period' alone.
    return period.role == PeriodRole.PRIOR and "nam truoc" in fold_text(header)


def fill_comparatives(panel: dict, corpus: Corpus) -> tuple[dict, list[dict]]:
    result = copy.deepcopy(panel)
    added: list[dict] = []
    analyzers: dict[tuple[str, int], TableAnalyzer] = {}
    for ticker, years in sorted(panel.items()):
        for year, metrics in sorted(years.items()):
            report_year = int(year)
            previous_year = str(report_year - 1)
            for key, source in sorted(metrics.items()):
                if key in result[ticker].get(previous_year, {}):
                    continue
                document = corpus._doc_by_id.get(str(source["doc_id"]))
                if document is None or document.scope != "consolidated" or document.report_year != report_year:
                    continue
                table_key = (document.doc_id, int(source["table_id"]))
                table = corpus.table(*table_key)
                if table_key not in analyzers:
                    analyzers[table_key] = TableAnalyzer(table.rows, context=table.context, report_year=report_year)
                analyzer = analyzers[table_key]
                row_idx = int(source["row_idx"])
                row = table.rows[row_idx]
                choices = []
                for col_idx, raw in enumerate(row):
                    if col_idx == int(source["col_idx"]):
                        continue
                    semantics = analyzer.cell(row_idx, col_idx)
                    if not is_prior_annual_column(semantics.column_header, report_year=report_year, balance=key.startswith("cdkt:")):
                        continue
                    # Header lineage must not turn a note reference or row
                    # code into an amount: only columns to the right of the
                    # currently selected reporting amount are comparatives.
                    # Reversed years are allowed only when both explicitly
                    # name the matching years (checked separately below).
                    if col_idx < int(source["col_idx"]):
                        if semantics.period.year != report_year - 1:
                            continue
                        current_period = analyzer.cell(row_idx, int(source["col_idx"])).period
                        if current_period.year != report_year:
                            continue
                    raw = raw.strip()
                    value = parse_vn_number(raw)
                    if value is None and raw in {"-", "–", "—"}:
                        value = 0.0
                    if value is None or raw.endswith("%"):
                        continue
                    normalized = value * semantics.unit_scale
                    if key in COST_KEYS:
                        normalized = abs(normalized)
                    choices.append({
                        "value": normalized, "raw": raw, "label": source["label"],
                        "doc_id": document.doc_id, "table_id": table.table_id,
                        "row_idx": row_idx, "col_idx": col_idx, "scale": semantics.unit_scale,
                        "scope": document.scope, "period_year": report_year - 1,
                        "period_role": semantics.period.role.value, "column_header": semantics.column_header,
                        "selection_method": "explicit_same_scope_comparative",
                    })
                # Ambiguity is an abstention, not a choice of the first number.
                if len(choices) != 1:
                    continue
                selected = choices[0]
                result[ticker].setdefault(previous_year, {})[key] = selected
                added.append({"ticker": ticker, "year": int(previous_year), "key": key, **selected})
    return result, added
