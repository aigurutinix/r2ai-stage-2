"""Deterministic evidence-pack builder for the Hard Codex-manual lane.

This module deliberately depends on the corpus/Cube/metric registries only.  It does not import
Hard Cube analytical frames, recipes, planners, or CP7 selection logic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import (
    GROUNDED_DERIVED_FORMULAS,
    GROUNDED_DERIVED_METRIC_NAMES,
    RATIOS,
    industry_l3_groups,
    metric_name,
)

HARD_MANUAL_PROTOCOL_VERSION = "hard-manual-v1"
PackKind = Literal["peer-window", "peer-period", "single-entity-window"]
_PACK_KINDS: tuple[PackKind, ...] = (
    "peer-window",
    "peer-period",
    "single-entity-window",
)
_TABLE_REF_RE = re.compile(r"^(?P<doc>.+)\|table_(?P<table_id>\d+)$")


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class PackCandidate:
    kind: PackKind
    tickers: tuple[str, ...]
    periods: tuple[str, ...]
    industry_l3: str | None = None

    @property
    def key(self) -> tuple[object, ...]:
        return (self.kind, self.industry_l3, self.tickers, self.periods)


def _contiguous_windows(years: set[str], *, minimum: int, maximum: int) -> list[tuple[str, ...]]:
    numeric = sorted(int(year) for year in years if year.isdigit())
    windows: list[tuple[str, ...]] = []
    for start in range(len(numeric)):
        for length in range(minimum, maximum + 1):
            chunk = numeric[start : start + length]
            if len(chunk) != length:
                continue
            if all(right == left + 1 for left, right in zip(chunk, chunk[1:])):
                windows.append(tuple(str(year) for year in chunk))
    return windows


def _rank(seed: int, candidate: PackCandidate) -> str:
    return hashlib.sha256(f"{seed}|{candidate.key!r}".encode()).hexdigest()


class HardManualPackBuilder:
    """Builds inventory/evidence only; it never chooses a computation or data-dependent key."""

    def __init__(
        self,
        *,
        cube: Cube,
        docs: list[DocumentRef],
        company_meta: dict[str, CompanyInfo],
        seed: int,
        source_commit: str,
        source_dirty: bool = False,
    ) -> None:
        self._cube = cube
        self._docs = docs
        self._company_meta = company_meta
        self._seed = seed
        self._source_commit = source_commit
        self._source_dirty = source_dirty
        self._docs_by_key = {(doc.ticker, doc.year, doc.doc_name): doc for doc in docs}

    def candidates(self) -> dict[PackKind, list[PackCandidate]]:
        available_years = {
            ticker: {year for year in self._cube.years(ticker) if self._cube.metric_keys(ticker, year)}
            for ticker in self._cube.tickers()
        }
        result: dict[PackKind, list[PackCandidate]] = {kind: [] for kind in _PACK_KINDS}

        for industry, members in sorted(industry_l3_groups(self._company_meta).items()):
            members_in_cube = tuple(sorted(ticker for ticker in members if ticker in available_years))
            all_years = sorted({year for ticker in members_in_cube for year in available_years[ticker]})
            for year in all_years:
                tickers = tuple(
                    ticker for ticker in members_in_cube if year in available_years[ticker]
                )
                if len(tickers) >= 3:
                    result["peer-period"].append(
                        PackCandidate("peer-period", tickers, (year,), industry)
                    )

            group_years = {year for ticker in members_in_cube for year in available_years[ticker]}
            for periods in _contiguous_windows(group_years, minimum=2, maximum=4):
                tickers = tuple(
                    ticker
                    for ticker in members_in_cube
                    if all(period in available_years[ticker] for period in periods)
                )
                if len(tickers) >= 3:
                    result["peer-window"].append(
                        PackCandidate("peer-window", tickers, periods, industry)
                    )

        for ticker, years in sorted(available_years.items()):
            for periods in _contiguous_windows(years, minimum=3, maximum=5):
                result["single-entity-window"].append(
                    PackCandidate("single-entity-window", (ticker,), periods)
                )

        for kind in _PACK_KINDS:
            result[kind].sort(key=lambda candidate: (_rank(self._seed, candidate), candidate.key))
        return result

    def build(self, count: int) -> list[dict[str, object]]:
        if count < 1:
            raise ValueError("count must be at least 1")
        candidates = self.candidates()
        selected: list[PackCandidate] = []
        offsets = {kind: 0 for kind in _PACK_KINDS}
        while len(selected) < count:
            progressed = False
            for kind in _PACK_KINDS:
                offset = offsets[kind]
                if offset >= len(candidates[kind]):
                    continue
                selected.append(candidates[kind][offset])
                offsets[kind] += 1
                progressed = True
                if len(selected) == count:
                    break
            if not progressed:
                raise ValueError(
                    f"Corpus produced only {len(selected)} eligible packs, fewer than count={count}"
                )
        return [self._build_pack(candidate) for candidate in selected]

    def _build_pack(self, candidate: PackCandidate) -> dict[str, object]:
        observations: list[dict[str, object]] = []
        evidence_cells: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for ticker in candidate.tickers:
            for period in candidate.periods:
                metrics: list[dict[str, object]] = []
                for key in self._cube.metric_keys(ticker, period):
                    cell = self._cube.cell(ticker, period, key)
                    if cell is None:
                        continue
                    metric = {
                        "metric_key": key,
                        "name": metric_name(key) or cell.label,
                        "table_ref": cell.table_ref,
                        "row_idx": cell.row_idx,
                        "col_idx": cell.col_idx,
                        "scale": cell.scale,
                        "label": cell.label,
                    }
                    metrics.append(metric)
                    evidence_cells.setdefault((ticker, period, cell.table_ref), []).append(
                        {
                            **metric,
                            "raw": cell.raw,
                        }
                    )
                observations.append(
                    {
                        "company": ticker,
                        "period": period,
                        "available_metrics": metrics,
                    }
                )

        pack_basis = {
            "protocol_version": HARD_MANUAL_PROTOCOL_VERSION,
            "source_commit": self._source_commit,
            "seed": self._seed,
            "kind": candidate.kind,
            "companies": candidate.tickers,
            "periods": candidate.periods,
            "industry_l3": candidate.industry_l3,
        }
        pack_id = stable_id("pack", pack_basis)
        return {
            "pack_id": pack_id,
            "protocol_version": HARD_MANUAL_PROTOCOL_VERSION,
            "source_commit": self._source_commit,
            "source_dirty": self._source_dirty,
            "seed": self._seed,
            "pack_kind": candidate.kind,
            "scope": {
                "companies": list(candidate.tickers),
                "periods": list(candidate.periods),
                "report_scope": "consolidated",
                "peer_basis": (
                    {"metadata_field": "industry_l3", "value": candidate.industry_l3}
                    if candidate.industry_l3 is not None
                    else None
                ),
            },
            "companies": [self._company_payload(ticker) for ticker in candidate.tickers],
            "metric_inventory": {
                "observations": observations,
                "approved_formulas": self._formula_inventory(),
            },
            "tables": self._table_evidence(evidence_cells),
        }

    def _company_payload(self, ticker: str) -> dict[str, str]:
        info = self._company_meta[ticker]
        return {
            "ticker": info.ticker,
            "name": info.name,
            "industry_l1": info.industry_l1,
            "industry_l2": info.industry_l2,
            "industry_l3": info.industry_l3,
            "exchange": info.exchange,
        }

    @staticmethod
    def _formula_inventory() -> list[dict[str, object]]:
        formulas = [
            {
                "metric_key": ratio.key,
                "name": ratio.name,
                "numerator": [list(term) for term in ratio.numerator],
                "denominator": [list(term) for term in ratio.denominator],
                "value_kind": ratio.value_kind,
            }
            for ratio in RATIOS
        ]
        formulas.extend(
            {
                "metric_key": key,
                "name": GROUNDED_DERIVED_METRIC_NAMES[key],
                "formula": formula,
                "value_kind": "percentage",
            }
            for key, formula in sorted(GROUNDED_DERIVED_FORMULAS.items())
        )
        return formulas

    def _table_evidence(
        self, evidence_cells: dict[tuple[str, str, str], list[dict[str, object]]]
    ) -> list[dict[str, object]]:
        tables: list[dict[str, object]] = []
        for (ticker, period, table_ref), cells in sorted(evidence_cells.items()):
            match = _TABLE_REF_RE.match(table_ref)
            if match is None:
                raise ValueError(f"invalid table_ref in Cube: {table_ref}")
            doc_name = match.group("doc")
            table_id = int(match.group("table_id"))
            doc = self._docs_by_key.get((ticker, period, doc_name))
            if doc is None:
                raise ValueError(f"Document not found for {ticker}/{period}/{table_ref}")
            table = load_table(
                doc.table_csv_path(table_id),
                ticker=ticker,
                year=period,
                doc_name=doc_name,
                table_id=table_id,
            )
            document = parse_document(doc.text_path) if doc.text_path is not None else None
            row_indices = sorted({int(cell["row_idx"]) for cell in cells})
            tables.append(
                {
                    "table_ref": table_ref,
                    "company": ticker,
                    "period": period,
                    "doc_name": doc_name,
                    "csv_path": str(table.csv_path.resolve()),
                    "header": list(table.header),
                    "raw_rows": [
                        {"row_idx": row_idx, "cells": list(table.rows[row_idx])}
                        for row_idx in row_indices
                    ],
                    "raw_cells": sorted(
                        cells,
                        key=lambda cell: (
                            str(cell["metric_key"]),
                            int(cell["row_idx"]),
                            int(cell["col_idx"]),
                        ),
                    ),
                    "anchor_context": (
                        document.table_anchor_context(table_id) if document is not None else ""
                    ),
                    "unit_evidence": (
                        list(document.table_unit_snippets(table_id)) if document is not None else []
                    ),
                }
            )
        return tables


def dump_pack(pack: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
