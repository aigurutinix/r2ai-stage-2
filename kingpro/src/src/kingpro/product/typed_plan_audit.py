"""Read-only typed-plan and dual-execution audit for the financial compiler.

The deterministic compiler is deliberately the acceptance gate: queries it
does not understand remain explicit refusals.  For every accepted query this
module builds a serialisable typed plan, evaluates that plan against the
normalised :class:`~kingpro.financial.statement_cube.FinancialCube`, and then
replays the compiler-generated Pandas in the isolated sandbox.

The typed evaluator never evaluates the compiler expression and never uses the
compiler answer as an operand.  It is therefore useful as an implementation-
diverse check of otherwise silent arithmetic, selector and unit failures.
Nothing in this module is imported by the live product router.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from kingpro.answering.pandas_answer import requested_unit
from kingpro.answering.sandbox import run_pandas_code
from kingpro.financial.panel_metrics import PanelMetricEngine, RATIO_FORMULAS, RAW_METRICS
from kingpro.financial.statement_cube import FinancialCube, StatementCell
from kingpro.product.deterministic_compiler import (
    CompiledFinancialQuery,
    DeterministicFinancialCompiler,
)
from kingpro.retrieval.bm25_index import extract_all_facets, fold


PlanKind = Literal["direct", "panel", "universe"]
_COST_KEYS = frozenset(
    {"kqkd:11", "kqkd:22", "kqkd:23", "kqkd:25", "kqkd:26", "kqkd:32", "kqkd:51"}
)
_MONETARY_DERIVED = frozenset(
    {"net_working_capital", "sga_expense", "net_finance_result", "net_other_result"}
)
_UNIVERSE_THRESHOLD_RE = re.compile(
    r"\bgiam\s+it\s+nhat\s+(?P<threshold>\d+(?:[.,]\d+)?)\s*"
    r"(?:%(?=\s|$)|phan\s+tram\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TypedOperand:
    """One source cell consumed by a typed plan."""

    role: str
    ticker: str
    year: str
    scope: str
    metric_key: str
    value_vnd: float
    label: str
    raw: str
    scale: float
    table_ref: str
    csv_path: str
    row_idx: int
    col_idx: int


@dataclass(frozen=True, slots=True)
class TypedFinancialPlan:
    """Stable JSON contract between parsing, retrieval and execution."""

    version: str
    kind: PlanKind
    operation: str
    metric: str
    companies: tuple[str, ...]
    periods: tuple[int, ...]
    scope: str
    output_unit: str
    parameters: dict[str, Any] = field(default_factory=dict)
    operands: tuple[TypedOperand, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OracleResult:
    value: float
    plan: TypedFinancialPlan


def _isclose(left: Any, right: Any) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=1e-10, abs_tol=1e-8
    )


def _scope(facets: dict[str, Any]) -> str:
    value = fold(str(facets.get("scope", "")))
    return (
        "physical:separate"
        if "cong ty me" in value or "rieng" in value
        else "consolidated"
    )


def _metric_dependencies(metric: str, year: int) -> list[tuple[str, int, str]]:
    """Return role/year/key dependencies without consulting compiler output."""

    if metric in RAW_METRICS:
        return [("value", year, RAW_METRICS[metric])]
    if metric in RATIO_FORMULAS:
        numerator, denominator, _multiplier = RATIO_FORMULAS[metric]
        return [
            (f"numerator:{coefficient:g}", year, key)
            for key, coefficient in numerator.items()
        ] + [
            (f"denominator:{coefficient:g}", year, key)
            for key, coefficient in denominator.items()
        ]
    if metric == "revenue_growth_pct":
        return [("current", year, "kqkd:10"), ("previous", year - 1, "kqkd:10")]
    if metric == "gross_margin_change_pp":
        return [
            ("current_gross_profit", year, "kqkd:20"),
            ("current_revenue", year, "kqkd:10"),
            ("previous_gross_profit", year - 1, "kqkd:20"),
            ("previous_revenue", year - 1, "kqkd:10"),
        ]
    if metric in {"roe_pct", "roa_pct"}:
        balance = "cdkt:400" if metric == "roe_pct" else "cdkt:270"
        return [
            ("profit", year, "kqkd:60"),
            ("current_balance", year, balance),
            ("previous_balance", year - 1, balance),
        ]
    if metric in {"equity_turnover_avg", "asset_turnover_avg"}:
        balance = "cdkt:400" if metric == "equity_turnover_avg" else "cdkt:270"
        return [
            ("revenue", year, "kqkd:10"),
            ("current_balance", year, balance),
            ("previous_balance", year - 1, balance),
        ]
    if metric == "net_working_capital":
        return [("current_assets", year, "cdkt:100"), ("current_liabilities", year, "cdkt:310")]
    if metric == "sga_expense":
        return [("selling_expense", year, "kqkd:25"), ("admin_expense", year, "kqkd:26")]
    if metric == "net_finance_result":
        return [("finance_revenue", year, "kqkd:21"), ("finance_expense", year, "kqkd:22")]
    if metric == "net_other_result":
        return [("other_income", year, "kqkd:31"), ("other_expense", year, "kqkd:32")]
    raise ValueError(f"typed oracle does not support metric {metric!r}")


def _operation(metric: str) -> str:
    if metric in RAW_METRICS:
        return "lookup"
    if metric in RATIO_FORMULAS:
        return "linear_ratio"
    return {
        "revenue_growth_pct": "growth_rate",
        "gross_margin_change_pp": "margin_change",
        "roe_pct": "ratio_to_average_balance",
        "roa_pct": "ratio_to_average_balance",
        "equity_turnover_avg": "ratio_to_average_balance",
        "asset_turnover_avg": "ratio_to_average_balance",
        "net_working_capital": "difference",
        "sga_expense": "sum",
        "net_finance_result": "difference",
        "net_other_result": "difference",
    }.get(metric, "unsupported")


class TypedPlanAuditor:
    """Offline auditor; construction and execution are read-only."""

    def __init__(self, root: str | Path, *, sandbox_timeout: float = 10.0) -> None:
        self.root = Path(root).resolve()
        self.compiler = DeterministicFinancialCompiler(self.root)
        self.cube = FinancialCube.read_jsonl(self.root / "build" / "statement_cube.jsonl")
        self.sandbox_timeout = float(sandbox_timeout)

    def _cell(self, ticker: str, year: int, scope: str, key: str) -> StatementCell:
        cell = self.cube.cell(ticker, year, key, scope)
        if cell is None:
            raise ValueError(f"missing cube cell {ticker}/{year}/{scope}/{key}")
        return cell

    @staticmethod
    def _operand(role: str, cell: StatementCell) -> TypedOperand:
        value = abs(float(cell.value)) if cell.metric_key in _COST_KEYS else float(cell.value)
        return TypedOperand(
            role=role,
            ticker=cell.ticker,
            year=cell.year,
            scope=cell.scope,
            metric_key=cell.metric_key,
            value_vnd=value,
            label=cell.label,
            raw=cell.raw,
            scale=float(cell.scale),
            table_ref=cell.table_ref,
            csv_path=cell.csv_path,
            row_idx=int(cell.row_idx),
            col_idx=int(cell.col_idx),
        )

    def _operands_for_metric(
        self, ticker: str, year: int, scope: str, metric: str, *, prefix: str = ""
    ) -> list[TypedOperand]:
        return [
            self._operand(f"{prefix}{role}", self._cell(ticker, target_year, scope, key))
            for role, target_year, key in _metric_dependencies(metric, year)
        ]

    def _metric_value(self, engine: PanelMetricEngine, ticker: str, year: int, metric: str) -> float:
        """Independent typed arithmetic over cube values."""

        if metric == "net_finance_result":
            revenue = engine.raw(ticker, year, "kqkd:21")
            expense = engine.raw(ticker, year, "kqkd:22")
            if revenue is None or expense is None:
                raise ValueError("missing net finance operand")
            return float(revenue) - abs(float(expense))
        if metric == "net_other_result":
            income = engine.raw(ticker, year, "kqkd:31")
            expense = engine.raw(ticker, year, "kqkd:32")
            if income is None or expense is None:
                raise ValueError("missing net other operand")
            return float(income) - abs(float(expense))
        value = engine.value(ticker, year, metric)
        if value is None:
            raise ValueError(f"typed metric unavailable: {ticker}/{year}/{metric}")
        if metric in RAW_METRICS and RAW_METRICS[metric] in _COST_KEYS:
            value = abs(float(value))
        return float(value)

    def _direct_oracle(
        self, question: str, facets: dict[str, Any], compiled: CompiledFinancialQuery
    ) -> OracleResult:
        ticker = str(facets["tickers"][0]).upper()
        year = int(facets["years"][0])
        scope = _scope(facets)
        engine = PanelMetricEngine(self.cube, scope=scope)
        value = self._metric_value(engine, ticker, year, compiled.metric)
        unit_name, unit_multiplier = requested_unit(question)
        if compiled.metric in RAW_METRICS or compiled.metric in _MONETARY_DERIVED:
            value /= float(unit_multiplier)
        operands = tuple(self._operands_for_metric(ticker, year, scope, compiled.metric))
        return OracleResult(
            value=value,
            plan=TypedFinancialPlan(
                version="typed-financial-plan/v1",
                kind="direct",
                operation=_operation(compiled.metric),
                metric=compiled.metric,
                companies=(ticker,),
                periods=(year,),
                scope=scope,
                output_unit=unit_name if compiled.metric in RAW_METRICS or compiled.metric in _MONETARY_DERIVED else compiled.unit,
                parameters={},
                operands=operands,
            ),
        )

    def _panel_oracle(
        self, question: str, facets: dict[str, Any], compiled: CompiledFinancialQuery
    ) -> OracleResult:
        tickers = tuple(str(value).upper() for value in facets.get("tickers", []))
        years = tuple(int(value) for value in facets.get("years", []))
        scope = _scope(facets)
        engine = PanelMetricEngine(self.cube, scope=scope)
        candidates = (
            tuple((tickers[0], year) for year in years)
            if len(tickers) == 1
            else tuple((ticker, years[0]) for ticker in tickers)
        )
        normalized = fold(question)
        descending = bool(re.search(r"\b(?:cao nhat|lon nhat)\b", normalized))
        ascending = bool(re.search(r"\b(?:thap nhat|nho nhat)\b", normalized))
        if descending == ascending:
            raise ValueError("panel direction is not unique")

        filter_specs: list[tuple[str, int]] = []
        if compiled.metric.startswith("signed_filter_extreme:"):
            topology = compiled.metric.split(":", 1)[1].split("->")
            if len(topology) != 3:
                raise ValueError("invalid signed panel topology")
            filter_metric, selector_metric, target_metric = topology
            parsed = self.compiler._signed_filter_extreme_metrics(normalized)
            if parsed is None:
                raise ValueError("signed panel parser disagreement")
            parsed_filters, parsed_selector, parsed_target, parsed_descending = parsed
            if (parsed_selector, parsed_target, parsed_descending) != (
                selector_metric,
                target_metric,
                descending,
            ):
                raise ValueError("signed panel typed-plan disagreement")
            filter_specs = list(parsed_filters)
        else:
            topology = compiled.metric.split(":", 1)[1].split("->")
            if len(topology) != 2:
                raise ValueError("invalid panel topology")
            selector_metric, target_metric = topology
            parsed = self.compiler._extreme_panel_metrics(normalized)
            if parsed != (selector_metric, target_metric, descending):
                raise ValueError("panel typed-plan disagreement")

        unit_name, unit_multiplier = requested_unit(question)
        target_scale = (
            float(unit_multiplier)
            if target_metric in RAW_METRICS or target_metric in _MONETARY_DERIVED
            else 1.0
        )
        rows: list[dict[str, Any]] = []
        operands: list[TypedOperand] = []
        for ticker, year in candidates:
            filters = [self._metric_value(engine, ticker, year, metric) for metric, _sign in filter_specs]
            selector = self._metric_value(engine, ticker, year, selector_metric)
            target = self._metric_value(engine, ticker, year, target_metric) / target_scale
            rows.append(
                {"ticker": ticker, "year": year, "filters": filters, "selector": selector, "target": target}
            )
            for metric, _sign in filter_specs:
                operands.extend(
                    self._operands_for_metric(ticker, year, scope, metric, prefix=f"candidate:{ticker}:{year}:filter:")
                )
            operands.extend(
                self._operands_for_metric(ticker, year, scope, selector_metric, prefix=f"candidate:{ticker}:{year}:selector:")
            )
            operands.extend(
                self._operands_for_metric(ticker, year, scope, target_metric, prefix=f"candidate:{ticker}:{year}:target:")
            )
        eligible = [
            row
            for row in rows
            if all(
                value > 0 if sign > 0 else value < 0
                for value, (_metric, sign) in zip(row["filters"], filter_specs)
            )
        ]
        if not eligible:
            raise ValueError("no eligible panel candidate")
        optimum = (max if descending else min)(row["selector"] for row in eligible)
        selected = [
            row for row in eligible if math.isclose(row["selector"], optimum, rel_tol=1e-12, abs_tol=1e-12)
        ]
        if len(selected) != 1:
            raise ValueError("non-unique panel optimum")
        return OracleResult(
            value=float(selected[0]["target"]),
            plan=TypedFinancialPlan(
                version="typed-financial-plan/v1",
                kind="panel",
                operation="select_max" if descending else "select_min",
                metric=target_metric,
                companies=tickers,
                periods=years,
                scope=scope,
                output_unit=unit_name if target_scale != 1 else compiled.unit,
                parameters={
                    "selector_metric": selector_metric,
                    "target_metric": target_metric,
                    "filters": [{"metric": metric, "sign": sign} for metric, sign in filter_specs],
                    "candidate_count": len(candidates),
                    "selected": {"ticker": selected[0]["ticker"], "year": selected[0]["year"]},
                },
                operands=tuple(operands),
            ),
        )

    def _universe_oracle(
        self, question: str, facets: dict[str, Any], compiled: CompiledFinancialQuery
    ) -> OracleResult:
        years = tuple(sorted(int(value) for value in facets.get("years", [])))
        if len(years) != 2:
            raise ValueError("universe plan requires exactly two years")
        match = _UNIVERSE_THRESHOLD_RE.search(fold(question))
        if match is None:
            raise ValueError("universe threshold is missing")
        threshold = float(match.group("threshold").replace(",", "."))
        previous_year, current_year = years
        scope = _scope(facets)
        engine = PanelMetricEngine(self.cube, scope=scope)
        tickers = tuple(
            sorted(
                {
                    cell.ticker
                    for cell in self.cube.iter_cells()
                    if cell.scope == scope and cell.year in {str(previous_year), str(current_year)}
                }
            )
        )
        rows: list[dict[str, Any]] = []
        operands: list[TypedOperand] = []
        for ticker in tickers:
            try:
                previous = self._metric_value(engine, ticker, previous_year, "inventory")
                current = self._metric_value(engine, ticker, current_year, "inventory")
                target = self._metric_value(engine, ticker, current_year, "cfo_margin_pct")
                candidate_operands = [
                    *self._operands_for_metric(ticker, previous_year, scope, "inventory", prefix=f"candidate:{ticker}:previous:"),
                    *self._operands_for_metric(ticker, current_year, scope, "inventory", prefix=f"candidate:{ticker}:current:"),
                    *self._operands_for_metric(ticker, current_year, scope, "cfo_margin_pct", prefix=f"candidate:{ticker}:target:"),
                ]
            except ValueError:
                continue
            if previous == 0:
                continue
            operands.extend(candidate_operands)
            rows.append(
                {
                    "ticker": ticker,
                    "inventory_change_pct": (current / previous - 1) * 100,
                    "target": target,
                }
            )
        eligible = [row for row in rows if row["inventory_change_pct"] <= -threshold]
        if not eligible:
            raise ValueError("no eligible universe candidate")
        optimum = max(row["target"] for row in eligible)
        selected = [row for row in eligible if math.isclose(row["target"], optimum, rel_tol=1e-12, abs_tol=1e-12)]
        if len(selected) != 1:
            raise ValueError("non-unique universe optimum")
        return OracleResult(
            value=float(optimum),
            plan=TypedFinancialPlan(
                version="typed-financial-plan/v1",
                kind="universe",
                operation="filter_threshold_then_max",
                metric="cfo_margin_pct",
                companies=tickers,
                periods=years,
                scope=scope,
                output_unit=compiled.unit,
                parameters={
                    "filter_metric": "inventory_growth_pct",
                    "predicate": "<=",
                    "threshold": -threshold,
                    "selector_metric": "cfo_margin_pct",
                    "candidate_count": len(rows),
                    "eligible_count": len(eligible),
                    "selected_ticker": selected[0]["ticker"],
                },
                operands=tuple(operands),
            ),
        )

    def typed_oracle(
        self, question: str, facets: dict[str, Any], compiled: CompiledFinancialQuery
    ) -> OracleResult:
        if compiled.metric.startswith("universe:"):
            return self._universe_oracle(question, facets, compiled)
        if compiled.metric.startswith("extreme:") or compiled.metric.startswith("signed_filter_extreme:"):
            return self._panel_oracle(question, facets, compiled)
        return self._direct_oracle(question, facets, compiled)

    @staticmethod
    def _provenance_key(item: dict[str, Any] | TypedOperand) -> tuple[Any, ...]:
        if isinstance(item, TypedOperand):
            return (item.ticker, item.year, item.metric_key, item.table_ref, item.row_idx, item.col_idx, item.raw)
        return (
            str(item.get("ticker", "")),
            str(item.get("year", "")),
            str(item.get("metric_key", "")),
            str(item.get("table_ref", "")),
            int(item.get("row_idx", -1)),
            int(item.get("col_idx", -1)),
            str(item.get("raw", "")),
        )

    def audit_row(self, row: dict[str, Any]) -> dict[str, Any]:
        question = str(row.get("question", ""))
        facets = extract_all_facets(question)
        base: dict[str, Any] = {
            "id": row.get("id"),
            "question": question,
            "facets": facets,
        }
        compiled = self.compiler.compile(question, facets)
        if compiled is None:
            return {
                **base,
                "compiler_accepted": False,
                "status": "refused",
                "refusal": {
                    "stage": "deterministic_compiler",
                    "code": "unsupported_or_ambiguous_query",
                    "message": "The constrained compiler declined this query; no typed answer was guessed.",
                },
                "typed_plan": None,
                "compiler": None,
                "typed_oracle": None,
                "pandas_execution": None,
            }
        compiler_payload = {
            "metric": compiled.metric,
            "answer": compiled.answer,
            "unit": compiled.unit,
            "table_refs": compiled.table_refs,
            "source_cell_count": len(compiled.source_cells),
        }
        try:
            oracle = self.typed_oracle(question, facets, compiled)
        except Exception as exc:
            return {
                **base,
                "compiler_accepted": True,
                "status": "typed_oracle_error",
                "refusal": None,
                "typed_plan": None,
                "compiler": compiler_payload,
                "typed_oracle": {"ok": False, "value": None, "error": f"{type(exc).__name__}: {exc}"},
                "pandas_execution": None,
            }
        replay = run_pandas_code(
            compiled.pandas_query,
            compiled.csv_paths,
            timeout=self.sandbox_timeout,
        )
        compiler_keys = {self._provenance_key(item) for item in compiled.source_cells}
        oracle_keys = {self._provenance_key(item) for item in oracle.plan.operands}
        compiler_oracle = _isclose(compiled.answer, oracle.value)
        pandas_oracle = bool(replay.get("ok")) and _isclose(replay.get("result"), oracle.value)
        provenance_match = compiler_keys == oracle_keys
        if not replay.get("ok"):
            status = "pandas_execution_error"
        elif not compiler_oracle:
            status = "compiler_oracle_mismatch"
        elif not pandas_oracle:
            status = "pandas_oracle_mismatch"
        elif not provenance_match:
            status = "provenance_mismatch"
        else:
            status = "verified"
        return {
            **base,
            "compiler_accepted": True,
            "status": status,
            "refusal": None,
            "typed_plan": oracle.plan.to_dict(),
            "compiler": compiler_payload,
            "typed_oracle": {"ok": True, "value": oracle.value, "error": None},
            "pandas_execution": {
                "ok": bool(replay.get("ok")),
                "value": replay.get("result"),
                "error": replay.get("error"),
            },
            "agreement": {
                "compiler_vs_oracle": compiler_oracle,
                "pandas_vs_oracle": pandas_oracle,
                "compiler_vs_pandas": bool(replay.get("ok")) and _isclose(compiled.answer, replay.get("result")),
            },
            "provenance": {
                "compiler_source_cells": len(compiler_keys),
                "oracle_source_cells": len(oracle_keys),
                "exact_match": provenance_match,
                "compiler_only": [list(key) for key in sorted(compiler_keys - oracle_keys)],
                "oracle_only": [list(key) for key in sorted(oracle_keys - compiler_keys)],
            },
        }

    def audit_rows(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = [self.audit_row(row) for row in rows]
        statuses = Counter(str(record["status"]) for record in records)
        kinds = Counter(
            str(record["typed_plan"]["kind"])
            for record in records
            if record.get("typed_plan") is not None
        )
        accepted = sum(bool(record["compiler_accepted"]) for record in records)
        covered = sum(record.get("typed_plan") is not None for record in records)
        verified = statuses.get("verified", 0)
        return {
            "schema_version": "typed-plan-dual-execution-audit/v1",
            "mode": "read_only_offline",
            "inputs": {
                "statement_cube": {
                    "path": str((self.root / "build" / "statement_cube.jsonl").resolve()),
                    "sha256": hashlib.sha256(
                        (self.root / "build" / "statement_cube.jsonl").read_bytes()
                    ).hexdigest(),
                }
            },
            "verification_scope": {
                "proves": [
                    "typed arithmetic agrees with the compiler answer",
                    "sandbox replay agrees with typed arithmetic",
                    "the compiler and typed evaluator bind the same source cells",
                ],
                "does_not_prove": [
                    "the compiler selected the semantically correct metric or topology",
                    "the normalized cube selected the correct physical source independently",
                    "correctness for compiler-refused questions",
                    "hidden/private-set correctness",
                ],
                "dependency_groups": {
                    "shared_semantic_gate": "compiler acceptance and compiled.metric/topology",
                    "shared_normalized_source": "FinancialCube",
                    "implementation_diverse_arithmetic": "typed fixed operators versus generated Pandas sandbox",
                },
                "blind_second_solver": False,
            },
            "summary": {
                "total_questions": len(records),
                "compiler_accepted": accepted,
                "compiler_refused": statuses.get("refused", 0),
                "typed_plan_covered": covered,
                "accepted_uncovered": accepted - covered,
                "verified": verified,
                "disagreements_or_errors": accepted - verified,
                "coverage_rate": (covered / accepted) if accepted else 1.0,
                "verification_rate": (verified / accepted) if accepted else 1.0,
                "status_counts": dict(sorted(statuses.items())),
                "plan_kind_counts": dict(sorted(kinds.items())),
            },
            "records": records,
        }


def write_audit(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
