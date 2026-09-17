"""Strict financial-plan IR and deterministic executor/compiler.

LLMs may propose this JSON representation, but they cannot provide source
numbers or executable Python.  Facts are resolved separately from the corpus;
the operator DAG is validated before deterministic evaluation or compilation.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


SCHEMA_VERSION = "1.0"
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")

SCOPES = {"consolidated", "separate", "aggregated", "any"}
PERIODS = {"end_or_flow", "start", "end", "flow"}
UNITS = {
    "number",
    "ratio",
    "percent",
    "count",
    "year",
    "shares",
    "VND_1",
    "VND_1e3",
    "VND_1e6",
    "VND_1e8",
    "VND_1e9",
    "VND_1e11",
    "VND_1e12",
    "VND_per_share",
    "USD_1",
    "USD_1e3",
    "USD_1e6",
}
SOURCE_PREFERENCES = {"auto", "primary_statement", "note_table"}
COMPARATORS = {"gt", "ge", "lt", "le", "eq", "ne"}

CONVENTION_VALUES = {
    "scope_policy": {"explicit", "consolidated_first", "separate_first"},
    "expense_sign": {"reported", "absolute"},
    "growth_denominator": {"reported", "absolute"},
    "balance_basis": {"explicit", "ending", "average"},
    "profit_attribution": {"total", "parent", "explicit"},
    "interest_source": {
        "auto",
        "primary_statement",
        "finance_cost_note",
        "borrowing_cost_note",
    },
    "zero_division": {"reject"},
}

UNARY_OPS = {"identity", "negate", "abs"}
BINARY_OPS = {"add", "subtract", "multiply", "divide"}
VECTOR_AGGREGATES = {"sum", "mean", "median", "min", "max", "count"}
VECTOR_TRANSFORMS = {"filter", "top_k", "bottom_k"}
VECTOR_SELECTORS = {"argmax_key", "argmin_key"}
PAIR_VECTOR_OPS = {"filter_by", "select_argmax", "select_argmin"}
OPERATORS = {
    "literal",
    "scale",
    "ratio_percent",
    "percent_change",
    "vector",
    "round",
    "count_if",
    *UNARY_OPS,
    *BINARY_OPS,
    *VECTOR_AGGREGATES,
    *VECTOR_TRANSFORMS,
    *VECTOR_SELECTORS,
    *PAIR_VECTOR_OPS,
}


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise ValueError(f"{context} fields mismatch; missing={missing}, extra={extra}")


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{context} must match {IDENTIFIER_RE.pattern}")
    return value


def _finite_number(value: Any, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ConventionProfile:
    scope_policy: str = "explicit"
    expense_sign: str = "reported"
    growth_denominator: str = "absolute"
    balance_basis: str = "explicit"
    profit_attribution: str = "explicit"
    interest_source: str = "auto"
    zero_division: str = "reject"

    def __post_init__(self) -> None:
        for name, allowed in CONVENTION_VALUES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"Invalid convention {name}={value!r}")

    @classmethod
    def from_dict(cls, raw: Any) -> "ConventionProfile":
        value = _require_object(raw, "conventions")
        _require_keys(value, set(CONVENTION_VALUES), "conventions")
        return cls(**value)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in CONVENTION_VALUES}


@dataclass(frozen=True)
class FactRequest:
    id: str
    ticker: str
    year: int
    metric: str
    scope: str
    period: str
    unit: str
    source_preference: str = "auto"
    row_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.id, "fact.id")
        if not TICKER_RE.fullmatch(self.ticker):
            raise ValueError(f"Invalid ticker {self.ticker!r}")
        if not isinstance(self.year, int) or isinstance(self.year, bool) or not 2010 <= self.year <= 2030:
            raise ValueError(f"Invalid fact year {self.year!r}")
        if (
            not isinstance(self.metric, str)
            or not self.metric.strip()
            or len(self.metric) > 240
            or any(character in self.metric for character in "{}[];\n\r")
        ):
            raise ValueError("fact.metric must be a short plain-text label")
        if self.scope not in SCOPES:
            raise ValueError(f"Invalid fact scope {self.scope!r}")
        if self.period not in PERIODS:
            raise ValueError(f"Invalid fact period {self.period!r}")
        if self.unit not in UNITS:
            raise ValueError(f"Invalid fact unit {self.unit!r}")
        if self.source_preference not in SOURCE_PREFERENCES:
            raise ValueError(f"Invalid source preference {self.source_preference!r}")
        if self.row_ref is not None and not re.fullmatch(r"t\d+r\d+", self.row_ref):
            raise ValueError(f"Invalid fact row_ref {self.row_ref!r}")

    @classmethod
    def from_dict(cls, raw: Any) -> "FactRequest":
        value = _require_object(raw, "fact")
        required = {
            "id",
            "ticker",
            "year",
            "metric",
            "scope",
            "period",
            "unit",
            "source_preference",
        }
        allowed = {*required, "row_ref"}
        if not required <= set(value) or not set(value) <= allowed:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - allowed)
            raise ValueError(f"fact fields mismatch; missing={missing}, extra={extra}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "ticker": self.ticker,
            "year": self.year,
            "metric": self.metric,
            "scope": self.scope,
            "period": self.period,
            "unit": self.unit,
            "source_preference": self.source_preference,
        }
        if self.row_ref is not None:
            result["row_ref"] = self.row_ref
        return result


@dataclass(frozen=True)
class PlanNode:
    id: str
    op: str
    inputs: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.id, "node.id")
        if self.op not in OPERATORS:
            raise ValueError(f"Unsupported operator {self.op!r}")
        if not isinstance(self.inputs, tuple) or any(
            not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value)
            for value in self.inputs
        ):
            raise ValueError(f"node {self.id}: inputs must be identifier strings")
        if not isinstance(self.params, dict):
            raise ValueError(f"node {self.id}: params must be an object")
        self._validate_shape()

    def _validate_shape(self) -> None:
        exact_arities = {
            "literal": 0,
            "round": 1,
            "scale": 1,
            "percent_change": 2,
            "ratio_percent": 2,
            "count_if": 1,
            "filter": 1,
            "top_k": 1,
            "bottom_k": 1,
            "argmax_key": 1,
            "argmin_key": 1,
            "filter_by": 2,
            "select_argmax": 2,
            "select_argmin": 2,
            **{op: 1 for op in UNARY_OPS | VECTOR_AGGREGATES},
            **{op: 2 for op in BINARY_OPS},
        }
        if self.op == "vector":
            if not self.inputs:
                raise ValueError(f"node {self.id}: vector needs at least one input")
        elif len(self.inputs) != exact_arities[self.op]:
            raise ValueError(
                f"node {self.id}: {self.op} needs {exact_arities[self.op]} inputs"
            )

        expected_params: set[str]
        if self.op == "literal":
            expected_params = {"value"}
            _finite_number(self.params.get("value"), f"node {self.id}.value")
        elif self.op == "vector":
            expected_params = {"labels"}
            labels = self.params.get("labels")
            if (
                not isinstance(labels, list)
                or len(labels) != len(self.inputs)
                or not all(isinstance(label, str) and label for label in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError(f"node {self.id}: vector labels must be unique strings")
        elif self.op in {"filter", "filter_by", "count_if"}:
            expected_params = {"comparator", "threshold"}
            if self.params.get("comparator") not in COMPARATORS:
                raise ValueError(f"node {self.id}: invalid comparator")
            _finite_number(self.params.get("threshold"), f"node {self.id}.threshold")
        elif self.op in {"top_k", "bottom_k"}:
            expected_params = {"k"}
            k = self.params.get("k")
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise ValueError(f"node {self.id}: k must be a positive integer")
        elif self.op == "round":
            expected_params = {"digits"}
            digits = self.params.get("digits")
            if not isinstance(digits, int) or isinstance(digits, bool) or not -12 <= digits <= 12:
                raise ValueError(f"node {self.id}: digits must be an integer from -12 to 12")
        elif self.op == "scale":
            expected_params = {"factor"}
            _finite_number(self.params.get("factor"), f"node {self.id}.factor")
        elif self.op == "percent_change":
            expected_params = {"denominator"}
            if self.params.get("denominator") not in {"absolute", "reported"}:
                raise ValueError(f"node {self.id}: invalid percent-change denominator")
        else:
            expected_params = set()
        if set(self.params) != expected_params:
            raise ValueError(
                f"node {self.id}: {self.op} params must be {sorted(expected_params)}"
            )

    @classmethod
    def from_dict(cls, raw: Any) -> "PlanNode":
        value = _require_object(raw, "node")
        _require_keys(value, {"id", "op", "inputs", "params"}, "node")
        if not isinstance(value["inputs"], list):
            raise ValueError("node.inputs must be an array")
        return cls(
            id=value["id"],
            op=value["op"],
            inputs=tuple(value["inputs"]),
            params=value["params"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "op": self.op, "inputs": list(self.inputs), "params": self.params}


@dataclass(frozen=True)
class FinancialPlan:
    question_id: int
    question: str
    facts: tuple[FactRequest, ...]
    nodes: tuple[PlanNode, ...]
    output: str
    output_unit: str = "number"
    conventions: ConventionProfile = field(default_factory=ConventionProfile)
    generator: str = "benchmark_locked"
    confidence: float = 1.0
    assumptions: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported FinancialPlan schema {self.schema_version!r}")
        if not isinstance(self.question_id, int) or isinstance(self.question_id, bool) or self.question_id < 1:
            raise ValueError("question_id must be a positive integer")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not self.facts:
            raise ValueError("A FinancialPlan needs at least one fact")
        if self.output_unit not in UNITS:
            raise ValueError(f"Invalid output unit {self.output_unit!r}")
        if not isinstance(self.generator, str) or not self.generator.strip() or len(self.generator) > 160:
            raise ValueError("generator must be a short non-empty string")
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(value, str) or not value.strip() or len(value) > 300
            for value in self.assumptions
        ):
            raise ValueError("assumptions must be short non-empty strings")
        self._validate_graph()

    def _validate_graph(self) -> None:
        known: dict[str, str] = {}
        for fact in self.facts:
            if fact.id in known:
                raise ValueError(f"Duplicate graph id {fact.id!r}")
            known[fact.id] = "scalar"
        for node in self.nodes:
            if node.id in known:
                raise ValueError(f"Duplicate graph id {node.id!r}")
            missing = [value for value in node.inputs if value not in known]
            if missing:
                raise ValueError(f"node {node.id}: unknown/forward inputs {missing}")
            input_types = [known[value] for value in node.inputs]
            known[node.id] = _output_type(node, input_types)
        if self.output not in known:
            raise ValueError(f"Unknown output reference {self.output!r}")
        if known[self.output] != "scalar":
            raise ValueError("FinancialPlan output must be scalar")

    @classmethod
    def from_dict(cls, raw: Any) -> "FinancialPlan":
        value = _require_object(raw, "FinancialPlan")
        keys = {
            "schema_version",
            "question_id",
            "question",
            "facts",
            "nodes",
            "output",
            "output_unit",
            "conventions",
            "generator",
            "confidence",
            "assumptions",
        }
        _require_keys(value, keys, "FinancialPlan")
        if not isinstance(value["facts"], list) or not isinstance(value["nodes"], list):
            raise ValueError("facts and nodes must be arrays")
        if not isinstance(value["assumptions"], list):
            raise ValueError("assumptions must be an array")
        return cls(
            schema_version=value["schema_version"],
            question_id=value["question_id"],
            question=value["question"],
            facts=tuple(FactRequest.from_dict(fact) for fact in value["facts"]),
            nodes=tuple(PlanNode.from_dict(node) for node in value["nodes"]),
            output=value["output"],
            output_unit=value["output_unit"],
            conventions=ConventionProfile.from_dict(value["conventions"]),
            generator=value["generator"],
            confidence=value["confidence"],
            assumptions=tuple(value["assumptions"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "FinancialPlan":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question_id": self.question_id,
            "question": self.question,
            "facts": [fact.to_dict() for fact in self.facts],
            "nodes": [node.to_dict() for node in self.nodes],
            "output": self.output,
            "output_unit": self.output_unit,
            "conventions": self.conventions.to_dict(),
            "generator": self.generator,
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _output_type(node: PlanNode, input_types: list[str]) -> str:
    if node.op == "literal":
        return "scalar"
    if node.op == "vector":
        if any(value != "scalar" for value in input_types):
            raise ValueError(f"node {node.id}: vector inputs must be scalar")
        return "vector"
    if node.op in UNARY_OPS | {"round", "scale"}:
        return input_types[0]
    if node.op in BINARY_OPS | {"ratio_percent", "percent_change"}:
        return "vector" if "vector" in input_types else "scalar"
    if node.op in VECTOR_AGGREGATES | VECTOR_SELECTORS | {"count_if"}:
        if input_types != ["vector"]:
            raise ValueError(f"node {node.id}: {node.op} requires one vector")
        return "scalar"
    if node.op in VECTOR_TRANSFORMS:
        if input_types != ["vector"]:
            raise ValueError(f"node {node.id}: {node.op} requires one vector")
        return "vector"
    if node.op in PAIR_VECTOR_OPS:
        if input_types != ["vector", "vector"]:
            raise ValueError(f"node {node.id}: {node.op} requires two vectors")
        return "vector" if node.op == "filter_by" else "scalar"
    raise AssertionError(node.op)


def _scalar(value: Any, context: str) -> float:
    if isinstance(value, dict):
        raise ValueError(f"{context} expected scalar, got vector")
    return _finite_number(value, context)


def _vector(value: Any, context: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{context} expected a non-empty vector")
    return {str(key): _scalar(item, context) for key, item in value.items()}


def _compare(value: float, comparator: str, threshold: float) -> bool:
    return {
        "gt": value > threshold,
        "ge": value >= threshold,
        "lt": value < threshold,
        "le": value <= threshold,
        "eq": value == threshold,
        "ne": value != threshold,
    }[comparator]


def _unary(value: Any, function) -> Any:
    if isinstance(value, dict):
        return {key: function(number) for key, number in value.items()}
    return function(_scalar(value, "unary operand"))


def _binary(left: Any, right: Any, function) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError("Element-wise vector operands have different labels")
        return {key: function(left[key], right[key]) for key in left}
    if isinstance(left, dict):
        scalar = _scalar(right, "right operand")
        return {key: function(value, scalar) for key, value in left.items()}
    if isinstance(right, dict):
        scalar = _scalar(left, "left operand")
        return {key: function(scalar, value) for key, value in right.items()}
    return function(_scalar(left, "left operand"), _scalar(right, "right operand"))


def evaluate_plan(plan: FinancialPlan | FinancialPlanV2, fact_values: Mapping[str, float]) -> float:
    """Evaluate a validated plan using externally grounded scalar facts."""

    expected = {fact.id for fact in plan.facts}
    if set(fact_values) != expected:
        raise ValueError(
            f"Fact bindings mismatch; missing={sorted(expected - set(fact_values))}, "
            f"extra={sorted(set(fact_values) - expected)}"
        )
    values: dict[str, Any] = {
        fact.id: _finite_number(fact_values[fact.id], f"fact {fact.id}")
        for fact in plan.facts
    }
    for node in plan.nodes:
        inputs = [values[value] for value in node.inputs]
        if node.op == "literal":
            result: Any = float(node.params["value"])
        elif node.op == "vector":
            result = {
                label: _scalar(value, f"node {node.id}")
                for label, value in zip(node.params["labels"], inputs)
            }
        elif node.op == "identity":
            result = inputs[0]
        elif node.op == "negate":
            result = _unary(inputs[0], lambda value: -value)
        elif node.op == "abs":
            result = _unary(inputs[0], abs)
        elif node.op in BINARY_OPS:
            functions = {
                "add": lambda left, right: left + right,
                "subtract": lambda left, right: left - right,
                "multiply": lambda left, right: left * right,
                "divide": lambda left, right: left / right if right != 0 else _raise_zero(),
            }
            result = _binary(inputs[0], inputs[1], functions[node.op])
        elif node.op == "scale":
            result = _unary(inputs[0], lambda value: value * float(node.params["factor"]))
        elif node.op == "ratio_percent":
            result = _binary(
                inputs[0],
                inputs[1],
                lambda left, right: left / right * 100.0 if right != 0 else _raise_zero(),
            )
        elif node.op == "percent_change":
            denominator = node.params["denominator"]
            result = _binary(
                inputs[0],
                inputs[1],
                lambda current, previous: (
                    (current - previous)
                    / (abs(previous) if denominator == "absolute" else previous)
                    * 100.0
                    if previous != 0
                    else _raise_zero()
                ),
            )
        elif node.op in VECTOR_AGGREGATES:
            vector = _vector(inputs[0], f"node {node.id}")
            numbers = list(vector.values())
            functions = {
                "sum": sum,
                "mean": statistics.fmean,
                "median": statistics.median,
                "min": min,
                "max": max,
                "count": lambda values: len(values),
            }
            result = float(functions[node.op](numbers))
        elif node.op in {"filter", "count_if"}:
            vector = _vector(inputs[0], f"node {node.id}")
            selected = {
                key: value
                for key, value in vector.items()
                if _compare(value, node.params["comparator"], float(node.params["threshold"]))
            }
            result = float(len(selected)) if node.op == "count_if" else selected
        elif node.op == "filter_by":
            vector = _vector(inputs[0], f"node {node.id}")
            selector = _vector(inputs[1], f"node {node.id}")
            if set(vector) != set(selector):
                raise ValueError(f"node {node.id}: filter vectors have different labels")
            result = {
                key: value
                for key, value in vector.items()
                if _compare(
                    selector[key],
                    node.params["comparator"],
                    float(node.params["threshold"]),
                )
            }
        elif node.op in {"top_k", "bottom_k"}:
            vector = _vector(inputs[0], f"node {node.id}")
            reverse = node.op == "top_k"
            ordered = sorted(vector.items(), key=lambda item: item[1], reverse=reverse)
            result = dict(ordered[: node.params["k"]])
        elif node.op in VECTOR_SELECTORS:
            vector = _vector(inputs[0], f"node {node.id}")
            function = max if node.op == "argmax_key" else min
            result = float(function(vector, key=vector.get))
        elif node.op in {"select_argmax", "select_argmin"}:
            vector = _vector(inputs[0], f"node {node.id}")
            selector = _vector(inputs[1], f"node {node.id}")
            if set(vector) != set(selector):
                raise ValueError(f"node {node.id}: selector vectors have different labels")
            function = max if node.op == "select_argmax" else min
            result = vector[function(selector, key=selector.get)]
        elif node.op == "round":
            result = _unary(inputs[0], lambda value: round(value, node.params["digits"]))
        else:  # pragma: no cover - PlanNode rejects unknown operators
            raise AssertionError(node.op)
        values[node.id] = result
    return _scalar(values[plan.output], "plan output")


def _raise_zero() -> float:
    raise ZeroDivisionError("FinancialPlan division by zero")


def _comparator_symbol(comparator: str) -> str:
    return {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}[comparator]


def compile_pandas(plan: FinancialPlan | FinancialPlanV2, fact_expressions: Mapping[str, str]) -> str:
    """Compile a validated plan from trusted fact-cell expressions to Pandas."""

    expected = {fact.id for fact in plan.facts}
    if set(fact_expressions) != expected:
        raise ValueError(
            f"Fact expressions mismatch; missing={sorted(expected - set(fact_expressions))}, "
            f"extra={sorted(set(fact_expressions) - expected)}"
        )
    expressions: dict[str, str] = {}
    for fact in plan.facts:
        expression = fact_expressions[fact.id]
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"fact {fact.id}: empty Pandas expression")
        expressions[fact.id] = expression.strip()
    for node in plan.nodes:
        inputs = [expressions[value] for value in node.inputs]
        if node.op == "literal":
            expression = repr(float(node.params["value"]))
        elif node.op == "vector":
            labels = repr(node.params["labels"])
            expression = f"pd.Series([{', '.join(f'({value})' for value in inputs)}], index={labels}, dtype='float64')"
        elif node.op == "identity":
            expression = f"({inputs[0]})"
        elif node.op == "negate":
            expression = f"-({inputs[0]})"
        elif node.op == "abs":
            expression = f"abs({inputs[0]})"
        elif node.op in BINARY_OPS:
            symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[node.op]
            expression = f"(({inputs[0]}) {symbol} ({inputs[1]}))"
        elif node.op == "scale":
            expression = f"(({inputs[0]}) * {repr(float(node.params['factor']))})"
        elif node.op == "ratio_percent":
            expression = f"(({inputs[0]}) / ({inputs[1]}) * 100.0)"
        elif node.op == "percent_change":
            denominator = (
                f"abs({inputs[1]})"
                if node.params["denominator"] == "absolute"
                else f"({inputs[1]})"
            )
            expression = f"((({inputs[0]}) - ({inputs[1]})) / {denominator} * 100.0)"
        elif node.op in {"sum", "mean", "median", "min", "max"}:
            expression = f"float(({inputs[0]}).{node.op}())"
        elif node.op == "count":
            expression = f"int(({inputs[0]}).count())"
        elif node.op in {"filter", "count_if"}:
            symbol = _comparator_symbol(node.params["comparator"])
            threshold = repr(float(node.params["threshold"]))
            mask = f"(({inputs[0]}) {symbol} {threshold})"
            expression = (
                f"int(({mask}).sum())"
                if node.op == "count_if"
                else f"({inputs[0]})[{mask}]"
            )
        elif node.op == "filter_by":
            symbol = _comparator_symbol(node.params["comparator"])
            threshold = repr(float(node.params["threshold"]))
            expression = f"({inputs[0]})[((({inputs[1]}) {symbol} {threshold}))]"
        elif node.op in {"top_k", "bottom_k"}:
            method = "nlargest" if node.op == "top_k" else "nsmallest"
            expression = f"({inputs[0]}).{method}({node.params['k']})"
        elif node.op in {"argmax_key", "argmin_key"}:
            method = "idxmax" if node.op == "argmax_key" else "idxmin"
            expression = f"float(({inputs[0]}).{method}())"
        elif node.op in {"select_argmax", "select_argmin"}:
            method = "idxmax" if node.op == "select_argmax" else "idxmin"
            expression = f"float(({inputs[0]}).loc[({inputs[1]}).{method}()])"
        elif node.op == "round":
            expression = f"round(({inputs[0]}), {node.params['digits']})"
        else:  # pragma: no cover - PlanNode rejects unknown operators
            raise AssertionError(node.op)
        expressions[node.id] = expression
    return expressions[plan.output]


def financial_plan_json_schema() -> dict[str, Any]:
    """Return the strict JSON schema supplied to local structured-output LLMs."""

    convention_properties = {
        name: {"type": "string", "enum": sorted(values)}
        for name, values in CONVENTION_VALUES.items()
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ViFinQA FinancialPlan",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "question_id",
            "question",
            "facts",
            "nodes",
            "output",
            "output_unit",
            "conventions",
            "generator",
            "confidence",
            "assumptions",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "question_id": {"type": "integer", "minimum": 1},
            "question": {"type": "string", "minLength": 1},
            "facts": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "ticker",
                        "year",
                        "metric",
                        "scope",
                        "period",
                        "unit",
                        "source_preference",
                    ],
                    "properties": {
                        "id": {"type": "string", "pattern": IDENTIFIER_RE.pattern},
                        "ticker": {"type": "string", "pattern": TICKER_RE.pattern},
                        "year": {"type": "integer", "minimum": 2010, "maximum": 2030},
                        "metric": {"type": "string", "minLength": 1, "maxLength": 240},
                        "scope": {"type": "string", "enum": sorted(SCOPES)},
                        "period": {"type": "string", "enum": sorted(PERIODS)},
                        "unit": {"type": "string", "enum": sorted(UNITS)},
                        "source_preference": {
                            "type": "string",
                            "enum": sorted(SOURCE_PREFERENCES),
                        },
                        "row_ref": {
                            "type": "string",
                            "pattern": "^t[0-9]+r[0-9]+$",
                        },
                    },
                },
            },
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "inputs", "params"],
                    "properties": {
                        "id": {"type": "string", "pattern": IDENTIFIER_RE.pattern},
                        "op": {"type": "string", "enum": sorted(OPERATORS)},
                        "inputs": {
                            "type": "array",
                            "items": {"type": "string", "pattern": IDENTIFIER_RE.pattern},
                        },
                        "params": {"type": "object"},
                    },
                },
            },
            "output": {"type": "string", "pattern": IDENTIFIER_RE.pattern},
            "output_unit": {"type": "string", "enum": sorted(UNITS)},
            "conventions": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(CONVENTION_VALUES),
                "properties": convention_properties,
            },
            "generator": {"type": "string", "minLength": 1, "maxLength": 160},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "assumptions": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 300},
            },
        },
    }

@dataclass(frozen=True)
class InferenceRequest:
    question: str
    request_id: str | int | None = None

@dataclass(frozen=True)
class FinancialPlanV2:
    question: str
    facts: tuple[FactRequest, ...]
    nodes: tuple[PlanNode, ...]
    output: str
    output_unit: str = "number"
    conventions: ConventionProfile = field(default_factory=ConventionProfile)
    generator: str = "benchmark_locked"
    confidence: float = 1.0
    assumptions: tuple[str, ...] = ()
    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        if self.schema_version != "2.0":
            raise ValueError(f"Unsupported FinancialPlanV2 schema {self.schema_version!r}")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not self.facts:
            raise ValueError("A FinancialPlanV2 needs at least one fact")
        if self.output_unit not in UNITS:
            raise ValueError(f"Invalid output unit {self.output_unit!r}")
        if not isinstance(self.generator, str) or not self.generator.strip() or len(self.generator) > 160:
            raise ValueError("generator must be a short non-empty string")
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(value, str) or not value.strip() or len(value) > 300
            for value in self.assumptions
        ):
            raise ValueError("assumptions must be short non-empty strings")
        self._validate_graph()

    def _validate_graph(self) -> None:
        known: dict[str, str] = {}
        for fact in self.facts:
            if fact.id in known:
                raise ValueError(f"Duplicate graph id {fact.id!r}")
            known[fact.id] = "scalar"
        for node in self.nodes:
            if node.id in known:
                raise ValueError(f"Duplicate graph id {node.id!r}")
            missing = [value for value in node.inputs if value not in known]
            if missing:
                raise ValueError(f"node {node.id}: unknown/forward inputs {missing}")
            input_types = [known[value] for value in node.inputs]
            known[node.id] = _output_type(node, input_types)
        if self.output not in known:
            raise ValueError(f"Unknown output reference {self.output!r}")
        if known[self.output] != "scalar":
            raise ValueError("FinancialPlanV2 output must be scalar")

    @classmethod
    def from_dict(cls, raw: Any) -> "FinancialPlanV2":
        value = _require_object(raw, "FinancialPlanV2")
        keys = {
            "schema_version",
            "question",
            "facts",
            "nodes",
            "output",
            "output_unit",
            "conventions",
            "generator",
            "confidence",
            "assumptions",
        }
        _require_keys(value, keys, "FinancialPlanV2")
        if not isinstance(value["facts"], list) or not isinstance(value["nodes"], list):
            raise ValueError("facts and nodes must be arrays")
        if not isinstance(value["assumptions"], list):
            raise ValueError("assumptions must be an array")
        return cls(
            schema_version=value["schema_version"],
            question=value["question"],
            facts=tuple(FactRequest.from_dict(fact) for fact in value["facts"]),
            nodes=tuple(PlanNode.from_dict(node) for node in value["nodes"]),
            output=value["output"],
            output_unit=value["output_unit"],
            conventions=ConventionProfile.from_dict(value["conventions"]),
            generator=value["generator"],
            confidence=value["confidence"],
            assumptions=tuple(value["assumptions"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "FinancialPlanV2":
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question": self.question,
            "facts": [fact.to_dict() for fact in self.facts],
            "nodes": [node.to_dict() for node in self.nodes],
            "output": self.output,
            "output_unit": self.output_unit,
            "conventions": self.conventions.to_dict(),
            "generator": self.generator,
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_v1(cls, v1_plan: "FinancialPlan") -> "FinancialPlanV2":
        return cls(
            question=v1_plan.question,
            facts=v1_plan.facts,
            nodes=v1_plan.nodes,
            output=v1_plan.output,
            output_unit=v1_plan.output_unit,
            conventions=v1_plan.conventions,
            generator=v1_plan.generator,
            confidence=v1_plan.confidence,
            assumptions=v1_plan.assumptions,
            schema_version="2.0",
        )
