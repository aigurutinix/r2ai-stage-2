"""Adapters from the source-controlled solvers to :mod:`financial_ir`.

These adapters create teacher plans for shadow comparison with LLM proposals.
They do not change any existing submission or retrieval choice.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Optional

from .financial_ir import (
    ConventionProfile,
    FactRequest,
    FinancialPlan,
    PlanNode,
    financial_plan_json_schema,
)
from .l1_fact_layer import load_companies, resolve_company
from .l2_formula import FORMULA_SPECS, FormulaSpec
from .set_reasoning import SetSpec


class FormulaGraphBuilder:
    """Translate a safe arithmetic expression into explicit IR nodes."""

    def __init__(self, fact_ids: set[str]):
        self.fact_ids = fact_ids
        self.nodes: list[PlanNode] = []

    def _node(self, op: str, inputs: tuple[str, ...], params: Optional[dict] = None) -> str:
        node_id = f"n{len(self.nodes) + 1}"
        self.nodes.append(PlanNode(node_id, op, inputs, params or {}))
        return node_id

    def visit(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id not in self.fact_ids:
                raise ValueError(f"Unknown formula symbol {node.id!r}")
            return node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"Unsupported formula literal {node.value!r}")
            return self._node("literal", (), {"value": float(node.value)})
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return self._node("negate", (self.visit(node.operand),))
            if isinstance(node.op, ast.UAdd):
                return self.visit(node.operand)
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            operations = {
                ast.Add: "add",
                ast.Sub: "subtract",
                ast.Mult: "multiply",
                ast.Div: "divide",
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise ValueError(f"Unsupported binary operator {type(node.op).__name__}")
            return self._node(operation, (self.visit(node.left), self.visit(node.right)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "abs"
            and len(node.args) == 1
            and not node.keywords
        ):
            return self._node("abs", (self.visit(node.args[0]),))
        raise ValueError(f"Unsupported formula AST node {type(node).__name__}")

    def parse(self, expression: str) -> str:
        tree = ast.parse(expression, mode="eval")
        return self.visit(tree.body)


def formula_spec_to_plan(
    question: dict[str, Any],
    ticker: str,
    spec: FormulaSpec,
) -> FinancialPlan:
    if int(question["id"]) != spec.question_id:
        raise ValueError("Question and FormulaSpec IDs differ")
    facts = tuple(
        FactRequest(
            id=component.name,
            ticker=ticker,
            year=component.year,
            metric=component.metric_text,
            scope=component.scope,
            period=component.period_kind,
            unit=component.target_unit,
            source_preference="auto",
        )
        for component in spec.components
    )
    if len({fact.id for fact in facts}) != len(facts):
        raise ValueError(f"q{spec.question_id}: duplicate formula component names")
    builder = FormulaGraphBuilder({fact.id for fact in facts})
    output = builder.parse(spec.formula)
    has_abs = any(node.op == "abs" for node in builder.nodes)
    output_unit = "percent" if "100.0" in spec.formula and "/" in spec.formula else "number"
    return FinancialPlan(
        question_id=spec.question_id,
        question=question["question"],
        facts=facts,
        nodes=tuple(builder.nodes),
        output=output,
        output_unit=output_unit,
        conventions=ConventionProfile(
            growth_denominator="absolute" if has_abs else "reported"
        ),
        generator="benchmark_locked:l2_formula_registry",
        confidence=1.0,
        assumptions=("Translated from the source-controlled L2 formula registry.",),
    )


def load_question_rows(path: Path, question_ids: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = int(row["id"])
            if question_id in question_ids:
                rows[question_id] = row
    if set(rows) != question_ids:
        raise ValueError(f"Question coverage mismatch; missing={sorted(question_ids - set(rows))}")
    return rows


def export_formula_plans(
    questions_path: Path,
    companies_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    questions = load_question_rows(questions_path, set(FORMULA_SPECS))
    companies = load_companies(companies_path)
    plans = []
    for question_id, spec in sorted(FORMULA_SPECS.items()):
        company, _alias = resolve_company(questions[question_id]["question"], companies)
        plans.append(formula_spec_to_plan(questions[question_id], company.ticker, spec))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            for plan in plans
        ),
        encoding="utf-8",
    )
    return {
        "plans": len(plans),
        "facts": sum(len(plan.facts) for plan in plans),
        "nodes": sum(len(plan.nodes) for plan in plans),
        "output": str(output_path),
    }


def direct_set_spec_to_plan(
    question: dict[str, Any],
    spec: SetSpec,
) -> FinancialPlan:
    """Translate a direct-fact set question into the constrained IR.

    L3 questions can contain rich per-item formulas.  This deliberately small
    adapter accepts only the safer subset where each item is one source fact.
    It is used as metadata for number-masked LLM retrieval, not as a new answer
    solver.
    """

    if int(question["id"]) != spec.question_id:
        raise ValueError("Question and SetSpec IDs differ")
    if any(
        item.expression != "a"
        or item.selector_expression is not None
        or len(item.facts) != 1
        or item.facts[0].name != "a"
        for item in spec.items
    ):
        raise ValueError(f"q{spec.question_id}: not a direct single-fact set question")

    facts = tuple(
        FactRequest(
            id=f"i{index:02d}_a",
            ticker=item.facts[0].ticker,
            year=item.facts[0].year,
            metric=item.facts[0].metric,
            scope=item.facts[0].scope,
            period=item.facts[0].period_kind,
            unit=item.facts[0].target_unit,
            source_preference="auto",
        )
        for index, item in enumerate(spec.items, 1)
    )
    labels = [item.key for item in spec.items]
    nodes = [
        PlanNode("items", "vector", tuple(fact.id for fact in facts), {"labels": labels})
    ]
    if spec.operation in {"sum", "mean", "max_value"}:
        operation = {"max_value": "max"}.get(spec.operation, spec.operation)
        nodes.append(PlanNode("answer", operation, ("items",)))
        units = {fact.unit for fact in facts}
        if len(units) != 1:
            raise ValueError(f"q{spec.question_id}: direct aggregate mixes units {units}")
        output_unit = next(iter(units))
    elif spec.operation == "argmax_key":
        if any(not str(label).isdigit() for label in labels):
            raise ValueError(f"q{spec.question_id}: non-numeric argmax label")
        nodes.append(PlanNode("answer", "argmax_key", ("items",)))
        output_unit = "year"
    elif spec.operation in {
        "count_gt",
        "count_positive",
        "count_nonzero",
        "count_negative",
    }:
        comparator, threshold = {
            "count_gt": ("gt", spec.threshold),
            "count_positive": ("gt", 0.0),
            "count_nonzero": ("ne", 0.0),
            "count_negative": ("lt", 0.0),
        }[spec.operation]
        if threshold is None:
            raise ValueError(f"q{spec.question_id}: count threshold is missing")
        nodes.append(
            PlanNode(
                "answer",
                "count_if",
                ("items",),
                {"comparator": comparator, "threshold": float(threshold)},
            )
        )
        output_unit = "count"
    else:
        raise ValueError(f"q{spec.question_id}: unsupported direct operation {spec.operation}")

    return FinancialPlan(
        question_id=spec.question_id,
        question=question["question"],
        facts=facts,
        nodes=tuple(nodes),
        output="answer",
        output_unit=output_unit,
        generator="benchmark_locked:l3_direct_set_registry",
        confidence=1.0,
        assumptions=(
            "Translated from the source-controlled L3 direct-fact registry for shadow retrieval.",
        ),
    )


def export_l3_direct_plans(
    questions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    from .l3_aggregate import L3_SPECS

    direct_ids = {
        question_id
        for question_id, spec in L3_SPECS.items()
        if all(
            item.expression == "a"
            and item.selector_expression is None
            and len(item.facts) == 1
            and item.facts[0].name == "a"
            for item in spec.items
        )
    }
    questions = load_question_rows(questions_path, direct_ids)
    plans = [
        direct_set_spec_to_plan(questions[question_id], L3_SPECS[question_id])
        for question_id in sorted(direct_ids)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            for plan in plans
        ),
        encoding="utf-8",
    )
    return {
        "plans": len(plans),
        "facts": sum(len(plan.facts) for plan in plans),
        "nodes": sum(len(plan.nodes) for plan in plans),
        "output": str(output_path),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    formulas = subparsers.add_parser("export-formulas")
    formulas.add_argument(
        "--questions",
        type=Path,
        default=Path("ViFinQA/questions/questions.jsonl"),
    )

    l3_direct = subparsers.add_parser("export-l3-direct")
    l3_direct.add_argument(
        "--questions",
        type=Path,
        default=Path("ViFinQA/questions/questions.jsonl"),
    )
    l3_direct.add_argument(
        "--output", type=Path, default=Path("outputs/financial-ir/l3-direct.jsonl")
    )
    formulas.add_argument(
        "--companies", type=Path, default=Path("ViFinQA/code_stock.csv")
    )
    formulas.add_argument(
        "--output", type=Path, default=Path("outputs/financial-ir/l2-formulas.jsonl")
    )

    schema = subparsers.add_parser("schema")
    schema.add_argument(
        "--output", type=Path, default=Path("analysis/financial_plan.schema.json")
    )

    args = parser.parse_args(argv)
    if args.command == "export-formulas":
        result = export_formula_plans(args.questions, args.companies, args.output)
    elif args.command == "export-l3-direct":
        result = export_l3_direct_plans(args.questions, args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(financial_plan_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"schema": str(args.output)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


def complex_set_spec_to_plan(
    question: dict[str, Any],
    spec: SetSpec,
) -> FinancialPlan:
    """Translate a complex (multi-fact) L3 SetSpec into the constrained IR.

    Handles items with expressions involving multiple facts (ratios, sums of
    components, etc.) and all set operations including aggregate_ratio.
    """
    if int(question["id"]) != spec.question_id:
        raise ValueError("Question and SetSpec IDs differ")

    facts: list[FactRequest] = []
    nodes: list[PlanNode] = []
    node_counter = 0

    def next_node_id() -> str:
        nonlocal node_counter
        node_counter += 1
        return f"n{node_counter}"

    def build_item_nodes(item_index: int, item: Any) -> str:
        """Parse item expression into IR nodes; return the output node id."""
        fact_ids = {}
        for f in item.facts:
            fid = f"i{item_index:02d}_{f.name}"
            fact_ids[f.name] = fid
            facts.append(FactRequest(
                id=fid,
                ticker=f.ticker,
                year=f.year,
                metric=f.metric,
                scope=f.scope,
                period=f.period_kind,
                unit=f.target_unit,
                source_preference="auto",
            ))

        # Use FormulaGraphBuilder with prefixed node IDs
        builder = _PrefixedGraphBuilder(fact_ids, next_node_id, nodes)
        return builder.parse(item.expression)

    if spec.operation == "aggregate_ratio":
        # Special case: sum all 'a' facts, sum all 'b' facts, then ratio
        num_ids = []
        den_ids = []
        for item_index, item in enumerate(spec.items, 1):
            for f in item.facts:
                fid = f"i{item_index:02d}_{f.name}"
                facts.append(FactRequest(
                    id=fid,
                    ticker=f.ticker,
                    year=f.year,
                    metric=f.metric,
                    scope=f.scope,
                    period=f.period_kind,
                    unit=f.target_unit,
                    source_preference="auto",
                ))
                if f.name == "a":
                    num_ids.append(fid)
                elif f.name == "b":
                    den_ids.append(fid)

        labels_num = [item.key for item in spec.items]
        labels_den = [item.key for item in spec.items]

        vec_num_id = next_node_id()
        nodes.append(PlanNode(vec_num_id, "vector", tuple(num_ids), {"labels": labels_num}))
        sum_num_id = next_node_id()
        nodes.append(PlanNode(sum_num_id, "sum", (vec_num_id,)))

        vec_den_id = next_node_id()
        nodes.append(PlanNode(vec_den_id, "vector", tuple(den_ids), {"labels": labels_den}))
        sum_den_id = next_node_id()
        nodes.append(PlanNode(sum_den_id, "sum", (vec_den_id,)))

        answer_id = next_node_id()
        nodes.append(PlanNode(answer_id, "ratio_percent", (sum_num_id, sum_den_id)))
        output_unit = "percent"

    else:
        # General case: evaluate each item expression, collect into vector, aggregate
        item_result_ids = []
        labels = []
        for item_index, item in enumerate(spec.items, 1):
            result_id = build_item_nodes(item_index, item)
            item_result_ids.append(result_id)
            labels.append(str(item.key))

        vec_id = next_node_id()
        nodes.append(PlanNode(vec_id, "vector", tuple(item_result_ids), {"labels": labels}))

        if spec.operation == "sum":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "sum", (vec_id,)))
            output_unit = facts[0].unit
        elif spec.operation == "mean":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "mean", (vec_id,)))
            output_unit = facts[0].unit
        elif spec.operation == "max_value":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "max", (vec_id,)))
            output_unit = facts[0].unit
        elif spec.operation == "argmax_key":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "argmax_key", (vec_id,)))
            output_unit = "year" if all(str(item.key).isdigit() for item in spec.items) else "number"
        elif spec.operation == "count_gt":
            if spec.threshold is None:
                raise ValueError(f"q{spec.question_id}: count_gt needs threshold")
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "count_if", (vec_id,),
                                  {"comparator": "gt", "threshold": float(spec.threshold)}))
            output_unit = "count"
        elif spec.operation == "count_positive":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "count_if", (vec_id,),
                                  {"comparator": "gt", "threshold": 0.0}))
            output_unit = "count"
        elif spec.operation == "count_nonzero":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "count_if", (vec_id,),
                                  {"comparator": "ne", "threshold": 0.0}))
            output_unit = "count"
        elif spec.operation == "count_negative":
            answer_id = next_node_id()
            nodes.append(PlanNode(answer_id, "count_if", (vec_id,),
                                  {"comparator": "lt", "threshold": 0.0}))
            output_unit = "count"
        else:
            raise ValueError(f"q{spec.question_id}: unsupported operation {spec.operation}")

    # Determine output unit for ratio expressions
    if spec.operation not in ("aggregate_ratio", "argmax_key", "count_gt",
                              "count_positive", "count_nonzero", "count_negative"):
        # Check if any item expression produces a percentage
        if any("100.0" in item.expression for item in spec.items):
            output_unit = "percent"

    return FinancialPlan(
        question_id=spec.question_id,
        question=question["question"],
        facts=tuple(facts),
        nodes=tuple(nodes),
        output=answer_id,
        output_unit=output_unit,
        conventions=ConventionProfile(expense_sign="absolute"),
        generator="benchmark_locked:l3_complex_set_registry",
        confidence=1.0,
        assumptions=(
            "Translated from the source-controlled L3 complex set registry for shadow retrieval.",
        ),
    )


class _PrefixedGraphBuilder:
    """Like FormulaGraphBuilder but with prefixed node IDs to avoid collisions."""

    def __init__(self, fact_ids: dict[str, str], next_id_fn, nodes: list):
        self.fact_ids = fact_ids
        self._next_id = next_id_fn
        self.nodes = nodes

    def _node(self, op: str, inputs: tuple, params: Optional[dict] = None) -> str:
        node_id = self._next_id()
        self.nodes.append(PlanNode(node_id, op, inputs, params or {}))
        return node_id

    def visit(self, node) -> str:
        import ast as _ast
        if isinstance(node, _ast.Name):
            if node.id not in self.fact_ids:
                raise ValueError(f"Unknown formula symbol {node.id!r}")
            return self.fact_ids[node.id]
        if isinstance(node, _ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"Unsupported formula literal {node.value!r}")
            return self._node("literal", (), {"value": float(node.value)})
        if isinstance(node, _ast.UnaryOp):
            if isinstance(node.op, _ast.USub):
                return self._node("negate", (self.visit(node.operand),))
            if isinstance(node.op, _ast.UAdd):
                return self.visit(node.operand)
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
        if isinstance(node, _ast.BinOp):
            operations = {
                _ast.Add: "add",
                _ast.Sub: "subtract",
                _ast.Mult: "multiply",
                _ast.Div: "divide",
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise ValueError(f"Unsupported binary operator {type(node.op).__name__}")
            return self._node(operation, (self.visit(node.left), self.visit(node.right)))
        if (
            isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Name)
            and node.func.id == "abs"
            and len(node.args) == 1
            and not node.keywords
        ):
            return self._node("abs", (self.visit(node.args[0]),))
        raise ValueError(f"Unsupported formula AST node {type(node).__name__}")

    def parse(self, expression: str) -> str:
        import ast as _ast
        tree = _ast.parse(expression, mode="eval")
        return self.visit(tree.body)


def export_l3_complex_plans(
    questions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Export complex L3 specs as FinancialPlan JSONL for retrieval context generation."""
    from .l3_aggregate import L3_SPECS

    complex_ids = {
        qid for qid, spec in L3_SPECS.items()
        if not all(
            item.expression == "a"
            and item.selector_expression is None
            and len(item.facts) == 1
            and item.facts[0].name == "a"
            for item in spec.items
        )
    }
    questions = load_question_rows(questions_path, complex_ids)
    plans = [
        complex_set_spec_to_plan(questions[qid], L3_SPECS[qid])
        for qid in sorted(complex_ids)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            for plan in plans
        ),
        encoding="utf-8",
    )
    return {
        "plans": len(plans),
        "facts": sum(len(plan.facts) for plan in plans),
        "nodes": sum(len(plan.nodes) for plan in plans),
        "output": str(output_path),
    }
