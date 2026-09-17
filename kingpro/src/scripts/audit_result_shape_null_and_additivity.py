"""Triage subtle dataframe-program risks learned from adjacent benchmarks.

The audit is deliberately read-only and is not a correctness oracle.  It finds
three families which ordinary execution tests often miss:

* missing/blank values silently converted to zero before arithmetic;
* one-to-many merge/join fan-out before SUM/COUNT/MEAN; and
* a result whose shape/type does not match a question asking for a count,
  entity, year, scalar amount, or percentage.

Every finding still needs to be reconciled with the exact BTC source cells.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ZERO_BLANK_RE = re.compile(
    r"(?:in\s*\(\s*['\"]['\"]\s*,\s*['\"]-['\"]\s*\)|fillna\(\s*0(?:\.0)?\s*\))",
    re.IGNORECASE,
)


def result_assignments(tree: ast.AST) -> list[ast.AST]:
    values: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets):
                values.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result" and node.value:
                values.append(node.value)
    return values


def call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def has_validated_merge(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"merge", "join"}:
            continue
        if any(keyword.arg == "validate" for keyword in node.keywords):
            return True
    return False


def dynamic_division_count(tree: ast.AST) -> int:
    """Count divisions whose denominator is data-derived, not a unit constant."""
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        denominator = node.right
        if isinstance(denominator, ast.Constant) and isinstance(denominator.value, (int, float)):
            continue
        count += 1
    return count


def inferred_intent(question: str) -> str:
    q = question.casefold()
    if "bao nhiêu công ty" in q or "bao nhiêu doanh nghiệp" in q or "số năm" in q:
        return "count"
    if "năm nào" in q:
        return "year"
    if any(token in q for token in ("công ty nào", "doanh nghiệp nào", "mã nào")):
        return "entity"
    if any(token in q for token in ("phần trăm", "bao nhiêu %", "tỷ lệ", "tỷ trọng", "biên ")):
        return "percentage"
    return "scalar"


def result_shape(values: list[ast.AST]) -> set[str]:
    shapes: set[str] = set()
    for value in values:
        if isinstance(value, ast.Constant):
            if value.value is None:
                shapes.add("none")
            elif isinstance(value.value, str):
                shapes.add("string")
            else:
                shapes.add("scalar")
        elif isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            shapes.add("collection")
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in {"int", "float", "round", "sum", "max", "min", "len"}:
                shapes.add("scalar")
            elif value.func.id == "str":
                shapes.add("string")
            else:
                shapes.add("unknown")
        elif isinstance(value, ast.Subscript):
            shapes.add("scalar_or_label")
        else:
            shapes.add("unknown")
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    submission = args.submission_dir.resolve()
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict] = []
    parse_failures: list[dict] = []

    for row in rows:
        query = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(query)
        except SyntaxError as error:
            parse_failures.append({"id": row["id"], "error": str(error)})
            continue

        calls = call_names(tree)
        reasons: list[str] = []
        score = 0
        # ``str.join`` is common in generated parsers and is not a dataframe
        # join.  ``merge`` is the unambiguous dataframe operation here.
        has_join = "merge" in calls
        has_selection_or_aggregate = bool(
            calls & {"sum", "mean", "count", "nunique", "idxmax", "idxmin", "max", "min"}
        )
        divisions = dynamic_division_count(tree)
        if has_join and has_selection_or_aggregate and not has_validated_merge(tree):
            score += 4
            reasons.append("merge/join precedes selection/aggregation without cardinality validation")
        if ZERO_BLANK_RE.search(query) and (divisions or has_selection_or_aggregate):
            score += 1
            reasons.append("blank/dash can become numeric zero on a ratio/aggregate path")

        values = result_assignments(tree)
        shapes = result_shape(values)
        intent = inferred_intent(str(row.get("question", "")))
        if "collection" in shapes:
            score += 3
            reasons.append("collection result may violate the scalar submission contract")
        if intent == "entity" and shapes == {"scalar"}:
            score += 2
            reasons.append("entity question returns only a numeric-looking scalar")
        if intent == "count" and not (
            calls & {"len", "sum", "count", "nunique"} or ".shape[0]" in query.replace(" ", "")
        ):
            score += 2
            reasons.append("count question lacks an explicit counting operator")
        if divisions and "ZeroDivisionError" not in query and "!= 0" not in query and "> 0" not in query:
            score += 1
            reasons.append(f"{divisions} data-derived division path(s) have no visible zero-denominator guard")

        if score >= 2:
            findings.append({
                "id": int(row["id"]),
                "score": score,
                "intent": intent,
                "result_shapes": sorted(shapes),
                "reasons": reasons,
                "question": row.get("question"),
                "answer": row.get("answer"),
                "relevant_tables": row.get("relevant_tables", []),
            })

    findings.sort(key=lambda item: (-item["score"], item["id"]))
    payload = {
        "submission": str(submission),
        "checked": len(rows),
        "finding_count": len(findings),
        "high_priority_count": sum(item["score"] >= 4 for item in findings),
        "parse_failure_count": len(parse_failures),
        "policy": "Read-only triage. Reconcile every finding with BTC source cells before changing an answer.",
        "findings": findings,
        "parse_failures": parse_failures,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"findings", "parse_failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
