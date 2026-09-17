"""Audit mean-of-ratios questions for aggregation-order mistakes.

``mean(numerator / denominator)`` and ``sum(numerator) / sum(denominator)``
answer different business questions.  The two values can be deceptively close
when the sampled companies have similar scale, so grader replay cannot prove
that the program follows the wording.

This audit is intentionally conservative.  It reports a finding only when a
question clearly asks for an average ratio while the result dependency graph
contains the canonical ratio-of-sums shape.  Programs that cannot be proved
statically are kept in an unresolved queue and are never auto-rewritten.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import unicodedata
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


AVERAGE = r"(?:trung binh|binh quan)"
RATIO = r"(?:ty le|ti le|ty trong|ti trong|ty suat|ti suat|he so|bien loi nhuan)"
AVERAGE_INPUT = (
    r"(?:(?:tong\s+)?(?:tai san|von chu so huu|hang ton kho)\s+"
    r"(?:thuan\s+)?(?:trung binh|binh quan)"
    r"|(?:trung binh|binh quan)\s+(?:tong\s+)?"
    r"(?:tai san|von chu so huu|hang ton kho))"
)


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _asks_for_mean_ratio(question: str) -> bool:
    text = _fold(question)
    if not re.search(AVERAGE, text) or not re.search(rf"(?:{RATIO}|%)", text):
        return False

    # An average balance is an input to ROA/ROE/inventory-days, not an output
    # average across companies or years.  Remove those noun phrases before
    # checking whether a separate average-ratio phrase remains.
    without_average_inputs = re.sub(AVERAGE_INPUT, "", text)
    output_mean = re.search(
        rf"(?:{AVERAGE}.{{0,45}}{RATIO}|{RATIO}.{{0,55}}{AVERAGE})",
        without_average_inputs,
    )
    return bool(output_mean)


def _target_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        try:
            return ast.unparse(node)
        except Exception:  # pragma: no cover - Python with no ast.unparse
            return None
    return None


class ProgramIndex:
    """Index top-level assignments while excluding parser/source helpers."""

    def __init__(self, tree: ast.Module) -> None:
        self.assignments: dict[str, ast.AST] = {}
        self.results: list[ast.AST] = []
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    key = _target_key(target)
                    is_result = isinstance(target, ast.Name) and target.id == "result"
                    if key and not is_result:
                        self.assignments[key] = statement.value
                    if is_result:
                        self.results.append(statement.value)
            elif isinstance(statement, ast.AnnAssign):
                key = _target_key(statement.target)
                is_result = isinstance(statement.target, ast.Name) and statement.target.id == "result"
                if key and statement.value is not None and not is_result:
                    self.assignments[key] = statement.value
                if (
                    isinstance(statement.target, ast.Name)
                    and statement.target.id == "result"
                    and statement.value is not None
                ):
                    self.results.append(statement.value)

    def resolve(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> ast.AST:
        key = _target_key(node)
        if key and key in self.assignments and key not in seen:
            return self.resolve(self.assignments[key], seen | {key})
        return node


def _unwrap(node: ast.AST, index: ProgramIndex) -> ast.AST:
    """Remove harmless result wrappers and resolve simple variables."""

    current = node
    for _ in range(20):
        resolved = index.resolve(current)
        if resolved is not current:
            current = resolved
            continue
        if isinstance(current, ast.Call):
            name = current.func.id if isinstance(current.func, ast.Name) else ""
            if name in {"round", "float", "abs"} and current.args:
                current = current.args[0]
                continue
        break
    return current


def _is_call(node: ast.AST, names: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in names
    if isinstance(node.func, ast.Name):
        return node.func.id in names
    return False


def _contains_call(node: ast.AST, names: set[str]) -> bool:
    return any(_is_call(child, names) for child in ast.walk(node))


def _resolved_expression_nodes(root: ast.AST, index: ProgramIndex) -> list[ast.AST]:
    """Return the dependency closure for names and dataframe-column targets."""

    output: list[ast.AST] = []
    stack = [root]
    seen_keys: set[str] = set()
    while stack:
        node = stack.pop()
        output.append(node)
        for child in ast.iter_child_nodes(node):
            key = _target_key(child)
            if key and key in index.assignments and key not in seen_keys:
                seen_keys.add(key)
                stack.append(index.assignments[key])
            stack.append(child)
    return output


def _mean_receivers(root: ast.AST, index: ProgramIndex) -> list[ast.AST]:
    receivers: list[ast.AST] = []
    for node in _resolved_expression_nodes(root, index):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mean"
        ):
            receivers.append(node.func.value)
    return receivers


def _looks_ratio_derived(node: ast.AST, index: ProgramIndex) -> bool:
    nodes = _resolved_expression_nodes(node, index)
    if any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div) for child in nodes):
        return True
    for child in nodes:
        key = _target_key(child)
        if key and re.search(r"ratio|margin|pct|rate|roe|roa|share", key, re.I):
            return True
        # Pivoted ratio columns often survive only as string literals such as
        # ``values='gross_margin_pct'`` or ``['interest_coverage']``.
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if re.search(r"ratio|margin|pct|rate|roe|roa|share|coverage", child.value, re.I):
                return True
    return False


def _sum_call_side(node: ast.AST, index: ProgramIndex) -> bool:
    node = _unwrap(node, index)
    return _contains_call(node, {"sum"})


def _ratio_of_sums(root: ast.AST, index: ProgramIndex) -> list[ast.BinOp]:
    findings: list[ast.BinOp] = []
    for node in _resolved_expression_nodes(root, index):
        node = _unwrap(node, index)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        if _sum_call_side(node.left, index) and _sum_call_side(node.right, index):
            findings.append(node)
    return findings


def _explicit_mean_of_ratios(root: ast.AST, index: ProgramIndex) -> bool:
    """Recognize ``(a/b + c/d + ...) / N`` in the result dependency graph."""

    for node in _resolved_expression_nodes(root, index):
        node = _unwrap(node, index)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        divisor = node.right
        if not (
            isinstance(divisor, ast.Constant)
            and isinstance(divisor.value, (int, float))
            and 2 <= float(divisor.value) <= 100
        ):
            continue
        ratio_terms = sum(
            isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div)
            for child in ast.walk(node.left)
        )
        if ratio_terms >= 2:
            return True
    return False


def _expression(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return node.__class__.__name__


def audit(submission_dir: Path) -> dict[str, object]:
    records = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    parse_failures: list[dict[str, object]] = []
    proven: list[int] = []
    checked = 0

    for record in records:
        question = str(record.get("question", ""))
        if not _asks_for_mean_ratio(question):
            continue
        checked += 1
        qid = int(record["id"])
        code = str(record.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            parse_failures.append({"id": qid, "error": f"{exc.msg} at line {exc.lineno}"})
            continue
        index = ProgramIndex(tree)
        if not index.results:
            unresolved.append({"id": qid, "question": question, "reason": "no result assignment"})
            continue
        semantic_results = [
            node
            for node in index.results
            if not any(isinstance(child, ast.Name) and child.id == "result" for child in ast.walk(node))
        ]
        root = semantic_results[-1] if semantic_results else index.results[-1]
        bad = _ratio_of_sums(root, index)
        if bad:
            findings.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": record.get("answer"),
                    "reason": "average-ratio wording but result contains sum(numerator) / sum(denominator)",
                    "expressions": sorted({_expression(node) for node in bad}),
                }
            )
            continue
        receivers = _mean_receivers(root, index)
        if any(_looks_ratio_derived(receiver, index) for receiver in receivers):
            proven.append(qid)
            continue
        if _explicit_mean_of_ratios(root, index):
            proven.append(qid)
            continue
        unresolved.append(
            {
                "id": qid,
                "question": question,
                "answer": record.get("answer"),
                "reason": "could not prove per-entity/per-period ratios are averaged before aggregation",
                "result_expression": _expression(_unwrap(root, index)),
            }
        )

    return {
        "kind": "mean_ratio_aggregation_order",
        "submission": str(submission_dir.resolve()),
        "questions_with_mean_ratio_language": checked,
        "proven_mean_of_ratios_count": len(proven),
        "proven_question_ids": proven,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "unresolved_count": len(unresolved),
        "parse_failure_count": len(parse_failures),
        "findings": findings,
        "unresolved": unresolved,
        "parse_failures": parse_failures,
        "claim_limit": (
            "A finding proves an aggregation-order contradiction, not the correct source values. "
            "Unresolved rows require source/program review and are never auto-rewritten."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.submission)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
