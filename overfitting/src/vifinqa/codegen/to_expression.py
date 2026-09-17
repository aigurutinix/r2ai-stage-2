"""Convert a straight-line PoT script into ONE pandas expression.

WHY THIS EXISTS (leaderboard-confirmed, submission #6):
The organizers evaluate `pandas_query` as an EXPRESSION (eval), not as a script.
Submission #5 used only single-line expressions -> EXECUTION_ACCURACY equalled
ANSWER_ACCURACY exactly (0.085/0.085). Submission #6 let the LLM emit multi-line
scripts: 233/1012 queries became scripts, every one of them raises
`SyntaxError` under eval, and EXECUTION_ACCURACY fell to 0.0613 while
ANSWER_ACCURACY rose to 0.1047. So: a correct answer whose query is a script is
scored as a crash.

Strategy: inline single-assignment straight-line code into the final expression.

    tmp = df1[df1['label'].str.contains('X', na=False)]
    val = tmp['value'].iloc[0]
    answer = round(float(val) * 1e6 / 1e9, 2)
        ->
    round(float(df1[df1['label'].str.contains('X', na=False)]['value'].iloc[0]) * 1e6 / 1e9, 2)

Only provably safe scripts are inlined (no control flow, no loops, no calls that
mutate state, each name bound exactly once). Everything else is reported as
un-inlinable so the caller can fall back instead of shipping a crash.
"""
from __future__ import annotations

import ast


class InlineError(Exception):
    pass


_ALLOWED_STMT = (ast.Assign, ast.Expr)


def to_single_expression(code: str) -> str:
    """Return an equivalent single expression. Raises InlineError if unsafe."""
    src = code.strip()
    if not src:
        raise InlineError("empty code")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise InlineError(f"unparsable: {e}") from e

    # already a single expression?
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        return _unparse(tree.body[0].value)

    assigns: dict[str, ast.AST] = {}
    order: list[str] = []
    final: ast.AST | None = None

    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            # a bare trailing expression is the result
            final = stmt.value
            continue
        if not isinstance(stmt, _ALLOWED_STMT):
            raise InlineError(f"unsupported statement: {type(stmt).__name__}")
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise InlineError("only single Name assignment targets are supported")
        name = stmt.targets[0].id
        if name in assigns:
            raise InlineError(f"variable reassigned: {name}")
        _reject_unsafe(stmt.value)
        assigns[name] = stmt.value
        order.append(name)
        if name == "answer":
            final = stmt.value

    if final is None:
        if not order:
            raise InlineError("no expression found")
        final = assigns[order[-1]]      # last assignment is the result

    # substitute in reverse binding order until no local names remain
    expr = final
    for name in reversed(order):
        if name == "answer":
            continue
        expr = _Substituter({name: assigns[name]}).visit(expr)

    leftover = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)} & set(order)
    if leftover:
        raise InlineError(f"could not inline names: {sorted(leftover)}")
    return _unparse(expr)


_FORBIDDEN_CALLS = {"exec", "eval", "compile", "open", "__import__", "globals",
                    "locals", "setattr", "getattr", "input", "print"}


def _reject_unsafe(node: ast.AST) -> None:
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                            ast.GeneratorExp, ast.Await, ast.Yield)):
            raise InlineError(f"unsupported construct: {type(sub).__name__}")
        if isinstance(sub, ast.NamedExpr):
            raise InlineError("walrus assignment is not inlinable")
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id in _FORBIDDEN_CALLS:
            raise InlineError(f"forbidden call: {sub.func.id}")


class _Substituter(ast.NodeTransformer):
    """Replace Name loads with their bound expression, parenthesised safely."""

    def __init__(self, mapping: dict[str, ast.AST]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(self.mapping[node.id], node)
        return node


def _unparse(node: ast.AST) -> str:
    expr = ast.unparse(node)
    return " ".join(expr.split())          # collapse any stray newlines


def try_to_expression(code: str) -> tuple[str, str | None]:
    """(expression, error). Never raises — callers decide the fallback."""
    try:
        return to_single_expression(code), None
    except InlineError as e:
        return code.strip(), str(e)
