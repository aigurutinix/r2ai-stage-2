"""Execute a generated pandas program under the organisers' runtime contract.

Mirrors `prompts/answering/program_system.txt`: only `pd` is in scope, imports
are forbidden, a single table binds to `df` while several bind to `dfs` keyed by
table_ref, the builtin set is fixed, and the program must leave exactly one
numeric scalar in `result`.

Tables load the way the scorer loads them — every cell a raw string, no dtype
inference, empty cells as `""` — because a program that quietly relies on pandas
having parsed "1.234.567" as a number here would fail there.

Scope note: this runs code in-process, so it contains mistakes, not malice. It
is the right tool for our own synthesised queries. Before running unvetted LLM
output at scale, wrap it in a subprocess with a wall-clock timeout: nothing here
stops `while True`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "round", "len", "min", "max", "sum", "sorted", "float", "int",
        "str", "bool", "list", "dict", "set", "range", "enumerate", "zip",
        "all", "any", "isinstance",
    )
}


def portability_problems(code: str) -> list[str]:
    """Reject constructs that work here but not on the grader.

    The scoring container is `codalab/codalab-legacy:py37`. Before PEP 709
    (Python 3.12) a comprehension compiled to its own function, and when `exec`
    receives separate globals and locals dicts that function resolves free names
    against *globals* — so a comprehension reading a variable defined moments
    earlier raises NameError there while running fine on 3.13 here. That single
    difference zeroed EXECUTION_ACCURACY on an otherwise correct run.
    """

    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            found.append(f"{type(node).__name__} at line {node.lineno} is unsafe under py37 exec scoping")
        elif isinstance(node, ast.Lambda):
            found.append(f"Lambda at line {node.lineno} is unsafe under py37 exec scoping")
        elif isinstance(node, ast.NamedExpr):
            found.append(f"walrus at line {node.lineno} requires py38+")
    return found


@dataclass(frozen=True, slots=True)
class ExecResult:
    ok: bool
    value: float | None
    error: str = ""

    @property
    def crashed(self) -> bool:
        return not self.ok


def _frame_as_objects(rows: list[list[str]]):
    """Every cell as the string it is in the CSV."""

    import pandas as pd

    header, body = rows[0], rows[1:]
    # Duplicate header labels are common in OCR tables; positional access stays
    # usable as long as we do not silently collapse them.
    return pd.DataFrame(body, columns=pd.Index(header), dtype=object).fillna("")


def frame_from_rows(rows: list[list[str]]):
    """Build the DataFrame exactly as the scorer does from our emitted CSV.

    "Exactly" means through `read_csv`, which infers a dtype per column — and
    that is not a detail. Building the frame as all-object made this harness
    disagree with the grader on the one thing it exists to predict: a program
    doing `cell.replace(".", "")` works on a string and raises
    `AttributeError: 'numpy.float64' object has no attribute 'replace'` on an
    inferred numeric column. Five such programs passed here and scored 0 there,
    and a sixth returned 0.01 locally against 0.17 on the grader.

    Round-tripping through CSV text rather than the grid also reproduces what
    `read_csv` does to duplicate headers, which OCR tables are full of.
    """

    import io

    import pandas as pd

    if not rows:
        return pd.DataFrame()
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    try:
        return pd.read_csv(io.StringIO(buffer.getvalue()))
    except Exception:  # noqa: BLE001
        # Ragged rows defeat the parser. The grader would hit the same wall, but
        # a string frame at least lets the program be judged on its logic rather
        # than dropped outright.
        return _frame_as_objects(rows)


def run_query(code: str, tables: dict[str, list[list[str]]]) -> ExecResult:
    """Execute `code` against the given tables and extract `result`."""

    import pandas as pd

    if not code or not code.strip():
        return ExecResult(False, None, "empty program")

    unsafe = portability_problems(code)
    if unsafe:
        return ExecResult(False, None, "; ".join(unsafe))

    frames = {ref: frame_from_rows(rows) for ref, rows in tables.items()}
    scope: dict[str, Any] = {"pd": pd, "dfs": frames}
    if len(frames) == 1:
        scope["df"] = next(iter(frames.values()))
    for position, frame in enumerate(frames.values(), start=1):
        scope.setdefault(f"df{position}", frame)

    try:
        exec(compile(code, "<query>", "exec"), {"__builtins__": SAFE_BUILTINS}, scope)
    except Exception as exc:  # noqa: BLE001 - any failure is a crash for scoring
        return ExecResult(False, None, f"{type(exc).__name__}: {exc}")

    if "result" not in scope:
        return ExecResult(False, None, "program did not assign `result`")

    value = scope["result"]
    if isinstance(value, bool):
        return ExecResult(False, None, "result is a bool, not a numeric scalar")
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        # numpy scalars and single-element Series still count as a scalar.
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if not isinstance(value, (int, float)):
        return ExecResult(False, None, f"result is {type(value).__name__}, not a numeric scalar")

    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return ExecResult(False, None, f"result is not finite: {number}")
    return ExecResult(True, number)
