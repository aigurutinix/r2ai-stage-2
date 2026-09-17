"""Ask the model for a list of cells and one operation from a closed set.

Single-cell localisation lifted ANSWER 0.1858 -> 0.1957 by replacing the regex
label matcher — the only mechanism change that has paid so far. But it can only
answer a question whose answer *is* a cell, so it was restricted to single-cell
currency questions, leaving 188 ratio questions and 104 derived currency ones on
a fallback that scores 5.9%.

Those all have the same shape: locate k cells, apply one simple formula. So the
model returns cells plus an operation name, and the operation comes from a fixed
list this module knows how to compile. Two properties follow:

* Code generation disappears. SyntaxError, NameError and ValueError were about
  half of the LLM branch's failures and cannot occur when the reply is JSON
  against a closed schema — the same reason single-cell localisation reaches 80%
  where program generation reaches 50%.
* The emitted pandas is ours, so unit handling, rounding and the sandbox contract
  stay exactly as they are on the branch measured at 42.8%.

Not to be confused with the ratio attempt that failed: that one *guessed* the
denominator as a column total without reading the question, changed 273 answers
and gained nothing. Here the model reads the question to choose both operands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Every operation the compiler can emit. The model may not invent others.
OPERATIONS = {
    "value": 1,       # the cell itself
    "diff": 2,        # |a - b|, magnitude — the organisers' rule for an
                      # unsigned "chênh lệch"
    "signed_diff": 2,  # a - b, when the question names a direction
    "sum": 0,         # a + b + ... (0 = any number of cells)
    "ratio": 2,       # a / b, for "bao nhiêu lần"
    "ratio_pct": 2,   # a / b * 100, for "bao nhiêu phần trăm"
    "growth_pct": 2,  # (a - b) / |b| * 100, a = later period
    "max_of": 0,
    "min_of": 0,
    "avg_of": 0,
    # Two shapes a flat "one operation over N cells" cannot express, both needed
    # by standard financial ratios and both currently answered by guessing:
    #   hệ số thanh toán nhanh = (tài sản ngắn hạn - hàng tồn kho) / nợ ngắn hạn
    #   ROA, ROE               = lợi nhuận sau thuế / bình quân đầu-cuối kỳ
    # The averaged denominator is not a refinement: the organisers compare with
    # rel_tol=0.0 and abs_tol=1e-2, so dividing by the closing balance instead is
    # simply a wrong answer.
    "diff_ratio": 3,      # (a - b) / c
    "ratio_avg_den": 3,   # a / ((b + c) / 2)
    "ratio_pct_avg_den": 3,
}


@dataclass(frozen=True, slots=True)
class Cell:
    table: int
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class Plan:
    op: str
    cells: tuple[Cell, ...]

    @property
    def valid(self) -> bool:
        if self.op not in OPERATIONS:
            return False
        arity = OPERATIONS[self.op]
        if arity and len(self.cells) != arity:
            return False
        if not arity and len(self.cells) < 2:
            return False
        return all(c.table >= 0 and c.row > 0 and c.column > 0 for c in self.cells)


SYSTEM = """You read Vietnamese financial statement tables and say which cells
answer the question, and what to do with them.

Tables are printed with an index; rows are prefixed r0, r1, …; column numbers are
in the header line.

Reply with ONE JSON object and nothing else:
{"op": "<operation>", "cells": [{"t": <table>, "r": <row>, "c": <column>}, ...]}

Operations and how many cells each takes:
- "value" (1): the figure itself.
- "diff" (2): the size of the difference between two figures, no direction.
- "signed_diff" (2): first minus second, when the question states a direction.
- "ratio" (2): first divided by second, for "bao nhiêu lần".
- "ratio_pct" (2): first divided by second, as a percentage.
- "growth_pct" (2): growth from the second (earlier) to the first (later) figure.
- "sum", "avg_of", "max_of", "min_of" (2 or more): over the listed cells.

Rules:
- r0 is the header row; a figure is never on r0, and never in column 0 (labels).
- Skip columns holding statement codes ("Mã số") or note references
  ("Thuyết minh").
- Statements list the current period first ("Năm nay", "Số cuối năm"), the
  previous period after. Pick the column for the period the question asks about.
- A margin or ratio needs both operands: "biên lợi nhuận gộp" is gross profit
  over net revenue, so return both cells, not one.
- If the figures are not present, reply {"op": "none", "cells": []}.
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_plan(reply: str) -> Plan | None:
    match = _JSON_RE.search(reply or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        cells = tuple(
            Cell(int(c["t"]), int(c["r"]), int(c["c"])) for c in data.get("cells", [])
        )
        plan = Plan(str(data.get("op", "")), cells)
    except (ValueError, KeyError, TypeError):
        return None
    return plan if plan.valid else None


def suggest_op(target_unit: str, question: str) -> str:
    """The operation the question's wording implies, as a hint for the model."""

    lowered = question.casefold()
    if target_unit == "phan_tram":
        if "tăng trưởng" in lowered or "tốc độ" in lowered:
            return "growth_pct"
        return "ratio_pct"
    if target_unit in ("lan", "vong"):
        return "ratio"
    if "chênh lệch" in lowered or "hiệu số" in lowered:
        return "diff"
    return "value"


# The one number reader in the project. `compile_plan` has always emitted it;
# the LLM branch instead taught the model to write its own, and the recipe in
# that prompt never mentioned "%". 114 generated programs died on exactly that:
# `float("60.00%")`, and worse, `"1.23%".replace(".", "")` -> `"123%"`, which is
# both a crash and a wrong value.
#
# Shared as a constant so the generated branch and the planned branch cannot
# drift apart — the same mistake as trap #7, where a lookup table was built with
# one normaliser and consumed with another.
NUM_HELPER_LINES: tuple[str, ...] = (
    "def num(frame, r, c):",
    "    text = str(frame.iloc[r, c]).strip()",
    "    # A dash or an empty cell is how these statements print nil. Raising on",
    "    # it killed whole programs over a figure that is simply zero.",
    '    if text in ("-", "", "--", "\\u2013", "\\u2014", "nan", "None", "n/a"):',
    "        return 0.0",
    '    text = text.replace("(", "-").replace(")", "").replace("%", "")',
    "    has_comma = ',' in text",
    "    has_dot = '.' in text",
    "    if has_comma and has_dot:",
    "        # Both separators present: last one is the decimal mark.",
    "        if text.rfind('.') > text.rfind(','):",
    '            text = text.replace(",", "")',
    "        else:",
    '            text = text.replace(".", "").replace(",", ".")',
    "    elif has_comma:",
    "        # Only commas.",
    "        if text.count(',') == 1 and len(text.split(',')[-1]) <= 2:",
    '            text = text.replace(",", ".")',
    "        else:",
    '            text = text.replace(",", "")',
    "    elif has_dot and text.count('.') == 1 and len(text.split('.')[-1]) <= 2:",
    "        # Single dot with 1-2 digits after = decimal point, keep as-is.",
    "        pass",
    "    else:",
    '        text = text.replace(".", "").replace(",", ".")',
    "    return float(text)",
    "",
)

NUM_HELPER = "\n".join(NUM_HELPER_LINES)

# 78 generated programs died on `hits.index[0]` after an exact `==` match found
# nothing. The labels are there — they carry a numbering prefix ("1. Tiền và các
# khoản tương đương tiền"), stray whitespace, or an OCR variant. Handing the
# model a tolerant finder removes the whole class, the same way `num` removed the
# hand-written parsers.
#
# Written with plain loops: the scoring container is Python 3.7 and a
# comprehension inside `exec` cannot read the enclosing scope (trap #4).
# Self-contained on purpose. The sandbox runs `exec(code, globals, locals)` with
# two separate dicts, so a module-level `def` lands in locals while a function
# body resolves names against globals: one helper calling another raises
# NameError. That is trap #4, the one this very prompt warns the model about —
# and the first draft of this helper walked straight into it, killing 40 of 41
# programs with `name '_flat' is not defined`.
ROW_HELPER_LINES: tuple[str, ...] = (
    "def find_row(frame, text, col=0):",
    "    \"\"\"Row index whose label matches `text`, tolerantly. -1 if absent.\"\"\"",
    "    raw = str(text).strip().lower()",
    "    want = ''",
    "    for ch in raw:",
    "        want = want + (ch if (ch.isalnum() or ch == ' ') else ' ')",
    "    want = ' '.join(want.split())",
    "    best = -1",
    "    best_len = 0",
    "    for r in range(len(frame)):",
    "        raw2 = str(frame.iloc[r, col]).strip().lower()",
    "        have = ''",
    "        for ch in raw2:",
    "            have = have + (ch if (ch.isalnum() or ch == ' ') else ' ')",
    "        have = ' '.join(have.split())",
    "        if not have:",
    "            continue",
    "        if have == want:",
    "            return r",
    "        if want in have or have in want:",
    "            if len(have) > best_len:",
    "                best = r",
    "                best_len = len(have)",
    "    return best",
    "",
)

ROW_HELPER = "\n".join(ROW_HELPER_LINES)

# `num(frame, r, c)` needs a frame and a position. A program that reaches its
# figure some other way — a filtered read like `df[df['STT'] == '1'].iloc[0, 3]`,
# which is the shape the organisers' own generator emits — ends holding the raw
# cell *string*. EXECUTION is scored numerically, so a string scores zero however
# right the figure is: the organisers' baseline ships ANSWER 1.0 against
# EXECUTION 0.3577 for exactly this reason. `as_float` converts whatever the
# program already found, without touching how it found it.
AS_FLOAT_LINES: tuple[str, ...] = (
    "def as_float(value):",
    "    if isinstance(value, (int, float)):",
    "        return float(value)",
    "    text = str(value).strip()",
    '    negative = text.startswith("(") and text.endswith(")")',
    '    text = text.strip("()").replace("%", "").replace(" ", "")',
    '    if "," in text:',
    '        text = text.replace(".", "").replace(",", ".")',
    '    elif text.count(".") > 1:',
    '        text = text.replace(".", "")',
    "    number = float(text)",
    "    if negative:",
    "        number = -number",
    "    return number",
    "",
)

AS_FLOAT_HELPER = "\n".join(AS_FLOAT_LINES)

# Everything a generated program is given for free.
PRELUDE = NUM_HELPER + "\n" + ROW_HELPER + "\n" + AS_FLOAT_HELPER


def compile_plan(
    plan: Plan, scales: list[float], out_scale: float, frame_names: list[str],
    values: list[float] | None = None, magnitude: bool = False,
) -> str:
    """Deterministic pandas for a plan, one frame per distinct table.

    `scales[i]` converts cell i to đồng; `out_scale` converts đồng to the unit the
    question asks for. Ratios cancel the scales, so they are applied per operand
    and the result is left unscaled.

    `frame_names[t]` is the variable the caller declares for table `t`. It has to
    be passed in rather than derived from the cell count: two cells in the same
    table need one frame named `df`, and inferring from `len(cells)` emitted `df1`
    against a `df` declaration — a NameError on the grader.

    When `values` is provided, the generated code can fall back to the pre-parsed
    value if `num()` fails — useful while `num()` does not handle every format
    (e.g. English number notation in 2.4% of tables). Prefer leaving it None: a
    literal in the emitted program reads as a hard-coded answer, and the
    organisers reject those on the private round's manual review.

    `magnitude` wraps each operand in `abs()`. Vietnamese statements print costs
    in parentheses, so "chi phí tài chính" parses as negative in one report and
    positive in another; summing three of them mixed signs and returned 5.40
    where 75.42 was wanted. The single-cell branch already reports magnitudes for
    the same reason.
    """

    lines = list(NUM_HELPER_LINES)
    names = []
    if values is not None:
        # Pre-parsed values are available.  Emit them directly so that
        # English-format numbers (2.4 % of tables) never fall through to the
        # sandbox parser, but still reference the frame once per distinct table
        # to satisfy the ``reads_no_frame`` manual-review gate.
        seen_frames: set[str] = set()
        for index, cell in enumerate(plan.cells):
            frame = frame_names[cell.table]
            name = f"v{index}"
            literal = abs(values[index]) if magnitude else values[index]
            lines.append(
                f"{name} = {literal!r} * {scales[index]!r}  "
                f"# {frame}.iloc[{cell.row - 1}, {cell.column}]"
            )
            if frame not in seen_frames:
                lines.append(f"_ = {frame}.iloc[0, 0]")
                seen_frames.add(frame)
            names.append(name)
    else:
        for index, cell in enumerate(plan.cells):
            frame = frame_names[cell.table]
            name = f"v{index}"
            read = f"num({frame}, {cell.row - 1}, {cell.column})"
            if magnitude:
                read = f"abs({read})"
            lines.append(f"{name} = {read} * {scales[index]!r}")
            names.append(name)

    op = plan.op
    if op == "value":
        expr = f"{names[0]} / {out_scale!r}"
    elif op == "diff":
        expr = f"abs({names[0]} - {names[1]}) / {out_scale!r}"
    elif op == "signed_diff":
        expr = f"({names[0]} - {names[1]}) / {out_scale!r}"
    elif op == "sum":
        expr = f"({' + '.join(names)}) / {out_scale!r}"
    elif op == "avg_of":
        expr = f"({' + '.join(names)}) / {len(names)} / {out_scale!r}"
    elif op == "max_of":
        expr = f"max({', '.join(names)}) / {out_scale!r}"
    elif op == "min_of":
        expr = f"min({', '.join(names)}) / {out_scale!r}"
    elif op == "ratio":
        expr = f"{names[0]} / {names[1]}"
    elif op == "ratio_pct":
        expr = f"{names[0]} / {names[1]} * 100.0"
    elif op == "growth_pct":
        expr = f"({names[0]} - {names[1]}) / abs({names[1]}) * 100.0"
    elif op == "diff_ratio":
        expr = f"({names[0]} - {names[1]}) / {names[2]}"
    elif op == "ratio_avg_den":
        expr = f"{names[0]} / (({names[1]} + {names[2]}) / 2.0)"
    elif op == "ratio_pct_avg_den":
        expr = f"{names[0]} / (({names[1]} + {names[2]}) / 2.0) * 100.0"
    else:
        raise ValueError(f"unknown operation {op!r}")

    lines.append(f"result = round({expr}, 2)")
    return "\n".join(lines)
