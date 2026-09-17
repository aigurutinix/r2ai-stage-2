"""Ask the model which cell answers the question — not to write a program.

Reading back the earlier experiments more carefully changes the diagnosis. The
metric panel lifted executable programs from 50% to 90.5%, and I read that as
"the OCR grid is too messy". But the panel changed two things at once: it was
tidy *and* the metric had already been identified for the model. So it does not
show the grid is unreadable. It shows that **once the right numbers are in front
of the model, it computes them correctly** — the failure is deciding *which*
numbers.

That is a localisation problem, and it is exactly where the regex label matcher
gives up: it finds no acceptable label for 250 of the 506 graded questions, and
those answer 5.9% correctly.

So this module replaces only the matcher. The model returns three integers —
table, row, column — and the existing deterministic path takes over from there:
same unit resolution, same emitted pandas, same sandbox. Two consequences worth
stating:

* Asking for three integers instead of a program removes the entire class of
  code-generation failures. SyntaxError, NameError and ValueError together were
  about half of all LLM failures, and none of them can happen here.
* Only one thing changes versus the current pipeline, so a score difference is
  attributable to localisation and nothing else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MAX_ROWS = 45
MAX_CELL = 34

SYSTEM = """You locate a single number inside Vietnamese financial statement tables.

You are given a question and several candidate tables. Each table is printed with
a numeric index, and every row is prefixed with its row number. Column numbers
are given in the header line.

Return ONLY a JSON object, no prose and no code fences:
{"table": <table index>, "row": <row number>, "column": <column number>}

Rules:
- Pick the cell whose value answers the question. Row 0 is the header; a figure
  is never on row 0.
- Column 0 holds the line-item label, so the answer is never in column 0.
- Prefer the column for the period the question asks about. Statements list the
  current period first ("Năm nay", "Số cuối năm") and the previous one after.
- Ignore columns holding statement codes ("Mã số") or note references
  ("Thuyết minh") — those are not figures.
- If no table contains the figure, return {"table": -1, "row": -1, "column": -1}.
"""


@dataclass(frozen=True, slots=True)
class Located:
    table: int
    row: int
    column: int

    @property
    def found(self) -> bool:
        return self.table >= 0 and self.row > 0 and self.column > 0


def render_candidates(grids: list[list[list[str]]], captions: list[str]) -> str:
    blocks = []
    for index, (grid, caption) in enumerate(zip(grids, captions)):
        if not grid:
            continue
        width = len(grid[0])
        header = " | ".join(f"[c{c}]" for c in range(width))
        lines = [f"TABLE {index} — {caption[:110]}", f"columns: {header}"]
        for row_index, row in enumerate(grid[:MAX_ROWS]):
            cells = " | ".join(str(cell)[:MAX_CELL] for cell in row[:width])
            lines.append(f"r{row_index}: {cells}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def parse_reply(reply: str) -> Located | None:
    match = _JSON_RE.search(reply or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        return Located(int(data["table"]), int(data["row"]), int(data["column"]))
    except (ValueError, KeyError, TypeError):
        return None


def locate(client, question: str, grids: list[list[list[str]]], captions: list[str]) -> Located | None:
    user = f"<question>\n{question}\n</question>\n\n{render_candidates(grids, captions)}"
    try:
        reply = client.complete(SYSTEM, user)
    except RuntimeError:
        return None
    return parse_reply(reply)
