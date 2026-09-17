"""LLM-generated pandas for the questions no lookup can decide.

Only 462 of the 1,012 questions ask for a single cell. The rest compare
entities, span years, or derive a ratio, and they need a program.

The system prompt is the organisers' own `prompts/answering/program_system.txt`
plus one clause they omit: their scoring container is Python 3.7, where a
comprehension inside `exec` cannot see names from the enclosing scope. Their own
baseline reports 100% correct answers but only 35.77% execution accuracy, which
is what that class of failure looks like from the outside.

Every candidate program is linted for that hazard and executed here before it is
accepted, and a failure is fed back for one repair attempt.
"""

from __future__ import annotations

import os

import re
from dataclasses import dataclass
from pathlib import Path

from vifin.answering.plan_cells import PRELUDE
from vifin.answering.sandbox import ExecResult, portability_problems, run_query

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

PY37_CLAUSE = """
<compatibility>
- The runtime is Python 3.7 and the program is executed with `exec` using
  separate globals and locals. A list/dict/set comprehension or generator
  expression CANNOT read a variable defined earlier in the program there: it
  raises NameError. Never use comprehensions, generator expressions, or lambdas.
  Use pandas vectorised operations or a plain `for` loop instead.
- Do not use syntax newer than Python 3.7: no walrus operator, no
  str.removeprefix/removesuffix, no dict union.
</compatibility>
"""

INDEX_CLAUSE = """
<frame_shape>
- Row labels are VALUES IN COLUMN 0, not the DataFrame index. The index is a
  plain 0..n-1 range. `df.loc["Tổng cộng"]` therefore raises KeyError.
- A helper `find_row(frame, text)` is ALREADY DEFINED above your code. It returns
  the row index whose column-0 label matches `text`, ignoring case, punctuation,
  spacing and any numbering prefix ("1. ", "V. "), or -1 when nothing matches.
  Use it instead of building a mask — an exact `==` match is how 78 programs
  died, because the real label carried a prefix the question did not:
      row = find_row(df, "Tiền và các khoản tương đương tiền")
      if row < 0:
          row = find_row(df, "Tiền")          # fall back to a shorter fragment
      value = num(df, row, 2)
- Always check `row < 0` before using it. Never write `.index[0]`.
- Address columns positionally with `df.iloc[row, n]` when a header repeats or
  is blank, which is common in this corpus.
- Many labels contain quotes and brackets, e.g. `(1) Công ty TNHH Anh Vũ ("Anh
  Vũ")`. Embedding one in a string literal breaks the program, so match on a
  short distinctive fragment instead of the whole label:
      hits = labels[labels.str.contains('Anh Vũ', regex=False, na=False)]
</frame_shape>
"""

# The previous version of this clause taught the model to roll its own parser and
# even said helper functions were unnecessary. The recipe it taught omitted "%",
# and 114 of 240 failed programs died on exactly that — `float("60.00%")`, or
# `"1.23%".replace(".", "")` producing "123%". The model is now handed the
# project's own reader instead of being taught to rebuild it badly.
NUMBER_CLAUSE = """
<parsing_numbers>
- A helper `num(frame, row, col)` is ALREADY DEFINED above your code. It returns
  a float from a raw Vietnamese cell, handling "." as thousands separator, ","
  as decimal point, "(1.234)" for negatives and a trailing "%".
- Use `num(...)` for EVERY numeric read. Do not define it, and do not write your
  own string cleaning — `float("60,00%")` and
  `float("1.729.803.613.573".replace(".", ""))` are the two ways that goes wrong.
      value = num(df, 12, 3)
      total = num(df1, 5, 1) + num(df1, 6, 1)
- For a whole column, loop with `num` and skip the cells that raise:
      values = []
      for r in range(len(df)):
          try:
              values.append(num(df, r, 2))
          except ValueError:
              pass
- Never initialise `result = 0` and leave it: if your lookup finds nothing the
  program silently reports zero, which is scored as a wrong answer. Compute
  `result` from a value you actually located.
</parsing_numbers>
"""

AGGREGATION_CLAUSE = """
<aggregating>
- `median`, `mean`, `np`, `numpy` and `statistics` DO NOT EXIST here and cannot
  be imported. Calling them raises NameError — this is the most common way these
  programs fail. Compute the statistic yourself from a plain list:
      values = []
      for name in names:
          values.append(ratios[name])
      values.sort()
      middle = len(values) // 2
      if len(values) % 2 == 1:
          median = values[middle]
      else:
          median = (values[middle - 1] + values[middle]) / 2.0
      average = sum(values) / len(values)
  `sum`, `min`, `max`, `sorted`, `len`, `round`, `abs` are available.
- A comprehension raises NameError on this runtime. Build lists with an explicit
  loop and `.append(...)`, never `[x for x in ...]`.
- Keep the program short and linear. Programs that fail here are twice as long
  as ones that work: resolve one figure at a time into a named variable, then
  combine those variables at the end.
</aggregating>
"""

UNIT_CLAUSE = """
<answer_unit>
- The question names the unit it wants. Convert to exactly that unit.
- IF THE QUESTION NAMES NO UNIT, or says only "đồng", DO NOT SCALE AT ALL: report
  the figure exactly as the statement carries it, in đồng. "Tổng vốn chủ sở hữu
  của IJC cuối năm 2025 là bao nhiêu?" wants 8064309392042, not 8064.31. This is
  146 of the 1012 questions and dividing by a billion out of habit is the single
  most common way to lose one.
- "phần trăm"/"%" means a percentage number: answer 90, never 0.9.
- Column headers often carry the table's unit ("2018Triệu VND", "Số cuối năm
  Triệu đồng"). Read the unit from the header when no separate unit line exists.
</answer_unit>
"""



MULTI_ENTITY_CLAUSE = """
<several_companies>
- When the question names more than one company, or asks to compare, rank, screen
  or count across a group, you must read a value FOR EACH company named, each from
  that company's own frame, before combining them. Read the frame refs to see which
  company each frame belongs to.
- Answering such a question from a single cell of one company is the single most
  common failure on this task. A hand audit of shipped answers found a screen over
  ten companies answered with one company's payables, and a difference between two
  banks answered from one bank's buildings column.
- If a company the question names has no frame here, say so by computing from the
  ones present rather than silently answering for one of them.
- Resolve one company at a time into its own named variable, then combine:
      hpg = num(df1, 12, 3)
      hsg = num(df2, 12, 3)
      result = round(abs(hpg - hsg) / 1e9, 2)
</several_companies>
"""

VARIABLES_CLAUSE = """
<variables>
- Each table is already loaded into the DataFrame variable named in its schema
  block. Use those exact names. There is no `dfs` dictionary and no variable
  named after a table_ref: `dfs["ABC_2023_consolidated|350"]` raises NameError.
- With a single table the variable is `df`; with several they are `df1`, `df2`,
  and so on, in the order listed below.
</variables>
"""


def build_prompts(repo_root: Path, bare: bool = False) -> tuple[str, str]:
    """The system and user prompts, optionally without our own additions.

    Six clauses and a pre-defined `num`/`find_row` prelude sit on top of the
    organisers' prompt. Adding the prelude took executable programs from 52.4% to
    76.9% and correct answers from 0.3241 to 0.3261 — +147 programs that run for
    +2 that are right. The reading at the time was "executable is not accurate".
    A second reading is worth testing: handed helpers that always return *a*
    number, the model stops deliberating over which cell it is reading, and the
    instructions crowd out the question. `bare=True` restores the organisers'
    prompt alone so the two can be compared on the same questions.
    """

    prompts = repo_root / "vifinqa-official" / "prompts" / "answering"
    system = (prompts / "program_system.txt").read_text(encoding="utf-8")
    user = (prompts / "common_user.txt").read_text(encoding="utf-8")
    if bare:
        return system + VARIABLES_CLAUSE + PY37_CLAUSE, user
    unit_clause = UNIT_CLAUSE if NO_UNIT_RULE else UNIT_CLAUSE.replace(
        UNIT_CLAUSE[UNIT_CLAUSE.index("- IF THE QUESTION NAMES NO UNIT"):
                    UNIT_CLAUSE.index("- \"phần trăm\"")], "")
    parts = [system, VARIABLES_CLAUSE, PY37_CLAUSE, INDEX_CLAUSE,
             NUMBER_CLAUSE, AGGREGATION_CLAUSE, unit_clause]
    if MULTI_ENTITY:
        parts.append(MULTI_ENTITY_CLAUSE)
    return "".join(parts), user


def variable_names(count: int) -> list[str]:
    """`df` alone, otherwise df1..dfn — matching what the packager declares.

    The grader binds each CSV to the name given in `evidence.variable`, so the
    program must use those names and nothing else. Keying tables by table_ref,
    as the organisers' prompt suggests, produces code that cannot run there.
    """

    return ["df"] if count == 1 else [f"df{i}" for i in range(1, count + 1)]


# Numbering every row with its DataFrame index looked like it should remove
# the counting the model has to do to name a cell. Measured on 100 gold
# questions it did the opposite: 28.0% against 33.0%, and paired per
# question it fixed 2 and broke 7. The prefix breaks the grid the model
# reads a CSV as, and it competes with the `find_row` helper the prompt
# already teaches. Off by default; `VIFIN_ROW_INDEX=1` re-enables it.
ROW_INDEX = os.environ.get("VIFIN_ROW_INDEX", "0") != "0"
# `VIFIN_BLANK_ROWS=0` restores dropping unlabelled rows from the schema.
# Naming the unlabelled total rows in the schema was measured at 38.0%
# against 39.3% without, so it is off; the 14.2% of answers that sit on
# such a row are real, but announcing them this way costs more than it
# returns. `VIFIN_BLANK_ROWS=1` re-enables it.
SHOW_BLANK_ROWS = os.environ.get("VIFIN_BLANK_ROWS", "0") != "0"
# A column-number line above the CSV body measured 19.3% against 22.0%
# without it on the program path, so it is off. It does help the *plan*
# path — but `locate.render_candidates` already prints one there, so there
# was nothing to add. Third annotation to be inserted into the body and
# third to lose points, after row indices (-6) and blank-row markers (-1.3):
# the model reads the body as a grid and anything spliced into it breaks
# that reading. `VIFIN_COLUMN_RULER=1` re-enables it.
COLUMN_RULER = os.environ.get("VIFIN_COLUMN_RULER", "0") != "0"
# Write the column index onto the header cell itself (`c1=Số cuối năm`) instead of
# leaving the model to map between the schema block and the body. Measured on the
# isolated cell probe: 50.0% -> 69.1%. `VIFIN_COL_HEADER_TAGS=0` disables it.
COL_HEADER_TAGS = os.environ.get("VIFIN_COL_HEADER_TAGS", "1") != "0"
# Greedy decoding on a long prompt loops. 0 disables the penalty entirely.
REP_PENALTY = float(os.environ.get("VIFIN_REP_PENALTY", "0") or 0)
# `VIFIN_NO_UNIT_RULE=0` drops the "no unit means đồng" sentence.
NO_UNIT_RULE = os.environ.get("VIFIN_NO_UNIT_RULE", "1") != "0"
# `VIFIN_MULTI_ENTITY=0` drops the several-companies clause.
MULTI_ENTITY = os.environ.get("VIFIN_MULTI_ENTITY", "1") != "0"


def describe_schema(variable: str, ref: str, rows: list[list[str]],
                    note: str = "") -> str:
    """Variable name, provenance, exact columns, and row labels for one table.

    KeyError was the single largest failure mode: the model invented plausible
    Vietnamese column names instead of using the ones present. Listing them
    verbatim, separately from the CSV body, removes the need to infer them from
    a possibly truncated preview.
    """

    if not rows:
        return f'<schema variable="{variable}" ref="{ref}">empty</schema>'
    columns = ", ".join(f"[{i}] {cell!r}" for i, cell in enumerate(rows[0]))
    # Row labels carry their DataFrame index. `num(df, r, c)` takes that index, and
    # without it the model has to count rows down a 35-to-60 line Vietnamese
    # statement to name one — an off-by-one there produces a plausible wrong
    # number, which is the shape of the organisers' largest error class (54.7% of
    # failures are a wrong cell). `frame_from_rows` consumes row 0 as the header,
    # so body row j is DataFrame index j.
    if ROW_INDEX:
        labels = ", ".join(
            f"[{j}] {str(row[0])!r}"
            for j, row in enumerate(rows[1:]) if row and str(row[0]).strip()
        )
    else:
        # An unlabelled row is not an empty row: Vietnamese statements leave the
        # label column blank on the total line, and 14.2% of the gold questions
        # whose answer is in a retrieved table have it on exactly such a row.
        # Dropping them from a list introduced as the table's row labels told the
        # model the line did not exist.
        shown = []
        for position, row in enumerate(rows[1:]):
            if not row:
                continue
            label = str(row[0]).strip()
            if label:
                shown.append(repr(label))
            elif SHOW_BLANK_ROWS and any(str(cell).strip() for cell in row[1:]):
                shown.append(f"<hàng {position} không có nhãn — thường là dòng cộng>")
        labels = ", ".join(shown)
    # The caption carries the unit ("Đơn vị: triệu đồng") where a table declares
    # one, and the statement title otherwise. The rule branch has always been
    # given it — `to_vnd` scales by `unit_page + unit_doc + caption` — while the
    # model branch was left to infer the unit from column headers that often do
    # not state it. There was no reason for that asymmetry.
    heading = f"caption_and_unit: {note}\n" if note else ""
    return (
        f'<schema variable="{variable}" ref="{ref}">\n'
        f"{heading}"
        f"columns: {columns}\n"
        f"row_labels_in_column_0: {labels}\n"
        f"</schema>"
    )


def render_tables(
    tables: dict[str, list[list[str]]],
    refs: dict[str, str],
    max_rows: int = 60,
    max_chars: int = 7000,
    notes: dict[str, str] | None = None,
) -> str:
    blocks = []
    for variable, rows in tables.items():
        # Same indexing in the body as in the schema: the header keeps no number,
        # every data line is prefixed with the DataFrame index to address it by.
        # A column-number line directly above the body. The indices are already
        # listed in the schema block, but separately, and the model then has to
        # map between the two: asked for a cell in an isolated table it answered
        # with the first *data* column 48 times out of 52 column errors. Printing
        # the numbers over the rows took cell accuracy from 37.3% to 50.7%.
        width = max((len(row) for row in rows[:max_rows]), default=0)
        lines = ["# " + ",".join(f"c{c}" for c in range(width))] if COLUMN_RULER else []
        for position, row in enumerate(rows[:max_rows]):
            # Bind the column index to the header text in the header line itself.
            # Both earlier attempts put the mapping somewhere the model had to
            # carry across a gap — the schema block above, or a ruler line whose
            # commas do not line up with the body's — and the isolated probe says
            # neither held: 49 of 58 column errors were exactly -1, the model
            # naming c0, the label column, where the answer sat in c1. Writing
            # `c1=Số cuối năm` on the header row took cell accuracy from 50.0% to
            # 69.1% and cut column errors from 26.4% to 7.7%.
            if position == 0 and COL_HEADER_TAGS:
                lines.append(",".join(f"c{c}={cell}"
                                      for c, cell in enumerate(row)))
                continue
            body = ",".join(str(cell) for cell in row)
            if position == 0 or not ROW_INDEX:
                lines.append(body)
            else:
                lines.append(f"[{position - 1}] {body}")
        block = "\n".join(lines)[:max_chars]
        blocks.append(
            f"{describe_schema(variable, refs[variable], rows, (notes or {}).get(variable, ''))}\n"
            f'<table variable="{variable}">\n{block}\n</table>'
        )
    return "\n\n".join(blocks)


RUNAWAY_INDENT_RE = re.compile(r"\n[ \t]{60,}\S")
# No legitimate program in this corpus is anywhere near this long; the median is
# 4.3k characters and the 90th percentile under 9k.
RUNAWAY_LENGTH = 20000


def degenerate(code: str) -> bool:
    """True when the reply is a repetition loop rather than a program.

    Greedy decoding on a long prompt walks into one: 21 of 74 failures in the
    150-question arm were `SyntaxError: too many levels of indentation`, and the
    offending reply was 62,114 characters that trailed off into blank space.
    """

    return bool(code) and (
        len(code) > RUNAWAY_LENGTH or RUNAWAY_INDENT_RE.search(code) is not None
    )


def extract_code(reply: str) -> str:
    """Pull runnable code out of a chat reply."""

    text = THINK_RE.sub("", reply).strip()
    fenced = CODE_FENCE_RE.findall(text)
    if fenced:
        # The last fence is usually the final answer after any reasoning.
        return fenced[-1].strip()
    return text


@dataclass(slots=True)
class Generated:
    code: str
    result: ExecResult
    attempts: int


def generate_query(
    client,
    system: str,
    user_template: str,
    question: str,
    tables: dict[str, list[list[str]]],
    refs: dict[str, str],
    max_attempts: int = 2,
    prelude: bool = True,
    notes: dict[str, str] | None = None,
) -> Generated:
    variable_hint = ", ".join(tables)
    user = (
        user_template.replace("{{QUESTION}}", question)
        .replace("{{VAR_HINT}}", variable_hint)
        .replace("{{TABLES}}", render_tables(tables, refs, notes=notes))
    )

    feedback = ""
    # Penalising only the retry recovered half the loops and left the rest: the
    # penalised second attempt loops too. `VIFIN_REP_PENALTY=1.05` applies it from
    # the first call instead.
    sampling: dict | None = (
        {"repetition_penalty": REP_PENALTY} if REP_PENALTY else None
    )
    last = Generated("", ExecResult(False, None, "no attempt"), 0)
    for attempt in range(1, max_attempts + 1):
        try:
            reply = client.complete(system, user + feedback, extra=sampling)
        except RuntimeError as exc:
            return Generated("", ExecResult(False, None, f"llm error: {exc}"), attempt)

        # The helper is prepended rather than injected into the namespace: the
        # scorer runs `pandas_query` standalone with only the declared frames
        # bound, so anything the program needs has to travel inside it.
        code = extract_code(reply)
        if prelude and code and "def num(" not in code:
            code = PRELUDE + "\n" + code
        unsafe = portability_problems(code)
        outcome = (
            ExecResult(False, None, "; ".join(unsafe)) if unsafe else run_query(code, tables)
        )
        last = Generated(code, outcome, attempt)
        # A program that returns exactly 0 is usually `result = 0` surviving a
        # lookup that matched nothing, not a genuine zero. Spend the remaining
        # attempt on it rather than shipping a silent miss.
        suspicious = outcome.ok and outcome.value == 0.0 and attempt < max_attempts
        if outcome.ok and not suspicious:
            return last
        # A looped program must not be echoed back. Feeding 62,114 characters of
        # runaway indentation into the retry prompt both blows the context and
        # shows the model the exact text to continue, so the second attempt loops
        # the same way. Ask again with a repetition penalty and say only what went
        # wrong. Widening the context window is what exposed this: the long
        # prompts used to be rejected by the server and never generated at all.
        if degenerate(code):
            sampling = {"repetition_penalty": 1.05, "temperature": 0.2}
            feedback = (
                "\n\n<previous_attempt_failed>\nChương trình trước lặp vô hạn: "
                f"{len(code)} ký tự, phần lớn là thụt lề trống, nên không biên "
                "dịch được.\n\nViết lại NGẮN GỌN: đọc thẳng các ô cần thiết bằng "
                "`num(df, r, c)` rồi gán `result`. Không lồng vòng lặp sâu, không "
                "quét toàn bảng.\n</previous_attempt_failed>"
            )
            continue
        available = "\n".join(
            describe_schema(variable, refs[variable], rows) for variable, rows in tables.items()
        )
        if suspicious:
            feedback = (
                f"\n\n<previous_attempt_suspicious>\nYour program:\n{code}\n\n"
                f"It returned exactly 0.0. That almost always means the row or column "
                f"lookup matched nothing and an initial `result = 0` survived.\n\n"
                f"These are the only columns and labels that exist. Use them verbatim:\n"
                f"{available}\n\n"
                f"Locate the figure properly and return only the corrected program.\n"
                f"</previous_attempt_suspicious>"
            )
            continue
        feedback = (
            f"\n\n<previous_attempt_failed>\nYour program:\n{code}\n\n"
            f"Error: {outcome.error}\n\n"
            f"These are the only columns and labels that exist. Use them verbatim:\n"
            f"{available}\n\n"
            f"Return only the corrected program, complete and untruncated.\n"
            f"</previous_attempt_failed>"
        )
    return last
