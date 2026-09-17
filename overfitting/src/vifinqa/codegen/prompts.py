"""Prompt construction for Text-to-Pandas (PoT) generation."""
from __future__ import annotations

SYSTEM = """You are an expert Vietnamese financial data analyst who writes pandas code.

You get a Vietnamese question about audited financial statements and one or more
DataFrames (df1, df2, ...). Each DataFrame is a normalized LONG-FORMAT view of one
OCR'd table with columns:
  row(int), label(str: the line-item text), code(str: VAS line code like 10, 60, 270),
  col(int: column index in the original table), col_name(str: original column header,
  often a year like '31/12/2023' or '2023'), value(float: the parsed number),
  unit_scale(float: multiplier converting `value` to VND, e.g. 1e6 if the table is in
  'triệu đồng')

Rules:
1. Use ONLY the provided variables (df1, df2, ...), pd and np. No imports, no I/O, no print.
2. Numbers are already parsed: negatives from '(1.839)' already carry a minus sign.
3. Find the right line item by fuzzy label match (str.contains with a distinctive
   lowercase substring, case=False) and/or by VAS code when obvious
   (e.g. net revenue code '10', profit after tax '60', total assets '270').
3b. When a CANONICAL CELL CONTRACT is provided, use every listed metric and
   address its concrete row and period explicitly (row/code/label AND col).
   Never substitute a parent total for a child row, a child row for a total, or
   another row merely because it is in the same table.
4. Pick the right column: match the asked year in col_name ('2023', '31/12/2023').
   For a report of the asked year, prefer 'Năm nay', 'Kỳ này', 'Số cuối năm' or
   'Cuối kỳ'. Never use metadata columns such as 'Mã số', 'Thuyết minh' or
   'Chỉ tiêu' as financial values.
5. Convert units: value_vnd = value * unit_scale. The question asks the answer in a
   specific unit -> answer = round(value_vnd / ANSWER_SCALE, 2).
5b. UNIT OF THE ANSWER (organizer-confirmed, a frequent source of wrong answers):
   the answer must be in the unit the QUESTION asks for.
     output_type=percent            -> return 90 for "90%", NEVER 0.9.
                                       a ratio must be multiplied by 100;
                                       a cell that is already a percentage must NOT.
     output_type=percentage_point   -> difference of two percentages (p1 - p2).
     output_type=ratio              -> "lần"/"vòng": plain a / b, do NOT x100.
     output_type=year               -> an integer year such as 2023.
     output_type=count              -> an integer count of companies/items.
     output_type=number             -> money divided by ANSWER_SCALE.
6. Robustness: ALWAYS pass na=False to .str.contains (labels can be NaN);
   the `code` column may load as numbers -> compare via df['code'].astype(str).
7. CRITICAL — the answer is graded by EVALUATING your code as a SINGLE PYTHON
   EXPRESSION. Write exactly ONE line of the form `answer = <expression>`.
   No intermediate variables, no comments, no extra statements, no if/else, no
   loops: those are SyntaxErrors when evaluated and score as a crash.
   BAD:
       rows = df1[df1['label'].str.contains('x', na=False)]
       answer = round(float(rows['value'].iloc[0]) / 1e9, 2)
   GOOD:
       answer = round(float(df1[df1['label'].str.contains('x', na=False) & (df1['col'] == 1)]['value'].iloc[0]) * 1e6 / 1e9, 2)
8. Output ONLY one ```python code block containing that single line."""

DEBUG_SYSTEM = SYSTEM

USER_TMPL = """Question (Vietnamese): {question}

Parsed intent: ticker={tickers}, years={years}, doc_type={doc_type}, \
output_type={output_type}, answer unit = {unit_name} \
(ANSWER_SCALE = {unit_scale:g}), percent={is_percent}

Available DataFrames:
{tables_block}

{plan_block}
CANDIDATE ROWS (pre-matched to the question's metric — prefer these; idx is for
your reading only, address rows in code by their label/code/col):
{shortlist_block}

FINAL TASK (this repeated anchor must remain actionable if long table previews
are truncated):
- Question: {question}
- Each df is long-format with row, label, code, col, col_name, value, unit_scale.
- Compute output_type={output_type} in {unit_name}; ANSWER_SCALE={unit_scale:g}.
- Use only df1..dfN, pd and np. Output ONLY one ```python block containing
  exactly `answer = <single expression>` rounded to 2 decimals."""

TABLE_TMPL = """--- {var} | source: {report_id}|{table_pos} | page {page} | \
table unit_scale={unit_scale} ({unit_note}) ---
{preview}
"""

DEBUG_TMPL = """Your previous code failed.

Question: {question}
Previous code:
```python
{code}
```
Error: {error}

Fix the code. Same rules apply: only df1..dfN, pd, np; ONE line
`answer = <single expression>` (no intermediate variables/comments/control flow);
output ONLY one ```python block."""


def build_user(question: str, route: dict, tables: list[dict],
               shortlist_block: str = "", plan_block: str = "") -> str:
    blocks = []
    for t in tables:
        us = t.get("unit_scale")
        note = "unit unknown, assume VND" if not us else (
            {1.0: "VND", 1e3: "nghìn đồng", 1e6: "triệu đồng", 1e9: "tỷ đồng",
             1e11: "trăm tỷ đồng",
             1e12: "nghìn tỷ đồng"}.get(float(us), f"x{us:g} VND"))
        blocks.append(TABLE_TMPL.format(
            var=t["var"], report_id=t["report_id"], table_pos=t["table_pos"],
            page=t.get("page", "?"), unit_scale=(us or 1.0), unit_note=note,
            preview=t["preview"],
        ))
    return USER_TMPL.format(
        question=question,
        tickers=",".join(route.get("tickers", [])) or "?",
        years=route.get("years", []),
        doc_type=route.get("doc_type", "consolidated"),
        unit_name=route.get("unit_name", "đồng"),
        unit_scale=float(route.get("unit_scale", 1.0)),
        is_percent=route.get("is_percent", False),
        output_type=route.get("output_type", "number"),
        tables_block="\n".join(blocks),
        shortlist_block=shortlist_block or "(none)",
        plan_block=plan_block or "",
    )


def build_debug(question: str, code: str, error: str) -> str:
    return DEBUG_TMPL.format(question=question, code=code, error=error)


# --------------------------------------------------------------------------
# SELECTION MODE — the model picks cells, we write the pandas.
# Audited on submission #12: when the model wrote pandas itself, 35% of queries
# had no column filter, 15% skipped the unit conversion and 90% omitted
# regex=False. All three are impossible here because the addressing, the column
# and the unit arithmetic are generated by vifinqa/codegen/selection.py.
# --------------------------------------------------------------------------

SELECT_SYSTEM = """You are a Vietnamese financial analyst. You do NOT write code.

You are given a question and a numbered list of CANDIDATE ROWS already located
in the financial statements. Each candidate is one concrete cell (a line item in
a specific report, column and period) with its value.

Your only job: choose which candidate row(s) answer the question, and which
arithmetic operation combines them. The system converts units and builds the
query — never do arithmetic or unit conversion yourself.

Answer with ONE JSON object and nothing else:
  {"op": "<operation>", "operands": [<candidate numbers>]}

Operations:
  lookup            one value as-is                      operands: [i]
  difference        A - B                                 operands: [A, B]
  growth_pct        percent growth, LATER period first    operands: [end, base]
  ratio             A / B as a percentage                 operands: [num, den]
  ratio_times       A / B in "lần"/"vòng" (no x100)       operands: [num, den]
  margin            profit / base as a percentage         operands: [num, den]
  percentage_point  difference of two percentages         operands: [A, B]
  sum               total of all listed                   operands: [i, j, ...]
  average           mean of all listed                    operands: [i, j, ...]
  ranking_max       the largest of the listed             operands: [i, j, ...]
  ranking_min       the smallest of the listed            operands: [i, j, ...]
  count             count selected entities/periods       operands: [i, j, ...]

Rules:
1. Pick candidates by their line-item meaning AND their period. The column
   header (col_name) tells you the period — never pick a row whose period
   contradicts the question.
2. For a comparison across companies, pick one candidate PER company.
3. For count questions, pick only candidates that satisfy the stated sign or
   threshold condition, with at most one proof row per counted entity/period.
4. If no candidate fits the question, answer {"op": "none", "operands": []}.
   Guessing a wrong row is worse than admitting it is not there.
5. Output only the JSON object. No explanation, no code, no markdown."""

SELECT_USER_TMPL = """Question (Vietnamese): {question}

Parsed intent: ticker={tickers}, years={years}, doc_type={doc_type}, \
output_type={output_type}, requested unit = {unit_name}
{plan_block}
CANDIDATE ROWS:
{shortlist_block}

Reply with one JSON object: {{"op": ..., "operands": [...]}}"""


def build_select_user(question: str, route: dict, shortlist_block: str,
                      plan_block: str = "") -> str:
    return SELECT_USER_TMPL.format(
        question=question,
        tickers=",".join(route.get("tickers", [])) or "?",
        years=route.get("years", []),
        doc_type=route.get("doc_type", "consolidated"),
        output_type=route.get("output_type", "number"),
        unit_name=route.get("unit_name", "đồng"),
        plan_block=("\n" + plan_block if plan_block else ""),
        shortlist_block=shortlist_block or "(none)",
    )
