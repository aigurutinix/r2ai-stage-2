# Generation Review Instructions

Apply every item below before accepting or merging a generated question:

1. **Terminal scalar:** the question has exactly one numerical, monetary, percentage, year, count, or Boolean answer.
2. **Arithmetic correctness:** deterministic execution produces the claimed answer with no hidden correction constant or unsupported rounding.
3. **Data grounding:** every operand comes from the declared evidence tables and exact rows/columns; the query touches no undeclared evidence.
4. **Financial validity:** the formula, unit conversion, sign convention, time basis, and reporting scope are financially valid.
5. **Real terminology:** every financial metric or term is standard and broadly verifiable in financial references. Reject invented names that merely sound plausible.
6. **Real dependency for Hard:** each adaptive or multi-hop step materially depends on an earlier result; reject decorative or independent extra steps.
7. **Tier/scenario fit:** the number of facts, operations, tables, reports, periods, and dependency depth match the requested public tier and scenario.
8. **Question-query alignment:** the natural-language request, locked computation plan, Pandas query, evidence order, answer type, and unit describe the same task.
9. **Naturalness:** the Vietnamese question is self-contained and natural, names entities/periods/scope when necessary, and does not leak file, table, row, column, Python, or Pandas details.
10. **No forced formula:** reject a computation chosen only because candidate cells happen to be available. The formula or ratio must be genuinely used in financial analysis.
11. **Independent verification:** rerun the Pandas query on the original CSV strings and re-check the result, selected references, and schema before publication.

Do not waive one criterion because the others pass. An ambiguous case is a rejection or an escalation, never an automatic fallback.
