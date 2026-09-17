"""The one Vietnamese-number parser, defined once and supplied to every program.

Asking the model to write this parser cost three of four programs: it left out the
bracketed negative, so every cost line raised. Asking it to copy a supplied parser cost
five of seven: it called `_num(...)` and did not repeat the definition.

So the definition is not the model's job at all. It is prepended before execution and
prepended again into the program that ships, which makes the parse identical in both
places and leaves the model only the part that needs judgement — which table, which row,
which arithmetic.

The text is kept as source rather than a function because it has to travel: into the
sandbox namespace here, and into the `pandas_query` field of the submission.
"""

from __future__ import annotations

SOURCE = '''def _num(x):
    """A Vietnamese figure as a float: dots group thousands, a comma is the decimal
    mark, brackets mark a negative, and a dash means nil."""
    t = str(x).strip().replace("%", "").replace(" ", "")
    if not t or t in ("-", "–", "—", "nan", "None", "n/a"):
        return 0.0
    neg = t.startswith("(") and t.endswith(")")
    if neg:
        t = t[1:-1]
    t = t.replace(".", "").replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return 0.0
    return -v if neg else v

'''
