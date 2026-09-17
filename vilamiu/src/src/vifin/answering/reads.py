"""Detect a program whose answer rests on a failed row lookup.

`find_row` returns -1 when no label matches. The prompt says to check for it and
many programs do not, so `num(frame, -1, c)` runs — and `frame.iloc[-1, c]` is the
LAST row of the table, returned without complaint. In a Vietnamese balance sheet
the last row is the total, so a failed lookup for "tài sản cố định vô hình" comes
back as total assets. A share-of-total question then divides the total by itself
and reports exactly 100.0, which is what twelve answers in one block of the
submission did.

Measured on `direct14b_repair.zip`: **75 of 1012 answers (7.4%)** read at least one
cell at a negative row index. They are wrong by construction — the program did not
find what it was looking for and reported a number anyway.

Editing the `num` in the prelude would only affect programs generated afterwards;
every shipped program carries its own copy. So the check watches the program run
instead, which works on a cache that already exists.
"""

from __future__ import annotations

import sys


def negative_reads(query: str, frames: dict) -> int:
    """How many cells the program reads at a negative row index.

    Traced rather than parsed: the row index is almost always a variable holding
    `find_row`'s result, so it is not visible in the source.
    """

    if not query or not frames:
        return 0

    # Callers hold tables in both shapes. `run_submit` keeps raw grids — lists of
    # lists — while the tracing scripts hold DataFrames. Binding a list as `df`
    # makes the program raise on its first `.iloc`, so every read is missed and
    # the gate silently passes everything. That is exactly what happened when
    # this was first wired in, and it is invisible: the count is simply 0.
    from vifin.answering.sandbox import frame_from_rows

    frames = {
        name: (table if hasattr(table, "iloc") else frame_from_rows(table))
        for name, table in frames.items()
    }

    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event != "call" or frame.f_code.co_name != "num":
            return None
        row = frame.f_locals.get("r")
        try:
            if int(row) < 0:
                count += 1
        except (TypeError, ValueError):
            pass
        return None

    namespace: dict = dict(frames)
    try:
        import pandas as pd

        namespace.setdefault("pd", pd)
    except ImportError:
        pass

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(query, namespace, namespace)  # noqa: S102 - our own generated code
    except Exception:  # noqa: BLE001 - a program that dies still made its reads
        pass
    finally:
        sys.settrace(previous)
    return count
