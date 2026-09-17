"""Recover the argument list behind each computed module constant in the lost
`run_submit.py`.

`_recover_pyc.py` prints a literal when the assignment is one `LOAD_CONST`, and
`<computed: CALL 1>` when it is not — which is every flag routed through the
`_flag(name, default)` helper, i.e. exactly the ones that decide behaviour. The
constants pushed before the call are still in the bytecode, so the name and the
default are recoverable; this walks backwards from each `STORE_NAME` to the
matching call and prints what was on the stack.
"""

from __future__ import annotations

import dis
import marshal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYC = ROOT / "scripts" / "__pycache__" / "run_submit.cpython-313.pyc"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    module = marshal.loads(PYC.read_bytes()[16:])
    instructions = list(dis.get_instructions(module))

    print(f"{PYC.name}\n")
    for index, ins in enumerate(instructions):
        if ins.opname != "STORE_NAME" or not ins.argval.isupper():
            continue
        if instructions[index - 1].opname == "LOAD_CONST":
            continue
        # Walk back to the start of the expression: the nearest preceding
        # LOAD_NAME/LOAD_GLOBAL that begins the call, bounded so a long
        # expression cannot swallow the previous statement.
        start = index
        for back in range(index - 1, max(-1, index - 14), -1):
            if instructions[back].opname in ("LOAD_NAME", "LOAD_GLOBAL", "PUSH_NULL"):
                start = back
        parts = []
        for step in instructions[start:index]:
            if step.opname in ("LOAD_CONST", "LOAD_NAME", "LOAD_GLOBAL"):
                parts.append(repr(step.argval))
            elif step.opname == "CALL_KW":
                parts.append("<kw>")
        print(f"  {ins.argval:24s} <- {' '.join(parts)}")

    print("\n=== helper functions present in the recovered module ===")
    for const in module.co_consts:
        if hasattr(const, "co_code") and const.co_name != "main":
            names = [c for c in const.co_consts if isinstance(c, str)]
            print(f"  def {const.co_name}({', '.join(const.co_varnames[:const.co_argcount])})"
                  f"   consts={names[:6]}")


if __name__ == "__main__":
    main()
