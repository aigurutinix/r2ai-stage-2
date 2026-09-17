"""Recover what the lost `run_submit.py` did, from the bytecode left in __pycache__.

There is no git history here, and the copy of `run_submit.py` on disk is older
than the submission it is supposed to reproduce — it declares a flat top-k where
the best zip declares a span-dependent one, and 90 answers differ. The only other
record of the newer code is `scripts/__pycache__/run_submit.cpython-313.pyc`,
compiled on 08/08 from a revision that still had the declare policy.

Bytecode keeps every constant and every global name, so the flag values are
recoverable exactly, and the arithmetic is recoverable by reading the
disassembly. This prints both.

Usage:  python scripts/_recover_pyc.py [pattern]
"""

from __future__ import annotations

import dis
import marshal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYC = ROOT / "scripts" / "__pycache__" / "run_submit.cpython-313.pyc"
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "DECLARE_K"


def walk(code, seen=None):
    seen = seen if seen is not None else set()
    if id(code) in seen:
        return
    seen.add(id(code))
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            yield from walk(const, seen)


def main() -> None:
    raw = PYC.read_bytes()
    # 3.7+ pyc header: magic, flags, mtime/hash, size — 16 bytes.
    module = marshal.loads(raw[16:])
    print(f"{PYC.name}: recovered module code object\n")

    top = None
    for code in walk(module):
        if code.co_name == "<module>":
            top = code

    print("=== module-level constants assigned to ALL-CAPS globals ===")
    instructions = list(dis.get_instructions(top))
    for i, ins in enumerate(instructions):
        if ins.opname != "STORE_NAME" or not ins.argval.isupper():
            continue
        # The value is whatever the preceding instructions pushed; a literal
        # assignment is a single LOAD_CONST, which covers every flag here.
        previous = instructions[i - 1]
        if previous.opname == "LOAD_CONST":
            print(f"  {ins.argval:24s} = {previous.argval!r}")
        else:
            print(f"  {ins.argval:24s} = <computed: {previous.opname} "
                  f"{previous.argval!r}>")

    print(f"\n=== global names referenced anywhere (filtered by {PATTERN!r}) ===")
    names = set()
    for code in walk(module):
        names.update(n for n in code.co_names if PATTERN in n)
    for name in sorted(names):
        print(f"  {name}")

    print(f"\n=== disassembly of `main` around {PATTERN!r} ===")
    for code in walk(module):
        if code.co_name != "main":
            continue
        lines = list(dis.get_instructions(code))
        marks = [
            i for i, ins in enumerate(lines)
            if PATTERN in str(ins.argval) or PATTERN in str(ins.argrepr)
        ]
        if not marks:
            print("  (pattern not referenced in main)")
            break
        lo, hi = max(0, marks[0] - 40), min(len(lines), marks[-1] + 40)
        for ins in lines[lo:hi]:
            print(f"  {ins.offset:6d} {ins.opname:24s} {ins.argrepr}")
        break


if __name__ == "__main__":
    main()
