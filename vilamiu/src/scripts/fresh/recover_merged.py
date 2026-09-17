"""Recover Mã số rows that OCR glued to their figure, judged by the accounting identities.

Inside the tables that already classify as one of the three primary statements, 1,561
rows on a 600-document sample carry a cell that is not a number on its own but becomes
one once one, two or three leading digits are removed — about 5,100 rows corpus-wide.
Those are addresses the parser loses inside statements it can otherwise read.

The split needs a judge, and the label is a poor one here: tried against the consensus
dictionary it confirmed 2.7% of candidates and most of those were wrong, because a
leading digit in a note row is an ordinal, not a code. The identities are the right
judge instead. If removing "110" from a cell yields a figure that makes
270 = 100 + 200 hold where the term was previously missing, three things are confirmed
at once — the code, the column and the unit scale — and no coincidence of two errors
can produce that.

So the rule is: propose every split, accept only the ones that complete an identity.
Anything else is left alone.

Reports how many identities go from unverifiable to holding, which is the only
measure of whether the repair is worth having.

Usage:  python scripts/fresh/recover_merged.py --limit 600
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from probe_merged_cells import merged_splits  # noqa: E402
from refine_codes import RELIABLE_LEN  # noqa: E402


def candidates(table: ps.Table, kind: str, scale: float) -> dict[str, list[float]]:
    """Every (code -> values) a merged-cell split could add for this statement."""

    want = RELIABLE_LEN[kind]
    hint = ps._header_ma_so_index(table.header)
    out: dict[str, list[float]] = {}
    for row in table.rows:
        # Rows the parser already reads are left alone; this only adds what it drops.
        if ps._find_ma_so_index(row, hint) is not None and \
                ps._value_cells(row, exclude=ps._find_ma_so_index(row, hint)):
            continue
        for cell in row:
            for head, rest in merged_splits(cell):
                if len(head) != want:
                    continue
                value = ps.parse_vn_number(rest)
                if value is None:
                    continue
                out.setdefault(head, []).append(value * scale)
    return out


def identity_state(cells: dict[str, float], kind: str) -> tuple[int, int, int]:
    """(identities holding, identities short by exactly one term, total for kind)."""

    holds = short = total = 0
    for identity_kind, target, plus, minus in IDENTITIES:
        if identity_kind != kind:
            continue
        total += 1
        needed = (target,) + plus + minus
        missing = [c for c in needed if c not in cells]
        if not missing:
            expected = cells[target]
            got = sum(cells[c] for c in plus) - sum(cells[c] for c in minus)
            if abs(expected - got) <= REL_TOL * max(abs(expected), abs(got), 1.0):
                holds += 1
        elif len(missing) == 1:
            short += 1
    return holds, short, total


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    files = sorted((ROOT / "data" / "financial_statements").glob("*/*/*/*.txt"))
    if args.limit:
        files = files[:args.limit]

    counters: Counter[str] = Counter()
    examples: list[str] = []
    for path in files:
        try:
            tables = ps.read_document(path)
        except Exception:  # noqa: BLE001
            continue
        for table in tables:
            statement = ps.parse_statement(table)
            if statement is None:
                continue
            for period in ("current", "prior"):
                base = {code: cell.value for code, cell
                        in getattr(statement, period).items()}
                before_holds, before_short, _ = identity_state(base, statement.kind)
                counters["dang thuc DUNG truoc khi cuu"] += before_holds
                if not before_short:
                    continue
                proposals = candidates(table, statement.kind, statement.scale)
                if not proposals:
                    continue
                accepted = 0
                for code, values in proposals.items():
                    if code in base:
                        continue
                    for value in values:
                        trial = dict(base)
                        trial[code] = value
                        holds, _, _ = identity_state(trial, statement.kind)
                        if holds > before_holds:
                            base[code] = value
                            before_holds = holds
                            accepted += 1
                            if len(examples) < args.show:
                                examples.append(
                                    f"  {statement.kind} {period}: them ma={code} "
                                    f"gia tri={value:,.0f} -> mot dang thuc thanh DUNG "
                                    f"({table.doc_name[:40]} t{table.table_id})")
                            break
                if accepted:
                    counters["o cuu duoc VA lam dang thuc DUNG"] += accepted
                    counters["ky bao cao duoc cuu"] += 1

    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print("\nvi du:")
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
