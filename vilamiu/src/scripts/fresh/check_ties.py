"""Do the note headings agree with the codes the arithmetic tied them to?

I said the note path had no verifier: the total tie confirms which statement line a
note details, but nothing confirms the row picked inside it, and nothing independently
confirms the tie itself.

The cleaned headings supply the missing check. A note tied to `cdkt/320` by arithmetic
turns out to be headed "Vay và nợ thuê tài chính ngắn hạn", which is that code's name;
one tied to `cdkt/132` is headed "Trả trước cho người bán ngắn hạn". The tie comes
from summing a column and matching a figure; the heading comes from text above the
anchor. Two unrelated signals, and where they agree the identification is not a
coincidence of arithmetic.

This measures that agreement across the corpus, which turns the note path from
unverified into measured.

Usage:  python scripts/fresh/check_ties.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from refine_codes import RELIABLE_LEN, agrees, tokens_exact  # noqa: E402


def verified(record: dict) -> bool:
    for kind, target, plus, minus in IDENTITIES:
        if record["kind"] != kind:
            continue
        for period in ("current", "prior"):
            cells = record[period]
            if not all(c in cells for c in (target,) + plus + minus):
                continue
            expected = cells[target][0]
            total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
            if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                return True
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not verified(record):
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) == want:
                label = tokens_exact(cell[1])
                if label:
                    votes[(record["kind"], code)][label] += 1
    labels = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    print(f"tu dien ma: {len(labels)}")

    counters: Counter[str] = Counter()
    disagreements = []
    for line in (ROOT / args.notes).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        note = json.loads(line)
        if not note.get("paired"):
            continue
        heading = tokens_exact(note.get("heading", ""))
        if not heading:
            counters["khong co tieu de de doi chieu"] += 1
            continue
        matched = False
        known = False
        for ref in note["paired"]:
            kind, code = ref.split("/")
            label = labels.get((kind, code))
            if label is None:
                continue
            known = True
            if agrees(heading, label):
                matched = True
                break
        if not known:
            counters["ma khong co trong tu dien"] += 1
        elif matched:
            counters["TIEU DE KHOP ma da noi"] += 1
        else:
            counters["tieu de LECH ma da noi"] += 1
            if len(disagreements) < args.show:
                disagreements.append(note)

    total = counters["TIEU DE KHOP ma da noi"] + counters["tieu de LECH ma da noi"]
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    if total:
        print(f"\ntrong so cac ca doi chieu duoc: "
              f"{100 * counters['TIEU DE KHOP ma da noi'] / total:.1f}% khop")
    print("\nvi du lech:")
    for note in disagreements:
        print(f"  TM{note['table_id']:<4d} noi {','.join(note['paired'][:2]):<18s} "
              f"{note['heading'][:74]}")


if __name__ == "__main__":
    main()
