"""Where do the 251 ratio questions fall out of the ratio branch?

The branch ships 26 answers. The exam asks 251 questions whose answer is a rate,
a share or a growth figure — a quarter of the paper — and the model-reading path
that was supposed to cover them just cost 0.38 questions per row on the board, so
this branch is what is left. Before changing it, find which gate drops what:

  unit      `target_unit` not one of phan_tram/lan/vong
  company   more than one ticker, or no year
  shape     the wording does not parse into (numerator, denominator)
  operand   one of the two operands has no row anywhere in the shortlist
  sanity    the quotient lands outside the plausible band

Prints the counts and a sample of the wording each gate rejects, because the
wording is what has to be handled.

Usage:  PYTHONPATH=src python scripts/_ratio_funnel.py --show 6
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import ratio as ratio_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

shape_spec = importlib.util.spec_from_file_location(
    "qs", ROOT / "scripts" / "_question_shape.py")
qs = importlib.util.module_from_spec(shape_spec)
shape_spec.loader.exec_module(qs)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=6)
    parser.add_argument("--resolve", action="store_true",
                        help="also run the operand lookup, which is slow")
    args = parser.parse_args()

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    ratio_questions = [q for q in questions if qs.shape(q.question) == "ty le"]
    print(f"{len(ratio_questions)} cau ty le trong {len(questions)} cau\n")

    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    def note(gate: str, question) -> None:
        counts[gate] += 1
        examples.setdefault(gate, []).append(question.question)

    passed = []
    for question in ratio_questions:
        if question.target_unit not in ("phan_tram", "lan", "vong"):
            note(f"don vi khac ({question.target_unit})", question)
            continue
        if len(question.tickers) != 1:
            note(f"so ma = {len(question.tickers)}", question)
            continue
        if not question.years:
            note("khong co nam", question)
            continue
        if ratio_mod.shape(question) is None:
            note("khong parse duoc tu/mau", question)
            continue
        passed.append(question)

    counts["qua het cong"] = len(passed)
    for gate, value in counts.most_common():
        print(f"  {gate}: {value}")

    for gate in sorted(examples):
        if gate.startswith("qua het"):
            continue
        print(f"\n--- {gate} ({counts[gate]} cau), {min(args.show, counts[gate])} vi du:")
        for text in examples[gate][:args.show]:
            print(f"    {text[:140]}")

    if args.resolve and passed:
        store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
        retriever = LexicalRetriever(store.frame)
        resolved = 0
        for question in passed:
            try:
                if ratio_mod.resolve(question, store, retriever) is not None:
                    resolved += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"\ntrong {len(passed)} cau qua cong, resolve ra chuong trinh: {resolved}")


if __name__ == "__main__":
    main()
