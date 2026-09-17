"""Class mix of any question set against the exam's, by the same classifier.

The distribution gap is the thing being fixed, so it has to be measurable on
whatever candidate set is proposed — generated records, SFT pairs, or a blend —
without editing a script each time.

Usage:
  python scripts/_mix.py artifacts/shapes.jsonl [artifacts/sft_locate2.jsonl ...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _probe_exam_classes import classify  # noqa: E402

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

# Shapes the model cannot see enough tables to answer, so they are excluded from
# the training target and the exam share is renormalised over what is left.
UNTRAINABLE = {"cohort screen / rank", "count over a group", "median filter"}


def questions_of(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line.replace(": NaN", ": null"))
        text = row.get("question") or row.get("meta", {}).get("question")
        if text:
            out.append(text)
    return out


def mix(texts: list[str], roster) -> collections.Counter:
    counts: collections.Counter[str] = collections.Counter()
    for text in texts:
        parsed = parse_question(0, text, roster)
        counts[classify(text, len(parsed.tickers), len(parsed.years))] += 1
    return counts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    exam = mix(questions_of(ROOT / "data" / "questions" / "questions.jsonl"), roster)
    trainable = sum(n for k, n in exam.items() if k not in UNTRAINABLE)

    sets = {}
    for arg in sys.argv[1:]:
        path = ROOT / arg
        sets[path.name] = mix(questions_of(path), roster)

    classes = [k for k, _ in exam.most_common()]
    width = max(len(c) for c in classes) + 2
    header = f"{'class':<{width}}{'exam':>8}{'target':>8}"
    for name in sets:
        header += f"{name[:16]:>18}"
    print(header)

    for name in classes:
        target = "-" if name in UNTRAINABLE else f"{exam[name] / trainable:.1%}"
        line = f"{name:<{width}}{exam[name] / sum(exam.values()):>8.1%}{target:>8}"
        for counts in sets.values():
            total = sum(counts.values()) or 1
            line += f"{counts[name] / total:>18.1%}"
        print(line)

    print(f"\n'target' renormalises the exam over the trainable classes only "
          f"({trainable}/{sum(exam.values())} questions); the excluded ones need "
          f"more tables than the prompt carries.")


if __name__ == "__main__":
    main()
