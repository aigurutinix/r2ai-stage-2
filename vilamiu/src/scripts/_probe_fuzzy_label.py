"""How much of the matcher's silence would a fuzzy label match recover?

`match_row` scores by token overlap, so it returns nothing when the question and
the table say the same thing with different tokens. Measured on the gold set, that
is 49% of the cases where `find` is silent, and it splits into three shapes:

    hỏi "tại Ngân hàng Nhà nước"        bảng "NHNN"                 viết tắt
    hỏi "xây dựng cơ bản chưa hoàn thành" bảng "dở dang"            đồng nghĩa
    hỏi "đầu tư xây dựng cơ bản đã hoàn thành" bảng "cơ bảnhoản thà"  OCR hỏng

Two of the three are reachable without a model. Character trigrams survive the OCR
damage that destroys tokens, and an initialism expands NHNN back to its words.
Synonyms are not reachable this way and are left to the fine-tune.

This measures the ceiling before anything is changed, because the number that
matters is not how many silences a fuzzy match *breaks* but how many it breaks
*correctly*. A loose matcher converts silence into confident error, which scores
the same as silence and also displaces the branch that would have answered.

Usage:  PYTHONPATH=src python scripts/_probe_fuzzy_label.py [limit]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster, strip_tones  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SOURCE = ROOT / "artifacts" / "easy_full.jsonl"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def parse_number(text: str) -> float | None:
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= 5e-4)


def trigrams(text: str) -> set[str]:
    flat = re.sub(r"\s+", " ", strip_tones(text).lower()).strip()
    return {flat[i:i + 3] for i in range(max(len(flat) - 2, 0))}


def similarity(a: str, b: str) -> float:
    """Trigram Jaccard, which degrades gracefully where tokens do not.

    "dau tu xay dung co ban da hoan thanh" against the OCR-mangled "co
    banhoan tha" shares most of its trigrams even though the token sets barely
    intersect, because the damage is a missing space rather than missing letters.
    """

    left, right = trigrams(a), trigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def initials(text: str) -> str:
    return "".join(word[0] for word in strip_tones(text).lower().split() if word)


def main() -> None:
    records = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ][:LIMIT]

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    stats: Counter[str] = Counter()
    examples: list[tuple[str, str, float, bool]] = []

    for index, record in enumerate(records):
        gold_answer = parse_number(record["answer"])
        if gold_answer is None:
            continue
        doc_name, table_id = record["relevant_tables"][0].rsplit("|table_", 1)
        grid = store.rows(TableKey(doc_name, int(table_id)))
        if not grid:
            continue
        question = parse_question(index, record["question"], roster)
        label_col = lookup_mod.label_column(grid)

        found = lookup_mod.find(grid, question)
        if found is not None and found.score >= lookup_mod.MIN_LABEL_SCORE:
            stats["matcher_already_answers"] += 1
            continue
        stats["silent"] += 1

        # Which rows actually hold the answer, so a fuzzy hit can be judged
        # rather than merely counted.
        gold_rows = {
            r_index
            for r_index, line in enumerate(grid[1:], start=1)
            for cell in line
            if (value := parse_number(cell)) is not None and value != 0
            and close(value, gold_answer)
        }
        if not gold_rows:
            stats["  answer_in_no_row"] += 1
            continue

        variants = list(lookup_mod.metric_variants(question.question))
        best_row, best_score, best_label = None, 0.0, ""
        for r_index, line in enumerate(grid[1:], start=1):
            if label_col >= len(line):
                continue
            label = str(line[label_col]).strip()
            if not label:
                continue
            for variant in variants:
                score = similarity(variant, label)
                # An initialism is a different kind of match: "NHNN" shares no
                # trigrams with "Ngan hang Nha nuoc" but is exactly its initials.
                if len(label) <= 8 and initials(variant).startswith(
                        strip_tones(label).lower()):
                    score = max(score, 0.9)
                if score > best_score:
                    best_row, best_score, best_label = r_index, score, label

        for threshold in (0.30, 0.40, 0.50, 0.60):
            if best_score >= threshold:
                hit = best_row in gold_rows
                stats[f"  fires@{threshold:.2f}"] += 1
                if hit:
                    stats[f"  CORRECT@{threshold:.2f}"] += 1
        if best_score >= 0.40 and len(examples) < 6:
            examples.append((variants[0] if variants else "", best_label,
                             best_score, best_row in gold_rows))

    silent = stats["silent"] or 1
    print(f"{stats['matcher_already_answers']} answered by the matcher, "
          f"{silent} silent\n")
    print(f"  {'threshold':12s} {'fires':>7} {'correct':>8} {'precision':>10} "
          f"{'of silences':>12}")
    for threshold in (0.30, 0.40, 0.50, 0.60):
        fires = stats[f"  fires@{threshold:.2f}"]
        right = stats[f"  CORRECT@{threshold:.2f}"]
        print(f"  {threshold:<12.2f} {fires:7d} {right:8d} "
              f"{right / max(fires, 1):9.1%} {fires / silent:11.1%}")

    print("\n  examples at 0.40:")
    for variant, label, score, hit in examples:
        mark = "OK " if hit else "BAD"
        print(f"    {mark} {score:.2f}  {variant[:38]!r} -> {label[:38]!r}")

    print("\n  Precision is the number that decides this. The matcher's own cell")
    print("  accuracy is ~70%, and the branches these questions fall through to")
    print("  score 5.9-13.3%, so a fuzzy match is worth shipping somewhere between")
    print("  those — but a threshold that fires on everything is worth nothing.")


if __name__ == "__main__":
    main()
