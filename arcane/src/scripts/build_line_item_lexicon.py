#!/usr/bin/env python3
"""Build the corpus-derived vocabulary used for line-item decomposition.

The vocabulary is data, not a hand-written shortcut list. Rebuilding it from
the organizer corpus keeps graph expansion explainable and makes a clean clone
independent of the generated copy committed under ``data/derived``.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from vifinqa.retrieval import load_reports, report_tables, tokenize

ROOT = Path(__file__).resolve().parents[1]


def row_label(row: tuple[str, ...]) -> str:
    """Return the longest leading cell, where statement row labels live."""
    return " ".join(tokenize(max(row[:2], key=len) if len(row) > 1 else row[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "derived" / "line_items.json")
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--min-reports", type=int, default=10)
    parser.add_argument("--max-question-share", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    reports = load_reports(args.dataset_root)
    report_frequency: Counter[str] = Counter()
    for number, report in enumerate(reports, 1):
        seen: set[str] = set()
        for table in report_tables(str(report.path), report.identity):
            for row in table.rows:
                if not row:
                    continue
                label = row_label(row)
                token_count = len(label.split())
                if (args.min_tokens <= token_count <= args.max_tokens
                        and not label.replace(" ", "").isdigit()):
                    seen.add(label)
        report_frequency.update(seen)
        report_tables.cache_clear()
        if args.progress_every and number % args.progress_every == 0:
            print(f"indexed {number}/{len(reports)} reports", file=sys.stderr, flush=True)

    frequent = {
        label for label, count in report_frequency.items()
        if count >= args.min_reports
    }
    questions = [
        json.loads(line)
        for line in (args.dataset_root / "questions" / "questions.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    question_frequency: Counter[str] = Counter()
    question_texts = [" ".join(tokenize(record["question"])) for record in questions]
    for text in question_texts:
        question_frequency.update(label for label in frequent if label in text)

    limit = args.max_question_share * len(questions)
    lexicon = {
        label: {
            "reports": report_frequency[label],
            "questions": question_frequency[label],
        }
        for label in sorted(frequent)
        if question_frequency[label] <= limit
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lexicon, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    dropped = sorted(
        (label for label in frequent if question_frequency[label] > limit),
        key=lambda label: (-question_frequency[label], label),
    )
    matched = sum(any(label in text for label in lexicon) for text in question_texts)
    print(json.dumps({
        "output": str(args.output),
        "labels_seen": len(report_frequency),
        "frequent_labels": len(frequent),
        "lexicon": len(lexicon),
        "dropped_as_boilerplate": dropped[:12],
        "questions_matching_any": matched,
        "questions": len(questions),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
