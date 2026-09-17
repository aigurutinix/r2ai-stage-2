"""In the 955 documents where no unit line was found, what does the text actually say?

The document scanner validated at 97.7% against documents whose scale is known from
their statements, so it is right when it fires. It fires on 963 of 1,918 documents and
finds nothing in 955, which is implausible: a Vietnamese financial statement almost
always prints "Đơn vị: VND" or "Đơn vị tính: triệu đồng" beside the balance sheet. So
the detector is likely missing them rather than the declarations being absent.

Three candidate causes, each with a different fix, so it is worth one measurement rather
than three guesses:

  line too long      the declaration shares an OCR line with other text and fails the
                     60-character limit
  in a table header  the unit sits inside a `<td>` of the table rather than on a text
                     line, which this scanner never looks at
  no marker word     the text says "(VND)" or "triệu đồng" with no "Đơn vị" before it

Prints, for a sample of the silent documents, every line that mentions đồng or VND at
all, with its length and whether it carries a marker word.

Usage:  python scripts/fresh/probe_missing_unit.py --n 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_scale import DONG_RE, MARKER_RE, document_scale, fold  # noqa: E402

SCALE_WORD_RE = re.compile(r"\btrieu\b|\bty\b|\bnghin\b")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--docs", type=int, default=120)
    args = parser.parse_args()

    where: dict[str, tuple[str, str]] = {}
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            where.setdefault(record["doc"], (record["ticker"], record["year"]))

    counters: Counter[str] = Counter()
    samples: list[tuple[str, list[str]]] = []
    seen = 0
    for doc in sorted(where):
        if seen >= args.docs:
            break
        ticker, year = where[doc]
        path = (ROOT / "data" / "official_corpus" / ticker / year / doc /
                f"{doc}_extracted.txt")
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scale, _votes = document_scale(text)
        if scale is not None:
            continue  # the scanner already handles this document
        seen += 1

        mentions = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            flat = fold(stripped)
            if not DONG_RE.search(flat):
                continue
            has_marker = bool(MARKER_RE.search(flat))
            has_scale = bool(SCALE_WORD_RE.search(flat))
            if has_marker and len(stripped) > 60:
                counters["co dau hieu nhung DONG QUA DAI"] += 1
            elif has_marker:
                counters["co dau hieu, ngan — le ra phai bat duoc"] += 1
            elif has_scale:
                counters["co tu ty le nhung KHONG co chu 'don vi'"] += 1
            else:
                counters["chi nhac dong/VND, khong phai khai bao"] += 1
            if len(mentions) < 4 and (has_marker or has_scale):
                mentions.append(f"[{len(stripped):3d}] {stripped[:110]}")
        if not mentions:
            counters["tai lieu KHONG he co dong nao nhac dong/VND"] += 1
        elif len(samples) < args.n:
            samples.append((doc, mentions))

    print(f"xem {seen} tai lieu ma bo quet im lang\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print("\nvi du cac dong co nhac:")
    for doc, mentions in samples:
        print(f"  {doc[:46]}")
        for line in mentions:
            print(f"     {line}")


if __name__ == "__main__":
    main()
