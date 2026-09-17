"""Does a window of raw OCR text contain the answer, without any table step?

The pipeline finds the right document 96.4% of the time and then throws most of that
away: extracting tables and choosing among eight of them lands on the right table
27.5% of the time end to end. Extraction also damages what it keeps — headers arrive
glued together as `'Số cuối nămTriệu đồng'`.

The raw `*_extracted.txt` has none of that. It carries the note titles, the row
labels and the figures in reading order. So the question is whether a window of it,
picked by matching the question's metric against the lines themselves, contains the
figure at all — because that is the ceiling of an architecture with no table step.

Only the presence of the digits is measured here. Whether a model can pick the right
one out of the window is the next question and needs a GPU; if the digits are not
there, that question does not arise.

Usage:
  PYTHONPATH=src python scripts/_probe_text_window.py --limit 400 --window 60 --regions 3
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import parse_number  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableStore  # noqa: E402

CORPUS = ROOT / "data" / "official_corpus"
NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\)?|\(?-?\d+[.,]\d+\)?")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def doc_path(doc_name: str):
    parts = doc_name.split("_")
    year = next((p for p in parts if p.isdigit() and len(p) == 4), None)
    if not parts or year is None:
        return None
    path = CORPUS / parts[0] / year / doc_name / f"{doc_name}_extracted.txt"
    return path if path.exists() else None


def pick_regions(lines, metric_tokens, window: int, regions: int):
    """Line windows whose text overlaps the metric most."""

    scored = []
    for index, line in enumerate(lines):
        tokens = set(fold(line).split())
        if not tokens:
            continue
        shared = len(metric_tokens & tokens)
        if shared:
            scored.append((shared / len(metric_tokens), index))
    scored.sort(reverse=True)
    chosen, taken = [], []
    for _, index in scored:
        if any(abs(index - other) < window for other in taken):
            continue
        taken.append(index)
        chosen.append((max(0, index - window // 3),
                       min(len(lines), index + window)))
        if len(chosen) >= regions:
            break
    return chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--regions", type=int, default=3)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    tally: collections.Counter[str] = collections.Counter()
    sizes = []
    seen = 0
    cache: dict[str, list[str]] = {}

    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc = refs[0].rpartition("|table_")[0]
        gold = parse_number(record.get("answer_raw_cell")
                            if record.get("answer_raw_cell") is not None
                            else record.get("answer"))
        if gold is None or gold == 0:
            continue
        if doc not in cache:
            path = doc_path(doc)
            cache[doc] = (path.read_text(encoding="utf-8", errors="replace")
                          .splitlines() if path else [])
        lines = cache[doc]
        if not lines:
            continue
        seen += 1

        parsed = parse_question(record.get("id", 0), record["question"], roster)
        metric = lookup_mod.extract_metric(record["question"])
        tokens = {t for t in fold(metric).split() if len(t) > 2}
        if not tokens:
            tally["không trích được chỉ tiêu"] += 1
            continue
        spans = pick_regions(lines, tokens, args.window, args.regions)
        if not spans:
            tally["không có vùng nào"] += 1
            continue
        text = "\n".join("\n".join(lines[a:b]) for a, b in spans)
        sizes.append(len(text))

        # Is the figure itself inside the window, on any scale the pipeline uses?
        found = False
        for match in NUMBER_RE.finditer(text):
            value = parse_number(match.group(0))
            if value is None or value == 0:
                continue
            for factor in (1.0, 1e3, 1e6, 1e9, 1e12):
                if abs(abs(value) - abs(gold)) <= 0.01 or \
                        abs(abs(value) * factor - abs(gold)) <= max(0.01, abs(gold) * 1e-9) or \
                        abs(abs(value) / factor - abs(gold)) <= max(0.01, abs(gold) * 1e-9):
                    found = True
                    break
            if found:
                break
        tally["CÓ số gold trong cửa sổ" if found else "không có số gold"] += 1

    total = max(seen, 1)
    sizes.sort()
    print(f"{seen} câu gold, cửa sổ {args.window} dòng × {args.regions} vùng\n")
    for name, count in tally.most_common():
        print(f"  {name:28s} {count:5d}  {count / total:6.1%}")
    if sizes:
        median = sizes[len(sizes) // 2]
        print(f"\n  ký tự/prompt: trung vị {median} (~{median / 2.6:.0f} token), "
              f"p90 {sizes[int(0.9 * len(sizes))]}")
    print("  (so: đọc đúng bảng end-to-end hiện tại 27,5%)")


if __name__ == "__main__":
    main()
