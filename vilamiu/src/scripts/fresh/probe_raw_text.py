"""Does the RAW OCR text carry the answer next to the words that ask for it?

Every reachability number I produced is dominated by oracle noise. The oracle is 39%
correct, a report holds tens of thousands of figures, so "the answer appears somewhere in
the document" is mostly measuring coincidence — and I built a strategy on it.

This measurement is designed to resist that. It does not ask whether the figure appears
in the report; it asks whether it appears NEXT TO the question's own metric words, inside
a window of a few dozen lines. A coincidental match is scattered anywhere in the
document, while the real figure sits beside its own label. Co-location is the part
coincidence does not reproduce.

It also tests the architecture two teams on the board appear to use. They score EXEC
0.6067 and 0.6601 with TABLES_F2 exactly 0.0 — and declaring a table is free once you
have located one, so a flat zero means they never locate one. That fits reading the raw
text directly. My own approach converted each report into a structured index, which
destroys precisely what makes a figure findable: the unit line above the table, the note
heading before it, the column header, the row label — all adjacent in the raw text, all
separated in my index and painfully reassembled one measurement at a time.

Reports three things:
  the figure appears near the metric words   the raw text is enough, and a text window
                                             plus a reader is the right architecture
  it appears only far away                   coincidence, and this says nothing
  it does not appear at all                  the answer is derived, not read

Usage:  python scripts/fresh/probe_raw_text.py --window 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

STOP = frozenset({"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "la",
                  "trong", "cho", "tai", "theo", "phan", "tram", "ctcp", "tong",
                  "gia", "tri", "muc", "khoan", "so", "du", "dong", "trieu", "ty"})


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def metric_tokens(text: str, ticker: str, name: str) -> set[str]:
    probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
    probe = re.sub(r"\([^)]*\)", " ", probe)
    probe = YEAR_RE.sub(" ", probe)
    for chunk in (name,) + STRIP_CHUNKS:
        probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
    return {t for t in fold(probe).split() if len(t) > 2 and t not in STOP}


def digit_forms(value: float) -> list[str]:
    """How a figure is printed in these reports, at each plausible unit scale."""

    forms = []
    for scale in (1.0, 1e-3, 1e-6, 1e-9):
        scaled = value * scale
        if scaled < 1:
            continue
        whole = int(round(scaled))
        if whole < 100:
            continue
        text = f"{whole:,}".replace(",", ".")
        forms.append(text)
    return forms


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    parser.add_argument("--window", type=int, default=40,
                        help="how many lines count as 'near'")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.oracle) as archive:
        oracle = {r["id"]: r.get("answer")
                  for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    corpus = ROOT / "data" / "official_corpus"
    cache: dict[Path, list[str]] = {}
    counters: Counter[str] = Counter()
    samples = []
    started = time.time()

    for number, question in enumerate(questions, start=1):
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        _name, unit = unit_of(text)
        if len(found) != 1 or not years or not unit:
            counters["ngoai pham vi"] += 1
            continue
        try:
            target = abs(float(oracle.get(question["id"]))) * unit
        except (TypeError, ValueError):
            continue
        if not target:
            continue

        ticker = next(iter(found))
        year = max(years)
        want_separate = bool(PARENT_RE.search(text))
        base = corpus / ticker / year
        if not base.is_dir():
            counters["khong co thu muc ma-nam"] += 1
            continue
        wanted = metric_tokens(text, ticker, resolver.tickers.get(ticker, ""))
        if not wanted:
            counters["khong con tu chi tieu"] += 1
            continue
        forms = digit_forms(target)
        if not forms:
            counters["dap an qua nho de tim chuoi"] += 1
            continue

        docs = sorted(p for p in base.iterdir() if p.is_dir())
        ordered = ([d for d in docs if ("separate" in d.name) == want_separate]
                   + [d for d in docs if ("separate" in d.name) != want_separate])

        verdict = "khong xuat hien trong van ban"
        for doc_dir in ordered:
            path = doc_dir / f"{doc_dir.name}_extracted.txt"
            if not path.exists():
                continue
            if path not in cache:
                cache[path] = path.read_text(
                    encoding="utf-8", errors="replace").splitlines()
            lines = cache[path]
            positions = [i for i, line in enumerate(lines)
                         if any(form in line for form in forms)]
            if not positions:
                continue
            verdict = "xuat hien nhung XA tu chi tieu"
            for position in positions:
                low = max(0, position - args.window)
                high = min(len(lines), position + args.window + 1)
                window = fold(" ".join(lines[low:high]))
                words = set(window.split())
                if len(wanted & words) / len(wanted) >= 0.5:
                    verdict = "xuat hien GAN tu chi tieu"
                    if len(samples) < args.show:
                        samples.append(
                            f"  id={question['id']:<5d} {forms[0]}\n"
                            f"     hoi : {text[:92]}\n"
                            f"     dong: {lines[position].strip()[:96]}")
                    break
            if verdict.startswith("xuat hien GAN"):
                break
        counters[verdict] += 1
        if number % 200 == 0:
            print(f"  {number}/{len(questions)}  {time.time() - started:.0f}s",
                  flush=True)

    measured = (counters["xuat hien GAN tu chi tieu"]
                + counters["xuat hien nhung XA tu chi tieu"]
                + counters["khong xuat hien trong van ban"])
    print(f"\n{sum(counters.values())} cau, {measured} cau do duoc\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    if measured:
        near = counters["xuat hien GAN tu chi tieu"]
        print(f"\nGAN tu chi tieu: {near}/{measured} = {100 * near / measured:.0f}%")
        print("oracle chi dung 39%, nen day la CHAN DUOI cua thong tin trong van ban tho.")
    for line in samples:
        print(line)


if __name__ == "__main__":
    main()
