"""How many of the 1012 questions can be reached by navigation rather than by matching?

The backbone measures well: a statement line's note pointer resolves 92% of the time and
the note's total ties back to the line 94% of the time, against 11% when deliberately
pointed at the wrong note. That says the mechanism is real. It does not say how much of the
exam it touches, and that is what decides whether to build on it.

A question is counted reachable when its own wording matches either

  * a statement row directly — answerable from the reconciled statement, or
  * a row inside a note that a statement row points at — answerable by navigation.

The matcher here is the crude token one, so this is a LOWER bound in the one direction that
matters: the design gives the label-variant job to a model precisely because a single
phrasing missed the row on 18% of questions in an earlier measurement. Whatever this
prints, the real figure is higher.

Usage:
  python scripts/fresh/measure_reach.py --limit 120
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from find_statements import locate_columns  # noqa: E402
from measure_pointers import POINTER_RE, contexts_of, heading_numbers  # noqa: E402
from plan_answers import PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

STOP = frozenset({"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "la",
                  "trong", "cho", "tai", "theo", "phan", "tram", "ctcp", "tong",
                  "gia", "tri", "muc", "khoan", "so", "du", "dong", "trieu", "ty",
                  "bang", "duoc", "cac", "va", "voi", "tu", "den", "co", "mot"})
OVERLAP = 0.6


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def metric_tokens(text: str, tickers: set[str], names: list[str]) -> set[str]:
    probe = text
    for ticker in tickers:
        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", probe)
    probe = re.sub(r"\([^)]*\)", " ", probe)
    probe = YEAR_RE.sub(" ", probe)
    for chunk in tuple(names) + STRIP_CHUNKS:
        if chunk:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
    return {t for t in fold(probe).split() if len(t) > 2 and t not in STOP}


def matches(wanted: set[str], label: str) -> bool:
    words = set(fold(label).split())
    if not wanted or not words:
        return False
    shared = wanted & words
    return bool(shared) and len(shared) / min(len(wanted), len(words)) >= OVERLAP


def matches_variant(names: list[str], label: str) -> bool:
    """A variant matches a row label by containment either way, or by token overlap.

    Containment is the strong test and the reason to ask a model for variants at all: a
    variant is a whole label as a report would print it, so "Chi phí quản lý DN" sits
    inside "9. Chi phí quản lý DN" without any token arithmetic.
    """

    flat_label = fold(label)
    if not flat_label:
        return False
    label_words = set(flat_label.split())
    for name in names:
        flat = fold(name)
        if not flat:
            continue
        if flat in flat_label or flat_label in flat:
            return True
        words = set(flat.split())
        shared = words & label_words
        if shared and len(shared) / min(len(words), len(label_words)) >= 0.7:
            return True
    return False


def load_industry() -> dict[str, str]:
    """Sector per ticker, so an unreachable question can be attributed.

    Banks and securities firms file on different statement forms, where the Mã số scheme
    does not carry the same lines. Whether the residue is those two sectors or ordinary
    companies decides whether the gap is structural or a bug.
    """

    out: dict[str, str] = {}
    path = ROOT / "data" / "file_filter.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv_mod.DictReader(handle):
            code = (row.get("Mã CK") or "").strip()
            sector = fold(row.get("Ngành cấp 1") or "")
            if not code:
                continue
            if "ngan hang" in sector:
                out[code] = "ngan hang"
            elif "chung khoan" in sector or "bao hiem" in sector:
                out[code] = "chung khoan/bao hiem"
            else:
                out[code] = "thuong"
    return out


INDUSTRY: dict[str, str] = {}


def industry(ticker: str) -> str:
    return INDUSTRY.get(ticker, "khong ro nganh")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    # Label variants from the one model call. With them the matcher stops depending on
    # the question happening to use the wording the report printed.
    parser.add_argument("--specs", default="")
    args = parser.parse_args()

    variants: dict[int, list[str]] = {}
    if args.specs:
        from score_model import parse as parse_json
        for line in (ROOT / args.specs).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            spec = parse_json(record.get("reply", ""))
            if not spec:
                continue
            names = []
            for item in spec.get("chi_tieu") or []:
                if isinstance(item, dict):
                    if item.get("ten"):
                        names.append(str(item["ten"]))
                    for text in item.get("bien_the") or []:
                        names.append(str(text))
            if names:
                variants[record["id"]] = names

    INDUSTRY.update(load_industry())
    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    corpus = ROOT / "data" / "official_corpus"
    counters: Counter[str] = Counter()
    cache: dict[Path, tuple] = {}

    def load(doc_dir: Path) -> tuple:
        """(statement rows with pointers, note tables by heading number)."""

        if doc_dir in cache:
            return cache[doc_dir]
        tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
        if not tables_dir.is_dir():
            cache[doc_dir] = ([], {})
            return cache[doc_dir]
        anchors = contexts_of(doc_dir / f"{doc_dir.name}_extracted.txt")
        grids = {}
        for path in sorted(tables_dir.glob("table_*.csv"),
                           key=lambda p: int(p.stem.split("_")[-1])):
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    grids[int(path.stem.split("_")[-1])] = list(
                        csv_mod.reader(handle))
            except OSError:
                continue
        by_number: dict[str, list[int]] = {}
        for table_id in grids:
            for number in heading_numbers(anchors.get(table_id, "")):
                by_number.setdefault(number, []).append(table_id)

        lines = []
        for table_id, grid in grids.items():
            if not grid:
                continue
            # Located by content when the header does not name them.
            ma_i, tm_i = locate_columns(grid)
            if ma_i is None:
                continue
            for row in grid[1:]:
                if len(row) <= ma_i:
                    continue
                code = str(row[ma_i]).strip()
                if not re.fullmatch(r"\d{1,3}", code):
                    continue
                label = str(row[0]).strip()
                pointer = ""
                if tm_i is not None and len(row) > tm_i:
                    candidate = str(row[tm_i]).strip()
                    if POINTER_RE.match(candidate):
                        pointer = re.sub(r"[\s.]", "", candidate).upper()
                lines.append((code, label, pointer))
        cache[doc_dir] = (lines, {k: [grids[t] for t in v if grids.get(t)]
                                 for k, v in by_number.items()})
        return cache[doc_dir]

    for question in questions:
        text = question["question"]
        tickers = resolver.resolve(text)
        years = sorted({y for y in YEAR_RE.findall(text)})
        if not tickers or not years:
            counters["khong xac dinh duoc ma/nam"] += 1
            continue
        names = [resolver.tickers.get(t, "") for t in tickers]
        wanted = metric_tokens(text, tickers, names)
        if not wanted:
            counters["khong con tu chi tieu"] += 1
            continue

        want_separate = bool(PARENT_RE.search(text))
        base = corpus / sorted(tickers)[0] / years[-1]
        if not base.is_dir():
            counters["khong co tai lieu"] += 1
            continue
        docs = sorted(p for p in base.iterdir() if p.is_dir())
        ordered = ([d for d in docs if ("separate" in d.name) == want_separate]
                   + [d for d in docs if ("separate" in d.name) != want_separate])
        if not ordered:
            counters["khong co tai lieu"] += 1
            continue

        # Every document for this company-year, not just the first. Loading only the
        # preferred-scope document counted a question as having no Mã số table whenever
        # the statements sat in the other file, which is a fault of the measurement
        # rather than of the corpus.
        lines, notes = [], {}
        for doc_dir in ordered:
            more_lines, more_notes = load(doc_dir)
            lines += more_lines
            for key, grids in more_notes.items():
                notes.setdefault(key, []).extend(grids)
        if not lines:
            counters[f"khong co bang Ma so ({industry(sorted(tickers)[0])})"] += 1
            continue

        names = variants.get(question["id"])
        hit = ((lambda label: matches_variant(names, label)) if names
               else (lambda label: matches(wanted, label)))
        direct = [(c, l, p) for c, l, p in lines if hit(l)]
        if direct:
            if any(p for _c, _l, p in direct):
                counters["CHAM: khop dong bao cao, co con tro thuyet minh"] += 1
            else:
                counters["CHAM: khop dong bao cao, khong con tro"] += 1
            continue

        # Not a statement line itself — try the rows of every pointed-to note.
        found = False
        for _code, _label, pointer in lines:
            if not pointer:
                continue
            for grid in notes.get(pointer, []):
                for row in grid[1:]:
                    if row and hit(str(row[0]).strip()):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        counters["CHAM: khop dong trong thuyet minh duoc tro toi" if found
                 else f"khong cham duoc ({industry(sorted(tickers)[0])})"] += 1

    total = sum(counters.values())
    reached = sum(v for k, v in counters.items() if k.startswith("CHAM"))
    for name, count in counters.most_common():
        print(f"  {count:5d} ({100 * count / total:3.0f}%)  {name}")
    print(f"\ncham duoc: {reached}/{total} = {100 * reached / total:.0f}% "
          f"(chan duoi — matcher tho)")


if __name__ == "__main__":
    main()
