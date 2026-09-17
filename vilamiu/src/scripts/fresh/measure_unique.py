"""How often does the pipeline land on exactly one candidate row?

This is the number that decides whether the navigation design is worth building. Reach —
whether a matching row exists at all — is 91% and is not the constraint. Landing on one
candidate is, because the organisers' own figures separate a model reading the right table
(87%) from a model choosing among ten (64%), and only a unique candidate puts us in the
first setting.

Three variants are reported side by side so the effect of each idea is visible:

  moc         plain matching, any row whose label overlaps the question's wording
  xep hang    ranked by how specifically the label matches (`label_match.score`)
  + con tro   ranked, and a statement row that carries a Thuyết minh pointer is preferred
              over one that does not, since the pointer is the document's own statement
              that this line is the detailed one

Usage:
  python scripts/fresh/measure_unique.py --limit 120 --specs artifacts/fresh/spec_replies.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from find_statements import locate_columns  # noqa: E402
from label_match import score  # noqa: E402
from measure_pointers import POINTER_RE, contexts_of, heading_numbers  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402
from score_model import parse as parse_json  # noqa: E402

CODE_RE = re.compile(r"^\d{1,3}$")


def variants_from(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
                names += [str(x) for x in (item.get("bien_the") or [])]
        if names:
            out[record["id"]] = names
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--specs", default="artifacts/fresh/spec_replies.jsonl")
    args = parser.parse_args()

    variants = variants_from(ROOT / args.specs)
    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    counters: Counter[str] = Counter()
    sizes: Counter[str] = Counter()

    for question in questions:
        names = variants.get(question["id"])
        if not names:
            counters["khong co spec"] += 1
            continue
        text = question["question"]
        tickers = sorted(resolver.resolve(text))
        years = sorted({y for y in YEAR_RE.findall(text)})
        if not tickers or not years:
            counters["khong xac dinh duoc ma/nam"] += 1
            continue
        base = ROOT / "data" / "official_corpus" / tickers[0] / years[-1]
        if not base.is_dir():
            counters["khong co tai lieu"] += 1
            continue
        want_separate = bool(PARENT_RE.search(text))
        docs = sorted(p for p in base.iterdir() if p.is_dir())
        docs = ([d for d in docs if ("separate" in d.name) == want_separate]
                + [d for d in docs if ("separate" in d.name) != want_separate])

        # (score, has_pointer, identity) over statement rows AND the rows of the notes
        # those statement rows point at. One pool, one ranking: a question naming a
        # sub-item should be able to win inside a note without the statement line
        # having to match first.
        found: list[tuple[float, bool, str]] = []
        for doc_dir in docs:
            tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
            if not tables_dir.is_dir():
                continue
            grids: dict[int, list[list[str]]] = {}
            for csv_path in sorted(tables_dir.glob("table_*.csv")):
                try:
                    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                        grids[int(csv_path.stem.split("_")[-1])] = list(
                            csv_mod.reader(handle))
                except OSError:
                    continue
            anchors = contexts_of(doc_dir / f"{doc_dir.name}_extracted.txt")
            by_number: dict[str, list[int]] = {}
            for table_id in grids:
                for number in heading_numbers(anchors.get(table_id, "")):
                    by_number.setdefault(number, []).append(table_id)

            pointed_tables: dict[int, str] = {}
            for table_id, grid in grids.items():
                if not grid:
                    continue
                code_col, pointer_col = locate_columns(grid)
                if code_col is None:
                    continue
                for row in grid[1:]:
                    if len(row) <= code_col:
                        continue
                    code = str(row[code_col]).strip()
                    if not CODE_RE.match(code):
                        continue
                    pointer = ""
                    if pointer_col is not None and len(row) > pointer_col:
                        candidate = str(row[pointer_col]).strip()
                        if POINTER_RE.match(candidate):
                            pointer = re.sub(r"[\s.]", "", candidate).upper()
                    value = score(names, str(row[0]).strip())
                    if value:
                        found.append((value, bool(pointer), f"ma:{code}"))
                    if pointer:
                        for target in by_number.get(pointer, []):
                            pointed_tables.setdefault(target, code)

            # Rows inside the notes a statement line points at. The parent code is
            # carried along, so a note row is a fully qualified answer.
            for table_id, parent in pointed_tables.items():
                for index, row in enumerate(grids.get(table_id, [])[1:]):
                    if not row:
                        continue
                    value = score(names, str(row[0]).strip())
                    if value:
                        found.append((value, True, f"tm:{parent}:{table_id}:{index}"))
            if found:
                break

        if not found:
            counters["0 ung vien"] += 1
            continue

        plain = {c for _s, _p, c in found}
        top = max(s for s, _p, _c in found)
        ranked = {c for s, _p, c in found if s == top}
        with_pointer = {c for s, p, c in found if s == top and p}
        best = with_pointer or ranked

        for label, group in (("moc", plain), ("xep hang", ranked),
                             ("xep hang + con tro", best)):
            sizes[f"{label}: {'1' if len(group) == 1 else ('2-3' if len(group) <= 3 else '4+')}"] += 1
        counters["do duoc"] += 1

    measured = counters["do duoc"]
    print(f"{measured} cau do duoc / {len(questions)}\n")
    for label in ("moc", "xep hang", "xep hang + con tro"):
        one = sizes[f"{label}: 1"]
        few = sizes[f"{label}: 2-3"]
        many = sizes[f"{label}: 4+"]
        print(f"  {label:22s} duy nhat {one:4d} ({100 * one / max(1, measured):3.0f}%)"
              f" | 2-3 ung vien {few:4d} | 4+ {many:4d}")
    print()
    for name, count in counters.most_common():
        if name != "do duoc":
            print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    main()
