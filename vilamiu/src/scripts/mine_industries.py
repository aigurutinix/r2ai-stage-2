"""Recover the industry grouping the generator needs, from the question set itself.

`CompanyInfo` carries `industry_l1/l2/l3` and `exchange`; our `code_stock.csv`
has only ticker and name, so those fields load as empty strings. That is not
cosmetic. Both derived tiers refuse to build a multi-company question without it:

    medium.py:179        if seed_company is None or not seed_company.industry_l3: return []
    intermediate.py:127  if seed_company is None or not seed_company.industry_l3: return []

Generating with empty industries therefore yields the **easy** tier only —
single-cell lookups, which is the half our label matcher already answers at
42.8%. Training on that teaches the model nothing about the half we actually lose
on, which is exactly the failure `probe_synth.py` was written to detect.

The grouping is recoverable without any external source, because the questions
state it outright:

    "Trong các doanh nghiệp ngành quản lý và phát triển bất động sản
     (bao gồm các công ty DIG, HPX, KBC, NVL, SCR, VIC, VPI, VRE) ..."

Each such question names an industry and enumerates its members, so mining them
reconstructs the organisers' own partition by construction — the same one the
test set was built against.

Usage:  PYTHONPATH=src python scripts/mine_industries.py [--write]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

# "ngành <name>" up to the parenthesis, the verb, or the clause end. The name is
# what the organisers call the industry; the tickers come from our own parser,
# which already resolves both codes and company names.
INDUSTRY_RE = re.compile(
    r"ngành\s+([^,.()]{3,60}?)\s*(?=\(|,|\.|\bgồm\b|\bbao gồm\b|\btrong\b|\bnăm\b|$)",
    re.I,
)
# Phrases that are not an industry even though they follow "ngành".
NOT_INDUSTRY = re.compile(r"^(nghề|này|đó|trên|của|có|và|hàng)\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="write data/file_filter.csv with the mined columns")
    args = parser.parse_args()

    questions = parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")

    votes: dict[str, Counter[str]] = defaultdict(Counter)
    named = 0
    for question in questions:
        if len(question.tickers) < 2:
            continue
        match = INDUSTRY_RE.search(question.question)
        if match is None:
            continue
        industry = " ".join(match.group(1).split()).lower()
        if NOT_INDUSTRY.match(industry) or len(industry) < 4:
            continue
        named += 1
        for ticker in question.tickers:
            votes[ticker][industry] += 1

    print(f"{len(questions)} questions, {named} name an industry with >=2 tickers")
    print(f"tickers with at least one industry vote: {len(votes)}")

    # Naming the industry outright is rare — 19 questions. Far more of them name
    # a peer group without the word: "Trong nhóm HPG, HSG, MSR và NKG",
    # "trong 3 doanh nghiệp Hoà Phát, Hoa Sen, Nam Kim". The organisers put those
    # tickers in one question because they are comparable, so co-occurrence is
    # the same partition observed through a different sentence. Build the graph
    # and let the components be the groups; the mined names then label whichever
    # components contain a named ticker.
    edges: Counter[tuple[str, str]] = Counter()
    cohort_questions = 0
    for question in questions:
        tickers = sorted(set(question.tickers))
        if not 2 <= len(tickers) <= 12:
            continue
        cohort_questions += 1
        for i, a in enumerate(tickers):
            for b in tickers[i + 1:]:
                edges[(a, b)] += 1
    print(f"questions naming a group of 2-12 tickers: {cohort_questions}")
    print(f"distinct co-occurring pairs: {len(edges)}")

    # A pair seen once may be an incidental comparison; requiring two sightings
    # keeps the components from merging into one blob through a single odd
    # question. The threshold is reported so its effect is visible.
    for floor in (1, 2, 3):
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (a, b), count in edges.items():
            if count >= floor:
                union(a, b)
        components: dict[str, list[str]] = defaultdict(list)
        for ticker in {t for pair in edges for t in pair}:
            components[find(ticker)].append(ticker)
        sizes = sorted((len(v) for v in components.values()), reverse=True)
        big = sum(1 for s in sizes if s >= 3)
        print(f"  min co-occurrence {floor}: {len(components)} components, "
              f"largest {sizes[:5]}, {big} with >=3 members")

    # Use the floor that keeps groups meaningful without collapsing them.
    parent = {}
    for (a, b), count in edges.items():
        if count >= 2:
            union(a, b)
    components = defaultdict(list)
    for ticker in {t for pair in edges for t in pair}:
        components[find(ticker)].append(ticker)

    assigned: dict[str, str] = {}
    for root, members in components.items():
        if len(members) < 2:
            continue
        named_votes: Counter[str] = Counter()
        for ticker in members:
            for industry, weight in votes.get(ticker, {}).items():
                named_votes[industry] += weight
        label = (named_votes.most_common(1)[0][0] if named_votes
                 else f"nhóm {min(members).lower()}")
        for ticker in members:
            assigned[ticker] = label

    groups: dict[str, list[str]] = defaultdict(list)
    for ticker, industry in assigned.items():
        groups[industry].append(ticker)

    print(f"\ndistinct industries mined: {len(groups)}")
    for industry, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(members):3d}  {industry}")
        print(f"       {', '.join(sorted(members))}")

    # The generator needs >=3 tickers in one industry to build a cohort.
    usable = {i: m for i, m in groups.items() if len(m) >= 3}
    covered = sum(len(m) for m in usable.values())
    print(f"\nindustries with >=3 members (a cohort can be built): {len(usable)}")
    print(f"tickers covered by those: {covered}")

    roster = {}
    with (ROOT / "data" / "code_stock.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("Mã CK") or "").strip()
            if ticker:
                roster[ticker] = (row.get("Tên công ty") or "").strip()
    print(f"\nroster: {len(roster)} companies, "
          f"{len(set(roster) & set(assigned))} of them get an industry")

    if not args.write:
        print("\n(pass --write to emit data/file_filter.csv)")
        return

    # `industry_l2 == "Tổ chức tín dụng"` selects the bank-specific formula set
    # (`intermediate_formulas/registry.py:136`) and the credit-institution branch
    # in `legacy_cli.py:153`. The mined *label* cannot decide that: the group of
    # 17 banks was named "công nghiệp chế biến" by a question about lending to
    # manufacturers. Membership decides it instead — a credit institution says so
    # in its registered name, and that is exact.
    banks = {t for t, name in roster.items() if "ngân hàng" in name.lower()}
    print(f"\ncredit institutions identified by registered name: {len(banks)}")

    out = ROOT / "data" / "file_filter.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Mã CK", "Tên công ty", "Ngành cấp 1",
                         "Ngành cấp 2", "Ngành cấp 3", "Sàn"])
        for ticker, name in sorted(roster.items()):
            group = assigned.get(ticker, "")
            if ticker in banks:
                group = group or "ngân hàng"
            # l3 is what the cohort builders group on, so it carries the mined
            # partition; only its identity matters there, not its wording.
            l2 = "Tổ chức tín dụng" if ticker in banks else group
            writer.writerow([ticker, name, group, l2, group, ""])
    covered = sum(1 for t in roster if assigned.get(t) or t in banks)
    print(f"wrote {out}  ({covered}/{len(roster)} companies carry a group)")


if __name__ == "__main__":
    main()
