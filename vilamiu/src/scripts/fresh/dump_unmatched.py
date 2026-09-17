"""What do the questions the Mã số dictionary cannot match actually ask?

The mapper leaves 255 questions in a bucket labelled "no code matched": each names a
single company, a year and a scope whose primary statements parsed, so the address
book reaches the right report — and nothing in the statutory chart answers them.

I asserted those were note-table metrics without checking, and the whole argument
about whether 0.60 is reachable rests on what is in that bucket. If they are note
items, the anchor-heading path is the lever. If they are ratios, growth figures or
superlatives, the lever is composition over reads instead, and the note path matters
less than I claimed.

Usage:  python scripts/fresh/dump_unmatched.py --show 16
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

from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from map_metric import contradicts  # noqa: E402
from refine_codes import RELIABLE_LEN, tokens_exact  # noqa: E402
from split_charts import chart_of  # noqa: E402

YEAR_RE = re.compile(r"(?:19|20)\d{2}")
PARENT_RE = re.compile(r"công ty mẹ", re.I)
STRIP_CHUNKS = ("công ty mẹ", "bao nhiêu", "đồng", "triệu", "tỷ", "nghìn", "trăm",
                "phần trăm", "cuối năm", "đầu năm", "cuối kỳ", "đầu kỳ", "đến ngày",
                "trong năm", "vào ngày", "tháng", "số dư", "tổng số", "giá trị",
                "của", "tại", "là")

FAMILIES = (
    ("ty le / phan tram / so lan", r"ph[ầa]n tr[ăa]m|%|t[ỷy] tr[ọo]ng|t[ỷy] l[ệe]|"
                                   r"bao nhi[êe]u l[ầa]n|t[ỷy] su[ấa]t|h[ệe] s[ốo]"),
    ("tang truong / chenh lech", r"t[ăa]ng tr[ưu][ởo]ng|ch[êe]nh l[ệe]ch|"
                                 r"bi[ếe]n [đd][ộo]ng|thay [đd][ổo]i|t[ốo]c [đd][ộo] t[ăa]ng"),
    ("nam nao / cao nhat", r"n[ăa]m n[àa]o|cao nh[ấa]t|th[ấa]p nh[ấa]t|l[ớo]n nh[ấa]t|"
                           r"nh[ỏo] nh[ấa]t|trung v[ịi]"),
    ("ben lien quan / cong ty con", r"b[êe]n li[êe]n quan|c[ôo]ng ty con|"
                                    r"c[ôo]ng ty li[êe]n k[ếe]t|c[ổo] [đd][ôo]ng|"
                                    r"s[ởo] h[ữu]u|bi[ểe]u quy[ếe]t"),
    ("chi tiet / co cau", r"chi ti[ếe]t|c[ơo] c[ấa]u|ph[âa]n lo[ạa]i|bao g[ồo]m|"
                          r"trong t[ổo]ng|thu[ộo]c"),
    ("ngoai te / co phieu / so luong", r"ngo[ạa]i t[ệe]|\bUSD\b|\bEUR\b|"
                                       r"c[ổo] phi[ếe]u|nh[âa]n vi[êe]n|s[ốo] l[ưu][ợo]ng"),
)


def verified(record: dict) -> bool:
    for kind, target, plus, minus in IDENTITIES:
        if record["kind"] != kind:
            continue
        for period in ("current", "prior"):
            cells = record[period]
            if not all(c in cells for c in (target,) + plus + minus):
                continue
            expected = cells[target][0]
            total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
            if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                return True
    return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements.jsonl")
    parser.add_argument("--show", type=int, default=16)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               (ROOT / args.index).read_text(encoding="utf-8").splitlines()
               if line.strip()]

    votes: dict[tuple, Counter] = defaultdict(Counter)
    available: dict[tuple, set[str]] = defaultdict(set)
    charts: dict[tuple, Counter] = defaultdict(Counter)
    for record in records:
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        key = (record["ticker"], record["year"], scope)
        available[key].add(record["kind"])
        charts[key][chart_of(record)] += 1
        if not verified(record):
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) != want:
                continue
            label_tokens = tokens_exact(cell[1])
            if label_tokens:
                votes[(chart_of(record), record["kind"], code)][label_tokens] += 1
    dictionary = {k: v.most_common(1)[0][0] for k, v in votes.items()}

    tickers = {}
    for line in (ROOT / "data" / "code_stock.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            tickers[code.strip()] = name.strip().strip('"')
    by_length = sorted(tickers.items(), key=lambda item: -len(item[1]))

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    unmatched = []
    for question in questions:
        text = question["question"]
        found = {c for c in tickers if re.search(rf"\b{re.escape(c)}\b", text)}
        for code, name in by_length:
            if name and name.casefold() in text.casefold():
                found.add(code)
        years = YEAR_RE.findall(text)
        if len(found) != 1 or not years:
            continue
        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        kinds = available.get((ticker, max(years), scope))
        if not kinds:
            continue
        chart = charts[(ticker, max(years), scope)].most_common(1)[0][0]

        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
        probe = re.sub(r"\([^)]*\)", " ", probe)
        probe = YEAR_RE.sub(" ", probe)
        for chunk in (tickers[ticker],) + STRIP_CHUNKS:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
        probe_tokens = tokens_exact(probe)
        if not probe_tokens:
            continue

        best = 0.0
        for (dict_chart, kind, _code), label_tokens in dictionary.items():
            if dict_chart != chart or kind not in kinds:
                continue
            shared = probe_tokens & label_tokens
            if not shared or contradicts(probe, " ".join(label_tokens)):
                continue
            if len(shared) / len(probe_tokens) < 0.5:
                continue
            best = max(best, len(shared) / len(probe_tokens | label_tokens))
        if best == 0.0:
            unmatched.append(question)

    print(f"so cau khong ma nao khop: {len(unmatched)}")
    counts: Counter[str] = Counter()
    for question in unmatched:
        hit = False
        for name, pattern in FAMILIES:
            if re.search(pattern, question["question"], re.I):
                counts[name] += 1
                hit = True
        if not hit:
            counts["khong roi vao nhom nao (co the la muc thuyet minh don thuan)"] += 1
    for name, count in counts.most_common():
        print(f"  {name}: {count}")

    print("\nvi du:")
    for question in unmatched[:args.show]:
        print(f"  id={question['id']:<5d} {question['question'][:116]}")


if __name__ == "__main__":
    main()
