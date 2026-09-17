"""Answer a rate question by finding two addresses instead of one.

171 questions ask for a %, a multiple or a turnover, and the pipeline skips all of
them because it converts a single cell into a money amount. They are the largest
tractable block left: nothing new has to be extracted, because a rate is two
addresses of the kind that already measure 52% correct, divided.

The split into numerator and denominator is the part a regular expression handles
badly, and the failure is worth stating plainly: Vietnamese has more ways to phrase a
share than an alternation has branches. So this handles the shapes that actually
appear and declines the rest, rather than growing the pattern list:

  "tỷ trọng A trên|trong B"     an explicit pair
  "A chiếm bao nhiêu % của B"   an explicit pair, other word order
  "ROA", "ROE", "biên lợi nhuận gộp", ...  named ratios with a fixed formula

The denominator of a share is usually a statement total — tổng tài sản, tổng nguồn
vốn, doanh thu thuần — which is the cleanest kind of address there is. A rate lands
in range or it does not, and out-of-range is a refusal rather than a guess: at 52%
per read a pair is right about a quarter of the time, so shipping the implausible
ones would cost more than they return.

Usage:  python scripts/fresh/plan_ratio.py --show 10
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
from plan_answers import OPENING_RE, PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from refine_codes import RELIABLE_LEN, tokens_exact  # noqa: E402
from score import Weighting  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

PAIR_RE = re.compile(
    r"\b(?:tỷ trọng|tỉ trọng|tỷ lệ|tỉ lệ|tỷ số|tỉ số|hệ số|tỷ suất)\s+"
    r"(?P<num>.{3,80}?)\s+(?:trên|trong|so với|chia cho|/)\s+(?P<den>.{3,80}?)"
    r"(?=\s+(?:của|tại|năm|vào|đến|là|đạt|cao|thấp)\b|[,?]|$)", re.I)
CHIEM_RE = re.compile(
    r"(?P<num>.{3,80}?)\s+(?:chiếm|bằng)\s+bao nhiêu\s*(?:%|phần trăm)\s*"
    r"(?:của|trên|trong)?\s*(?P<den>.{3,80}?)(?=\s+(?:của|tại|năm|vào|đến)\b|[,?]|$)",
    re.I)
NAMED = (
    (re.compile(r"\bROA\b|lợi nhuận sau thuế trên tổng tài sản", re.I),
     "lợi nhuận sau thuế", "tổng cộng tài sản"),
    (re.compile(r"\bROE\b|lợi nhuận sau thuế trên vốn chủ", re.I),
     "lợi nhuận sau thuế", "vốn chủ sở hữu"),
    (re.compile(r"biên lợi nhuận gộp", re.I), "lợi nhuận gộp", "doanh thu thuần"),
    (re.compile(r"biên lợi nhuận (?:ròng|thuần)|\bROS\b", re.I),
     "lợi nhuận sau thuế", "doanh thu thuần"),
    (re.compile(r"hệ số nợ|tỷ lệ nợ trên tổng tài sản", re.I),
     "nợ phải trả", "tổng cộng tài sản"),
)
PERCENT_RE = re.compile(r"%|phần trăm", re.I)
TIMES_RE = re.compile(r"bao nhiêu lần|vòng quay", re.I)

# Where the rate is a FILTER rather than the answer. "Trong giai đoạn 2021–2024 của
# DIG, năm mà có tỷ lệ nợ phải trả trên vốn chủ sở hữu cao nhất, lợi nhuận … là bao
# nhiêu" asks for the profit, not the ratio: the ratio only picks the year. Computing
# and reporting the ratio there is a confident wrong answer, and it is what this
# planner did on almost every sample before the guard existed.
FILTER_CLAUSE_RE = re.compile(
    r"giai đoạn|trong các năm|trong số|xét các năm|xét những|năm mà|tại năm|"
    r"vào năm|năm nào|năm có|cao nhất|thấp nhất|lớn nhất|nhỏ nhất|trung vị|"
    r"liền trước|liền sau", re.I)


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


def operands(text: str) -> tuple[str, str] | None:
    for pattern, numerator, denominator in NAMED:
        if pattern.search(text):
            return numerator, denominator
    for pattern in (PAIR_RE, CHIEM_RE):
        match = pattern.search(text)
        if match:
            numerator = match.group("num").strip(" ,.;:")
            denominator = match.group("den").strip(" ,.;:")
            if len(numerator) > 2 and len(denominator) > 2:
                return numerator, denominator
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/ratio_plan.jsonl")
    parser.add_argument("--min-score", type=float, default=0.34)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    cells: dict[tuple, dict] = defaultdict(dict)
    votes: dict[tuple, Counter] = defaultdict(Counter)
    raw_labels: dict[tuple, Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        family = "bank" if record["bank"] else "firm"
        for period in ("current", "prior"):
            key = (record["ticker"], record["year"], scope, record["kind"], period)
            for code, cell in record[period].items():
                cells[key].setdefault(code, {
                    "value": cell[0], "row": cell[2], "col": cell[3],
                    "csv": record["csv"], "doc": record["doc"],
                    "table_id": record["table_id"],
                    "table_ref": record["table_ref"], "scale": record["scale"],
                })
        if not verified(record):
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) != want:
                continue
            key3 = (family, record["kind"], code)
            label = tokens_exact(cell[1])
            if label:
                votes[key3][label] += 1
                raw_labels[key3][" ".join(str(cell[1]).split())] += 1
    dictionary = {k: (c.most_common(1)[0][0], raw_labels[k].most_common(1)[0][0])
                  for k, c in votes.items()}
    weights = Weighting(tokens for tokens, _label in dictionary.values())
    print(f"tu dien: {len(dictionary)}")

    resolver = TickerResolver()
    banks = {r["ticker"] for r in
             (json.loads(line) for line in
              (ROOT / args.index).read_text(encoding="utf-8").splitlines()
              if line.strip()) if r["bank"]}

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    def address(probe_text: str, ticker: str, year: str, scope: str, family: str,
                period: str):
        probe = re.sub(r"\([^)]*\)", " ", probe_text)
        probe = YEAR_RE.sub(" ", probe)
        for chunk in STRIP_CHUNKS:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
        probe_tokens = tokens_exact(probe)
        if not probe_tokens:
            return None
        best = None
        for (dict_family, kind, code), (label_tokens, label) in dictionary.items():
            if dict_family != family:
                continue
            entry = cells.get((ticker, year, scope, kind, period), {}).get(code)
            if entry is None:
                continue
            shared = probe_tokens & label_tokens
            if not shared or contradicts(probe, label):
                continue
            if len(shared) / len(probe_tokens) < 0.5:
                continue
            score = weights.overlap(probe_tokens, label_tokens)
            if best is None or score > best[0]:
                best = (score, kind, code, label, entry)
        if best is None or best[0] < args.min_score:
            return None
        return best

    counters: Counter[str] = Counter()
    plan, samples = [], []
    for question in questions:
        text = question["question"]
        if not (PERCENT_RE.search(text) or TIMES_RE.search(text)):
            continue
        counters["cau ty le"] += 1
        if FILTER_CLAUSE_RE.search(text):
            counters["ty le chi la BO LOC, khong phai dap an"] += 1
            continue
        pair = operands(text)
        if pair is None:
            counters["khong tach duoc tu/mau"] += 1
            continue
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        if len(found) != 1 or not years:
            counters["nhieu ma hoac khong nam"] += 1
            continue
        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        period = "prior" if OPENING_RE.search(text) else "current"
        family = "bank" if ticker in banks else "firm"

        top = address(pair[0], ticker, year, scope, family, period)
        bottom = address(pair[1], ticker, year, scope, family, period)
        if top is None or bottom is None:
            counters["khong dinh vi du hai toan hang"] += 1
            continue
        if (top[1], top[2]) == (bottom[1], bottom[2]):
            counters["tu va mau trung mot o"] += 1
            continue
        numerator = abs(top[4]["value"])
        denominator = abs(bottom[4]["value"])
        if not denominator:
            counters["mau bang 0"] += 1
            continue
        as_percent = bool(PERCENT_RE.search(text))
        value = numerator / denominator * (100.0 if as_percent else 1.0)
        # A rate outside this band means one operand is the wrong line, and at 52%
        # per read the pair is already the weaker half of the build.
        if not value or abs(value) > (100.0 if as_percent else 100.0):
            counters["ket qua ngoai khoang hop ly"] += 1
            continue

        counters["CO CAP DIA CHI"] += 1
        plan.append({
            "id": question["id"], "op": "pct" if as_percent else "times",
            "value": round(value, 2),
            "num": {"kind": top[1], "code": top[2], "label": top[3],
                    **{k: top[4][k] for k in ("row", "col", "csv", "doc",
                                              "table_id", "table_ref", "scale")}},
            "den": {"kind": bottom[1], "code": bottom[2], "label": bottom[3],
                    **{k: bottom[4][k] for k in ("row", "col", "csv", "doc",
                                                 "table_id", "table_ref", "scale")}},
        })
        if len(samples) < args.show:
            samples.append(
                f"  id={question['id']:<5d} {value:.2f}{'%' if as_percent else 'x'}"
                f"  {top[1]}/{top[2]} / {bottom[1]}/{bottom[2]}\n"
                f"     hoi : {text[:100]}\n"
                f"     tu  : {top[3][:70]}\n"
                f"     mau : {bottom[3][:70]}")

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    print()
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print("\nvi du:")
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
