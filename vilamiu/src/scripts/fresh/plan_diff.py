"""Answer "chênh lệch giữa hai kỳ" by reading one code twice. No model needed.

30% of the answers in the shipped build have the wrong shape — a single cell for a
question that needs several — and the largest tractable slice of that is the two-period
difference: "Chênh lệch vốn chủ sở hữu giữa cuối năm 2025 và cuối năm 2024".

This group is the cheapest thing in the whole exam to get right, for a reason worth
stating: both cells carry the SAME `Mã số`. Line choice is the step running at ~23%, and
here it is paid once instead of twice — pick the code correctly and both reads follow.

Adjacent years are cheaper still. A statement prints its own prior period, so
"giữa cuối năm 2025 và cuối năm 2024" is `current` and `prior` of the SAME table: one
row, one unit scale, no second retrieval, and the two figures are printed side by side
by the company itself. Only a gap of two or more years needs a second report.

Detects the shape, resolves the code with the existing matcher, and emits a subtraction.

Usage:  python scripts/fresh/plan_diff.py --show 10
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
from plan_answers import PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from refine_codes import RELIABLE_LEN, tokens_exact  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

# "chênh lệch … giữa <năm> và <năm>", "từ <năm> đến <năm>", "<năm> so với <năm>".
DIFF_RE = re.compile(
    r"(?:chênh lệch|biến động|thay đổi|mức tăng|mức giảm|tăng|giảm)"
    r"[^?]{0,110}?\b((?:19|20)\d{2})\b[^?]{0,50}?"
    r"\b(?:và|đến|so với|sang)\b[^?]{0,50}?\b((?:19|20)\d{2})\b", re.I)
# A superlative or a median means the years are a search space, not two endpoints.
FILTER_RE = re.compile(
    r"cao nh[ấa]t|th[ấa]p nh[ấa]t|l[ớo]n nh[ấa]t|nh[ỏo] nh[ấa]t|trung v[ịi]|"
    r"n[ăa]m n[àa]o|n[ăa]m m[àa]|x[ée]t c[áa]c n[ăa]m", re.I)
PERCENT_RE = re.compile(r"%|phần trăm|bao nhiêu lần", re.I)


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
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/diff_plan.jsonl")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--show", type=int, default=8)
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
                    "value": cell[0], "label": cell[1], "row": cell[2],
                    "col": cell[3], "csv": record["csv"], "doc": record["doc"],
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

    banks = set()
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["bank"]:
                banks.add(record["ticker"])

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    plan, samples = [], []
    for question in questions:
        text = question["question"]
        match = DIFF_RE.search(text)
        if not match:
            continue
        counters["cau co dang chenh lech hai nam"] += 1
        if FILTER_RE.search(text):
            counters["  bo qua — cac nam la khong gian tim, khong phai hai moc"] += 1
            continue
        if PERCENT_RE.search(text):
            counters["  bo qua — hoi ty le tang truong, khong phai hieu"] += 1
            continue
        found = resolver.resolve(text)
        if len(found) != 1:
            counters["  bo qua — nhieu cong ty"] += 1
            continue
        ticker = next(iter(found))
        later, earlier = sorted((match.group(1), match.group(2)), reverse=True)
        if later == earlier:
            counters["  bo qua — hai moc cung nam"] += 1
            continue
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        family = "bank" if ticker in banks else "firm"

        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
        probe = re.sub(r"\([^)]*\)", " ", probe)
        probe = YEAR_RE.sub(" ", probe)
        for chunk in (resolver.tickers.get(ticker, ""),) + STRIP_CHUNKS + (
                "chênh lệch", "biến động", "thay đổi", "mức tăng", "mức giảm",
                "giữa", "so với", "từ", "đến", "sang", "tăng", "giảm"):
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
        probe_tokens = tokens_exact(probe)
        if not probe_tokens:
            counters["  bo qua — khong con tu de doi chieu"] += 1
            continue

        # Adjacent years live in one table: `current` and `prior` of the later report.
        adjacent = int(later) - int(earlier) == 1
        best = None
        for (dict_family, kind, code), (label_tokens, label) in dictionary.items():
            if dict_family != family:
                continue
            if adjacent:
                slot = cells.get((ticker, later, scope, kind, "current"), {}).get(code)
                other = cells.get((ticker, later, scope, kind, "prior"), {}).get(code)
            else:
                slot = cells.get((ticker, later, scope, kind, "current"), {}).get(code)
                other = cells.get((ticker, earlier, scope, kind, "current"), {}).get(code)
            if slot is None or other is None:
                continue
            shared = probe_tokens & label_tokens
            if not shared or contradicts(probe, label):
                continue
            if len(shared) / len(probe_tokens) < 0.5:
                continue
            score = len(shared) / len(probe_tokens | label_tokens)
            if best is None or score > best[0]:
                best = (score, kind, code, label, slot, other)
        if best is None:
            counters["  bo qua — khong ma nao co ca hai ky"] += 1
            continue
        score, kind, code, label, slot, other = best
        if score < args.min_score:
            counters["  bo qua — khop yeu"] += 1
            continue

        counters["CO CAP DIA CHI CUNG MA"] += 1
        counters["  cung mot bang (hai nam lien nhau)" if adjacent
                 else "  hai bao cao khac nhau"] += 1
        plan.append({
            "id": question["id"], "op": "hieu", "kind": kind, "code": code,
            "label": label, "adjacent": adjacent,
            "later": {"year": later, **{k: slot[k] for k in
                                        ("row", "col", "csv", "doc", "table_id",
                                         "table_ref", "scale")}},
            "earlier": {"year": earlier, **{k: other[k] for k in
                                            ("row", "col", "csv", "doc", "table_id",
                                             "table_ref", "scale")}},
        })
        if len(samples) < args.show:
            samples.append(
                f"  id={question['id']:<5d} {kind}/{code} {earlier}->{later} "
                f"{'cung bang' if adjacent else 'hai bao cao'} diem={score:.2f}\n"
                f"     hoi : {text[:96]}\n"
                f"     nhan: {label[:70]}")

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print()
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
