"""Turn each question into a cell address, or refuse it.

Joins the three things that are now established:

  the address book   (statement kind, Mã số, current|prior) over the organisers' own
                     CSVs, verified at 97–99.6% on the printed accounting identities
                     and 90.9% by comparing year Y's `current` with year Y+1's
                     `prior` across two independently unit-declaring documents
  the scope rule     from the generator's own prompt: consolidated is the default and
                     is never mentioned, so `công ty mẹ` means the separate statements
  the period rule    also from that prompt, which forbids "tại ngày" and writes a
                     balance as "cuối năm", "đầu năm" or "đến ngày" — so "đầu năm"
                     is the prior column and everything else is current

The metric-to-code step scores a question against the consensus label of each code,
using Jaccard over tokens WITH diacritics. Coverage of the label alone rewards the
shortest generic candidate: it answered "vay ngắn hạn" with 310 "Nợ ngắn hạn" and tied
"lưu chuyển tiền thuần từ hoạt động kinh doanh" against 50 "Lưu chuyển tiền thuần
trong năm". Folding diacritics is worse still, merging bán with bản so that "chi phí
bán hàng" matched "Chi phí xây dựng cơ bản dở dang".

Three refusals, each one measured into existence:

  a contrastive clash    "sau thuế" must not be answered with "trước thuế", nor
                         "phải nộp" with "đã nộp" or "hoãn lại"
  a tie                  no margin over the runner-up means the evidence names no
                         single code, and either choice is a coin flip
  thin coverage          the label must account for half of what the question asks,
                         or a candidate from the wrong statement wins by default

Credit institutions are kept apart: 21 of the 100 tickers are flagged
`Tổ chức tín dụng` in `data/file_filter.csv`, they use a different chart, and they
fail the enterprise identities 100% of the time.

Usage:  python scripts/fresh/plan_answers.py --show 12
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
from map_metric import CONTRASTIVE, contradicts  # noqa: E402
from refine_codes import RELIABLE_LEN, tokens_exact  # noqa: E402
from score import Weighting  # noqa: E402

PARENT_RE = re.compile(r"công ty mẹ", re.I)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
OPENING_RE = re.compile(r"đầu năm|đầu kỳ|1/1/", re.I)
STRIP_CHUNKS = ("công ty mẹ", "bao nhiêu", "đồng", "triệu", "tỷ", "nghìn", "trăm",
                "phần trăm", "cuối năm", "đầu năm", "cuối kỳ", "đầu kỳ", "đến ngày",
                "trong năm", "vào ngày", "tháng", "số dư", "tổng số", "giá trị",
                "của", "tại", "là")


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
    parser.add_argument("--out", default="artifacts/fresh/answer_plan.jsonl")
    parser.add_argument("--min-score", type=float, default=0.34)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               (ROOT / args.index).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    print(f"{len(records)} bang bao cao chinh")

    # One statement per (ticker, year, scope, kind): a statement split over pages
    # arrives as several tables, and the first table to carry a code owns it.
    cells: dict[tuple, dict] = defaultdict(dict)
    votes: dict[tuple, Counter] = defaultdict(Counter)
    raw_labels: dict[tuple, Counter] = defaultdict(Counter)
    for record in records:
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        family = "bank" if record["bank"] else "firm"
        for period in ("current", "prior"):
            key = (record["ticker"], record["year"], scope, record["kind"], period)
            for code, cell in record[period].items():
                cells[key].setdefault(code, {
                    "value": cell[0], "label": cell[1], "row": cell[2],
                    "col": cell[3], "raw": cell[4], "csv": record["csv"],
                    "doc": record["doc"], "table_id": record["table_id"],
                    "table_ref": record["table_ref"], "scale": record["scale"],
                })
        if not verified(record):
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) != want:
                continue
            key3 = (family, record["kind"], code)
            label_tokens = tokens_exact(cell[1])
            if label_tokens:
                votes[key3][label_tokens] += 1
                raw_labels[key3][" ".join(str(cell[1]).split())] += 1
    dictionary = {k: (c.most_common(1)[0][0], raw_labels[k].most_common(1)[0][0])
                  for k, c in votes.items()}
    # Token weights from the dictionary's own labels. "ngắn", "hạn", "chi", "phí"
    # appear in most of them and separate nothing; "vay" appears in a handful and is
    # the whole difference between 310 and 320.
    weights = Weighting(tokens for tokens, _label in dictionary.values())
    print(f"tu dien: {len(dictionary)} (ho, loai, ma)")

    tickers = {}
    for line in (ROOT / "data" / "code_stock.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            tickers[code.strip()] = name.strip().strip('"')
    by_length = sorted(tickers.items(), key=lambda item: -len(item[1]))
    banks = {r["ticker"] for r in records if r["bank"]}

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    plan, samples = [], []
    for question in questions:
        text = question["question"]
        found = {c for c in tickers if re.search(rf"\b{re.escape(c)}\b", text)}
        for code, name in by_length:
            if name and name.casefold() in text.casefold():
                found.add(code)
        years = YEAR_RE.findall(text)
        if len(found) != 1 or not years:
            counters["ngoai pham vi (nhieu ma / khong nam)"] += 1
            continue
        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        year = max(years)
        period = "prior" if OPENING_RE.search(text) else "current"
        family = "bank" if ticker in banks else "firm"

        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
        probe = re.sub(r"\([^)]*\)", " ", probe)
        probe = YEAR_RE.sub(" ", probe)
        for chunk in (tickers[ticker],) + STRIP_CHUNKS:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
        probe_tokens = tokens_exact(probe)
        if not probe_tokens:
            counters["khong con tu de doi chieu"] += 1
            continue

        scored = []
        for (dict_family, kind, code), (label_tokens, label) in dictionary.items():
            if dict_family != family:
                continue
            entry = cells.get((ticker, year, scope, kind, period), {}).get(code)
            if entry is None:
                continue
            shared = probe_tokens & label_tokens
            if not shared or contradicts(probe, label):
                continue
            # The floor stays UNWEIGHTED. Weighting it too took coverage from 105
            # addresses to 68: a rare word left in the probe dominates its mass, so a
            # label that names the right line but not that word fails the floor. The
            # weights belong in the ranking, where the question is which of two
            # similar labels to prefer.
            if len(shared) / len(probe_tokens) < 0.5:
                continue
            # MEASURED AND REVERTED: ranking by IDF-weighted overlap. The reasoning
            # was that "ngắn" and "hạn" appear in most labels while "vay" appears in
            # a handful, so weighting should prefer 320 "Vay và nợ thuê tài chính
            # ngắn hạn" over 310 "Nợ ngắn hạn" for a question about "vay ngắn hạn".
            # It did not: addresses fell 105 -> 96, id=16 still chose 310, and id=25
            # ("Giá vốn hàng hóa" -> kqkd/11) was lost. 310 wins because its label is
            # a SUBSET of the probe, which no reweighting of a symmetric overlap
            # changes; only a coverage floor does, and that floor costs more
            # addresses than the accuracy it buys.
            scored.append((len(shared) / len(probe_tokens | label_tokens),
                           kind, code, label, entry))
        if not scored:
            counters["khong ma nao khop tren bao cao co san"] += 1
            continue
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        if best[0] < args.min_score:
            counters["khop yeu"] += 1
            continue
        if best[0] - runner < args.min_margin:
            counters["hoa, tu choi"] += 1
            continue

        score, kind, code, label, entry = best
        counters["CO DIA CHI"] += 1
        plan.append({
            "id": question["id"], "kind": kind, "code": code, "period": period,
            "score": round(score, 3), "label": label,
            "doc": entry["doc"], "table_id": entry["table_id"],
            "table_ref": entry["table_ref"], "csv": entry["csv"],
            "row": entry["row"], "col": entry["col"], "scale": entry["scale"],
            "value": entry["value"], "raw": entry["raw"],
        })
        if len(samples) < args.show:
            samples.append(
                f"  id={question['id']:<5d} {kind}/{code} {period:7s} diem={score:.2f}"
                f"  o=({entry['row']},{entry['col']}) raw={entry['raw'][:18]}\n"
                f"     hoi  : {text[:100]}\n"
                f"     nhan : {label[:80]}")

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
