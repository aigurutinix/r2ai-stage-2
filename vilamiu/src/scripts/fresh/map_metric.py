"""Map a question's metric onto a Mã số, and see how much of the exam that reaches.

Everything upstream is now measured. A read addressed as (statement kind, Mã số,
current|prior) is verified at 97–99.6% on the printed accounting identities, which
check the code and the column, and at 90.9% by comparing `current` for year Y against
`prior` for year Y+1 across two independently unit-declaring documents. Scale errors
run at 1.6%. That is the substrate.

What is missing is the one step between a question and an address: the question names
a metric in natural Vietnamese, and the address needs a code.

The dictionary for that comes from the corpus rather than from a hand-written chart.
Each code appears in hundreds of reports, so the majority rendering of its label is
clean even where individual OCR is not — and only statements whose arithmetic already
verified contribute, so the dictionary cannot be polluted by a misparse. Candidate
codes are further restricted to the statement kinds that exist for the company, year
and scope the question names, and to the chart of accounts its labels imply, because
a credit institution and an enterprise use different codes for the same position.

Scoring is coverage of the LABEL's distinguishing words by the question. A label is
short and specific ("Doanh thu thuần về bán hàng và cung cấp dịch vụ"); a question
additionally carries a company name, a year and a unit that no label contains, so
scoring against the question's own token count would penalise every correct code
equally.

Reports how many questions get a confident code and prints samples to be read by
hand. It does not answer anything yet — the unit convention of the gold answer is
still unverified, and that needs the leaderboard, not more inference.

Usage:  python scripts/fresh/map_metric.py --show 14
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
from refine_codes import RELIABLE_LEN, fold, tokens, tokens_exact  # noqa: E402
from split_charts import chart_of  # noqa: E402

PARENT_RE = re.compile(r"công ty mẹ", re.I)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# The generator forbids "tại ngày" and writes balances as "cuối năm", "đầu năm" or
# "đến ngày", and flows by year. That is a contract from its prompt, not a guess.
CLOSING_RE = re.compile(r"cuối năm|đến ngày|cuối kỳ|31/12", re.I)
OPENING_RE = re.compile(r"đầu năm|đầu kỳ|1/1/", re.I)

# Pairs that share nearly every word and name opposite things. Token overlap cannot
# separate them, and all three errors in the first sample of this mapper were one of
# these: "lợi nhuận sau thuế" answered with code 50 "Lợi nhuận kế toán trước thuế",
# and "thuế TNDN phải nộp" (a balance) answered with "Thuế TNDN đã nộp" (a flow).
# Where either side commits to one member, the other must not commit to the opposite.
CONTRASTIVE = (
    ("trước thuế", "sau thuế"),
    ("phải nộp", "đã nộp"),
    ("phải trả", "đã trả"),
    ("phải thu", "phải trả"),
    ("phải nộp", "hoãn lại"),
    ("giá gốc", "dự phòng"),
    ("nguyên giá", "khấu hao"),
    ("ngắn hạn", "dài hạn"),
    ("vô hình", "hữu hình"),
    ("đầu năm", "cuối năm"),
    ("đầu kỳ", "cuối kỳ"),
)


def contradicts(probe: str, label: str) -> bool:
    left, right = fold(probe), fold(label)
    for one, other in CONTRASTIVE:
        a, b = fold(one), fold(other)
        if a in left and b in right and a not in right:
            return True
        if b in left and a in right and b not in right:
            return True
    return False


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
    parser.add_argument("--show", type=int, default=12)
    # Jaccard runs lower than coverage did, so the threshold moves with it: an exact
    # match of a four-word label against a probe carrying two extra words scores 0.67.
    parser.add_argument("--min-score", type=float, default=0.34)
    parser.add_argument("--min-margin", type=float, default=0.08)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]

    # Dictionary: (chart, kind, code) -> consensus label tokens, from verified only.
    votes: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    raw: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    available: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    charts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for record in records:
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        key = (record["ticker"], record["year"], scope)
        available[key].add(record["kind"])
        chart = chart_of(record)
        charts[key][chart] += 1
        if not verified(record):
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) != want:
                continue
            key3 = (chart, record["kind"], code)
            label_tokens = tokens_exact(cell[1])
            if label_tokens:
                votes[key3][label_tokens] += 1
                raw[key3][" ".join(str(cell[1]).split())] += 1
    dictionary = {k: (c.most_common(1)[0][0], raw[k].most_common(1)[0][0])
                  for k, c in votes.items()}
    print(f"tu dien: {len(dictionary)} (bieu, loai, ma)")

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

    counters: Counter[str] = Counter()
    samples: list[str] = []
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
        key = (ticker, year, scope)
        kinds = available.get(key)
        if not kinds:
            counters["khong co bao cao chinh cho ma-nam-pham vi"] += 1
            continue
        chart = charts[key].most_common(1)[0][0]

        # Strip what no row label carries, so the remaining words are the metric.
        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
        probe = re.sub(r"\((?:[^)]*)\)", " ", probe)
        probe = YEAR_RE.sub(" ", probe)
        # Period and framing words belong to the question, never to a row label, so
        # leaving them in inflates the union and drags every Jaccard score down —
        # "Số dư vay ngắn hạn … cuối năm 2025" scored 0.38 against its own label.
        for chunk in (tickers[ticker], "công ty mẹ", "bao nhiêu", "là", "đồng",
                      "triệu", "tỷ", "nghìn", "trăm", "phần trăm",
                      "cuối năm", "đầu năm", "cuối kỳ", "đầu kỳ", "đến ngày",
                      "trong năm", "vào ngày", "tháng", "số dư", "tổng số",
                      "giá trị", "của", "tại"):
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
        probe_tokens = tokens_exact(probe)
        if not probe_tokens:
            counters["khong con tu nao de doi chieu"] += 1
            continue

        scored = []
        for (dict_chart, kind, code), (label_tokens, label) in dictionary.items():
            if kind not in kinds or dict_chart != chart:
                continue
            shared = probe_tokens & label_tokens
            if not shared:
                continue
            if contradicts(probe, label):
                continue
            # The label must also explain most of what the question asks for. Jaccard
            # alone let "tổng chi phí bán hàng" through as cdkt/242 "Chi phí xây dựng
            # cơ bản dở dang" at 0.38, because PLX 2017 has no parseable income
            # statement and the best remaining candidate came from the wrong one.
            if len(shared) / len(probe_tokens) < 0.5:
                continue
            # Jaccard, not coverage of the label. Coverage rewards the shortest
            # generic label: asked about "vay ngắn hạn" it returns 310 "Nợ ngắn
            # hạn" at 1.00 because both of that label's words are present, and
            # asked about "lưu chuyển tiền thuần từ hoạt động kinh doanh" it ties
            # 50 "Lưu chuyển tiền thuần trong năm" with the right answer. Counting
            # the union punishes a label that ignores the question's own
            # distinguishing words as well as one that adds its own.
            hit = len(shared) / len(probe_tokens | label_tokens)
            scored.append((hit, kind, code, label))
        if not scored:
            counters["khong ma nao khop"] += 1
            continue
        scored.sort(reverse=True)
        best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        # A tie means the evidence does not name one code, and shipping either is a
        # coin flip. Refusing leaves the question to something else.
        if best[0] - runner < args.min_margin:
            counters["hoa hoac gan hoa, tu choi"] += 1
            continue
        period = "prior" if OPENING_RE.search(text) and not CLOSING_RE.search(text) \
            else "current"
        if best[0] < args.min_score:
            counters[f"khop yeu (<{args.min_score})"] += 1
            continue
        counters["CO MA tin cay"] += 1
        if len(samples) < args.show:
            samples.append(
                f"  id={question['id']:<5d} {best[1]}/{best[2]} {period:7s} "
                f"diem={best[0]:.2f} (nhi {runner:.2f})\n"
                f"     hoi  : {text[:104]}\n"
                f"     nhan : {best[3][:88]}")

    print()
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print("\nvi du:")
    for line in samples:
        print(line)


if __name__ == "__main__":
    main()
