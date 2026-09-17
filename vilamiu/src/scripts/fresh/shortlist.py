"""Five candidate lines per question, and a measurement of whether choosing among them
could win anything.

The three commonest errors are all near-misses between labels that share most of their
words: 310 "Nợ ngắn hạn" against 320 "Vay và nợ thuê tài chính ngắn hạn", lctt/30
"hoạt động đầu tư" against lctt/40 "hoạt động tài chính", kqkd/60 "sau thuế" against 61
"của công ty mẹ". Token overlap cannot separate those; telling them apart is reading.

The failed model pass handed over two hundred lines and asked for one. This narrows the
question to the five nearest, which is a discrimination rather than a search — and it
makes the prompt about three hundred tokens instead of forty-seven hundred, so a small
model can run the whole exam.

But it is only worth running if the right line is IN the five. That is what this
measures, using the best shipped submission as a noisy oracle: it is 39% correct, so
where one of the candidates reproduces its answer the right line is probably among
them. The lift from top-1 to top-5 is the ceiling on what any discriminator can add;
if there is no lift, choosing better among these five cannot help and the shortlist is
the wrong shortlist.

Usage:  python scripts/fresh/shortlist.py --k 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from plan_answers import OPENING_RE, PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from refine_codes import tokens_exact  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def close(a: float, b: float) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) <= 0.01:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 1e-4


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--oracle", default="submissions/aimed.zip")
    parser.add_argument("--out", default="artifacts/fresh/shortlist.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    statements: dict[tuple, list[dict]] = defaultdict(list)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        for period in ("current", "prior"):
            for code, cell in record[period].items():
                statements[(record["ticker"], record["year"], scope)].append({
                    "period": period, "kind": record["kind"], "code": code,
                    "value": cell[0], "label": cell[1], "row": cell[2],
                    "col": cell[3], "csv": record["csv"], "doc": record["doc"],
                    "table_id": record["table_id"],
                    "table_ref": record["table_ref"], "scale": record["scale"],
                })

    with zipfile.ZipFile(ROOT / args.oracle) as archive:
        oracle = {r["id"]: r.get("answer")
                  for r in json.loads(archive.read("submission.json"))}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    hit_at = Counter()
    written = 0
    samples = []
    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for question in questions:
            text = question["question"]
            found = resolver.resolve(text)
            years = YEAR_RE.findall(text)
            if len(found) != 1 or not years:
                counters["ngoai pham vi"] += 1
                continue
            ticker = next(iter(found))
            scope = "separate" if PARENT_RE.search(text) else "consolidated"
            year = max(years)
            period = "prior" if OPENING_RE.search(text) else "current"
            _unit_name, unit = unit_of(text)

            probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
            probe = re.sub(r"\([^)]*\)", " ", probe)
            probe = YEAR_RE.sub(" ", probe)
            for chunk in (resolver.tickers.get(ticker, ""),) + STRIP_CHUNKS:
                probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
            probe_tokens = tokens_exact(probe)
            if not probe_tokens:
                counters["khong con tu de doi chieu"] += 1
                continue

            scored = []
            for key in ((ticker, year, scope),
                        (ticker, year, "consolidated" if scope == "separate"
                         else "separate")):
                for cell in statements.get(key, ()):
                    if cell["period"] != period:
                        continue
                    label_tokens = tokens_exact(cell["label"])
                    shared = probe_tokens & label_tokens
                    if not shared:
                        continue
                    scored.append((len(shared) / len(probe_tokens | label_tokens),
                                   cell))
                if scored:
                    break
            if not scored:
                counters["khong ung vien nao"] += 1
                continue
            scored.sort(key=lambda item: -item[0])
            top = scored[:args.k]
            counters["co danh sach ung vien"] += 1

            # Where does the oracle's answer sit in this ranking?
            target = oracle.get(question["id"])
            try:
                target = float(target) * (unit or 1.0)
            except (TypeError, ValueError):
                target = None
            rank = None
            if target is not None:
                for position, (_score, cell) in enumerate(scored, start=1):
                    if close(abs(cell["value"]), abs(target)):
                        rank = position
                        break
            if rank is None:
                hit_at["khong o dau trong danh sach"] += 1
            elif rank == 1:
                hit_at["hang 1"] += 1
            elif rank <= args.k:
                hit_at[f"hang 2..{args.k}"] += 1
            else:
                hit_at[f"sau hang {args.k}"] += 1

            handle.write(json.dumps({
                "id": question["id"], "question": text, "ticker": ticker,
                "year": year, "scope": scope, "period": period,
                "candidates": [{"kind": c["kind"], "code": c["code"],
                                "label": " ".join(str(c["label"]).split())[:80],
                                "value": c["value"], "score": round(s, 3),
                                "row": c["row"], "col": c["col"], "csv": c["csv"],
                                "doc": c["doc"], "table_id": c["table_id"],
                                "table_ref": c["table_ref"], "scale": c["scale"]}
                               for s, c in top],
            }, ensure_ascii=False) + "\n")
            written += 1
            if len(samples) < args.show and rank and rank > 1:
                samples.append(
                    f"  id={question['id']:<5d} oracle o hang {rank}\n"
                    f"     hoi : {text[:92]}\n" + "\n".join(
                        f"     {'>' if i + 1 == rank else ' '} {i + 1}. "
                        f"{c['kind']}/{c['code']} {str(c['label'])[:52]}"
                        for i, (_s, c) in enumerate(top)))

    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    total = sum(hit_at.values())
    print(f"\nvi tri dap an cua oracle trong danh sach ({total} cau):")
    for name, count in hit_at.most_common():
        print(f"  {name}: {count} ({100 * count / max(1, total):.0f}%)")
    one = hit_at["hang 1"]
    more = hit_at[f"hang 2..{args.k}"]
    if one + more:
        print(f"\n  top-1 = {100 * one / total:.0f}%   "
              f"top-{args.k} = {100 * (one + more) / total:.0f}%   "
              f"do nhac = {100 * more / total:.0f} diem")
        print("  do nhac chinh la chan tren cua viec phan biet trong 5 ung vien.")
    print(f"\n{written} cau co danh sach -> {ROOT / args.out}")
    for line in samples:
        print(line)


if __name__ == "__main__":
    main()
