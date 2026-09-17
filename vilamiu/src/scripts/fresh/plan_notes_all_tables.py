"""Plan thuyết-minh reads from all_tables.jsonl (heading + row label).

Extends tied-note planner to tables without statement tie — same ticker/year/scope
filter, match question tokens against heading ∩ row label.

Usage:
  python scripts/fresh/plan_notes_all_tables.py
  python scripts/fresh/plan_notes_all_tables.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import parse_statements as ps  # noqa: E402
from plan_answers import OPENING_RE, PARENT_RE, YEAR_RE  # noqa: E402
from plan_notes import PERIOD_LABEL_RE  # noqa: E402
from refine_codes import tokens_exact  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

STRIP_CHUNKS = (
    "bao nhiêu", "là bao nhiêu", "cho biết", "hãy cho", "theo", "đơn vị",
    "triệu đồng", "tỷ đồng", "nghìn đồng", "đồng", "vnd", "công ty", "cty",
    "năm", "kỳ", "báo cáo", "hợp nhất", "riêng lẻ", "công ty mẹ",
)


def probe_tokens(question: str, ticker: str, names: dict[str, str]) -> set[str]:
    probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", question)
    probe = re.sub(r"\([^)]*\)", " ", probe)
    probe = YEAR_RE.sub(" ", probe)
    for chunk in (names.get(ticker, ""),) + STRIP_CHUNKS:
        if chunk:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
    return tokens_exact(probe)


def load_tables_by_key(path: Path) -> dict[tuple, list[dict]]:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("scale") is None:
            continue
        key = (rec["ticker"], rec["year"], rec["scope"])
        by_key[key].append(rec)
    return by_key


def score_row(remaining: set[str], heading: str, label: str) -> float:
    hay = tokens_exact(f"{heading} {label}")
    if not remaining or not hay:
        return 0.0
    return len(remaining & hay) / len(remaining | hay)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--skip", default="artifacts/fresh/greedy_plan.jsonl")
    parser.add_argument("--existing", default="artifacts/fresh/note_plan.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/note_plan_all.jsonl")
    parser.add_argument("--min-row-score", type=float, default=0.45)
    parser.add_argument("--min-gap", type=float, default=0.06)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    skip_ids: set[int] = set()
    for path in (args.skip, args.existing):
        p = ROOT / path
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    skip_ids.add(json.loads(line)["id"])

    existing: dict[int, dict] = {}
    exist_path = ROOT / args.existing
    if exist_path.exists():
        for line in exist_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["id"]] = rec

    print("loading all_tables index…", flush=True)
    started = time.time()
    by_key = load_tables_by_key(ROOT / args.tables)
    print(f"  {len(by_key)} report keys in {time.time() - started:.1f}s", flush=True)

    tickers: dict[str, str] = {}
    for line in (ROOT / "data/code_stock.csv").read_text(encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            tickers[code.strip()] = name.strip().strip('"')
    by_length = sorted(tickers.items(), key=lambda item: -len(item[1]))

    questions = [
        json.loads(line)
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        questions = questions[: args.limit]

    resolver = TickerResolver()
    counters: Counter[str] = Counter()
    plan: list[dict] = []

    for question in questions:
        qid = question["id"]
        if qid in skip_ids and qid not in existing:
            counters["skip_maso"] += 1
            continue
        if qid in existing and float(existing[qid].get("row_score") or 0) >= 0.75:
            counters["keep_tied"] += 1
            continue

        text = question["question"]
        if not re.search(
            r"thuyết minh|thù lao|lãi tiền gửi|cho vay.{0,20}ngành|"
            r"cổ tức|cổ đông|phạt|dự phòng|doanh thu từ|chi phí khác|"
            r"lãi vay phải trả|trả trước|tiền gửi|phải thu|phải trả",
            text, re.I,
        ):
            counters["not_note_shape"] += 1
            continue

        found = {c for c in tickers if re.search(rf"\b{re.escape(c)}\b", text)}
        for code, name in by_length:
            if name and name.casefold() in text.casefold():
                found.add(code)
        years = YEAR_RE.findall(text)
        if len(found) != 1 or not years:
            counters["scope"] += 1
            continue

        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        key = (ticker, max(years), scope)
        tables = by_key.get(key)
        if not tables:
            counters["no_tables"] += 1
            continue

        remaining = probe_tokens(text, ticker, tickers)
        if len(remaining) < 2:
            counters["thin_probe"] += 1
            continue

        scored: list[tuple[float, dict, dict, int, int]] = []
        for tab in tables:
            heading = tab.get("heading") or ""
            for row in tab.get("rows") or []:
                label = row.get("label") or ""
                if PERIOD_LABEL_RE.match(label.strip()):
                    continue
                hit = score_row(remaining, heading, label)
                if hit < args.min_row_score:
                    continue
                cols = row.get("cols") or []
                if not cols:
                    continue
                wanted = 1 if OPENING_RE.search(text) else 0
                if len(cols) <= wanted:
                    continue
                col = cols[wanted][0]
                scored.append((hit, tab, row, row["row"], col))

        if not scored:
            counters["no_row"] += 1
            continue
        scored.sort(key=lambda t: -t[0])
        best, tab, row, ridx, col = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        if best - runner < args.min_gap:
            counters["tie_row"] += 1
            continue

        entry = {
            "id": qid,
            "source": "note",
            "row_score": round(best, 3),
            "doc": tab["doc"],
            "table_id": tab["table_id"],
            "table_ref": tab["table_ref"],
            "csv": tab["csv"],
            "row": ridx,
            "col": col,
            "scale": tab["scale"],
            "row_label": row["label"],
            "heading": (tab.get("heading") or "")[:120],
            "period": "prior" if OPENING_RE.search(text) else "current",
        }
        prev = existing.get(qid)
        if prev and float(prev.get("row_score") or 0) >= best:
            counters["existing_better"] += 1
            continue
        plan.append(entry)
        counters["planned"] += 1

    out = ROOT / args.out
    merged = {r["id"]: r for r in plan}
    for qid, rec in existing.items():
        if qid not in merged:
            merged[qid] = rec
    out.write_text(
        "".join(json.dumps(merged[i], ensure_ascii=False) + "\n"
                for i in sorted(merged)),
        encoding="utf-8",
    )
    print()
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"-> {out}  ({len(merged)} total plans, +{len(plan)} new)")


if __name__ == "__main__":
    main()
