"""Human-readable agent trace dump for manual review.

Reads agent_trace.jsonl and prints question, resolve context, specialist
attempts, chosen answer, and the cell read from corpus when available.

Usage:
  python scripts/fresh/agent_review.py
  python scripts/fresh/agent_review.py --trace artifacts/fresh/agent_trace.jsonl \\
      --out artifacts/fresh/agent_review.txt --only-changed
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from validate_bctc import resolve_corpus_csv, reread_from_corpus  # noqa: E402


def cell_snippet(csv_path: Path, row: int, col: int, label_chars: int = 50) -> str:
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            grid = list(csv_mod.reader(handle))
    except OSError:
        return "(csv missing)"
    if row >= len(grid):
        return f"(row {row} OOR)"
    r = grid[row]
    label = str(r[0]).strip()[:label_chars] if r else ""
    val = str(r[col]).strip() if col < len(r) else ""
    hdr = str(grid[0][col]).strip()[:30] if grid and col < len(grid[0]) else ""
    return f"r{row} c{col} [{hdr}] {label!r} = {val!r}"


def render_trace(rec: dict, rows_by_id: dict[int, dict]) -> list[str]:
    qid = rec["id"]
    lines = ["=" * 96, f"id={qid}", f"HOI: {rec['question']}"]
    res = rec.get("resolve") or {}
    shape = res.get("shape") or rec.get("shape")
    lines.append(
        f"RESOLVE: tickers={res.get('tickers')} years={res.get('years')} "
        f"scope={res.get('scope')} unit={res.get('unit')} "
        f"shape={shape} block={res.get('block')}"
    )
    lines.append(f"OUTCOME: {rec.get('outcome')}  chosen={rec.get('chosen')}")
    lines.append(f"ANSWER: incumbent={rec.get('incumbent')} -> {rec.get('answer')}")

    sub = rows_by_id.get(qid) or {}
    if sub.get("pandas_query"):
        code = sub["pandas_query"].strip().splitlines()
        lines.append("PROGRAM: " + " / ".join(l.strip() for l in code[:3])[:280])

    for att in rec.get("attempts") or []:
        plan = att.get("plan") or {}
        spec = att.get("specialist", "?")
        st = att.get("status", "?")
        detail = ""
        if plan.get("table"):
            detail = f" table={plan['table']}"
        if plan.get("code"):
            detail += f" code={plan['code']}"
        lines.append(f"  try {spec}: {st}{detail}")

    evidence = sub.get("evidence") or []
    if evidence and rec.get("outcome") == "accepted":
        ev = evidence[0]
        path = resolve_corpus_csv(ev.get("csv_path", ""))
        meta = (rec.get("attempts") or [{}])[-1]
        for att in rec.get("attempts") or []:
            if att.get("specialist") == rec.get("chosen"):
                meta = att
                break
        plan = meta.get("plan") or {}
        row = plan.get("row")
        col = plan.get("col")
        if path and path.is_file() and row is not None and col is not None:
            lines.append(f"CELL: {cell_snippet(path, int(row), int(col))}")
        elif path and path.is_file():
            reread, status, source = reread_from_corpus(sub)
            lines.append(f"REREAD: {reread} ({status}, {source}) from {path.name}")
        rr, status, source = reread_from_corpus(sub)
        lines.append(f"BCTC_REREAD: {rr} status={status} source={source}")

    lines.append("")
    return lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="artifacts/fresh/agent_trace.jsonl")
    parser.add_argument("--zip", default="submissions/agent_smoke.zip")
    parser.add_argument("--out", default="artifacts/fresh/agent_review.txt")
    parser.add_argument("--only-changed", action="store_true")
    parser.add_argument("--ids", default="", help="Filter to comma-separated ids")
    args = parser.parse_args()

    trace_path = ROOT / args.trace
    if not trace_path.exists():
        print(f"missing trace: {trace_path}", file=sys.stderr)
        sys.exit(1)

    traces = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))

    id_filter: set[int] | None = None
    if args.ids:
        id_filter = {int(x) for x in args.ids.split(",") if x.strip()}

    zip_path = ROOT / args.zip
    rows_by_id: dict[int, dict] = {}
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            rows_by_id = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    out = io.StringIO()
    out.write(f"Agent review — {len(traces)} questions\n")
    out.write(f"trace={trace_path.name} zip={zip_path.name}\n\n")

    shown = 0
    for rec in traces:
        qid = rec["id"]
        if id_filter is not None and qid not in id_filter:
            continue
        if args.only_changed and rec.get("outcome") != "accepted":
            continue
        for line in render_trace(rec, rows_by_id):
            out.write(line + "\n")
        shown += 1

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out.getvalue(), encoding="utf-8")
    print(f"{shown} traces -> {dest} ({len(out.getvalue())} chars)")


if __name__ == "__main__":
    main()
