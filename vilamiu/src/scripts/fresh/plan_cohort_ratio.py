"""Consume the cohort and rate replies, keeping every check the single-cell path has.

Two model passes land here.

The cohort pass names a line, an operation and the companies to read it for. Where the
operation is `mot_ma` the model has already done the filtering with the figures in front
of it, so the plan is a single-cell address and the existing single-cell program serves
it unchanged — which is most of them, because a cohort question usually filters and then
asks for one company's figure. The combining operations get a multi-cell program.

The rate pass names a numerator line and a denominator line, and feeds the two-frame
program already in the builder.

Both keep the same discipline: a code outside the block is dropped rather than trusted,
the cell comes from the address book the identities check at 97–99.6%, and the sign rule
and unit conversion stay deterministic. A bad pick costs coverage, never a fabricated
number.

Usage:
  python scripts/fresh/plan_cohort_ratio.py \
      --cohort artifacts/fresh/replies_cohort.jsonl \
      --ratio artifacts/fresh/replies_ratio.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_block import Corpus  # noqa: E402
from score_model import TM_RE, parse  # noqa: E402

COMBINE = {"tong", "hieu", "trung_binh", "lon_nhat", "nho_nhat"}


def cell_for(corpus: Corpus, ticker: str, year: str, scope: str,
             kind: str, code: str, period: str):
    """The addressed cell, trying the other scope when this one has no statements."""

    for candidate in (scope, "consolidated" if scope == "separate" else "separate"):
        slot = corpus.rows.get((ticker, year, candidate), {}).get((kind, code))
        if slot:
            cell = slot.get(period) or slot.get("current") or slot.get("prior")
            if cell:
                return cell
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--cohort", default="")
    parser.add_argument("--cohort-prompts",
                        default="artifacts/fresh/prompts_cohort.jsonl")
    parser.add_argument("--ratio", default="")
    parser.add_argument("--ratio-prompts",
                        default="artifacts/fresh/prompts_ratio.jsonl")
    parser.add_argument("--out-cohort", default="artifacts/fresh/cohort_plan.jsonl")
    parser.add_argument("--out-ratio", default="artifacts/fresh/ratio_model_plan.jsonl")
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)

    def load(path: str) -> dict:
        out = {}
        if not path:
            return out
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["id"]] = record
        return out

    if args.cohort:
        meta = {k: v["meta"] for k, v in load(args.cohort_prompts).items()}
        counters: Counter[str] = Counter()
        plan = []
        for qid, record in sorted(load(args.cohort).items()):
            info = meta.get(qid)
            spec = parse(record["reply"])
            if info is None or not spec:
                counters["khong doc duoc JSON"] += 1
                continue
            kind = str(spec.get("nguon", "")).strip()
            code = str(spec.get("ma", "")).strip()
            period = str(spec.get("ky", "")).strip()
            op = str(spec.get("phep", "")).strip()
            tickers = [str(t).strip() for t in (spec.get("ma_ck") or [])]
            if kind not in ("cdkt", "kqkd", "lctt") or not code or not tickers:
                counters["dau ra khong hop le"] += 1
                continue
            if period not in ("current", "prior"):
                period = "current"

            cells = []
            for ticker in tickers:
                cell = cell_for(corpus, ticker, info["year"], info["scope"],
                                kind, code, period)
                if cell is None:
                    break
                cells.append((ticker, cell))
            if len(cells) != len(tickers):
                counters["thieu o cho mot trong cac ma"] += 1
                continue

            if op == "mot_ma" or len(cells) == 1:
                # The model filtered with the figures it could see and named the
                # winner; the answer is that one company's cell, so the ordinary
                # single-cell program serves it.
                ticker, cell = cells[0]
                plan.append({"id": qid, "source": "maso", "kind": kind,
                             "code": code, "period": period,
                             "label": cell["label"], "row": cell["row"],
                             "col": cell["col"], "csv": cell["csv"],
                             "doc": cell["doc"], "table_id": cell["table_id"],
                             "table_ref": cell["table_ref"],
                             "scale": cell["scale"], "score": 1.0,
                             "picked_by": "model_cohort", "op": "mot_ma"})
                counters["mot ma (loc roi bao)"] += 1
                continue
            if op not in COMBINE:
                counters["phep khong hop le"] += 1
                continue
            if op == "hieu" and len(cells) != 2:
                counters["hieu nhung khong dung 2 ma"] += 1
                continue
            plan.append({"id": qid, "source": "cohort", "kind": kind, "code": code,
                         "period": period, "op": op,
                         "cells": [{"ticker": t, **{k: c[k] for k in
                                                    ("label", "row", "col", "csv",
                                                     "doc", "table_id", "table_ref",
                                                     "scale")}}
                                   for t, c in cells],
                         "picked_by": "model_cohort"})
            counters[f"phep {op}"] += 1

        (ROOT / args.out_cohort).write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
            encoding="utf-8")
        print("=== cau nhom")
        for name, count in counters.most_common():
            print(f"  {name}: {count}")
        print(f"  tong: {len(plan)} -> {args.out_cohort}")

    if args.ratio:
        meta = {k: v["meta"] for k, v in load(args.ratio_prompts).items()}
        counters = Counter()
        plan = []
        for qid, record in sorted(load(args.ratio).items()):
            info = meta.get(qid)
            spec = parse(record["reply"])
            if info is None or not spec:
                counters["khong doc duoc JSON"] += 1
                continue
            top = spec.get("tu") or {}
            bottom = spec.get("mau") or {}
            period = str(spec.get("ky", "current")).strip() or "current"
            op = str(spec.get("phep", "phan_tram")).strip()
            sides = []
            for side in (top, bottom):
                kind = str(side.get("nguon", "")).strip()
                code = str(side.get("ma", "")).strip()
                if kind not in ("cdkt", "kqkd", "lctt") or not code:
                    sides = []
                    break
                cell = cell_for(corpus, info["ticker"], info["year"],
                                info["scope"], kind, code, period)
                if cell is None:
                    sides = []
                    break
                sides.append((kind, code, cell))
            if len(sides) != 2:
                counters["khong dinh vi du hai toan hang"] += 1
                continue
            if (sides[0][0], sides[0][1]) == (sides[1][0], sides[1][1]):
                counters["tu va mau trung mot dong"] += 1
                continue
            plan.append({
                "id": qid, "op": "pct" if op == "phan_tram" else "times",
                "num": {"kind": sides[0][0], "code": sides[0][1],
                        **{k: sides[0][2][k] for k in
                           ("label", "row", "col", "csv", "doc", "table_id",
                            "table_ref", "scale")}},
                "den": {"kind": sides[1][0], "code": sides[1][1],
                        **{k: sides[1][2][k] for k in
                           ("label", "row", "col", "csv", "doc", "table_id",
                            "table_ref", "scale")}},
                "picked_by": "model_ratio"})
            counters["CO CAP DIA CHI"] += 1

        (ROOT / args.out_ratio).write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
            encoding="utf-8")
        print("\n=== cau ty le")
        for name, count in counters.most_common():
            print(f"  {name}: {count}")
        print(f"  tong: {len(plan)} -> {args.out_ratio}")


if __name__ == "__main__":
    main()
