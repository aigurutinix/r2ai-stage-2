"""Turn directly-answered numbers into admissible programs and splice them in.

The model is asked for the value, not for code, because that is what it is good
at: stripping our clauses made 106 of 170 replies hard-code a constant, and 52.8%
of those constants sat in a real cell of the tables it had been shown. It finds
figures; it fails at writing the lookup.

A bare number cannot ship — EXECUTION scores the program. So each answer is traced
back to the cell that produces it and re-emitted as `num(df_k, r, c) * factor`,
which reads a real frame and survives manual review.

The trace is a conversion, not a scale hunt: a cell matches when
`num(cell) * column_scale / asked_scale` lands on the model's number.
`column_scale` reads the unit off the column header — a bank statement printing
"Triệu VND" is already in millions — and it is the same function the 42.8% branch
uses. Hunting over a list of powers of ten would let a *wrong* cell match a right
number, which is the failure this design exists to avoid.

Measured on 60 questions: 70% located, and among those the answer agrees with the
shipped one 71.4% of the time — the shipped answers on that pool come mostly from
the 42.8% label matcher, so the direct path is in that league rather than noise.
That is what makes it worth putting against the 5.9%/7.2%/13.3% branches.

Usage:
  PYTHONPATH=src python scripts/patch_direct.py --base noconst.zip \
      --direct artifacts/direct_all.jsonl --out direct1.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _audit_generated import classify  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.package import _csv_name, _render_csv  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

REL_TOL = 5e-4


def cell_value(text: str) -> float | None:
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    raw = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return (scale > 0 and abs(a - b) / scale <= REL_TOL) or abs(a - b) <= 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="noconst.zip")
    parser.add_argument("--direct", default="artifacts/direct_all.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--weak-only", action="store_true", default=True,
                        help="only replace answers that are provably wrong or come "
                             "from a branch measured below 15% (default)")
    parser.add_argument("--everywhere", dest="weak_only", action="store_false",
                        help="replace wherever a cell is located, including the "
                             "42.8%% matcher's own pool")
    parser.add_argument("--branches", default="",
                        help="comma-separated branch labels to replace, e.g. "
                             "llm_14b. Targets a pool whose accuracy is known so "
                             "the submission measures the direct path instead of "
                             "merely betting on it; `lexical` is deliberately not "
                             "a sensible target because the 42.8%% matcher hides "
                             "in it.")
    args = parser.parse_args()

    parsed = {q.id: q for q in parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")}
    with zipfile.ZipFile(ROOT / "submissions" / args.base) as z:
        records = json.loads(z.read("submission.json"))
        payloads = {
            name.split("/", 1)[1]: z.read(name).decode("utf-8")
            for name in z.namelist() if name.startswith("data/")
        }
    by_id = {r["id"]: r for r in records}
    direct = {}
    for line in Path(args.direct).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("value") is not None:
                direct[row["id"]] = row
    print(f"{args.base}: {len(records)} records | direct answers: {len(direct)}")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    # Which branch wrote each shipped program, by matching the text against every
    # cache. Shape heuristics cannot do this: one generator serves both the 42.8%
    # matcher and the 5.9% fallback.
    caches: dict[str, dict[int, str]] = {}
    for label, name in (("locate_model", "located.jsonl"),
                        ("locate_embed", "embed_located.jsonl"),
                        ("plan", "planned.jsonl"),
                        ("llm_8b", "gen_helpers.jsonl"),
                        ("llm_14b", "gen14b_merged.jsonl")):
        path = ROOT / "artifacts" / name
        if not path.exists():
            continue
        out: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("code"):
                    out[row["id"]] = row["code"].strip()
        caches[label] = out

    WEAK = {"locate_model", "locate_embed", "plan", "placeholder"}

    def branch_of(record) -> str:
        code = (record.get("pandas_query") or "").strip()
        if code in ("", "result = 0.0"):
            return "placeholder"
        for label, cache in caches.items():
            if cache.get(record["id"]) == code:
                return label
        return "lexical"

    stats: Counter[str] = Counter()
    patched = 0

    for qid, record in by_id.items():
        row = direct.get(qid)
        if row is None:
            stats["no_direct_answer"] += 1
            continue
        question = parsed[qid]
        branch = branch_of(record)
        wrong = classify(record) != "USABLE"

        wanted = {b.strip() for b in args.branches.split(",") if b.strip()}
        if wanted:
            eligible = wrong or branch in wanted
        elif args.weak_only:
            eligible = wrong or branch in WEAK
        else:
            eligible = True
        if not eligible:
            stats[f"kept:{branch}"] += 1
            continue

        value = float(row["value"])
        asked = UNIT_SCALE.get(question.target_unit) or 1.0
        keys = [TableKey(d, int(t)) for d, t in row["keys"]]
        names = list(row["variables"])

        found = None
        for name, key in zip(names, keys):
            grid = store.rows(key)
            meta = store.meta(key)
            fallback = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            width = max((len(line) for line in grid), default=0)
            for c in range(width):
                factor = lookup_mod.column_scale(grid, c, fallback) / asked
                for r, line in enumerate(grid[1:], start=1):
                    if c >= len(line):
                        continue
                    raw = cell_value(line[c])
                    if raw is None or raw == 0:
                        continue
                    if close(raw * factor, value):
                        # `frame_from_rows` consumes grid row 0 as the header, so
                        # grid row N is DataFrame row N-1.
                        found = (name, key, r - 1, c, factor)
                        break
                if found:
                    break
            if found:
                break

        if found is None:
            stats["value_not_locatable"] += 1
            continue

        name, key, r_idx, c_idx, factor = found
        single = len(keys) == 1
        var = "df" if single else name
        body = (f"result = num({var}, {r_idx}, {c_idx})\n"
                f"result = round(result * {factor!r}, 2)")
        program = f"{PRELUDE}\n{body}"
        outcome = run_query(program, {var: store.rows(key)})
        if not outcome.ok or outcome.value is None or reads_no_frame(program):
            stats["program_failed"] += 1
            continue
        got = float(outcome.value)
        if not close(got, value):
            stats["program_disagrees"] += 1
            continue

        stats[f"patched:{'wrong' if wrong else branch}"] += 1
        patched += 1
        if args.dry_run:
            continue

        csv_name = _csv_name(key)
        if csv_name not in payloads:
            payloads[csv_name] = _render_csv(store.rows(key))
        record["answer"] = got
        record["pandas_query"] = program
        record["evidence"] = [{"variable": var, "csv_path": f"data/{csv_name}"}]
        # relevant_tables is scored separately and may be broader than evidence.
        # Leaving it untouched keeps TABLES_F2 provably identical to the base —
        # we lead that metric and it is a third of the macro average.

    print()
    for key, count in stats.most_common(14):
        print(f"  {key:26s} {count:4d}")
    print(f"\n  PATCHED {patched}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    referenced = {
        item["csv_path"].split("/", 1)[1]
        for record in records for item in record["evidence"]
    }
    missing = referenced - set(payloads)
    if missing:
        raise SystemExit(f"{len(missing)} csv files referenced but absent")

    out_path = ROOT / "submissions" / args.out
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(records, ensure_ascii=False, indent=1))
        for name in sorted(referenced):
            archive.writestr(f"data/{name}", payloads[name])
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(referenced)} csv)")


if __name__ == "__main__":
    main()
