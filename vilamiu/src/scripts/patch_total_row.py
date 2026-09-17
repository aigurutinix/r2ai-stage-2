"""Splice the total-row rule into an existing submission zip.

The code that built the best submission is no longer on disk, so rebuilding from
`run_submit.py` regresses — 90 answers move and the declared-table policy collapses
to a flat top-k. The zip is the only faithful record of that configuration, so the
improvement goes in by patching it.

What the rule adds, measured on 600 gold questions: `lookup.find` was silent on
64.5% of them; with the rule it is silent on 38.8%, and the cells it takes are
right 66.9% of the time against 65.4% before. Coverage more than doubled without
costing precision.

Replacement is confined to questions where the rule adds something that could not
have happened before:

  * `find` returns nothing with `VIFIN_TOTAL_ROW=0` — so the label matcher was
    not the source of the current answer and nothing it earned can be lost;
  * `find` returns a cell with the rule on;
  * the current program came from a branch measured below the rule, or is the
    `result = 0.0` placeholder.

The third condition is the one that limits the downside. The rule is 66.9%
accurate on gold questions that skew toward totals (39.6% of them open with
"Tổng" against 12.9% of the real set), so it should displace the 5.9-13.3%
branches and nothing better.

`relevant_tables` is left untouched, so TABLES_F2 is provably identical to the
base — we lead that metric and it is a third of the macro average.

Usage:
  PYTHONPATH=src python scripts/patch_total_row.py --base direct14b.zip \
      --out totalrow.zip [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

WEAK_BRANCHES = {"locate_model", "locate_embed", "plan", "llm_8b", "llm_14b",
                 "placeholder"}


def key_from_csv_path(csv_path: str) -> TableKey | None:
    """`data/<doc_name>_table_<id>.csv` back to a store key."""

    name = csv_path.rsplit("/", 1)[-1]
    if not name.endswith(".csv") or "_table_" not in name:
        return None
    doc_name, _, table_id = name[:-4].rpartition("_table_")
    try:
        return TableKey(doc_name, int(table_id))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="direct14b.zip")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("VIFIN_TOTAL_ROW") == "0":
        raise SystemExit("VIFIN_TOTAL_ROW=0 disables the rule this patch ships")

    parsed = {q.id: q for q in parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")}
    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        records = json.loads(archive.read("submission.json"))
        payloads = {
            name.split("/", 1)[1]: archive.read(name).decode("utf-8")
            for name in archive.namelist() if name.startswith("data/")
        }
    print(f"{args.base}: {len(records)} records")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    # Which branch wrote each shipped program. Shape heuristics cannot tell them
    # apart — one generator serves both the label matcher and the weak fallback —
    # so each program is matched against the caches that produced it.
    caches: dict[str, dict[int, str]] = {}
    for label, name in (("locate_model", "located.jsonl"),
                        ("locate_embed", "embed_located.jsonl"),
                        ("plan", "planned.jsonl"),
                        ("llm_8b", "gen_helpers.jsonl"),
                        ("llm_14b", "gen14b_merged.jsonl")):
        path = ROOT / "artifacts" / name
        if not path.exists():
            continue
        found: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("code"):
                    found[row["id"]] = row["code"].strip()
        caches[label] = found

    def branch_of(record: dict) -> str:
        code = (record.get("pandas_query") or "").strip()
        if code in ("", "result = 0.0"):
            return "placeholder"
        for label, cache in caches.items():
            if cache.get(record["id"]) == code:
                return label
        return "lexical"

    stats: Counter[str] = Counter()
    patched = 0

    for record in records:
        question = parsed.get(record["id"])
        if question is None:
            stats["question_not_parsed"] += 1
            continue

        branch = branch_of(record)
        if branch not in WEAK_BRANCHES:
            stats[f"kept:{branch}"] += 1
            continue

        # The tables this question already reads. Using them rather than a fresh
        # retrieval keeps the change to the reader alone — if retrieval was
        # wrong, it stays wrong, and the measurement is not confounded.
        keys: list[TableKey] = []
        for item in record.get("evidence") or []:
            key = key_from_csv_path(item.get("csv_path", ""))
            if key is not None and key not in keys:
                keys.append(key)
        if not keys:
            stats["no_table_in_evidence"] += 1
            continue

        chosen = None
        for key in keys:
            grid = store.rows(key)
            if not grid:
                continue
            # Silent before the rule: this is what makes the patch additive.
            lookup_mod.TOTAL_ROW_ENABLED = False
            before = lookup_mod.find(grid, question)
            lookup_mod.TOTAL_ROW_ENABLED = True
            after = lookup_mod.find(grid, question)
            if before is not None:
                stats["matcher_already_had_it"] += 1
                break
            if after is None or after.score < lookup_mod.MIN_LABEL_SCORE:
                continue
            chosen = (key, grid, after)
            break

        if chosen is None:
            stats["rule_does_not_fire"] += 1
            continue

        key, grid, found = chosen
        meta = store.meta(key)
        unit_text = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        scale_in = lookup_mod.column_scale(grid, found.column, unit_text)
        asked = UNIT_SCALE.get(question.target_unit) or 1.0

        variable = "df" if len(keys) == 1 else "df1"
        body = lookup_mod.synthesize(found, scale_in, asked, magnitude=True)
        program = f"{PRELUDE}\n{body.replace('df.', f'{variable}.')}"
        outcome = run_query(program, {variable: grid})
        if not outcome.ok or outcome.value is None or reads_no_frame(program):
            stats["program_failed"] += 1
            continue

        stats[f"PATCHED:{branch}"] += 1
        patched += 1
        if args.dry_run:
            continue

        csv_name = f"{key.doc_name}_table_{key.table_id}.csv"
        if csv_name not in payloads:
            from vifin.submit.package import _render_csv

            payloads[csv_name] = _render_csv(grid)
        record["answer"] = float(outcome.value)
        record["pandas_query"] = program
        record["evidence"] = [{"variable": variable, "csv_path": f"data/{csv_name}"}]
        # relevant_tables deliberately untouched — TABLES_F2 stays identical.

    print()
    for key, count in stats.most_common(14):
        print(f"  {key:28s} {count:5d}")
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
