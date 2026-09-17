"""Splice new programs into an existing submission zip.

The code that built the best submission is no longer on disk (see the warning at
the top of IMPLEMENTATION.md), so rebuilding from `run_submit.py` regresses: 90
answers move and the declared-table policy collapses to a flat top-k. The zip
itself is intact and is the only faithful record of that configuration, which
makes patching it — not rebuilding — the safe way to ship an improvement.

What gets replaced is deliberately narrow. A question is a candidate only if its
current answer is *provably* wrong, so substitution cannot cost a point that was
already being earned:

  * a rate unit (%/lần/vòng) carrying a figure above a million — a đồng amount
    handed back as a rate;
  * a counting question ("có bao nhiêu công ty…") whose answer exceeds any
    possible count of the companies it names;
  * `result = 0.0`, the placeholder for a question no branch could serve.

Declared tables are handled to hold TABLES_F2 steady: the new program's tables go
to the front and the base list follows, truncated to the length it already had.
The count of declared refs per question is therefore unchanged — only their order
and membership move, so precision's denominator is fixed and only its numerator
can improve.

Usage:
  PYTHONPATH=src python scripts/patch_submission.py \
      --base screen_ratio_gated.zip --cache artifacts/gen14b_all.jsonl \
      --out llm14b.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.package import _csv_name, _render_csv  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

RATE_UNITS = frozenset({"phan_tram", "lan", "vong"})
# A percentage may legitimately pass 1000 (an eleven-fold rise is 1000%); nothing
# that is a rate at all passes a million.
RATE_CEILING = 1e6
COUNT_RE = re.compile(
    r"\bcó\s+bao\s+nhiêu\s+"
    r"(công ty|doanh nghiệp|ngân hàng|đơn vị|năm|quý|khoản|mã|cổ đông|thành viên)\b",
    re.I,
)


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# `UNIT_SCALE` in the parser maps currency words only, on the ground that share
# counts carry no VND conversion. True for the rescale pass — but it also leaves
# `unit_scale` None for "bao nhiêu triệu cổ phiếu", so nothing ever divides by a
# million and the figure ships in units of one share.
MILLION_UNITS = {"trieu_co_phieu": 1e6, "trieu_usd": 1e6}
MILLION_CEILING = 1e5


def wrong_for_sure(question, record) -> str | None:
    """Why this answer cannot be right, or None if it might be."""

    code = (record.get("pandas_query") or "").strip()
    if code in ("", "result = 0.0"):
        return "placeholder"
    answer = as_float(record.get("answer"))
    if question.target_unit in RATE_UNITS and abs(answer) > RATE_CEILING:
        return "rate_unit_is_dong"
    if question.target_unit in MILLION_UNITS and abs(answer) > MILLION_CEILING:
        # Counting questions ("có bao nhiêu công ty … trên 500 triệu cổ phiếu")
        # pick up the same unit word from their own threshold, but their answers
        # are single digits and never reach this ceiling.
        return "million_unit_unscaled"
    if COUNT_RE.search(question.question):
        # The answer counts members of a set the question names; it cannot
        # exceed that set, and 60 is far above any roster in this corpus.
        if abs(answer) > 60:
            return "count_too_large"
    return None


def rescaled(question, record) -> tuple[float, str] | None:
    """Deterministic repair for a figure asked in millions and shipped in units.

    Only for `million_unit_unscaled`, and only as a fallback when no generated
    program is usable. It does not re-pick the cell — if the base program read
    the wrong row, this puts a wrong figure in the right magnitude. That is still
    a free roll, because the unscaled figure scores nothing either way.
    """

    scale = MILLION_UNITS.get(question.target_unit)
    if scale is None:
        return None
    code = (record.get("pandas_query") or "").rstrip()
    if not code or "result" not in code:
        return None
    return as_float(record["answer"]) / scale, f"{code}\nresult = result / {scale:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="screen_ratio_gated.zip")
    parser.add_argument("--cache", default="artifacts/gen14b_all.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ids", default="",
                        help="JSON file holding the exact question ids to replace; "
                             "produced by _label_branches.py, which identifies the "
                             "serving branch by matching the shipped program text "
                             "against each cache rather than guessing from shape")
    parser.add_argument("--weak-branches", action="store_true",
                        help="also replace answers that came from the best-effort "
                             "fallback (5.9%%) or the cell-plan branch (7.2%%)")
    parser.add_argument("--promote-refs", action="store_true",
                        help="also move the new program's tables to the front of "
                             "relevant_tables (changes TABLES_F2; off by default)")
    args = parser.parse_args()

    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    base_path = ROOT / "submissions" / args.base
    with zipfile.ZipFile(base_path) as z:
        records = json.loads(z.read("submission.json"))
        payloads = {
            name.split("/", 1)[1]: z.read(name).decode("utf-8")
            for name in z.namelist() if name.startswith("data/")
        }
    by_id = {r["id"]: r for r in records}
    print(f"{args.base}: {len(records)} records, {len(payloads)} csv files")

    generated: dict[int, dict] = {}
    for line in Path(args.cache).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("value") is not None:
                generated[row["id"]] = row
    print(f"{args.cache}: {len(generated)} usable programs")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    # Which branch answered a question is not recorded in the zip, and the code
    # that wrote it is gone. It is not recoverable from the program's shape
    # either: 501 of the 1,012 programs come from one generator that serves both
    # the 42.8% label matcher and the 5.9% best-effort fallback, and 490 share a
    # prelude across the llm, plan and locate branches. What *is* recoverable is
    # the decision itself — `Corroborator.choose` is intact, and the strong
    # branch runs only where it returns a table. Replaying it separates the two
    # populations that share a generator without needing `run_submit.py` back.
    weak: set[int] = set()
    if args.ids:
        weak = set(json.loads(Path(args.ids).read_text(encoding="utf-8")))
        print(f"explicit id list: {len(weak)} questions")
    if args.weak_branches:
        from vifin.answering.corroborate import Corroborator  # noqa: PLC0415

        corroborator = Corroborator(store)
        ranked: dict[int, list[TableKey]] = {}
        rank_path = ROOT / "artifacts" / "anchor_rank.jsonl"
        for line in rank_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                ranked[row["id"]] = [
                    TableKey(r["doc_name"], int(r["table_id"])) for r in row["refs"]
                ][:20]

        planned = {
            row["id"]
            for line in (ROOT / "artifacts" / "planned.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
            for row in [json.loads(line)]
            if row.get("ok") and row.get("value") is not None
        }
        old_llm = {
            row["id"]
            for line in (ROOT / "artifacts" / "gen_helpers.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
            for row in [json.loads(line)]
            if row.get("ok") and row.get("value") is not None
        }
        def shape(record) -> str:
            code = (record.get("pandas_query") or "").strip()
            if code in ("", "result = 0.0"):
                return "placeholder"
            if "def num(" in code or "def find_row(" in code:
                return "prelude"
            return "lookup_style"

        fell_through = plan_served = 0
        for qid, question in parsed.items():
            record = by_id.get(qid)
            if record is None:
                continue
            kind = shape(record)
            # `run_submit` calls `choose` only for a question that names a
            # currency unit and that no earlier branch has answered. Replaying it
            # on all 1,012 declines 575 — most of them rate questions that never
            # entered this branch at all, and compositions that were already
            # answered. Both gates have to be replayed too, or the "weak" set
            # swallows the 42.8% branch it exists to protect.
            if (question.unit_scale
                    and kind == "lookup_style"
                    and corroborator.choose(question, ranked.get(qid, [])) is None):
                weak.add(qid)
                fell_through += 1
            # The plan branch sits after the llm branch and fires only where the
            # llm had nothing, so a question the old cache could not serve but
            # `planned.jsonl` could was answered at 7.2%.
            if kind == "prelude" and qid in planned and qid not in old_llm:
                weak.add(qid)
                plan_served += 1
        print(f"weak-branch questions: {len(weak)}  "
              f"(corroborator declined {fell_through}, plan-served {plan_served})")

    reasons: dict[str, int] = {}
    patched = skipped_no_program = skipped_dead = rescale_count = 0
    for qid, question in parsed.items():
        record = by_id.get(qid)
        if record is None:
            continue
        why = wrong_for_sure(question, record)
        if why is None and qid in weak:
            why = "weak_branch"
        if why is None:
            continue
        reasons[why] = reasons.get(why, 0) + 1

        def usable_program() -> tuple[float, str, list, list] | None:
            """The generated program, if it runs and does not repeat the defect."""

            row = generated.get(qid)
            if row is None:
                return None
            code = row["code"]
            # A program that hard-codes its answer fails the organisers' manual
            # review whatever it scores here.
            if reads_no_frame(code):
                return None
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            names = list(row["variables"])
            try:
                tables = {n: store.rows(k) for n, k in zip(names, keys)}
                outcome = run_query(code, tables)
            except Exception:
                return None
            if not outcome.ok or outcome.value is None:
                return None
            value = float(outcome.value)
            if value == 0.0 or value != value or abs(value) == float("inf"):
                return None
            if wrong_for_sure(question, {"pandas_query": code, "answer": value}):
                return None
            return value, code, keys, names

        found = usable_program()
        if found is None:
            # Arithmetic repair is the last resort and only exists for the one
            # defect it can actually fix.
            repair = rescaled(question, record) if why == "million_unit_unscaled" else None
            if repair is None:
                if qid in generated:
                    skipped_dead += 1
                else:
                    skipped_no_program += 1
                continue
            rescale_count += 1
            patched += 1
            if not args.dry_run:
                record["answer"], record["pandas_query"] = repair
            continue
        value, code, keys, names = found

        if args.dry_run:
            patched += 1
            continue

        evidence = []
        single = len(keys) == 1
        for position, (name, key) in enumerate(zip(names, keys), start=1):
            csv_name = _csv_name(key)
            if csv_name not in payloads:
                payloads[csv_name] = _render_csv(store.rows(key))
            evidence.append({
                "variable": "df" if single else f"df{position}",
                "csv_path": f"data/{csv_name}",
            })

        record["answer"] = value
        record["pandas_query"] = code
        record["evidence"] = evidence

        # `relevant_tables` is scored separately from `evidence` and may be
        # broader than what the program reads, so the declaration does not have
        # to follow the new program. Leaving it alone makes TABLES_F2 provably
        # identical to the base rather than merely equal in length: we lead that
        # metric, it is a third of the macro average, and promoting the new
        # program's tables would trade a certain hold for a speculative gain on
        # questions whose answers were wrong anyway.
        if args.promote_refs:
            width = len(record["relevant_tables"])
            fresh: list[str] = []
            for key in keys:
                ref = f"{key.doc_name}|{int(store.meta(key).start_line)}"
                if ref not in fresh:
                    fresh.append(ref)
            refs = (fresh + [r for r in record["relevant_tables"]
                             if r not in fresh])[:width]
            docs: list[str] = []
            for ref in refs:
                doc = ref.split("|", 1)[0]
                if doc not in docs:
                    docs.append(doc)
            for doc in record["relevant_docs"]:
                if doc not in docs:
                    docs.append(doc)
            record["relevant_tables"] = refs
            record["relevant_docs"] = docs
        patched += 1

    print(f"\nprovably wrong today: {sum(reasons.values())}  {reasons}")
    print(f"  no generated program : {skipped_no_program}")
    print(f"  program unusable     : {skipped_dead}")
    print(f"  repaired by rescale  : {rescale_count}")
    print(f"  PATCHED              : {patched}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    # Every csv_path referenced must exist; a missing one scores the question
    # zero on execution even though the program is right.
    referenced = {
        item["csv_path"].split("/", 1)[1]
        for record in records for item in record["evidence"]
    }
    missing = referenced - set(payloads)
    if missing:
        raise SystemExit(f"{len(missing)} csv files referenced but absent: "
                         f"{sorted(missing)[:5]}")

    out_path = ROOT / "submissions" / args.out
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(records, ensure_ascii=False, indent=1))
        for name in sorted(referenced):
            archive.writestr(f"data/{name}", payloads[name])
    print(f"\nwrote {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB, {len(referenced)} csv)")


if __name__ == "__main__":
    main()
