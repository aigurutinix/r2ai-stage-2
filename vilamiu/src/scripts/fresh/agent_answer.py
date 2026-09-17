"""Single-flow BCTC agent — resolve → plan → read → verify → fallback.

Unlike method_pipeline (multi-specialist router), this module exposes one
deterministic agent loop with a full trace per question for manual review.

Stages:
  1. resolve   ticker / years / scope / unit / shape (no numbers)
  2. plan      rule-first specialist order by shape (JSON plan metadata only)
  3. read      corpus CSV iloc / CellBook compute → candidate Patch
  4. verify    answer_gate + reexec + BCTC corpus re-read + identity
  5. fallback  keep vote3 incumbent when all specialists refuse

Usage:
  python scripts/fresh/agent_answer.py --dest submissions/agent_v1.zip
  python scripts/fresh/agent_answer.py --ids 29,56,124 --dest submissions/agent_smoke.zip
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import hard_hop as hh  # noqa: E402
import table_norm as tn  # noqa: E402
from blocks import block_of  # noqa: E402
from build_submission import unit_of as unit_of_sub  # noqa: E402
from method_pipeline import (  # noqa: E402
    Patch,
    accept,
    load_address_zip,
    load_btc_catalog,
    load_dual_ids,
    load_panel_det_cache,
    merge_note_plans,
    shape_of,
    try_address_book,
    try_catalog_ratio,
    try_cellbook_dag,
    try_dual_tab,
    try_maso_plan,
    try_note_plan,
)
from build_dualread_zip import load_jsonl  # noqa: E402
from validate_bctc import resolve_corpus_csv  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

# BCTC-first specialist order — no panel_det; screen → vote3 only (DAG unverified).
AGENT_ORDER: dict[str, tuple[str, ...]] = {
    "single": ("dual_tab", "maso_plan", "note_plan"),
    "ratio": ("catalog_ratio", "maso_plan"),
    "multi_year": ("cellbook_dag", "catalog_ratio"),
    "screen": (),
    "other": ("maso_plan", "note_plan"),
}


NOTE_STOP = frozenset("của và trong năm tổng số dư cuối đầu kỳ các khoản mục".split())


def note_row_matches(patch: Patch) -> bool:
    """Reject note plans whose row index does not contain the planned label."""

    if patch.layer != "note_plan":
        return True
    meta = patch.meta or {}
    row = meta.get("row")
    expected = str(meta.get("row_label") or "")
    if row is None or not expected:
        return True
    evidence = patch.evidence or []
    if not evidence:
        return False
    path = resolve_corpus_csv(evidence[0].get("csv_path", ""))
    if path is None or not path.is_file():
        return True
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            grid = list(csv_mod.reader(handle))
    except OSError:
        return False
    if int(row) >= len(grid):
        return False
    actual = str(grid[int(row)][0]).strip()
    if not actual:
        return False
    exp_fold = expected.casefold()
    act_fold = actual.casefold()
    if exp_fold in act_fold or act_fold in exp_fold:
        return True
    exp_tok = [
        t for t in re.split(r"[^0-9a-zà-ỹ]+", exp_fold)
        if len(t) >= 3 and t not in NOTE_STOP
    ]
    if not exp_tok:
        return False
    hits = sum(1 for t in exp_tok if t in act_fold)
    if hits / len(exp_tok) < 0.8:
        return False
    return tn.label_overlaps_question(expected, actual, min_token=2)


def ratio_magnitude_ok(patch: Patch, incumbent: float) -> bool:
    """Reject catalog_ratio splices that jump >4× vs vote3."""

    if patch.layer != "catalog_ratio":
        return True
    if abs(incumbent) < 1e-6:
        return True
    ratio = abs(patch.answer / incumbent)
    return 0.25 <= ratio <= 4.0


def agent_accept(
    patch: Patch | None, question: str, incumbent: float,
) -> tuple[Patch | None, str]:
    """accept() plus agent-specific BCTC semantic gates."""

    if patch is None:
        return None, "no_candidate"
    if not note_row_matches(patch):
        return None, "note_row_mismatch"
    if not ratio_magnitude_ok(patch, incumbent):
        return None, "ratio_magnitude"
    return accept(patch, question, incumbent)


@dataclass
class ResolveCtx:
    tickers: list[str]
    years: list[str]
    scope: str
    unit_name: str | None
    unit_div: float | None
    block: str
    shape: str
    n_companies: int


@dataclass
class PlanAttempt:
    specialist: str
    status: str
    plan: dict = field(default_factory=dict)


@dataclass
class AgentTrace:
    id: int
    question: str
    resolve: dict
    attempts: list[dict]
    chosen: str | None
    incumbent: float
    answer: float
    outcome: str
    verify_status: str | None = None


def resolve_question(question: str, resolver: TickerResolver) -> ResolveCtx:
    tickers = hh.resolve_cohort(question, resolver)
    years = hh.years_of(question)
    scope = "separate" if hh.PARENT_RE.search(question) else "consolidated"
    unit_name, unit_div = unit_of_sub(question)
    n_co = len(tickers)
    block = block_of(question, companies=max(1, n_co))
    shape = shape_of(block)
    return ResolveCtx(
        tickers=tickers,
        years=years,
        scope=scope,
        unit_name=unit_name,
        unit_div=unit_div,
        block=block,
        shape=shape,
        n_companies=n_co,
    )


def plan_meta(specialist: str, patch: Patch | None, reject: str) -> dict:
    if patch is None:
        return {"specialist": specialist, "reject": reject}
    meta = dict(patch.meta or {})
    meta.update({
        "specialist": specialist,
        "layer": patch.layer,
        "code": meta.get("code") or meta.get("row_label"),
        "kind": meta.get("kind"),
        "row": meta.get("row"),
        "col": meta.get("col"),
        "table": (patch.relevant_tables or [None])[0],
        "doc": (patch.relevant_docs or [None])[0],
        "candidate_answer": patch.answer,
    })
    return meta


def run_specialist(
    name: str,
    qid: int,
    question: str,
    ctx: ResolveCtx,
    *,
    book: hh.CellBook,
    resolver: TickerResolver,
    catalog,
    tab,
    dual_ids,
    maso_plans,
    note_plans,
    topk,
    address_rows,
    address_blobs,
) -> Patch | None:
    if name == "catalog_ratio":
        return try_catalog_ratio(book, question, resolver, catalog)
    if name == "cellbook_dag":
        return try_cellbook_dag(book, question, resolver)
    if name == "dual_tab":
        return try_dual_tab(qid, question, tab, dual_ids)
    if name == "maso_plan":
        return try_maso_plan(qid, question, maso_plans, topk)
    if name == "note_plan":
        return try_note_plan(qid, question, note_plans)
    if name == "address_book":
        return try_address_book(qid, address_rows, address_blobs)
    return None


def agent_one(
    qid: int,
    question: str,
    incumbent: float,
    ctx: ResolveCtx,
    resources: dict,
    use_address: bool,
) -> tuple[AgentTrace, Patch | None]:
    order = list(AGENT_ORDER.get(ctx.shape, AGENT_ORDER["other"]))
    if use_address:
        order.append("address_book")

    attempts: list[PlanAttempt] = []
    chosen: Patch | None = None
    chosen_name: str | None = None

    for name in order:
        cand = run_specialist(name, qid, question, ctx, **resources)
        patch, status = agent_accept(cand, question, incumbent)
        plan = plan_meta(name, cand, status)
        attempts.append(PlanAttempt(specialist=name, status=status, plan=plan))
        if status == "ok" and patch is not None:
            chosen = patch
            chosen_name = name
            break

    if chosen is None:
        outcome = "fallback_vote3"
        answer = incumbent
        verify_status = None
    else:
        outcome = "accepted"
        answer = chosen.answer
        verify_status = "ok"

    trace = AgentTrace(
        id=qid,
        question=question,
        resolve={
            "tickers": ctx.tickers,
            "years": ctx.years,
            "scope": ctx.scope,
            "unit": ctx.unit_name,
            "block": ctx.block,
            "shape": ctx.shape,
        },
        attempts=[asdict(a) for a in attempts],
        chosen=chosen_name,
        incumbent=incumbent,
        answer=answer,
        outcome=outcome,
        verify_status=verify_status,
    )
    return trace, chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--dest", default="submissions/agent_v1.zip")
    parser.add_argument("--trace", default="artifacts/fresh/agent_trace.jsonl")
    parser.add_argument("--ids", default="", help="Comma-separated ids (smoke)")
    parser.add_argument("--use-address", action="store_true")
    args = parser.parse_args()

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    resolver = TickerResolver()
    book = hh.CellBook()
    catalog = load_btc_catalog()
    dual_ids = load_dual_ids()
    tab = load_jsonl(ROOT / "artifacts/fresh/tab_plan.jsonl")
    maso_plans = load_jsonl(ROOT / "artifacts/fresh/greedy_plan.jsonl")
    note_plans = merge_note_plans(
        ROOT / "artifacts/fresh/note_plan.jsonl",
        ROOT / "artifacts/fresh/note_plan_all.jsonl",
    )
    topk_path = ROOT / "artifacts/fresh/dense_topk.json"
    topk: dict[int, list[dict]] = {}
    if topk_path.exists():
        topk = {
            int(k): v[:9]
            for k, v in json.loads(topk_path.read_text(encoding="utf-8")).items()
        }
    address_rows, address_blobs = load_address_zip(ROOT / "submissions/_wide_maso.zip")

    resources = dict(
        book=book,
        resolver=resolver,
        catalog=catalog,
        tab=tab,
        dual_ids=dual_ids,
        maso_plans=maso_plans,
        note_plans=note_plans,
        topk=topk,
        address_rows=address_rows,
        address_blobs=address_blobs,
    )

    base_path = ROOT / args.base
    with zipfile.ZipFile(base_path) as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        files = {n: z.read(n) for n in z.namelist() if n != "submission.json"}

    id_filter: set[int] | None = None
    if args.ids:
        id_filter = {int(x) for x in args.ids.split(",") if x.strip()}
        print(f"agent smoke: {len(id_filter)} ids", flush=True)

    stats: Counter[str] = Counter()
    traces: list[dict] = []
    manifest: list[dict] = []

    for qid in sorted(qs):
        if id_filter is not None and qid not in id_filter:
            continue
        question = qs[qid]
        incumbent = float(rows[qid].get("answer") or 0)
        ctx = resolve_question(question, resolver)
        trace, patch = agent_one(
            qid, question, incumbent, ctx, resources, args.use_address)

        traces.append(asdict(trace))
        stats[trace.outcome] += 1
        if trace.chosen:
            stats[f"layer_{trace.chosen}"] += 1

        if patch is None:
            stats["unchanged"] += 1
            continue

        rows[qid].update({
            "answer": patch.answer,
            "pandas_query": patch.pandas_query,
            "evidence": patch.evidence,
            "relevant_docs": patch.relevant_docs,
            "relevant_tables": patch.relevant_tables,
        })
        files.update(patch.files)
        manifest.append({
            "id": qid,
            "shape": ctx.shape,
            "layer": patch.layer,
            "old_answer": incumbent,
            "new_answer": patch.answer,
        })
        print(f"{qid} [{ctx.shape}] {patch.layer} {incumbent} -> {patch.answer}",
              flush=True)

    dest = ROOT / args.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    trace_path = ROOT / args.trace
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces),
        encoding="utf-8",
    )

    man_path = ROOT / "artifacts/fresh/agent_manifest.jsonl"
    man_path.write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
        encoding="utf-8",
    )

    print(f"\nagent done: changed {len(manifest)} / {len(traces)}")
    print(f"  stats={dict(stats)}")
    print(f"  -> {dest}")
    print(f"  trace -> {trace_path}")
    print(f"  manifest -> {man_path}")


if __name__ == "__main__":
    main()
