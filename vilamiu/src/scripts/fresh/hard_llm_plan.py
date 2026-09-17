"""Hard multi-hop: Qwen3-14B plans a closed DAG; CellBook executes it.

Model never invents the number. It only picks op + metrics from an allow-list.
Execution binds Circular-200 cells and emits pandas that re-reads those cells.

Usage:
  PYTHONPATH=src python scripts/fresh/hard_llm_plan.py --limit 5
  PYTHONPATH=src python scripts/fresh/hard_llm_plan.py --workers 6
  PYTHONPATH=src python scripts/fresh/hard_llm_plan.py --splice
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import re
import sys
import threading
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from vifin.llm.client import ChatClient  # noqa: E402

import hard_hop as hh  # noqa: E402

MODEL = "qwen/qwen3-14b"
assert MODEL in __import__("vifin.llm.client", fromlist=["ALLOWED_MODELS"]).ALLOWED_MODELS

METRIC_KEYS = sorted(
    set(hh.ATOM) | set(hh.RATIOS) | set(hh.YOY_LEVEL)
)
OPS = (
    "argmin_lookup",
    "argmax_lookup",
    "below_median_mean",
    "above_median_mean",
    "median_then_rank_lookup",
    "median_then_extreme",
    "year_argmin",
    "year_argmax",
    "pred_mean",
    "pred_count",
    "refuse",
)

SYSTEM = f"""You plan multi-hop financial questions over Vietnamese statements (Circular 200).
Return ONE JSON object only. No markdown fences. No commentary.

Allowed ops: {list(OPS)}
Allowed metrics: {METRIC_KEYS}

Schema:
{{
  "op": "<one of allowed ops>",
  "tickers": ["AAA", "BBB"],
  "years": ["2022", "2023", "2024"],
  "year": "2024",
  "scope": "consolidated" | "separate",
  "filter": "<metric used to rank/filter, or null>",
  "rank": "<metric for second rank after median, or null>",
  "target": "<metric finally reported, or null>",
  "preds": [["metric", ">"|">="|"<"|"<="|"==", number], ...],
  "next_year": false,
  "unit": "nghìn tỷ đồng" | "tỷ đồng" | "triệu đồng" | "lần" | "%" | null
}}

Rules:
- Prefer consolidated unless the question says công ty mẹ.
- year = the reporting year for cohort screens. years = full list for year_argmin/argmax.
- next_year=true when the answer is for the year AFTER the extreme year.
- preds are hard filters before ranking (e.g. net_income>0, current_ratio>1.5, cfo_margin<0).
- For "có bao nhiêu doanh nghiệp ..." use pred_count; target may be null.
- For median then extreme of target among survivors use median_then_extreme
  (filter=median metric, target=extreme metric).
- For median then rank on B then lookup C use median_then_rank_lookup
  (filter=A, rank=B, target=C).
- If the question needs DOH/CCC/CAGR/hypothetical nếu/EBIT proxy, set op=refuse.
- Numbers in preds use dots as decimals (1.5 not 1,5).
- Tickers must be exchange codes named or implied in the question.
"""

# Narrow refuse for LLM path — HARD_REFUSE blocks too many solvable hops
# (inventory days, mean-delta wording) that CellBook can still execute.
LLM_REFUSE = re.compile(
    r"\bnếu\b|CAGR|chu kỳ tiền mặt|\bCCC\b|EBIT proxy|"
    r"đòn bẩy kinh doanh|biên an toàn",
    re.I,
)

HOP_CUE = re.compile(
    r"của doanh nghiệp có|của công ty có|"
    r"thấp hơn (?:mức )?trung vị|cao hơn (?:mức )?trung vị|cao hơn ngưỡng|"
    r"năm sau năm|năm kế tiếp|năm đầu tiên|tại năm có|ở năm |"
    r"có bao nhiêu doanh nghiệp|"
    r"CFO margin âm|bình quân của các|"
    r"ghi nhận.{0,40}là bao nhiêu|"
    r"trong nhóm.{0,80}(cao|thấp) nhất|"
    r"xét các (?:công ty|doanh nghiệp).{0,120}trung vị",
    re.I,
)

JSON_RE = re.compile(r"\{.*\}", re.S)


def unit_scale(hint: str | None, question: str) -> float | None:
    if hint:
        for name, scale in hh.UNIT_SCALES:
            if name in hint.casefold():
                return scale
        if hint.strip() in ("%", "lần", "vòng"):
            return None
    return hh.unit_of(question)


def parse_plan(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(plan, dict):
        return None
    op = plan.get("op")
    if op not in OPS:
        return None
    if op == "refuse":
        return plan
    for key in ("filter", "rank", "target"):
        value = plan.get(key)
        if value is not None and value not in METRIC_KEYS and value != "count":
            return None
    preds = plan.get("preds") or []
    cleaned = []
    for pred in preds:
        if not (isinstance(pred, (list, tuple)) and len(pred) == 3):
            return None
        metric, rel, thresh = pred
        if metric not in METRIC_KEYS or rel not in (">", ">=", "<", "<=", "=="):
            return None
        cleaned.append((metric, rel, float(thresh)))
    plan["preds"] = cleaned
    tickers = plan.get("tickers") or []
    if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
        return None
    plan["tickers"] = [t.strip().upper() for t in tickers if t.strip()]
    years = plan.get("years") or []
    if years and not all(isinstance(y, str) and re.fullmatch(r"20[0-2]\d", y) for y in years):
        return None
    year = plan.get("year")
    if year is not None and not re.fullmatch(r"20[0-2]\d", str(year)):
        return None
    if year is not None:
        plan["year"] = str(year)
    plan["scope"] = "separate" if plan.get("scope") == "separate" else "consolidated"
    plan["next_year"] = bool(plan.get("next_year"))
    return plan


def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def _score(book: hh.CellBook, tickers: list[str], year: str, scope: str,
           metric: str) -> list[tuple[str, float, list]]:
    out = []
    for ticker in tickers:
        got = hh.compute(book, ticker, year, scope, metric)
        if got is None:
            continue
        out.append((ticker, got[0], got[1]))
    return out


def _apply_preds(book: hh.CellBook, tickers: list[str], year: str, scope: str,
                 preds: list[tuple[str, str, float]]) -> list[str]:
    keep = []
    for ticker in tickers:
        ok = True
        for pred in preds:
            passed = hh.passes_pred(book, ticker, year, scope, pred)
            if passed is None:
                ok = False
                break
            if not passed:
                ok = False
                break
        if ok:
            keep.append(ticker)
    return keep


def execute_plan(book: hh.CellBook, question: str, plan: dict) -> dict | None:
    op = plan["op"]
    if op == "refuse":
        return None
    scope = plan["scope"]
    unit = unit_scale(plan.get("unit"), question)
    tickers = plan["tickers"]
    preds = plan.get("preds") or []
    year = plan.get("year") or (plan.get("years") or ["2024"])[-1]

    if op in ("year_argmin", "year_argmax"):
        if len(tickers) != 1:
            return None
        years = plan.get("years") or []
        if len(years) < 2:
            return None
        filt = plan.get("filter")
        target = plan.get("target")
        if not filt or not target:
            return None
        scored = []
        for y in years:
            if any(p[0] == "net_income" and p[1] == ">" for p in preds):
                ni = hh.compute(book, tickers[0], y, scope, "net_income")
                if ni is None or ni[0] <= 0:
                    continue
            got = hh.compute(book, tickers[0], y, scope, filt)
            if got is None:
                continue
            scored.append((y, got[0], got[1]))
        if not scored:
            return None
        winner = (min if op == "year_argmin" else max)(scored, key=lambda x: x[1])
        look = str(int(winner[0]) + 1) if plan.get("next_year") else winner[0]
        return hh.pack_answer(op, filt, target, tickers[0], winner[1], winner[2],
                              book, look, scope, unit, {"from_year": winner[0]})

    if not tickers:
        return None
    eligible = _apply_preds(book, tickers, year, scope, preds) if preds else list(tickers)

    if op == "pred_count":
        cells = []
        for ticker in eligible:
            metric = preds[0][0] if preds else "net_income"
            got = hh.compute(book, ticker, year, scope, metric)
            if got:
                cells.extend(got[1])
        if not cells and not eligible:
            # still allow zero count with at least one attempted cell from cohort
            for ticker in tickers[:1]:
                got = hh.compute(book, ticker, year, scope, "total_assets")
                if got:
                    cells.extend(got[1])
        if not cells:
            return None
        return {
            "op": op, "filter": (preds[0][0] if preds else "count"),
            "target": "count", "survivors": eligible, "year": year,
            **hh._pack_rate(float(len(eligible)), cells, filter_cells=[]),
        }

    if len(eligible) < 2 and op != "pred_mean":
        return None

    filt = plan.get("filter")
    target = plan.get("target")
    rank = plan.get("rank")

    if op == "pred_mean":
        if not target:
            return None
        values, cells = [], []
        for ticker in eligible:
            got = hh.compute(book, ticker, year, scope, target)
            if got is None:
                continue
            values.append(got[0])
            cells.extend(got[1])
        if len(values) < 1:
            return None
        mean = sum(values) / len(values)
        return {
            "op": op, "filter": (preds[0][0] if preds else filt), "target": target,
            "survivors": eligible, "year": year,
            **hh._pack_rate(mean, cells, filter_cells=[]),
        }

    if op == "median_then_rank_lookup":
        if not filt or not rank or not target:
            return None
        scored = _score(book, eligible, year, scope, filt)
        if len(scored) < 2:
            return None
        med = _median([v for _, v, _ in scored])
        side = plan.get("side") or "above"
        if side == "above":
            survivors = [t for t, v, _ in scored if v > med]
        else:
            survivors = [t for t, v, _ in scored if v < med]
        ranked = _score(book, survivors, year, scope, rank)
        if not ranked:
            return None
        ext = plan.get("ext") or "argmax"
        winner = (max if ext == "argmax" else min)(ranked, key=lambda x: x[1])
        look = plan.get("look_year") or year
        return hh.pack_answer(
            op, filt, target, winner[0], winner[1], winner[2],
            book, look, scope, unit, {"rank": rank, "median": med},
        )

    if op in ("argmin_lookup", "argmax_lookup"):
        if not filt or not target or filt == target:
            return None
        scored = _score(book, eligible, year, scope, filt)
        if len(scored) < 2:
            return None
        winner = (min if op == "argmin_lookup" else max)(scored, key=lambda x: x[1])
        return hh.pack_answer(op, filt, target, winner[0], winner[1], winner[2],
                              book, year, scope, unit)

    if op in ("below_median_mean", "above_median_mean", "median_then_extreme"):
        if not filt or not target:
            return None
        scored = _score(book, eligible, year, scope, filt)
        if len(scored) < 2:
            return None
        med = _median([v for _, v, _ in scored])
        if op == "above_median_mean" or (
                op == "median_then_extreme" and plan.get("side") == "above"):
            survivors = [(t, c) for t, v, c in scored if v > med]
        else:
            survivors = [(t, c) for t, v, c in scored if v < med]
        # below_median_mean / above_median_mean defaults
        if op == "below_median_mean":
            survivors = [(t, c) for t, v, c in scored if v < med]
        elif op == "above_median_mean":
            survivors = [(t, c) for t, v, c in scored if v > med]
        if not survivors:
            return None
        if op == "median_then_extreme" or plan.get("extreme"):
            want_max = (plan.get("ext") or "argmax") == "argmax"
            ranked = []
            for ticker, _ in survivors:
                got = hh.compute(book, ticker, year, scope, target)
                if got is None:
                    continue
                ranked.append((ticker, got[0], got[1]))
            if not ranked:
                return None
            winner = (max if want_max else min)(ranked, key=lambda x: x[1])
            return hh.pack_answer(
                "median_then_extreme", filt, target, winner[0],
                med, winner[2], book, year, scope, unit)
        values, cells = [], []
        for ticker, _ in survivors:
            got = hh.compute(book, ticker, year, scope, target)
            if got is None:
                continue
            values.append(got[0])
            cells.extend(got[1])
        if not values:
            return None
        mean = sum(values) / len(values)
        return {
            "op": op, "filter": filt, "target": target,
            "median": med, "survivors": [t for t, _ in survivors], "year": year,
            **hh._pack_rate(mean, cells, filter_cells=[]),
        }
    return None


def enrich_plan(question: str, plan: dict, resolver: hh.TickerResolver) -> dict:
    """Fill missing tickers/years/side from the question when the model omits them."""

    if not plan.get("tickers"):
        plan["tickers"] = hh.resolve_cohort(question, resolver)
    years = hh.years_of(question)
    if not plan.get("years") and years:
        plan["years"] = years
    if not plan.get("year"):
        start = re.search(r"^Năm\s+(20[0-2]\d)", question)
        inside = re.search(r"(?:trong|vào)\s+năm\s+(20[0-2]\d)", question, re.I)
        if start:
            plan["year"] = start.group(1)
        elif inside:
            plan["year"] = inside.group(1)
        elif len(years) == 1:
            plan["year"] = years[0]
        elif years:
            plan["year"] = years[-1]
        else:
            plan["year"] = "2024"
    if hh.PARENT_RE.search(question):
        plan["scope"] = "separate"
    if "side" not in plan or not plan["side"]:
        if re.search(r"cao hơn|trên trung vị|cao hơn ngưỡng", question, re.I):
            plan["side"] = "above"
        elif re.search(r"thấp hơn|dưới trung vị", question, re.I):
            plan["side"] = "below"
    if "ext" not in plan or not plan["ext"]:
        if re.search(r"cao nhất|lớn nhất", question, re.I):
            plan["ext"] = "argmax"
        elif re.search(r"thấp nhất|nhỏ nhất", question, re.I):
            plan["ext"] = "argmin"
    if re.search(r"năm sau năm|năm kế tiếp|năm ngay sau", question, re.I):
        plan["next_year"] = True
    if re.search(r"từ năm 2024 đến năm 2025", question, re.I):
        plan["look_year"] = "2025"
    return plan


def build_user(question: str, resolver: hh.TickerResolver) -> str:
    tickers = hh.resolve_cohort(question, resolver)
    years = hh.years_of(question)
    return (
        f"Question:\n{question}\n\n"
        f"Resolved tickers (may be incomplete): {tickers}\n"
        f"Years mentioned: {years}\n"
        "Return the plan JSON."
    )


def screening_block_ids() -> list[int]:
    path = ROOT / "artifacts/fresh/blocks.json"
    if not path.exists():
        return []
    blocks = json.loads(path.read_text(encoding="utf-8"))
    ids: set[int] = set()
    for name in (
        "nhieu cong ty", "ty le — sang loc", "khac — sang loc",
        "tien — sang loc", "tong hop nhieu o", "dem cong ty",
        "ty le — nhieu nam", "so lan",
    ):
        ids.update(blocks.get(name, []))
    return sorted(ids)


def candidate_ids(qs: dict[int, str], resolver: hh.TickerResolver,
                  book: hh.CellBook, *, screening: bool = False) -> list[int]:
    """Hop-like questions the rule solver cannot accept."""

    if screening:
        out = []
        for qid in screening_block_ids():
            text = qs.get(qid)
            if not text:
                continue
            if LLM_REFUSE.search(text) or hh.HARD_REFUSE.search(text):
                continue
            try:
                rule = hh.solve_one(book, text, resolver)
            except Exception:
                rule = None
            if rule is not None and hh.confidence_hit(text, rule):
                continue
            out.append(qid)
        return out
    out = []
    for qid, text in sorted(qs.items()):
        if not HOP_CUE.search(text):
            continue
        tickers = hh.resolve_cohort(text, resolver)
        years = hh.years_of(text)
        if not (len(tickers) >= 2 or (len(tickers) == 1 and len(years) >= 2)):
            continue
        if LLM_REFUSE.search(text):
            continue
        try:
            rule = hh.solve_one(book, text, resolver)
        except Exception:
            rule = None
        if rule is not None and hh.confidence_hit(text, rule):
            continue
        out.append(qid)
    return out


def reexec_ok(hit: dict) -> bool:
    namespace: dict = {"pd": pd}
    try:
        for item in hit["evidence"]:
            path = ROOT / item["csv_path"]
            if not path.exists():
                # payloads shipped in-memory
                name = item["csv_path"].split("/", 1)[-1]
                blob = hit["csv_payloads"].get(name)
                if blob is None:
                    return False
                namespace[item["variable"]] = pd.read_csv(
                    io.StringIO(blob), dtype=str, keep_default_na=False)
            else:
                namespace[item["variable"]] = pd.read_csv(
                    path, dtype=str, keep_default_na=False)
        # Prefer in-memory payloads always
        for item in hit["evidence"]:
            name = item["csv_path"].split("/", 1)[-1]
            blob = hit["csv_payloads"].get(name)
            if blob is not None:
                namespace[item["variable"]] = pd.read_csv(
                    io.StringIO(blob), dtype=str, keep_default_na=False)
        exec(hit["pandas_query"], namespace, namespace)  # noqa: S102
        return abs(float(hit["answer"]) - float(namespace["result"])) <= 0.01
    except Exception:
        return False


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--screening", action="store_true",
                        help="plan all multi-company / screening-block questions")
    parser.add_argument("--blocks", default="artifacts/fresh/blocks.json")
    parser.add_argument("--out", default="artifacts/fresh/hard_llm_plans.jsonl")
    parser.add_argument("--splice", action="store_true")
    parser.add_argument("--base", default="submissions/vote3_hardhop.zip")
    parser.add_argument("--dest", default="submissions/vote3_hardllm.zip")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--mag-max", type=float, default=2.0,
                        help="Skip splice when |new/old| exceeds this (vs base)")
    args = parser.parse_args()

    if args.model not in __import__(
            "vifin.llm.client", fromlist=["ALLOWED_MODELS"]).ALLOWED_MODELS:
        raise SystemExit(f"model {args.model!r} not allowed (<=14B open-weight)")

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    resolver = hh.TickerResolver()
    book = hh.CellBook()
    client = ChatClient.from_env(ROOT, model=args.model, max_tokens=args.max_tokens,
                                 disable_reasoning=True)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = candidate_ids(qs, resolver, book, screening=args.screening)
    if args.limit:
        ids = ids[: args.limit]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    todo = [
        qid for qid in ids
        if qid not in done or done[qid].get("status") in {
            "error", "bad_json", "exec_fail", "reexec_fail", "gate_drop",
        }
    ]
    print(f"model={args.model} candidates={len(ids)} todo={len(todo)} "
          f"cached={len(done)}", flush=True)

    lock = threading.Lock()
    handle = out_path.open("a", encoding="utf-8")

    def work(qid: int) -> None:
        question = qs[qid]
        row = {"id": qid, "question": question}
        try:
            raw = client.complete(SYSTEM, build_user(question, resolver))
            plan = parse_plan(raw)
            row["raw"] = raw[:2000]
            if plan is None:
                row["status"] = "bad_json"
                return
            plan = enrich_plan(question, plan, resolver)
            row["plan"] = {k: v for k, v in plan.items() if k != "preds"} | {
                "preds": [list(p) for p in plan.get("preds") or []],
            }
            if plan["op"] == "refuse":
                row["status"] = "refuse"
                return
            if LLM_REFUSE.search(question):
                row["status"] = "refuse_pattern"
                return
            hit = execute_plan(book, question, plan)
            if hit is None:
                row["status"] = "exec_fail"
                return
            if not hh.accept_hit(question, hit):
                row["status"] = "gate_drop"
                row["answer"] = hit["answer"]
                return
            if not hh.confidence_hit(question, hit):
                row["status"] = "conf_drop"
                row["answer"] = hit["answer"]
                return
            if not reexec_ok(hit):
                row["status"] = "reexec_fail"
                row["answer"] = hit["answer"]
                return
            row["status"] = "ok"
            row["answer"] = hit["answer"]
            row["op"] = hit.get("op")
            row["filter"] = hit.get("filter")
            row["target"] = hit.get("target")
            row["winner"] = hit.get("winner")
            row["hit"] = {
                "answer": hit["answer"],
                "pandas_query": hit["pandas_query"],
                "evidence": hit["evidence"],
                "relevant_docs": hit["relevant_docs"],
                "relevant_tables": hit["relevant_tables"],
                "csv_payloads": hit["csv_payloads"],
                "op": hit.get("op"),
                "filter": hit.get("filter"),
                "target": hit.get("target"),
                "winner": hit.get("winner"),
            }
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)[:300]
        finally:
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done[qid] = row
                print(f"id={qid} {row.get('status')} "
                      f"ans={row.get('answer')} op={row.get('op')}", flush=True)

    if todo:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
    handle.close()

    ok_rows = [done[i] for i in ids if done.get(i, {}).get("status") == "ok"]
    print(f"ok={len(ok_rows)} / planned={len(ids)}", flush=True)

    if not args.splice:
        return

    base = ROOT / args.base
    if not base.exists():
        base = ROOT / "submissions" / "vote3.zip"
    dest = ROOT / args.dest
    with zipfile.ZipFile(base) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}

    spliced = 0
    skipped_mag = 0
    for row in ok_rows:
        hit = row["hit"]
        qid = row["id"]
        inc = float(rows[qid].get("answer") or 0)
        if abs(hit["answer"] - inc) <= 0.01:
            continue
        new = float(hit["answer"])
        if inc != 0 and new != 0:
            ratio = max(abs(new / inc), abs(inc / new))
            if ratio > args.mag_max:
                skipped_mag += 1
                print(f"skip mag {qid} {new} vs {inc} ratio={ratio:.2f}")
                continue
        rows[qid].update({
            "answer": hit["answer"],
            "pandas_query": hit["pandas_query"],
            "evidence": hit["evidence"],
            "relevant_docs": hit["relevant_docs"],
            "relevant_tables": hit["relevant_tables"],
        })
        for name, text in hit["csv_payloads"].items():
            files[f"data/{name}"] = text.encode("utf-8")
        spliced += 1
        print(f"splice {qid} {hit.get('op')} {hit['answer']} (was {inc})")

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)
    print(f"spliced {spliced} skipped_mag={skipped_mag} -> {dest}")


if __name__ == "__main__":
    main()
