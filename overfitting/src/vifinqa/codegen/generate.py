"""Codegen orchestration: batch prompts -> execute -> majority vote ->
self-debug round(s) -> rule-based fallback.

Batch-first design so vLLM sees big batches (fast on Kaggle T4 x2).
Outputs codegen_results.jsonl records:
    {id, answer, pandas_query, used_vars:[{var, report_id, table_pos}],
     status, source, votes, attempts, detail}
"""
from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

from ..config import CODEGEN_K
from ..extraction.build_store import Store
from ..router.evidence import evidence_coverage
from ..retrieval.serialize import tidy_csv_text, df_roundtrip
from ..utils.io import read_jsonl, write_jsonl
from .executor import run_code, extract_code
from .to_expression import try_to_expression
from .formulas import describe_for_prompt
from ..retrieval.shortlist import (
    build_shortlist,
    candidate_matches_requirement,
    requirement_linking_variants,
    render_shortlist,
)
from .prompts import SYSTEM, build_user, SELECT_SYSTEM, build_select_user
from .rule_codegen import try_rule_answer
from .rule_composite import try_composite_answer
from .formula_solver import requires_formula_solver, try_formula_answer
from .arbitrate import arbitrate
from .selection import (
    parse_selection,
    requirement_coverage,
    selection_matches_route,
    synthesize,
    confidence as sel_conf,
)
from .semantic import (
    all_dataframe_refs,
    answer_dataframe_refs,
    validate_generated_answer,
)


_DERIVED_EVIDENCE_FORMULAS = {
    "net_financial_result": (
        "financial_income - abs(financial_expense); return in the requested money unit"
    ),
    "effective_income_tax_rate": (
        "abs(income_tax_expense_total) / abs(pretax_profit) * 100"
    ),
    "depreciation_amortization_cost_share": (
        "abs(depreciation_amortization_expense) / "
        "abs(production_business_cost_by_element_total) * 100"
    ),
    "related_party_receivables_payables_total": (
        "related_party_receivables + related_party_payables; "
        "return in the requested money unit"
    ),
}


class QuestionBundle:
    def __init__(self, rec: dict, store: Store, k: int = CODEGEN_K,
                 run_signature: str = ""):
        self.id = rec["id"]
        self.question = rec["question"]
        self.route = rec["route"]
        self.run_signature = run_signature
        self.cands = rec["candidates"][:k]
        self.evidence = evidence_coverage(
            self.route.get("evidence_requirements", []), self.cands
        )
        self.route["evidence"] = self.evidence
        self.tables: list[dict] = []
        self._shortlist = None
        self.dfs: dict = {}
        by_ticker: dict[str, list[dict]] = {}
        for c in self.cands:
            by_ticker.setdefault(c["ticker"], []).append(c)
        meta_lookup = {}
        for ticker, cs in by_ticker.items():
            tdf = store.tables_of(ticker, list({c["report_id"] for c in cs}))
            for m in tdf.to_dict("records"):
                meta_lookup[(m["report_id"], int(m["table_pos"]))] = m
        for i, c in enumerate(self.cands, start=1):
            m = meta_lookup.get((c["report_id"], int(c["table_pos"])))
            if m is None:
                continue
            csv_text = tidy_csv_text(m)
            var = f"df{i}"
            us = m.get("unit_scale")
            us = None if (us is None or us != us) else float(us)
            self.tables.append({
                "var": var, "report_id": c["report_id"],
                "table_pos": int(c["table_pos"]), "page": c.get("page"),
                "unit_scale": us, "unit_source": m.get("unit_source", "none"),
                "report_year": int(m["year"]),
                "context": str(m.get("context") or ""),
                "grid_json": str(m.get("grid_json") or "[]"),
                "csv_text": csv_text,
            })
            self.dfs[var] = df_roundtrip(csv_text)

    def shortlist(self, encoder=None, top_n: int = 8):
        """Row-level schema linking (P1.1): pre-match candidate cells so the
        model chooses among ~8 rows instead of scanning every table."""
        if self._shortlist is None:
            variants = self.route.get("metric_variants") or [self.route.get("metric_norm", "")]
            generic = build_shortlist(
                self.tables, variants, self.route.get("years") or [],
                top_n=max(24, top_n * 2), encoder=encoder,
                question=self.question)
            self._shortlist = self._requirement_shortlist(generic, encoder)
        requirement_count = len(self.route.get("evidence_requirements", []))
        effective_top_n = max(top_n, min(24, requirement_count))
        return self._shortlist[:effective_top_n]

    def _requirement_shortlist(self, generic, encoder=None):
        """Reserve row candidates per operand, then fill with global matches."""
        requirements = self.route.get("evidence_requirements", [])
        if not requirements:
            return generic
        selected, seen = [], set()
        for requirement in requirements:
            ticker = str(requirement.get("ticker") or "").upper()
            year = requirement.get("year")
            tables = [
                table for table in self.tables
                if (not ticker or str(table["report_id"]).split("_")[0].upper() == ticker)
                and (year is None or int(table.get("report_year") or 0)
                     in {int(year), int(year) + 1})
            ]
            if not tables:
                continue
            matches = build_shortlist(
                tables,
                requirement_linking_variants(requirement),
                [int(year)] if year is not None else [],
                top_n=4,
                encoder=encoder,
                min_score=62.0,
                question=str(requirement.get("metric_label") or ""),
            )
            matches = [
                candidate for candidate in matches
                if candidate_matches_requirement(candidate, requirement)
            ]
            if not matches:
                continue
            candidate = matches[0]
            key = (candidate.var, candidate.row, candidate.col)
            if key not in seen:
                selected.append(candidate)
                seen.add(key)
        for candidate in generic:
            key = (candidate.var, candidate.row, candidate.col)
            if key not in seen:
                selected.append(candidate)
                seen.add(key)
        return selected

    def _evidence_plan_block(self) -> str:
        requirements = self.route.get("evidence_requirements", [])
        if not requirements:
            return ""
        needed = "\n".join(
            f"- {r.get('ticker')}/{r.get('year')}/{r.get('metric_key')} "
            f"(exact line item: {r.get('metric_label')})"
            for r in requirements[:24]
        )
        state = self.evidence
        keys = list(dict.fromkeys(self.route.get("metric_keys") or []))
        formulas = [
            f"{key} = {_DERIVED_EVIDENCE_FORMULAS[key]}"
            for key in keys if key in _DERIVED_EVIDENCE_FORMULAS
        ]
        formula_block = (
            "DERIVED FORMULA CONTRACT: " + "; ".join(formulas) + "\n"
            if formulas else ""
        )
        return (
            "CANONICAL CELL CONTRACT (use every exact operand below):\n"
            f"{needed}\n"
            f"{formula_block}"
            f"RETRIEVAL COVERAGE: {state['covered']}/{state['required']}"
            f" complete={state['complete']}\n"
        )

    def select_messages(self, encoder=None) -> list[dict]:
        """Selection mode: the model only picks shortlist rows + an operation."""
        plan = self.route.get("plan") or {}
        op = plan.get("op", "lookup")
        plan_block = ""
        if op != "lookup" or len(plan.get("facts", [])) > 1:
            plan_block = ("HINT — the question looks like: " + op + "\n"
                          "ENTITIES/PERIODS NEEDED: " + ", ".join(
                              f"{f.get('ticker')}/{f.get('year')}"
                              for f in plan.get("facts", [])[:8]))
        plan_block += self._evidence_plan_block()
        return [{"role": "system", "content": SELECT_SYSTEM},
                {"role": "user", "content": build_select_user(
                    self.question, self.route,
                    render_shortlist(self.shortlist(encoder, top_n=12)),
                    plan_block)}]

    def prompt_messages(self, encoder=None) -> list[dict]:
        from ..retrieval.serialize import preview_for_prompt
        tables = [{**t, "preview": preview_for_prompt(t["csv_text"])} for t in self.tables]
        plan = self.route.get("plan") or {}
        op = plan.get("op", "lookup")
        plan_block = ""
        if op != "lookup" or len(plan.get("facts", [])) > 1:
            plan_block = (describe_for_prompt(op, len(plan.get("facts", []))) + "\n"
                          "FACTS TO LOCATE: " + ", ".join(
                              f"{f.get('ticker')}/{f.get('year')}/{f.get('doc_type')}"
                              for f in plan.get("facts", [])[:8]) + "\n")
        plan_block += self._evidence_plan_block()
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user(
                    self.question, self.route, tables,
                    shortlist_block=render_shortlist(self.shortlist(encoder)),
                    plan_block=plan_block)}]

    def used_vars(self, code: str) -> list[dict]:
        # Submission replay must receive every dataframe the program may access,
        # including intermediates that do not flow into the final answer.  The
        # separate semantic guard still requires the answer itself to be grounded.
        try:
            refs = sorted(all_dataframe_refs(code))
        except SyntaxError:
            refs = []
        by_var = {t["var"]: t for t in self.tables}
        used = [by_var[v] for v in refs if v in by_var]
        return [{"var": t["var"], "report_id": t["report_id"],
                 "table_pos": t["table_pos"]} for t in used]

    def debug_messages(self, code: str, error: str) -> list[dict]:
        """Retry with the original route and table previews still in context."""
        messages = self.prompt_messages()
        messages.extend([
            {"role": "assistant", "content": f"```python\n{code}\n```"},
            {"role": "user", "content": (
                "The previous program failed execution or semantic validation.\n"
                f"Error: {error}\n"
                "Return a corrected Python code block grounded in the provided "
                "DataFrames."
            )},
        ])
        return messages


def _vote(samples_results: list[tuple[str, dict]]):
    """samples_results: [(code, exec_result)] -> (answer, code) by majority on
    the rounded value; ties broken by first occurrence."""
    ok = [(c, r) for c, r in samples_results if r["status"] == "ok"]
    if not ok:
        return None
    counts = Counter(round(r["value"], 2) for _c, r in ok)
    winner, n = counts.most_common(1)[0]
    for c, r in ok:
        if round(r["value"], 2) == winner:
            return {"answer": float(winner), "code": c, "votes": int(n),
                    "n_ok": len(ok), "semantic": r.get("semantic")}
    return None


def run_codegen(retrieval_path: Path, store_dir: Path, out_path: Path, client,
                k: int = CODEGEN_K, n_samples: int = 4, temperature: float = 0.7,
                debug_rounds: int = 1, limit: int = 0, use_rule_fallback: bool = True,
                rule_first: bool = False, max_tokens: int = 768,
                checkpoint_every: int = 32, resume: bool = True,
                time_budget_s: float = 0.0, run_signature: str = "",
                use_dense: bool = False, dense_model: str = "",
                llm_target: str = "all", llm_mode: str = "code",
                llm_ids: set[int] | None = None) -> None:
    """Crash-safe codegen.

    Order of operations (important on Kaggle where a session can die at any time):
      1. build bundles
      2. compute the deterministic RULE baseline for every question and FLUSH it
         -> out_path is a complete, submittable file from minute ~2 onward
      3. run the LLM in chunks of `checkpoint_every`, overwriting entries with
         better answers, flushing after every chunk
      4. stop cleanly when `time_budget_s` is exceeded (keeps what it has)

    resume=True picks up an existing out_path and skips ids already answered by
    the LLM (source startswith 'llm'). When run_signature is non-empty, only
    records produced by the exact same semantic run configuration are reused.
    """
    t_start = time.time()
    store = Store(store_dir, cache_size=4)
    # optional BGE-M3 row matcher (P1.4). None -> lexical only, never fatal.
    encoder = None
    if use_dense:
        from ..retrieval.dense import load_encoder, DEFAULT_MODEL
        encoder = load_encoder(dense_model or DEFAULT_MODEL,
                               cache_dir=Path(store_dir) / "label_index")
    recs = read_jsonl(retrieval_path)
    if limit:
        recs = recs[:limit]

    bundles, results = [], {}
    print(f"building {len(recs)} bundles...", flush=True)
    for rec in recs:
        b = QuestionBundle(rec, store, k, run_signature=run_signature)
        if not b.tables:
            results[b.id] = _empty_result(b, "no candidate tables")
            continue
        bundles.append(b)

    # ---------- resume: read the previous file BEFORE overwriting it ----------
    prev = _load_previous(out_path) if resume else {}
    already = {
        qid for qid, r in prev.items()
        if str(r.get("source", "")).startswith("llm")
        and (not run_signature or r.get("run_signature") == run_signature)
    }
    if already:
        print(f"resume: {len(already)} questions already solved by the LLM", flush=True)
    elif prev and run_signature:
        print("resume: previous output exists but has a different/missing run "
              "signature; its LLM answers will not be reused", flush=True)

    # ---------- step 2: rule baseline for everything, flush immediately ----------
    print(f"rule baseline over {len(bundles)} questions...", flush=True)
    rule_conf, rule_answers = {}, {}
    for b in bundles:
        if b.id in already:                     # keep the better LLM answer
            results[b.id] = prev[b.id]
            rule_conf[b.id] = 0.0
            continue
        r = _rule_result(b, encoder) if use_rule_fallback else None
        results[b.id] = r if r is not None else _empty_result(b, "rule found nothing")
        rule_conf[b.id] = r["detail_conf"] if r else 0.0
        if r is not None:
            rule_answers[b.id] = {"answer": r["answer"],
                                  "pandas_query": r["pandas_query"],
                                  "confidence": r["detail_conf"],
                                  "source": r["source"]}
    _flush(out_path, recs, results)
    print(f"  baseline written ({_srcs(recs, results)}) -> {out_path}", flush=True)

    if client.name == "none":
        print(f"codegen done (rule only): {_srcs(recs, results)} -> {out_path}")
        return

    # GPU budget targeting. A Kaggle session fits ~1000 questions; spending it
    # on questions the rule already answers confidently is waste, because
    # arbitration keeps the rule answer anyway when the two disagree.
    #   all   - every question (default, most thorough)
    #   empty - ONLY questions with no deterministic answer  (best value/hour)
    #   weak  - empty + rule answers flagged AMBIGUOUS / low confidence
    def _wanted(b) -> bool:
        if llm_ids is not None and int(b.id) not in llm_ids:
            return False
        if rule_first and rule_conf.get(b.id, 0) >= 90:
            return False
        if llm_target == "empty":
            return b.id not in rule_answers
        if llm_target == "weak":
            return b.id not in rule_answers or rule_conf.get(b.id, 0) < 78
        return True

    llm_todo = [b for b in bundles if b.id not in already and _wanted(b)]
    skipped = len(bundles) - len(llm_todo) - len(already)
    print(f"LLM queue: {len(llm_todo)} questions "
          f"(mode={llm_mode}, target={llm_target}, skipped {skipped}, "
          f"n={n_samples}, T={temperature})", flush=True)

    # ---------- step 3: chunked generation with checkpoints ----------
    chunk = max(1, checkpoint_every)
    n_chunks = (len(llm_todo) + chunk - 1) // chunk
    for ci in range(n_chunks):
        elapsed_before = time.time() - t_start
        if time_budget_s and elapsed_before >= time_budget_s:
            print(f"time budget {time_budget_s/60:.0f}min reached before chunk "
                  f"{ci+1}; keeping the complete checkpoint", flush=True)
            break
        part = llm_todo[ci * chunk:(ci + 1) * chunk]
        t0 = time.time()
        # keep the no-arg call path: test doubles implement prompt_messages()
        if llm_mode == "select":
            convs = [b.select_messages(encoder) for b in part]
        else:
            convs = [(b.prompt_messages(encoder) if encoder is not None
                      else b.prompt_messages()) for b in part]
        try:
            gens = client.chat_batch(convs, n=n_samples, temperature=temperature,
                                     max_tokens=max_tokens)
        except Exception as exc:
            # The rule baseline and all earlier chunks are already durable. Make
            # that guarantee explicit before propagating OOM/backend failures.
            backend_error = f"{type(exc).__name__}: {exc}"[:1000]
            for b in part:
                results[b.id]["llm_diagnostics"] = {
                    "requested": int(n_samples),
                    "returned": 0,
                    "statuses": {"backend_error": int(n_samples)},
                    "first_error": backend_error,
                    "sample_preview": [],
                }
                if results[b.id].get("source") == "none":
                    results[b.id]["detail"] = f"llm backend failed: {backend_error}"[:500]
            _flush(out_path, recs, results)
            print(f"[chunk {ci+1}/{n_chunks}] generation failed; checkpoint kept "
                  f"at {out_path}; {backend_error}", flush=True)
            raise
        if len(gens) != len(part):
            _flush(out_path, recs, results)
            raise RuntimeError(
                f"LLM returned {len(gens)} result groups for {len(part)} prompts"
            )
        debug_queue = []
        if llm_mode == "select":
            for b, samples in zip(part, gens):
                rec = _selection_result(b, samples, encoder)
                if rec is not None:
                    results[b.id] = _arbitrated(b, rule_answers.get(b.id), rec,
                                                int(rec.get("votes", 1)),
                                                max(1, len(samples)))
            _flush(out_path, recs, results)
            dt, elapsed = time.time() - t0, time.time() - t_start
            print(f"[chunk {ci+1}/{n_chunks}] {len(part)}q in {dt/60:.1f}min | "
                  f"elapsed {elapsed/60:.1f}min | {_srcs(recs, results)}", flush=True)
            if time_budget_s and elapsed + dt > time_budget_s:
                print("time budget reached -> stopping cleanly", flush=True)
                break
            continue

        for b, samples in zip(part, gens):
            sr = [(extract_code(s), None) for s in samples]
            sr = [(c, _run_validated(b, c)) for c, _ in sr]
            win = _vote(sr)
            if win:
                llm_rec = _final(b, win["answer"], win["code"], "llm",
                                 votes=win["votes"], n_ok=win["n_ok"],
                                 semantic=win.get("semantic"))
                results[b.id] = _arbitrated(b, rule_answers.get(b.id), llm_rec,
                                            win.get("votes", 1), n_samples)
            else:
                err = next((r["error"] for _c, r in sr if r["error"]), "no output")
                status_counts = Counter(r.get("status", "unknown") for _c, r in sr)
                diagnostics = {
                    "requested": int(n_samples),
                    "returned": len(samples),
                    "statuses": dict(status_counts),
                    "first_error": err,
                    "sample_preview": [str(sample)[:240] for sample in samples[:2]],
                }
                results[b.id]["llm_diagnostics"] = diagnostics
                if results[b.id].get("source") == "none":
                    results[b.id]["detail"] = (
                        f"llm failed: returned={len(samples)}/{n_samples}; "
                        f"statuses={dict(status_counts)}; first_error={err}"
                    )[:500]
                debug_queue.append((b, sr[0][0] if sr else "", err))

        # Preserve all successful round-1 answers before the optional debug
        # pass. A debug OOM must not discard useful work from this chunk.
        _flush(out_path, recs, results)

        for rnd in range(debug_rounds):
            if not debug_queue:
                break
            convs = [b.debug_messages(code, err) for b, code, err in debug_queue]
            try:
                gens = client.chat_batch(convs, n=1, temperature=0.2,
                                         max_tokens=max_tokens)
            except Exception:
                _flush(out_path, recs, results)
                print(f"[chunk {ci+1}/{n_chunks}] debug round {rnd+1} failed; "
                      "round-1 checkpoint kept", flush=True)
                raise
            if len(gens) != len(debug_queue):
                _flush(out_path, recs, results)
                raise RuntimeError(
                    f"LLM returned {len(gens)} debug groups for "
                    f"{len(debug_queue)} prompts"
                )
            nxt = []
            for (b, old_code, _err), samples in zip(debug_queue, gens):
                code = extract_code(samples[0]) if samples else ""
                r = _run_validated(b, code)
                if r["status"] == "ok":
                    llm_rec = _final(
                        b, round(r["value"], 2), code, "llm_debug",
                        semantic=r.get("semantic"),
                    )
                    results[b.id] = _arbitrated(b, rule_answers.get(b.id),
                                                llm_rec, 1, 1)
                else:
                    nxt.append((b, code or old_code, r["error"] or "no output"))
            debug_queue = nxt
        # questions the LLM could not fix keep their rule-baseline entry

        _flush(out_path, recs, results)
        dt, elapsed = time.time() - t0, time.time() - t_start
        eta = dt * (n_chunks - ci - 1)
        print(f"[chunk {ci+1}/{n_chunks}] {len(part)}q in {dt/60:.1f}min | "
              f"elapsed {elapsed/60:.1f}min | ETA {eta/60:.1f}min | "
              f"{_srcs(recs, results)}", flush=True)

        if time_budget_s and elapsed + dt > time_budget_s:
            print(f"time budget {time_budget_s/60:.0f}min reached -> stopping cleanly "
                  f"(file is complete & submittable; rerun with --resume to continue)",
                  flush=True)
            break

    _flush(out_path, recs, results)
    print(f"codegen done: {_srcs(recs, results)} -> {out_path}")


def _flush(out_path: Path, recs: list[dict], results: dict) -> None:
    """Write every question in input order; atomic-ish via temp file + replace."""
    ordered = [results[rec["id"]] for rec in recs if rec["id"] in results]
    tmp = Path(str(out_path) + ".tmp")
    write_jsonl(tmp, ordered)
    os.replace(tmp, out_path)


def _load_previous(out_path: Path) -> dict:
    if not Path(out_path).exists():
        return {}
    try:
        return {r["id"]: r for r in read_jsonl(out_path)}
    except Exception:  # noqa: BLE001 - corrupt/partial file must not kill the run
        return {}


def _srcs(recs: list[dict], results: dict) -> dict:
    return dict(Counter(results[r["id"]]["source"] for r in recs if r["id"] in results))


def _run_validated(b: QuestionBundle, code: str) -> dict:
    result = run_code(code, b.dfs)
    if result["status"] != "ok":
        return result
    check = validate_generated_answer(
        code, b.dfs.keys(), result["value"], route=getattr(b, "route", {}),
    )
    result["semantic"] = check.to_dict()
    if not check.ok:
        result["status"] = "semantic_error"
        result["error"] = "; ".join(check.errors)
    return result


def _selection_result(b: QuestionBundle, samples, encoder) -> dict | None:
    """Turn the model's JSON pick into a verified answer + generated query.

    The model never writes pandas here, so the three failure classes audited in
    submission #12 (missing column filter, missing unit conversion, regex
    patterns) cannot occur: selection.synthesize emits the expression.
    """
    cands = b.shortlist(encoder, top_n=12)
    if not cands:
        return None
    valid = []
    for text in samples:
        sel = parse_selection(text)
        if sel is None or sel.op == "none":
            continue
        requirements = b.route.get("evidence_requirements", [])
        if not selection_matches_route(sel, b.route, requirements):
            continue
        coverage = requirement_coverage(
            sel, cands, requirements,
        )
        if not coverage["complete"]:
            continue
        answer, query, err = synthesize(sel, cands, b.route)
        if err or answer is None:
            continue
        ex = _run_validated(b, query)          # the query must reproduce it
        if ex["status"] != "ok" or abs(ex["value"] - answer) > 0.011:
            continue
        key = (sel.op, tuple(sel.operands))
        valid.append((key, sel, answer, query, ex, coverage))
    if not valid:
        return None

    winner, votes = Counter(item[0] for item in valid).most_common(1)[0]
    # Self-consistency must mean agreement, not merely "at least one sample
    # parsed". For n=3 this accepts 2/3 or 3/3 and rejects a three-way split.
    if len(samples) > 1 and votes <= len(samples) // 2:
        return None
    _key, sel, answer, query, ex, coverage = next(
        item for item in valid if item[0] == winner
    )
    out = _final(
        b, round(ex["value"], 2), query, "llm_select",
        votes=votes, n_ok=len(valid), semantic=ex.get("semantic"),
    )
    out["detail"] = f"op={sel.op} operands={sel.operands} consensus={votes}/{len(samples)}"
    out["detail_conf"] = sel_conf(sel, cands, answer, b.route)
    out["selection_evidence"] = coverage
    return out


def _arbitrated(b: QuestionBundle, rule: dict | None, llm_rec: dict,
                votes: int, n_samples: int) -> dict:
    """Merge the deterministic answer with the LLM's, recording the decision.

    Self-consistency is folded into the LLM's confidence: an answer that won
    2/2 votes is trusted more than one that won 1/4.
    """
    llm_conf = 50.0 + 40.0 * (votes / max(1, n_samples))
    verdict = arbitrate(rule, {"answer": llm_rec["answer"],
                               "pandas_query": llm_rec["pandas_query"],
                               "confidence": llm_conf,
                               "source": llm_rec["source"]})
    if verdict is None or verdict.used == "llm":
        out = llm_rec
    else:
        out = _final(b, verdict.answer, verdict.pandas_query, verdict.source)
        out["detail_conf"] = verdict.confidence
    out["arbitration"] = {"reason": verdict.reason if verdict else "no verdict",
                          "used": verdict.used if verdict else "llm",
                          "rule_answer": (rule or {}).get("answer"),
                          "llm_answer": llm_rec["answer"],
                          "rule_conf": (rule or {}).get("confidence", 0.0),
                          "llm_conf": round(llm_conf, 1)}
    return out


def _rule_result(b: QuestionBundle, encoder=None) -> dict | None:
    """Deterministic answer: formula/composite solvers, then lookup rule."""
    fa = try_formula_answer(b.route, b.tables, encoder=encoder)
    if fa.ok:
        ex = _run_validated(b, fa.pandas_query)
        if ex["status"] == "ok":
            out = _final(b, round(ex["value"], 2), fa.pandas_query,
                         "rule_formula", semantic=ex.get("semantic"))
            out["detail"] = fa.detail
            out["detail_conf"] = fa.confidence
            return out

    if requires_formula_solver(b.route):
        return None

    ca = try_composite_answer(b.route, b.tables, encoder=encoder)
    if ca.ok:
        ex = _run_validated(b, ca.pandas_query)
        if ex["status"] == "ok":
            out = _final(b, round(ex["value"], 2), ca.pandas_query,
                         "rule_composite", semantic=ex.get("semantic"))
            out["detail"] = ca.detail
            out["detail_conf"] = ca.confidence
            return out

    ra = try_rule_answer(b.route, b.tables)
    if not ra.ok:
        return None
    ex = _run_validated(b, ra.pandas_query)
    if ex["status"] != "ok":
        return None
    ans = round(ex["value"], 2)  # answer == what the query reproduces
    out = _final(b, ans, ra.pandas_query, "rule", semantic=ex.get("semantic"))
    out["detail"] = ra.detail
    out["detail_conf"] = ra.confidence
    return out


def _final(b: QuestionBundle, answer: float, code: str, source: str,
           votes: int = 1, n_ok: int = 1, semantic: dict | None = None) -> dict:
    # The grader EVALUATES pandas_query as an expression: a multi-line script is
    # a SyntaxError == crash (leaderboard-confirmed, submission #6). Inline the
    # straight-line script into one expression right where it is produced, so
    # every downstream consumer (checkpoints, resume, submission) is already
    # in the graded form. used_vars is derived from the ORIGINAL code so that
    # evidence still covers every DataFrame the model touched.
    used = b.used_vars(code)
    query, inline_err = try_to_expression(code)
    return {"id": b.id, "question": b.question, "answer": float(answer),
            "pandas_query": query, "used_vars": used,
            "inline_error": inline_err or "",
            "status": "ok", "source": source, "votes": votes, "n_ok": n_ok,
            "detail": "", "detail_conf": 0.0,
            "semantic": semantic or {},
            "run_signature": getattr(b, "run_signature", "")}


def _empty_result(b, reason: str) -> dict:
    return {"id": b.id, "question": b.question, "answer": 0.0,
            "pandas_query": "0.0",
            "used_vars": ([{"var": t["var"], "report_id": t["report_id"],
                            "table_pos": t["table_pos"]} for t in b.tables[:1]]
                          if getattr(b, "tables", None) else []),
            "status": "failed", "source": "none", "votes": 0, "n_ok": 0,
            "detail": reason, "detail_conf": 0.0,
            "semantic": {},
            "run_signature": getattr(b, "run_signature", "")}
