"""Assemble the final submission: results.json + data/*.csv -> submission.zip

Format (per BTC spec):
[
  {"id", "question", "answer", "relevant_docs": [report_id],
   "relevant_tables": ["report_id|pos"], "evidence": [{"variable","csv_path"}],
   "pandas_query"}
]
- relevant_tables = top SUBMISSION_K retrieval candidates (F2 -> generous k)
- evidence        = only the tables the pandas_query actually uses
- csv files       = normalized long-format tables (see retrieval/serialize.py)
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path

import pandas as pd

from ..codegen.executor import run_code
from ..codegen.to_expression import try_to_expression
from ..codegen.semantic import all_dataframe_refs
from ..config import SUBMISSION_K, TABLE_POS_BASE, TABLE_POS_MODE
from ..extraction.build_store import Store
from ..retrieval.serialize import tidy_csv_text
from ..utils.io import read_jsonl, write_json, ensure_dir


class _PositionMapper:
    """(report_id, table_pos:order) -> submitted position.

    mode='line' : position = 1-based line number where <table> starts in the
                  OCR .txt — THE OFFICIAL SCHEME, confirmed by the organizers
                  (requires a store built with the line_no column).
    mode='order': legacy debugging mode (table_pos + pos_base).
    """

    def __init__(self, store: Store, mode: str, pos_base: int):
        assert mode in ("order", "line"), mode
        self.store, self.mode, self.pos_base = store, mode, pos_base
        self._maps: dict[str, dict] = {}

    def __call__(self, report_id: str, table_pos: int) -> int:
        if self.mode == "order":
            return int(table_pos) + self.pos_base
        ticker = report_id.split("_")[0]
        if ticker not in self._maps:
            tdf = self.store._load("tables", ticker)
            if "line_no" not in tdf.columns:
                raise SystemExit(
                    "store has no line_no column - rebuild it first: "
                    "python scripts/01_build_store.py")
            self._maps[ticker] = {(r.report_id, int(r.table_pos)): int(r.line_no)
                                  for r in tdf.itertuples()}
        key = (report_id, int(table_pos))
        if key not in self._maps[ticker]:
            raise KeyError(
                f"cannot map evidence table to official line number: {report_id}|{table_pos}"
            )
        return self._maps[ticker][key]


def build_submission(retrieval_path: Path, codegen_path: Path, store_dir: Path,
                     out_dir: Path, sub_k: int = SUBMISSION_K,
                     pos_base: int = TABLE_POS_BASE, pos_mode: str = TABLE_POS_MODE,
                     questions_path: Path | None = None, expand_docs: bool = False,
                     offline_eval: bool = False,
                     json_name: str = "results.json") -> Path:
    out_dir = ensure_dir(out_dir)
    data_dir = ensure_dir(out_dir / "data")
    store = Store(store_dir, cache_size=6)
    position = _PositionMapper(store, pos_mode, pos_base)

    retr = _read_unique_by_id(retrieval_path, "retrieval")
    # restrict to the official test question ids when provided (gold=506 != 1012)
    if questions_path:
        keep = {q["id"] for q in read_jsonl(questions_path)}
        missing = keep - set(retr)
        if missing:
            print(f"[WARN] {len(missing)} official question ids absent from "
                  f"retrieval output: {sorted(missing)[:10]}...")
        retr = {qid: r for qid, r in retr.items() if qid in keep}

    if not codegen_path or not Path(codegen_path).exists():
        raise FileNotFoundError(f"codegen results not found: {codegen_path}")
    codegen = _read_unique_by_id(codegen_path, "codegen")
    missing_codegen = set(retr) - set(codegen)
    if missing_codegen:
        sample = sorted(missing_codegen)[:10]
        raise ValueError(
            f"codegen is incomplete: missing {len(missing_codegen)} question ids; "
            f"first ids={sample}"
        )

    written_csv: set[str] = set()
    entries = []
    for qid, rec in sorted(retr.items()):
        cands = rec["candidates"][:sub_k]
        rel_tables, rel_docs, seen_docs = [], [], set()
        for c in cands:
            rel_tables.append(
                f"{c['report_id']}|{position(c['report_id'], c['table_pos'])}")
            if c["report_id"] not in seen_docs:
                seen_docs.add(c["report_id"])
                rel_docs.append(c["report_id"])
        if expand_docs:
            # recall-oriented (F2): also list the sibling doc_type report and
            # the year+1 report (its prior-year column carries the same figure)
            route = rec.get("route", {})
            for t in route.get("tickers", []):
                for y in route.get("years", []):
                    for dt in ("consolidated", "separate"):
                        for yy in (y, y + 1):
                            rid = store.report_index.get((t, yy, dt))
                            if rid and rid not in seen_docs:
                                seen_docs.add(rid)
                                rel_docs.append(rid)

        cg = codegen[qid]
        evidence, pandas_query, answer = [], "0.0", 0.0
        answer = float(cg.get("answer", 0.0) or 0.0)
        if not math.isfinite(answer):
            raise ValueError(f"question {qid}: answer is not finite: {answer!r}")
        pandas_query = _to_expression(cg.get("pandas_query") or "0.0")
        used_vars = cg.get("used_vars", [])
        _validate_codegen_evidence(qid, pandas_query, used_vars, cg.get("status"))
        seen_vars = set()
        for uv in used_vars:
            var = uv["var"]
            if var in seen_vars:
                raise ValueError(f"question {qid}: duplicate evidence variable {var}")
            seen_vars.add(var)
            pos_out = position(uv["report_id"], uv["table_pos"])
            table_ref = f"{uv['report_id']}|{pos_out}"
            # Execution evidence must also be declared as a relevant table/doc.
            # This matters when CODEGEN_K > SUBMISSION_K.
            if table_ref not in rel_tables:
                rel_tables.append(table_ref)
            if uv["report_id"] not in seen_docs:
                seen_docs.add(uv["report_id"])
                rel_docs.append(uv["report_id"])
            csv_name = f"{uv['report_id']}_table_{pos_out}.csv"
            evidence.append({"variable": var, "csv_path": f"data/{csv_name}"})
            if csv_name not in written_csv:
                _write_table_csv(store, uv["report_id"], uv["table_pos"],
                                 data_dir / csv_name)
                written_csv.add(csv_name)

        entries.append({
            "id": qid,
            "question": rec["question"],
            "answer": round(answer, 2),
            "relevant_docs": rel_docs,
            "relevant_tables": rel_tables,
            "evidence": evidence,
            "pandas_query": pandas_query,
        })

    json_path = out_dir / json_name
    _validate_expression_form(entries)
    _validate_replay(entries, out_dir)
    write_json(json_path, entries)

    _warn_if_not_official(entries, offline_eval, out_dir)

    zip_path = out_dir / ("OFFLINE_EVAL_DO_NOT_UPLOAD.zip" if offline_eval
                          else "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, json_name)
        for name in sorted(written_csv):
            z.write(data_dir / name, f"data/{name}")
    _validate_zip_layout(zip_path, json_name, written_csv)
    print(f"submission: {len(entries)} entries, {len(written_csv)} csv files "
          f"(pos_mode={pos_mode}) -> {zip_path}")
    return zip_path


def _warn_if_not_official(entries: list[dict], offline_eval: bool, out_dir: Path) -> None:
    """Guard against uploading a synthetic-eval submission to the leaderboard.

    The offline eval suite invents its own question ids (1..N) and its own
    question text. The grader matches by id, so uploading it scores 0.0 on every
    metric AND burns one of the daily submission slots. This check compares the
    built ids/questions with the official questions.jsonl and shouts when they
    do not line up.
    """
    from ..config import QUESTIONS_JSONL
    marker = out_dir / "DO_NOT_UPLOAD.txt"
    try:
        official = {q["id"]: q["question"] for q in read_jsonl(QUESTIONS_JSONL)}
    except Exception:  # noqa: BLE001 - dataset may be absent on a worker box
        official = {}
    mismatch = 0
    if official:
        for e in entries:
            q = official.get(e["id"])
            if q is None or q.strip() != str(e.get("question", "")).strip():
                mismatch += 1
    synthetic = offline_eval or (official and mismatch > len(entries) * 0.5)
    if synthetic:
        marker.write_text(
            "This folder was built from a SYNTHETIC offline-eval question set.\n"
            "Uploading it to the leaderboard scores 0.0 and wastes a submission "
            "slot. Use it only with scripts/07_evaluate.py.\n", encoding="utf-8")
        stale = out_dir / "submission.zip"
        if stale.exists():      # a previous run left an uploadable-looking file
            try:
                stale.unlink()
            except OSError:
                pass
        print("=" * 72)
        print("[STOP] This submission does NOT match the official questions "
              f"({mismatch}/{len(entries)} differ).")
        print("       It is an OFFLINE EVAL artifact - do NOT upload it.")
        print("=" * 72)
    elif marker.exists():
        marker.unlink()


def _to_expression(code: str) -> str:
    """`pandas_query` MUST be a single expression.

    LEADERBOARD-CONFIRMED (submission #6): the grader evaluates the query as an
    expression. Multi-line scripts raise SyntaxError and are scored as crashes —
    233 such queries cut EXECUTION_ACCURACY from 0.085 to 0.0613 even though
    ANSWER_ACCURACY rose. So every script is inlined into one expression here;
    the value is re-verified against `answer` by _validate_replay afterwards.
    """
    q = (code or "").strip()
    if not q:
        return "0.0"
    if "\n" not in q and q.startswith("answer"):
        head, sep, rhs = q.partition("=")
        if sep and "==" not in head:
            return rhs.strip()
    expr, err = try_to_expression(q)
    if err:
        _EXPR_FAILURES.append(err)
    return expr


_EXPR_FAILURES: list[str] = []


def _validate_expression_form(entries: list[dict]) -> None:
    """Loud check: every query must compile in 'eval' mode (grader semantics)."""
    bad = []
    for e in entries:
        try:
            compile(e["pandas_query"], "<q>", "eval")
        except SyntaxError:
            bad.append(e["id"])
    if bad:
        print(f"[WARN] {len(bad)} pandas_query are NOT single expressions and will "
              f"be scored as crashes by the grader: ids={bad[:10]}"
              f"{'...' if len(bad) > 10 else ''}")
    else:
        print(f"expression-form check: all {len(entries)} queries eval-compilable")


def _write_table_csv(store: Store, report_id: str, table_pos: int, path: Path) -> None:
    ticker = report_id.split("_")[0]
    tdf = store.tables_of(ticker, [report_id])
    row = tdf[tdf.table_pos == table_pos]
    if not len(row):
        raise KeyError(f"missing evidence table in store: {report_id}|{table_pos}")
    path.write_text(tidy_csv_text(row.iloc[0].to_dict()), encoding="utf-8")


def _read_unique_by_id(path: Path, label: str) -> dict[int, dict]:
    rows = read_jsonl(path)
    out: dict[int, dict] = {}
    duplicates = []
    for row in rows:
        if "id" not in row:
            raise ValueError(f"{label} row has no id: {row}")
        qid = int(row["id"])
        if qid in out:
            duplicates.append(qid)
        out[qid] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate ids: {sorted(set(duplicates))[:10]}")
    return out


def _validate_codegen_evidence(qid: int, code: str, used_vars: list[dict], status) -> None:
    try:
        refs = all_dataframe_refs(code)
    except SyntaxError as exc:
        raise ValueError(
            f"question {qid}: pandas_query has invalid syntax: {exc.msg}"
        ) from exc
    evidence_vars = {str(uv.get("var", "")) for uv in used_vars}
    missing = refs - evidence_vars
    if missing:
        raise ValueError(
            f"question {qid}: pandas_query references variables without evidence: "
            f"{sorted(missing)}"
        )
    if status == "ok" and not refs:
        raise ValueError(
            f"question {qid}: successful pandas_query does not reference any dataframe"
        )


def _validate_zip_layout(zip_path: Path, json_name: str, csv_names: set[str]) -> None:
    expected = {json_name, *(f"data/{name}" for name in csv_names)}
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    if names != expected:
        missing, extra = expected - names, names - expected
        raise ValueError(
            f"invalid submission zip layout: missing={sorted(missing)[:5]} "
            f"extra={sorted(extra)[:5]}"
        )


def _validate_replay(entries: list[dict], out_dir: Path) -> None:
    """Re-run every submitted query against exactly its submitted CSV files."""
    failures = []
    for entry in entries:
        dfs = {}
        for ev in entry["evidence"]:
            csv_path = out_dir / Path(ev["csv_path"])
            if not csv_path.is_file():
                failures.append((entry["id"], f"missing {ev['csv_path']}"))
                continue
            dfs[ev["variable"]] = pd.read_csv(csv_path)
        result = run_code(entry["pandas_query"], dfs)
        if result["status"] != "ok":
            failures.append((entry["id"], result["error"] or result["status"]))
            continue
        if not math.isclose(
            round(float(result["value"]), 2), float(entry["answer"]),
            rel_tol=0.0, abs_tol=1e-9,
        ):
            failures.append(
                (entry["id"], f"replay={result['value']} answer={entry['answer']}")
            )
    if failures:
        raise ValueError(
            f"submission replay failed for {len(failures)} questions; "
            f"first={failures[:5]}"
        )
