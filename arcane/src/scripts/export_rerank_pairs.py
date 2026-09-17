#!/usr/bin/env python3
"""Export question/table candidate pairs for cross-encoder reranking off-box.

Reranking is the one stage the teams above us all run and we do not. A
cross-encoder scores a question against a table representation, so the pairs have
to be materialized where the ranking is known and scored where a GPU is: this
writes them, a notebook scores them, and apply_rerank_scores.py fuses the result.

Only the ordering of already-retrieved candidates changes. The document gate, the
candidate set, and the table IDs are untouched, so a bad reranker can be dropped
by ignoring the scores file.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


from docs import load_companies, load_reports as load_doc_reports, parse_question, required_report_years, retrieve_docs
from vifinqa.answers import fold
from vifinqa.jsonl import load_jsonl
from vifinqa.rerank import INVENTORY, table_representation
from vifinqa.retrieval import load_reports, retrieve_rows, score_bm25, unicode_tokenize


def canonical_scope(scope: str) -> str:
    """Match the document gate's treatment of aggregated reports."""
    return "consolidated" if scope == "aggregated" else scope


def dense_report_ids(by_id: dict, selected_docs: list[str], policy: str) -> set[str] | None:
    """Reports an embedding candidate may come from under one explicit policy."""
    if policy == "any":
        return None
    if policy == "selected-reports":
        return set(selected_docs)
    selected = [by_id[report_id] for report_id in selected_docs if report_id in by_id]
    triples = {
        (report.identity.ticker, report.identity.year, canonical_scope(report.identity.scope))
        for report in selected
    }
    return {
        report_id for report_id, report in by_id.items()
        if (report.identity.ticker, report.identity.year, canonical_scope(report.identity.scope)) in triples
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--questions", type=Path, help="defaults to the full question set")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "rerank" / "pairs.jsonl")
    parser.add_argument("--depth", type=int, default=50, help="candidates per question to rerank")
    parser.add_argument("--table-mode", default="report-coverage")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inventory", type=int, default=INVENTORY, help="characters of the table's line-item list to include; 0 keeps the matched row alone")
    parser.add_argument(
        "--hierarchy", action="store_true",
        help="include source-derived row and column paths in every reranker document",
    )
    parser.add_argument(
        "--table-family-prior", action="store_true",
        help="use the opt-in multi-label statement-family retrieval view",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dense", type=Path, help="directory from kaggle/embed_tables.py; adds dense candidates the report gate never offered")
    parser.add_argument("--dense-depth", type=int, default=100, help="dense candidates per question; deduplication against the sparse pool removes some, so the pool is smaller than depth + dense-depth")
    parser.add_argument(
        "--dense-policy", choices=("same-triple", "selected-reports", "any"),
        default="same-triple",
        help="metadata boundary for dense candidates; same-triple retains alternate report files "
             "for the gated ticker/year/scope while rejecting other scopes and issuers",
    )
    parser.add_argument("--dense-tables", type=Path, help="where to record which report each dense candidate came from; run.py needs it to materialize a table its own gate never selected (defaults beside --output)")
    args = parser.parse_args()

    dense = load_dense(args.dense) if args.dense else None
    from_report: dict[str, str] = {}

    path = args.questions or args.dataset_root / "questions" / "questions.jsonl"
    questions = load_jsonl(path)
    if args.limit:
        questions = questions[: args.limit]

    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")
    table_reports = load_reports(args.dataset_root)
    by_id = {report.identity.report_id: report for report in table_reports}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for number, question in enumerate(questions, 1):
            parsed = parse_question(question["question"], companies)
            if not parsed.tickers and parsed.candidate_tickers:
                parsed.tickers = parsed.candidate_tickers[:1]
            docs, _ = retrieve_docs(parsed, doc_reports)
            result = retrieve_rows(
                question["question"],
                {
                    "tickers": parsed.tickers,
                    "years": parsed.years,
                    "slot_years": required_report_years(parsed),
                    "scope": parsed.scope,
                },
                table_reports,
                top_k=args.depth,
                report_ids=docs,
                mode=args.table_mode,
                hierarchy=args.hierarchy,
                family_prior=args.table_family_prior,
            )
            candidates = []
            for rank, table in enumerate(result["tables"], 1):
                report = by_id[table["report_id"]]
                tables = report_tables_cached(report)
                ordinal = next((n for n, item in enumerate(tables) if item.table_id == table["table_id"]), None)
                if ordinal is None:
                    continue
                candidates.append({
                    "table_id": table["table_id"],
                    "sparse_rank": rank,
                    "text": table_representation(
                        tables[ordinal], table["row_index"], inventory=args.inventory,
                        identity=report.identity, hierarchy=args.hierarchy,
                    ),
                })
            if dense is not None:
                dense_ranked = dense_candidates(
                    dense, question["id"], by_id, args.dense_depth,
                    dense_report_ids(by_id, docs, args.dense_policy),
                    set(unicode_tokenize(question["question"])), args.inventory,
                    args.hierarchy,
                )
                from_report.update(
                    (item["table_id"], item["report_id"])
                    for item in dense_ranked
                )
                candidates = merge_dense_candidates(candidates, dense_ranked)
            identifier = question.get("id", question.get("task_id"))
            if identifier is None:
                raise ValueError("each question needs an id or task_id")
            handle.write(json.dumps({
                "id": identifier,
                "question": question["question"],
                # Which reports the gate selected. Kept for provenance, and no
                # longer a guarantee: with --dense a candidate can come from a
                # report outside this list, which is why the candidate text now
                # opens with the filing it belongs to.
                "selected_docs": docs,
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")
            pairs += len(candidates)
            if args.progress_every and number % args.progress_every == 0:
                print(f"exported {number}/{len(questions)} questions, {pairs} pairs", flush=True, file=sys.stderr)

    summary = {
        "output": str(args.output),
        "questions": len(questions),
        "pairs": pairs,
        "megabytes": round(args.output.stat().st_size / 1e6, 1),
    }
    if dense is not None:
        # run.py re-runs its own gated retrieval and then reorders what that
        # returned, so a dense candidate from an ungated report would be dropped
        # on the way to the submission unless run.py is told where to find it.
        sidecar = args.dense_tables or args.output.with_suffix(".dense_tables.json")
        sidecar.write_text(json.dumps(from_report, ensure_ascii=False), encoding="utf-8")
        summary["dense_tables"] = str(sidecar)
        summary["dense_candidates"] = len(from_report)
        summary["dense_policy"] = args.dense_policy
    summary["hierarchy"] = args.hierarchy
    print(json.dumps(summary, ensure_ascii=False, indent=2))


_TABLES_BY_PATH: dict[str, list] = {}


def report_tables_cached(report):
    """Read each report's tables once while exporting its candidate rows."""
    from vifinqa.retrieval import report_tables

    path = str(report.path)
    if path not in _TABLES_BY_PATH:
        _TABLES_BY_PATH[path] = report_tables(path, report.identity)
    return _TABLES_BY_PATH[path]


def load_dense(directory: Path) -> dict:
    """Read the four files kaggle/embed_tables.py wrote."""
    import numpy as np

    # Convert once. Converting the 267 MB matrix inside every question made a
    # full export spend most of its time copying the same vectors 1,012 times.
    tables = np.load(directory / "table_vectors.npy").astype(np.float32)
    identifiers = json.loads((directory / "table_ids.json").read_text("utf-8"))
    if len(identifiers) != len(tables):
        raise ValueError(f"{len(identifiers)} ids against {len(tables)} vectors")
    questions = np.load(directory / "question_vectors.npy").astype(np.float32)
    order = {identity: row for row, identity
             in enumerate(json.loads((directory / "question_ids.json").read_text("utf-8")))}
    report_rows: dict[str, list[int]] = {}
    for index, (report_id, _) in enumerate(identifiers):
        report_rows.setdefault(report_id, []).append(index)
    print(f"dense index: {len(tables)} tables, {tables.shape[1]} dimensions",
          file=sys.stderr)
    return {
        "tables": tables, "ids": identifiers, "questions": questions,
        "order": order, "report_rows": report_rows,
    }


def row_for(table, question_tokens: set) -> int:
    """The row a candidate is shown through.

    A dense candidate has no matched row, because nothing matched — the whole
    table was scored at once. It gets the row the sparse ranker would have
    picked, so that a dense candidate and a sparse one reach the reranker in the
    same shape and the arms differ in what was retrieved rather than in how it
    was rendered.
    """
    if not table.rows:
        return 0
    scored = score_bm25(question_tokens,
                        [unicode_tokenize(" ".join(row)) for row in table.rows])
    return max(range(len(table.rows)), key=lambda index: scored[index])


def merge_dense_candidates(sparse: list[dict], dense: list[dict]) -> list[dict]:
    """Attach dense ranks to overlaps and append genuinely new candidates."""
    merged = [dict(candidate) for candidate in sparse]
    by_table = {candidate["table_id"]: candidate for candidate in merged}
    for candidate in dense:
        existing = by_table.get(candidate["table_id"])
        if existing is None:
            added = dict(candidate)
            merged.append(added)
            by_table[added["table_id"]] = added
            continue
        existing["dense_rank"] = candidate["dense_rank"]
        existing["dense_score"] = candidate["dense_score"]
    return merged


def dense_candidates(dense: dict, question_id, by_id: dict,
                     depth: int, allowed_reports: set[str] | None = None,
                     question_tokens: set = frozenset(),
                     inventory: int = INVENTORY,
                     hierarchy: bool = False) -> list[dict]:
    """The nearest tables, including candidates also returned by sparse.

    Both vector sets are unit length, so the dot product is the cosine. The
    corpus is small enough that this is one matrix-vector product against every
    table rather than an approximate index — 146k by 1024 in float16 is 300 MB
    and the exact answer costs milliseconds, so there is nothing for an ANN
    structure to buy and one fewer approximation to explain.

    Each carries a dense rank and no sparse rank. Which of the two the ordering
    reads is the arm: the sparse arm ranks only what BM25 returned, the dense arm
    only what the index returned, and the hybrid arm sums both and so rewards the
    tables they agree on. Three systems out of one scored pool.
    """
    import numpy as np

    row = dense["order"].get(question_id)
    if row is None:
        return []
    if allowed_reports is None:
        pool = np.arange(len(dense["ids"]))
    else:
        # A set's iteration order changes with PYTHONHASHSEED. Keep the global
        # vector-row order so equal cosine scores cannot swap candidates across
        # runs (or move a duplicate filing across the depth boundary).
        pool = np.fromiter(
            (index for report_id in sorted(allowed_reports)
             for index in dense["report_rows"].get(report_id, ())),
            dtype=np.int64,
        )
        pool.sort()
    if not len(pool):
        return []
    scores = dense["tables"][pool] @ dense["questions"][row]
    added = []
    # Invalid source entries are rare, but inspect beyond the requested depth so
    # one broken table cannot shorten the arm.
    # Global vector row is the deterministic tie-breaker for equal similarities.
    for position in np.lexsort((pool, -scores))[: depth * 4]:
        index = pool[position]
        report_id, table_id = dense["ids"][index]
        if (report_id not in by_id
                or allowed_reports is not None and report_id not in allowed_reports):
            continue
        tables = report_tables_cached(by_id[report_id])
        ordinal = next((n for n, item in enumerate(tables) if item.table_id == table_id), None)
        if ordinal is None:
            continue
        added.append({
            "table_id": table_id,
            "report_id": report_id,
            # Sparse rank is attached later when both retrievers found the table.
            "sparse_rank": None,
            "dense_rank": len(added) + 1,
            "dense_score": round(float(scores[position]), 4),
            "text": table_representation(
                tables[ordinal], row_for(tables[ordinal], question_tokens),
                inventory=inventory,
                identity=by_id[report_id].identity,
                hierarchy=hierarchy,
            ),
        })
        if len(added) >= depth:
            break
    return added


if __name__ == "__main__":
    main()
