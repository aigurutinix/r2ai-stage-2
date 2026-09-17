"""Evaluate lexical/semantic retrieval on source-audited Easy questions."""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from road2ai_vifinqa.corpus import Corpus, load_questions  # noqa: E402
from road2ai_vifinqa.easy_solver import EASY_AUDITED_OVERRIDES  # noqa: E402
from road2ai_vifinqa.retrieval import (  # noqa: E402
    STOPWORDS,
    _idf_weights,
    metric_phrase,
    retrieve_rows,
    retrieve_rows_cascade,
    retrieve_tables,
)
from road2ai_vifinqa.text import fold_text  # noqa: E402


def _legacy_rows(corpus: Corpus, question: str) -> list[tuple[float, str, int, int]]:
    documents = corpus.documents_for_question(question, include_prior=False)
    rows = corpus.rows_for_documents(documents)
    years = set(corpus.infer_years(question))
    folded_question = fold_text(question)
    phrase = metric_phrase(question, tickers=corpus.infer_tickers(question))
    qtokens = set(phrase.split()) or (set(fold_text(question).split()) - STOPWORDS)
    weights = _idf_weights(rows, qtokens)
    doc_by_id = {doc.doc_id: doc for doc in documents}
    table_cache = {}
    result: list[tuple[float, str, int, int]] = []
    for row in rows:
        rtokens = set(row.folded_text.split())
        overlap = qtokens & rtokens
        if not overlap:
            continue
        weighted_recall = sum(weights[token] for token in overlap) / max(sum(weights.values()), 1e-9)
        precision = len(overlap) / max(len(rtokens & (qtokens | STOPWORDS)), 1)
        sequence = SequenceMatcher(None, phrase, row.folded_text).ratio()
        exact = 1.0 if phrase and phrase in row.folded_text else 0.0
        short_bonus = 1.0 / (1.0 + max(0, len(rtokens) - len(qtokens)) / 8.0)
        score = 7.0 * weighted_recall + 1.5 * precision + 2.5 * sequence + 5.0 * exact + short_bonus
        key = (row.doc_id, row.table_id)
        if key not in table_cache:
            table_cache[key] = corpus.table(*key)
        table = table_cache[key]
        context_tokens = set(fold_text(table.context).split())
        folded_context = fold_text(table.context)
        score += 2.2 * len(context_tokens & qtokens) / max(len(qtokens), 1)
        if any(marker in folded_question for marker in ("cuoi nam", "den ngay", "vao ngay")):
            if any(marker in folded_context for marker in ("bang can doi", "bao cao tinh hinh tai chinh")):
                score += 2.4
        hit_year = doc_by_id[row.doc_id].report_year
        score += 1.5 if hit_year in years else (-0.5 if years else 0.0)
        result.append((score, row.doc_id, row.table_id, row.row_idx))
    return sorted(result, key=lambda item: (-item[0], doc_by_id[item[1]].report_year, item[2], item[3]))


def _rank(sequence: list[tuple[str, int, int]], gold: set[tuple[str, int, int]]) -> int | None:
    return next((idx for idx, item in enumerate(sequence, 1) if item in gold), None)


def benchmark(index_path: Path) -> dict[str, object]:
    questions = {int(row["id"]): str(row["question"]) for row in load_questions()}
    records: list[dict[str, object]] = []
    with Corpus(index_path) as corpus:
        for qid, (_operation, coordinates, _reason) in sorted(EASY_AUDITED_OVERRIDES.items()):
            gold = {(doc, table, row) for doc, table, row, _col in coordinates}
            question = questions[qid]
            legacy = _legacy_rows(corpus, question)
            semantic = retrieve_rows(corpus, question, limit=200, semantic=True)
            cascade = retrieve_rows_cascade(corpus, question, table_limit=20, row_limit=200)
            tables = retrieve_tables(corpus, question, limit=50)
            legacy_rank = _rank([(doc, table, row) for _score, doc, table, row in legacy], gold)
            semantic_rank = _rank([(hit.row.doc_id, hit.row.table_id, hit.row.row_idx) for hit in semantic], gold)
            cascade_rank = _rank(
                [(hit.row.doc_id, hit.row.table_id, hit.row.row_idx) for hit in cascade],
                gold,
            )
            gold_tables = {(doc, table) for doc, table, _row in gold}
            table_rank = next(
                (
                    idx
                    for idx, hit in enumerate(tables, 1)
                    if (hit.table.doc_id, hit.table.table_id) in gold_tables
                ),
                None,
            )
            records.append(
                {
                    "id": qid,
                    "legacy_row_rank": legacy_rank,
                    "semantic_row_rank": semantic_rank,
                    "cascade_row_rank": cascade_rank,
                    "semantic_table_rank": table_rank,
                }
            )

    def metrics(field: str, cutoffs: tuple[int, ...]) -> dict[str, object]:
        ranks = [item[field] for item in records]
        finite = [int(value) for value in ranks if value is not None]
        return {
            **{f"recall_at_{cutoff}": sum(value is not None and int(value) <= cutoff for value in ranks) / len(ranks) for cutoff in cutoffs},
            "mrr": sum(1.0 / value for value in finite) / len(ranks),
            "missing": len(ranks) - len(finite),
        }

    return {
        "schema_version": 1,
        "questions": len(records),
        "legacy_rows": metrics("legacy_row_rank", (1, 5, 10, 20, 50)),
        "semantic_rows": metrics("semantic_row_rank", (1, 5, 10, 20, 50)),
        "cascade_rows": metrics("cascade_row_rank", (1, 5, 10, 20, 50)),
        "semantic_tables": metrics("semantic_table_rank", (1, 3, 5, 10, 20)),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.index)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "questions",
                    "legacy_rows",
                    "semantic_rows",
                    "cascade_rows",
                    "semantic_tables",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
