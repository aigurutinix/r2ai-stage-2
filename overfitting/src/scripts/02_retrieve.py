"""Step 2 (local, CPU): structured lookup + BM25 retrieval -> retrieval.jsonl

  python scripts/02_retrieve.py
  python scripts/02_retrieve.py --limit 30            # smoke test
  python scripts/02_retrieve.py --questions artifacts/validation/val_questions.jsonl \
         --out artifacts/val_retrieval.jsonl          # on the synthetic val set
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.retrieval.retrieve import run_retrieval
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(config.QUESTIONS_JSONL))
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    ap.add_argument("--out", default=str(config.RETRIEVAL_JSONL))
    ap.add_argument("--depth", type=int, default=config.RETRIEVE_DEPTH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="",
                    help="comma-separated question IDs to refresh")
    ap.add_argument("--ids-file", default="",
                    help="optional text/JSON list of question IDs to refresh")
    ap.add_argument("--base", default="",
                    help="optional retrieval JSONL whose untouched rows are preserved")
    ap.add_argument("--row-rerank", action="store_true",
                    help="rerank table candidates with row-level shortlist scores")
    ap.add_argument("--row-score-weight", type=float, default=0.18,
                    help="weight for row-level shortlist score when --row-rerank is set")
    args = ap.parse_args()

    question_ids = {
        int(value.strip()) for value in args.ids.split(",") if value.strip()
    }
    if args.ids_file:
        raw = Path(args.ids_file).read_text(encoding="utf-8").strip()
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = raw.replace(",", " ").split()
        if isinstance(values, dict):
            values = values.get("ids", [])
        if not isinstance(values, list):
            raise SystemExit("--ids-file must contain a JSON list/object or text ids")
        question_ids.update(int(value) for value in values)

    run_retrieval(Path(args.questions), Path(args.store_dir), Path(args.code_stock),
                  Path(args.out), args.depth, args.limit,
                  row_rerank=args.row_rerank,
                  row_score_weight=args.row_score_weight,
                  question_ids=question_ids or None,
                  base_path=Path(args.base) if args.base else None)


if __name__ == "__main__":
    main()
