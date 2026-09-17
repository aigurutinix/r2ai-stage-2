"""Step 2 (local, CPU): build the multi-class offline evaluation suite.

  python scripts/09_gen_eval_suite.py --per-class 60

!!!! NEVER UPLOAD THE EVAL SUBMISSION TO THE LEADERBOARD !!!!
The questions here are SYNTHETIC with their own ids (1..N). The grader matches
by id, so submitting this scores 0.0 on every metric and burns a daily slot.
Always pass --offline-eval to 05_build_submission.py: the zip is then named
OFFLINE_EVAL_DO_NOT_UPLOAD.zip and a DO_NOT_UPLOAD.txt marker is written.

Then run the pipeline against it and score per class:

  python scripts/02_retrieve.py --questions artifacts/eval/eval_questions.jsonl \
      --out artifacts/eval/eval_retrieval.jsonl
  python scripts/03_rule_baseline.py --retrieval artifacts/eval/eval_retrieval.jsonl \
      --out artifacts/eval/eval_codegen.jsonl
  python scripts/05_build_submission.py --retrieval artifacts/eval/eval_retrieval.jsonl \
      --codegen artifacts/eval/eval_codegen.jsonl --out-dir artifacts/eval/eval_submission \
      --offline-eval
  python scripts/07_evaluate.py --submission artifacts/eval/eval_submission \
      --gold artifacts/eval/eval_gold.json --by-class
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.validation.gen_multiclass import generate
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    ap.add_argument("--out-dir", default=str(config.ART_DIR / "eval"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--max-tickers", type=int, default=0,
                    help="limit the scan for a quick smoke run")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    generate(Path(args.store_dir), Path(args.code_stock), Path(args.out_dir),
             per_class=args.per_class, seed=args.seed, max_tickers=args.max_tickers)


if __name__ == "__main__":
    main()
