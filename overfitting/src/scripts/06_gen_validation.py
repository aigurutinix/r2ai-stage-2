"""Step 6 (local, optional but recommended): synthetic validation set.

  python scripts/06_gen_validation.py --n 300
Then evaluate a submission built on it:
  python scripts/02_retrieve.py --questions artifacts/validation/val_questions.jsonl --out artifacts/val_retrieval.jsonl
  python scripts/03_rule_baseline.py --retrieval artifacts/val_retrieval.jsonl --out artifacts/val_codegen.jsonl
  python scripts/05_build_submission.py --retrieval artifacts/val_retrieval.jsonl --codegen artifacts/val_codegen.jsonl --out-dir artifacts/val_submission
  python scripts/07_evaluate.py --submission artifacts/val_submission --gold artifacts/validation/val_gold.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.validation.gen_questions import generate
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    ap.add_argument("--out-dir", default=str(config.VALIDATION_DIR))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    generate(Path(args.store_dir), Path(args.code_stock), Path(args.out_dir), args.n)


if __name__ == "__main__":
    main()
