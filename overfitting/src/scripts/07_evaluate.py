"""Step 7 (local): offline metrics on the synthetic validation submission.

  python scripts/07_evaluate.py --submission artifacts/val_submission --gold artifacts/validation/val_gold.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.validation.evaluate import evaluate
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--by-class", action="store_true",
                    help="break metrics down by question class (eval suite)")
    args = ap.parse_args()
    evaluate(Path(args.submission), Path(args.gold), by_class=args.by_class)


if __name__ == "__main__":
    main()
