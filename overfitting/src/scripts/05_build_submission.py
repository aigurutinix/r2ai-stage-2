"""Step 5 (local): build submission.zip from retrieval + codegen results.

  python scripts/05_build_submission.py                        # uses artifacts/codegen_results.jsonl
  python scripts/05_build_submission.py --codegen downloads/codegen_results.jsonl
  python scripts/05_build_submission.py --sub-k 7 --out-dir artifacts/submission_k7

Official table positions use ``--pos-mode line`` (the default). ``--pos-base``
only applies to the legacy ``--pos-mode order`` debugging mode.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.submission.build import build_submission
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default=str(config.RETRIEVAL_JSONL))
    ap.add_argument("--codegen", default=str(config.CODEGEN_JSONL))
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--out-dir", default=str(config.SUBMISSION_DIR))
    ap.add_argument("--sub-k", type=int, default=config.SUBMISSION_K)
    ap.add_argument("--pos-base", type=int, default=config.TABLE_POS_BASE)
    ap.add_argument("--pos-mode", choices=["order", "line"],
                    default=config.TABLE_POS_MODE,
                    help="'line' (default) = 1-based line number of <table> in "
                         "the .txt — OFFICIAL scheme confirmed by the organizers; "
                         "'order' is a legacy debug mode")
    ap.add_argument("--questions", default="",
                    help="official test questions.jsonl -> only submit those ids")
    ap.add_argument("--offline-eval", action="store_true",
                    help="building against the synthetic eval suite: names the zip "
                         "OFFLINE_EVAL_DO_NOT_UPLOAD.zip so it cannot be submitted "
                         "by accident")
    ap.add_argument("--expand-docs", action="store_true",
                    help="add sibling doc_type + year+1 reports to relevant_docs "
                         "(recall/F2 experiment, one variable per submission!)")
    args = ap.parse_args()

    build_submission(Path(args.retrieval), Path(args.codegen), Path(args.store_dir),
                     Path(args.out_dir), args.sub_k, args.pos_base,
                     pos_mode=args.pos_mode,
                     questions_path=Path(args.questions) if args.questions else None,
                     expand_docs=args.expand_docs, offline_eval=args.offline_eval)


if __name__ == "__main__":
    main()
