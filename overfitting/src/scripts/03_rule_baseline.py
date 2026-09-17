"""Step 3 (local, CPU, NO GPU NEEDED): deterministic rule-based baseline.

Produces a codegen_results.jsonl WITHOUT any LLM — a submittable end-to-end
baseline you can build while the Kaggle run is being set up.

  python scripts/03_rule_baseline.py
  python scripts/03_rule_baseline.py --limit 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.codegen.generate import run_codegen
from vifinqa.codegen.llm_client import NoLLM
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default=str(config.RETRIEVAL_JSONL))
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--out", default=str(config.CODEGEN_JSONL))
    ap.add_argument("--k", type=int, default=config.CODEGEN_K)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run_codegen(Path(args.retrieval), Path(args.store_dir), Path(args.out),
                client=NoLLM(), k=args.k, limit=args.limit,
                use_rule_fallback=True)


if __name__ == "__main__":
    main()
