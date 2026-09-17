"""Audit retrieval and solver stages of a formula offline-eval run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.utils.io import setup_stdout
from vifinqa.validation.audit_formula_eval import audit_formula_eval


def main() -> None:
    setup_stdout()
    default_dir = config.ART_DIR / "formula_eval"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(default_dir / "formula_gold.json"))
    parser.add_argument("--retrieval",
                        default=str(default_dir / "formula_retrieval.jsonl"))
    parser.add_argument("--codegen",
                        default=str(default_dir / "formula_codegen_k15.jsonl"))
    parser.add_argument("--out", default=str(default_dir / "formula_audit_k15.json"))
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()
    audit_formula_eval(
        Path(args.gold), Path(args.retrieval), Path(args.codegen),
        k=args.k, out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
