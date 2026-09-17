"""End-to-end pipeline orchestrator for ViFinQA.

This script acts as the main entry point to route execution to the
appropriate pipeline (general, benchmark-locked, or benchmark_evaluator).
"""

import argparse
from pathlib import Path

from vifinqa.pipelines.benchmark_locked import run_benchmark_locked
from vifinqa.pipelines.general import run_general
from vifinqa.pipelines.benchmark_evaluator import run_benchmark_evaluator

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["general", "benchmark-locked", "evaluate"], required=True,
                        help="The execution mode to run.")
    parser.add_argument("--data-root", type=Path, default=Path("ViFinQA"),
                        help="Path to the ViFinQA OCR dataset directory.")
    parser.add_argument("--questions", type=Path,
                        help="Path to questions file (for general/evaluate mode).")
    parser.add_argument("--config", type=Path, default=Path("configs/release.json"),
                        help="Path to release config (for benchmark-locked mode).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submission"),
                        help="Directory for outputs.")
    parser.add_argument("--database", type=Path, default=Path("outputs/vifinqa.db"),
                        help="Path to the SQLite database.")
    
    args = parser.parse_args()

    if args.mode == "benchmark-locked":
        run_benchmark_locked(
            data_root=args.data_root,
            config_path=args.config,
            output_dir=args.output_dir,
            database_path=args.database,
        )
    elif args.mode == "general":
        run_general(
            questions_file=args.questions,
            output_dir=args.output_dir,
            database_path=args.database,
        )
    elif args.mode == "evaluate":
        run_benchmark_evaluator(
            questions_file=args.questions,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
