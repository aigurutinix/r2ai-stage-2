"""Evaluation pipeline to compare general output with benchmark_locked truth."""

from pathlib import Path

import json
from pathlib import Path

def run_benchmark_evaluator(
    questions_file: Path,
    output_dir: Path,
) -> None:
    # A basic evaluator that will later compare output_dir against benchmark golden
    print("Running evaluation on generated outputs...")
    # Read the golden labels (for now, just release_plan.jsonl)
    golden_path = Path("ViFinQA/release_plan.jsonl")
    if not golden_path.exists():
        print("Golden plan not found. Evaluation skipped.")
        return
        
    # Later this will compare the output of the general mode with the golden plans
    # and compute entity/year/scope accuracy, and numeric answer accuracy.
    print(f"Evaluator initialized. General outputs expected in: {output_dir}")
    print("Evaluation logic will be expanded in later milestones.")
