"""Script to freeze the current benchmark-locked baseline.

This script runs the benchmark-locked pipeline, computes SHA256 checksums of 
key artifacts (execution plan, submission zip, database, and questions), 
and saves them to a manifest file.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # Output paths
    output_dir = ROOT / "outputs" / "submission"
    baseline_dir = ROOT / "outputs" / "baselines" / "benchmark_locked"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = ROOT / "artifacts" / "vifinqa.db"
    if not database_path.exists():
        database_path = ROOT / "outputs" / "vifinqa.db"

    # Run the pipeline
    print("--- Running Benchmark Locked Pipeline ---")
    subprocess.run([
        sys.executable, "-m", "vifinqa.run_pipeline",
        "--mode", "benchmark-locked",
        "--data-root", "ViFinQA",
        "--output-dir", str(output_dir),
        "--database", str(database_path),
    ], cwd=ROOT, env=env, check=True)

    # Move output submission zip
    submission_zip = ROOT / "outputs" / "submission.zip"
    if not submission_zip.exists():
        # Fallback to the known naming pattern
        submission_zip = output_dir / "compliance_submission_v1.zip"
        if not submission_zip.exists():
            print("Could not find the generated submission zip!")
            sys.exit(1)

    baseline_zip = baseline_dir / "submission.zip"
    import shutil
    shutil.copyfile(submission_zip, baseline_zip)

    # Compute checksums
    questions_file = ROOT / "ViFinQA" / "questions" / "questions.jsonl"
    execution_plan = ROOT / "configs" / "release_plan.jsonl"
    
    manifest = {
        "event": "freeze_baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checksums": {
            "database": sha256_file(database_path) if database_path.exists() else None,
            "questions_jsonl": sha256_file(questions_file) if questions_file.exists() else None,
            "execution_plan": sha256_file(execution_plan) if execution_plan.exists() else None,
            "submission_zip": sha256_file(baseline_zip) if baseline_zip.exists() else None,
        }
    }

    manifest_path = baseline_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBaseline successfully frozen at: {baseline_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
