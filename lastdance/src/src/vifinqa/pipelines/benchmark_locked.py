"""Pipeline for reproducing the benchmark-locked (0.6522) baseline submission."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..release import build_from_execution_plan, load_config, resolve_path, sha256_file
from ..submission_validation import replay_archive

ROOT = Path(__file__).resolve().parents[3]


def run_benchmark_locked(
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    database_path: Path,
) -> None:
    config = load_config(resolve_path(str(config_path)))
    work_dir = output_dir.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    database_path = database_path.absolute()
    
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # Step 1: Build Database
    if not database_path.exists():
        print("--- Building SQLite Database ---")
        subprocess.run([
            sys.executable, "-m", "vifinqa.corpus_db", "build",
            "--data-root", str(data_root.absolute()),
            "--database", str(database_path),
        ], cwd=ROOT, env=env, check=True)
    else:
        print(f"Database {database_path} already exists, skipping build.")

    execution_plan = resolve_path(config["paths"]["execution_plan"])

    print(f"--- Building Release from {execution_plan.name} ---")
    
    submission_zip = build_from_execution_plan(config, database_path, output_dir)

    print("--- Replaying and Validating Submission ---")
    replay_archive(submission_zip, database_path)

    print(f"\nSUCCESS! Final submission packaged at: {submission_zip}")
