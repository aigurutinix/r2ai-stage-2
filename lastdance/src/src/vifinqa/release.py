"""Reproduce, validate and package an immutable ViFinQA product release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional

from .experiments import SubmissionArchive
from .submission_validation import replay_archive
from .submission_validation import validate_query_ast


ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_VERSION = "vifinqa-release-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported release config schema")
    required = {"release_id", "paths", "expected", "models", "inference"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Release config is missing {sorted(missing)}")
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("pandas", "rapidfuzz"):
        result[name] = metadata.version(name)
    return result


def _csv_bytes(grid: list[list[Any]]) -> bytes:
    width = max((len(row) for row in grid), default=0)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([f"col_{index}" for index in range(width)])
    for row in grid:
        writer.writerow([*row, *("" for _ in range(width - len(row)))])
    return output.getvalue().encode("utf-8")


def provenance_rows(
    archive: SubmissionArchive, database: Path
) -> Iterable[dict[str, Any]]:
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    cache: dict[str, dict[str, Any] | None] = {}
    try:
        for item in archive.items:
            candidates: dict[str, list[dict[str, Any]]] = {}
            for evidence_key in item["relevant_tables"]:
                if evidence_key not in cache:
                    row = connection.execute(
                        """SELECT evidence_key, document_id, source_line_1, grid_json
                           FROM tables WHERE evidence_key = ?""",
                        (evidence_key,),
                    ).fetchone()
                    cache[evidence_key] = dict(row) if row else None
                source = cache[evidence_key]
                if source is None:
                    continue
                source_bytes = _csv_bytes(json.loads(source["grid_json"]))
                csv_hash = hashlib.sha256(source_bytes).hexdigest()
                candidates.setdefault(csv_hash, []).append(source)
            for evidence in item["evidence"]:
                csv_path = evidence["csv_path"]
                csv_hash = hashlib.sha256(archive.members[csv_path]).hexdigest()
                matches = candidates.get(csv_hash, [])
                if not matches:
                    raise ValueError(
                        f"q{item['id']}: no BTC provenance for {csv_path}"
                    )
                source = matches[0]
                yield {
                    "question_id": int(item["id"]),
                    "variable": evidence["variable"],
                    "csv_path": csv_path,
                    "document_id": source["document_id"],
                    "source_line_1": int(source["source_line_1"]),
                    "evidence_key": source["evidence_key"],
                    "source_grid_sha256": hashlib.sha256(
                        source["grid_json"].encode("utf-8")
                    ).hexdigest(),
                    "csv_sha256": csv_hash,
                    "extraction": "full_source_table_grid",
                    "extractor_version": EXTRACTOR_VERSION,
                }
    finally:
        connection.close()


def freeze_execution_plan(
    config: dict[str, Any], output_path: Optional[Path] = None
) -> Path:
    """Persist retrieval/code decisions without answers or financial values."""

    paths = config["paths"]
    archive = SubmissionArchive.load(resolve_path(paths["golden_submission"]))
    database = resolve_path(paths["database"])
    provenance = {
        (int(row["question_id"]), row["csv_path"]): row
        for row in provenance_rows(archive, database)
    }
    plans = []
    for item in archive.items:
        question_id = int(item["id"])
        validate_query_ast(item["pandas_query"])
        evidence = []
        for source in item["evidence"]:
            row = provenance[(question_id, source["csv_path"])]
            evidence.append({
                "variable": source["variable"],
                "csv_path": source["csv_path"],
                "evidence_key": row["evidence_key"],
                "source_grid_sha256": row["source_grid_sha256"],
            })
        plans.append({
            "id": question_id,
            "question": item["question"],
            "answer_type": "int" if isinstance(item["answer"], int) else "float",
            "relevant_docs": item["relevant_docs"],
            "relevant_tables": item["relevant_tables"],
            "evidence": evidence,
            "pandas_query": item["pandas_query"],
        })
    target = output_path or resolve_path(paths["execution_plan"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(plan, ensure_ascii=False) + "\n" for plan in plans),
        encoding="utf-8",
    )
    return target


def load_execution_plan(path: Path, expected_count: int) -> list[dict[str, Any]]:
    plans = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(plan["id"]) for plan in plans] != list(range(1, expected_count + 1)):
        raise ValueError("Execution plan must cover consecutive question IDs")
    for plan in plans:
        if "answer" in plan:
            raise ValueError(f"q{plan['id']}: execution plan must not store an answer")
        validate_query_ast(plan["pandas_query"])
    return plans


def _execute_query(query: str, frames: dict[str, Any]) -> Any:
    import pandas as pd

    validate_query_ast(query)
    return eval(  # noqa: S307 - AST and namespaces are allowlisted
        query,
        {
            "__builtins__": {}, "pd": pd, "float": float, "int": int,
            "str": str, "abs": abs, "round": round,
        },
        frames,
    )


def build_from_execution_plan(
    config: dict[str, Any], database: Path, output_dir: Path
) -> Path:
    """Build the official ZIP from BTC tables and a value-free execution plan."""

    import pandas as pd

    plan_path = resolve_path(config["paths"]["execution_plan"])
    plan_hash = sha256_file(plan_path)
    if plan_hash != config["expected"]["execution_plan_sha256"]:
        raise ValueError(f"Execution plan hash changed: {plan_hash}")
    plans = load_execution_plan(plan_path, int(config["expected"]["question_count"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)
    connection = sqlite3.connect(str(database))
    items = []
    try:
        for plan in plans:
            frames = {}
            evidence = []
            for source in plan["evidence"]:
                row = connection.execute(
                    "SELECT grid_json FROM tables WHERE evidence_key = ?",
                    (source["evidence_key"],),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"q{plan['id']}: missing source {source['evidence_key']}"
                    )
                grid_json = row[0]
                if hashlib.sha256(grid_json.encode("utf-8")).hexdigest() != source[
                    "source_grid_sha256"
                ]:
                    raise ValueError(f"q{plan['id']}: source grid checksum changed")
                csv_path = output_dir / source["csv_path"]
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_path.write_bytes(_csv_bytes(json.loads(grid_json)))
                frames[source["variable"]] = pd.read_csv(csv_path)
                evidence.append({
                    "variable": source["variable"], "csv_path": source["csv_path"]
                })
            answer = _execute_query(plan["pandas_query"], frames)
            if plan["answer_type"] == "int":
                answer = int(answer)
            else:
                answer = float(answer)
            items.append({
                "id": int(plan["id"]), "question": plan["question"],
                "answer": answer, "relevant_docs": plan["relevant_docs"],
                "relevant_tables": plan["relevant_tables"], "evidence": evidence,
                "pandas_query": plan["pandas_query"],
            })
    finally:
        connection.close()
    submission_path = output_dir / "submission.json"
    submission_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for path in sorted(data_dir.glob("*.csv")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    replay_archive(zip_path, database)
    return zip_path


def _validate_models(config: dict[str, Any]) -> list[str]:
    cutoff = date.fromisoformat(config["model_release_cutoff"])
    warnings = []
    for model in config["models"]:
        missing = {
            "id", "revision", "created_at", "license", "role", "download_url"
        } - set(model)
        if missing:
            raise ValueError(f"Model metadata missing {sorted(missing)}")
        created = datetime.fromisoformat(
            model["created_at"].replace("Z", "+00:00")
        ).date()
        if created >= cutoff:
            raise ValueError(
                f"Model {model['id']} was not released before {cutoff.isoformat()}"
            )
        if len(model["revision"]) != 40:
            raise ValueError(f"Model {model['id']} revision is not pinned")
    distribution = config.get("data_distribution", {})
    if not distribution.get("share_url"):
        warnings.append("DATA_SHARE_URL_NOT_SET")
    if not distribution.get("access_verified_at"):
        warnings.append("DATA_SHARE_ACCESS_NOT_VERIFIED")
    return warnings


def validate_locked_release(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    questions = resolve_path(paths["questions"])
    companies = resolve_path(paths["companies"])
    database = resolve_path(paths["database"])
    submission = resolve_path(paths["golden_submission"])
    execution_plan = resolve_path(paths["execution_plan"])
    for path in (questions, companies, database, submission, execution_plan):
        if not path.is_file():
            raise FileNotFoundError(path)
    question_count = sum(
        bool(line.strip()) for line in questions.read_text(encoding="utf-8").splitlines()
    )
    if question_count != int(config["expected"]["question_count"]):
        raise ValueError(f"Unexpected question count {question_count}")
    submission_hash = sha256_file(submission)
    if submission_hash != config["expected"]["submission_sha256"]:
        raise ValueError(
            f"Golden submission hash changed: {submission_hash}"
        )
    validation = replay_archive(submission, database)
    load_execution_plan(execution_plan, question_count)
    execution_plan_hash = sha256_file(execution_plan)
    if execution_plan_hash != config["expected"]["execution_plan_sha256"]:
        raise ValueError(f"Execution plan hash changed: {execution_plan_hash}")
    return {
        "validation": validation,
        "warnings": _validate_models(config),
        "inputs": {
            "questions": {"path": paths["questions"], "sha256": sha256_file(questions)},
            "companies": {"path": paths["companies"], "sha256": sha256_file(companies)},
            "database": {"path": paths["database"], "sha256": sha256_file(database)},
            "submission": {"path": paths["golden_submission"], "sha256": submission_hash},
            "execution_plan": {
                "path": paths["execution_plan"], "sha256": execution_plan_hash
            },
        },
    }


def package_release(config: dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    audit = validate_locked_release(config)
    paths = config["paths"]
    database = resolve_path(paths["database"])
    target = output_dir or resolve_path(paths["release_dir"])
    target.mkdir(parents=True, exist_ok=True)
    staging_dir = target / ".submission_build" / "submission"
    generated_submission = build_release(config, staging_dir)
    target_submission = target / "submission.zip"
    shutil.copyfile(generated_submission, target_submission)
    shutil.rmtree(staging_dir.parent)
    archive = SubmissionArchive.load(target_submission)
    provenance_path = target / "evidence_provenance.jsonl"
    rows = list(provenance_rows(archive, database))
    provenance_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    git_state = _git_state()
    warnings = list(audit["warnings"])
    if git_state["dirty"]:
        warnings.append("GIT_WORKTREE_DIRTY")
    manifest = {
        "schema_version": 1,
        "release_id": config["release_id"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "competition": config.get("competition"),
        "git": git_state,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "inputs": audit["inputs"],
        "models": config["models"],
        "inference": config["inference"],
        "leaderboard": config["expected"]["leaderboard"],
        "validation": audit["validation"],
        "warnings": warnings,
        "artifacts": {
            "submission": {
                "path": target_submission.name,
                "sha256": sha256_file(target_submission),
            },
            "provenance": {
                "path": provenance_path.name,
                "sha256": sha256_file(provenance_path),
                "rows": len(rows),
            },
        },
    }
    manifest_path = target / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_commands(
    config: dict[str, Any], work_dir: Path, database: Path
) -> list[list[str]]:
    """Return the deterministic raw-data-to-submission stage commands."""

    p = config["paths"]
    q, companies = resolve_path(p["questions"]), resolve_path(p["companies"])
    python = sys.executable
    stage = lambda name: work_dir / name  # noqa: E731 - compact declarative pipeline
    commands: list[list[str]] = []

    def add(module: str, *args: object) -> None:
        commands.append([python, "-m", module, *(str(value) for value in args)])

    add("vifinqa.l1_fact_layer", "candidates", "--questions", q, "--companies", companies,
        "--database", database, "--output-dir", stage("01_l1_facts"), "--overrides",
        ROOT / "analysis/l1_manual_overrides.csv")
    add("vifinqa.l1_fact_layer", "submission", "--top-facts",
        stage("01_l1_facts/top_facts.jsonl"), "--database", database, "--output-dir",
        stage("02_l1_submission"), "--confidence", "all", "--line-base", 1)
    add("vifinqa.l2_temporal", "candidates", "--questions", q, "--companies", companies,
        "--database", database, "--output-dir", stage("03_l2_temporal_facts"),
        "--overrides", ROOT / "analysis/l2_manual_overrides.csv")
    add("vifinqa.l2_temporal", "submission", "--top-pairs",
        stage("03_l2_temporal_facts/top_pairs.jsonl"), "--database", database,
        "--l1-submission", stage("02_l1_submission/submission.json"), "--output-dir",
        stage("04_l2_temporal_submission"), "--line-base", 1)

    registry = [
        ("vifinqa.l2_formula", "05_l2_formula_facts", "06_l2_formula_submission",
         "analysis/l2_formula_manual_overrides.csv", "04_l2_temporal_submission"),
        ("vifinqa.l2_cross_entity", "07_l2_cross_facts", "08_l2_cross_submission",
         "analysis/l2_cross_manual_overrides.csv", "06_l2_formula_submission"),
        ("vifinqa.l3_aggregate", "09_l3_facts", "10_l3_submission",
         "analysis/l3_manual_overrides.csv", "08_l2_cross_submission"),
        ("vifinqa.l4_selector", "11_l4_facts", "12_l4_submission",
         "analysis/l4_manual_overrides.csv", "10_l3_submission"),
        ("vifinqa.l5_screening", "13_l5_screening_facts", "14_l5_screening_submission",
         "analysis/l5_manual_overrides.csv", "12_l4_submission"),
        ("vifinqa.l5_advanced", "15_l5_advanced_facts", "16_l5_advanced_submission",
         "analysis/l5_advanced_manual_overrides.csv", "14_l5_screening_submission"),
        ("vifinqa.l5_scenarios", "17_l5_scenario_facts", "18_l5_scenario_submission",
         "analysis/l5_scenario_manual_overrides.csv", "16_l5_advanced_submission"),
        ("vifinqa.l5_currency", "19_l5_currency_facts", "20_final_submission",
         "analysis/l5_currency_manual_overrides.csv", "18_l5_scenario_submission"),
    ]
    for module, facts, output, overrides, base in registry:
        add(module, "candidates", "--questions", q, "--database", database,
            "--output-dir", stage(facts), "--overrides", ROOT / overrides)
        submission_args: list[object] = [
            "submission", "--plans", stage(f"{facts}/plans.jsonl"),
            "--database", database, "--base-submission", stage(f"{base}/submission.json"),
            "--output-dir", stage(output),
        ]
        if module == "vifinqa.l2_formula":
            submission_args.extend([
                "--temporal-pairs", stage("03_l2_temporal_facts/top_pairs.jsonl")
            ])
        add(module, *submission_args)
    return commands


def reproduce_release(
    config: dict[str, Any], work_dir: Path, rebuild_database: bool = False,
    start_stage: int = 1,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    database = (
        work_dir / "vifinqa.db"
        if rebuild_database else resolve_path(config["paths"]["database"])
    )
    commands = build_commands(config, work_dir, database)
    if rebuild_database:
        commands.insert(0, [
            sys.executable, "-m", "vifinqa.corpus_db", "build", "--data-root",
            str(resolve_path(config["paths"]["data_root"])), "--database", str(database),
        ])
    if not 1 <= start_stage <= len(commands):
        raise ValueError(f"start_stage must be between 1 and {len(commands)}")
    if rebuild_database and start_stage != 1:
        raise ValueError("Cannot combine --rebuild-database with --start-stage > 1")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for index, command in enumerate(commands[start_stage - 1:], start_stage):
        print(json.dumps({"stage": index, "of": len(commands), "command": command}))
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    final_zip = work_dir / "20_final_submission.zip"
    replay_archive(final_zip, database)
    golden_path = resolve_path(config["paths"]["golden_submission"])
    if golden_path.is_file():
        golden = SubmissionArchive.load(golden_path)
        generated = SubmissionArchive.load(final_zip)
        if generated.items != golden.items or generated.members.keys() != golden.members.keys():
            raise ValueError("Reproduced submission differs from the locked golden submission")
        for name in generated.members:
            if generated.members[name] != golden.members[name]:
                raise ValueError(f"Reproduced member differs from golden: {name}")
    return final_zip


def build_release(
    config: dict[str, Any], output_dir: Path, rebuild_database: bool = False
) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    database = resolve_path(config["paths"]["database"])
    if rebuild_database:
        database = output_dir.parent / "vifinqa.db"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([
            sys.executable, "-m", "vifinqa.corpus_db", "build", "--data-root",
            str(resolve_path(config["paths"]["data_root"])), "--database", str(database),
        ], cwd=ROOT, env=env, check=True)
    result = build_from_execution_plan(config, database, output_dir)
    golden_path = resolve_path(config["paths"]["golden_submission"])
    if golden_path.is_file():
        golden = SubmissionArchive.load(golden_path)
        generated = SubmissionArchive.load(result)
        if generated.items != golden.items:
            raise ValueError("Release build JSON differs from golden")
        for name in generated.members:
            if generated.members[name] != golden.members[name]:
                raise ValueError(f"Release build member differs from golden: {name}")
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/release.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    freeze = sub.add_parser("freeze-plan")
    freeze.add_argument("--output", type=Path)
    package = sub.add_parser("package")
    package.add_argument("--output-dir", type=Path)
    reproduce = sub.add_parser("research-rebuild")
    reproduce.add_argument("--work-dir", type=Path, default=Path("outputs/reproduce"))
    reproduce.add_argument("--rebuild-database", action="store_true")
    reproduce.add_argument(
        "--start-stage", type=int, default=1,
        help="Resume an existing work directory from this 1-based command index",
    )
    build = sub.add_parser("build")
    build.add_argument(
        "--output-dir", type=Path, default=Path("outputs/release_build/submission")
    )
    build.add_argument("--rebuild-database", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(resolve_path(str(args.config)))
    if args.command == "check":
        print(json.dumps(validate_locked_release(config), ensure_ascii=False, indent=2))
    elif args.command == "freeze-plan":
        path = freeze_execution_plan(config, args.output)
        print(json.dumps({"execution_plan": str(path)}, ensure_ascii=False))
    elif args.command == "package":
        path = package_release(config, args.output_dir)
        print(json.dumps({"manifest": str(path)}, ensure_ascii=False))
    elif args.command == "research-rebuild":
        path = reproduce_release(
            config, args.work_dir, args.rebuild_database, args.start_stage
        )
        print(json.dumps({"submission": str(path)}, ensure_ascii=False))
    else:
        path = build_release(config, args.output_dir, args.rebuild_database)
        print(json.dumps({"submission": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
