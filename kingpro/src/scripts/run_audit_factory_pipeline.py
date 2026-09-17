"""One-command fast-screen -> priority -> deep-audit pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FACTORY = ROOT / "scripts/run_question_audit_factory.py"
DEFAULT_SUBMISSION = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
DEFAULT_LEDGER = ROOT / "knowledge/vothuong/question_source_verdicts.json"
DEFAULT_OUT = ROOT / "build/v234_audit_factory_pipeline"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def queue_ids(path: Path, closed: set[int]) -> list[int]:
    return [
        int(item["id"])
        for item in load(path).get("queue", [])
        if int(item["id"]) not in closed
    ]


def select_deep_ids(
    priority_rows: list[dict], target_ids: list[int], limit: int
) -> list[int]:
    target = set(target_ids)
    ranked = [int(item["id"]) for item in priority_rows if int(item["id"]) in target]
    seen = set(ranked)
    ranked.extend(qid for qid in target_ids if qid not in seen)
    return ranked[: max(0, limit)]


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--target", choices=("public", "batch"), default="batch")
    parser.add_argument("--public-queue", type=Path, default=ROOT / "build/v227_residual_public100_fused_v217.json")
    parser.add_argument("--batch-queue", type=Path, default=ROOT / "build/v227_residual_batch150_fused_v217.json")
    parser.add_argument("--fast-workers", type=int, default=16)
    parser.add_argument("--deep-workers", type=int, default=12)
    parser.add_argument("--deep-limit", type=int, default=150)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    started = time.perf_counter()
    output = args.out.resolve()
    fast_out = output / "fast"
    deep_out = output / "deep"
    output.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(FACTORY),
        "--submission",
        str(args.submission.resolve()),
        "--ledger",
        str(args.ledger.resolve()),
    ]
    run(
        common
        + [
            "--queue",
            "all",
            "--mode",
            "fast",
            "--limit",
            "1012",
            "--workers",
            str(max(1, args.fast_workers)),
            "--out",
            str(fast_out),
        ]
    )

    ledger = load(args.ledger.resolve()).get("verdicts", {})
    closed = {int(value) for value in ledger}
    target_path = args.public_queue if args.target == "public" else args.batch_queue
    targets = queue_ids(target_path.resolve(), closed)
    priority_rows = load(fast_out / "PRIORITY_QUEUE.json")
    selected = select_deep_ids(priority_rows, targets, args.deep_limit)
    if selected:
        run(
            common
            + [
                "--ids",
                ",".join(map(str, selected)),
                "--queue",
                args.target,
                "--mode",
                "deep",
                "--limit",
                str(len(selected)),
                "--workers",
                str(max(1, args.deep_workers)),
                "--out",
                str(deep_out),
            ]
        )

    fast_summary = load(fast_out / "summary.json")
    deep_summary = load(deep_out / "summary.json") if selected else None
    deep_priority = load(deep_out / "PRIORITY_QUEUE.json") if selected else []
    shards = [{"agent": index + 1, "questions": []} for index in range(3)]
    for index, item in enumerate(deep_priority):
        shards[index % len(shards)]["questions"].append(item)
    (output / "AGENT_SHARDS.json").write_text(
        json.dumps(shards, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    report = {
        "target": args.target,
        "target_pending": len(targets),
        "deep_selected": selected,
        "fast": fast_summary,
        "deep": deep_summary,
        "agent_shards": shards,
        "pipeline_wall_seconds": round(elapsed, 4),
        "claim_limit": "Factory prioritization only; no verdict or answer mutation is automatic.",
    }
    (output / "pipeline_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Audit Factory Pipeline",
        "",
        f"- Target: **{args.target}**",
        f"- Pending target questions: **{len(targets)}**",
        f"- Deep audited: **{len(selected)}**",
        f"- Pipeline wall time: **{elapsed:.2f}s**",
        f"- Fast throughput: **{fast_summary['questions_per_second']:.2f} q/s**",
    ]
    if deep_summary:
        lines.append(f"- Deep throughput: **{deep_summary['questions_per_second']:.2f} q/s**")
        lines.append(f"- Deep triage: `{deep_summary['triage_counts']}`")
    lines.extend(
        [
            "",
            f"- Fast dashboard: `{fast_out.relative_to(ROOT) / 'DASHBOARD.md'}`",
            f"- Deep dashboard: `{deep_out.relative_to(ROOT) / 'DASHBOARD.md'}`",
            "",
            "> Fail-closed: this pipeline does not write verdicts or alter answers.",
            "",
        ]
    )
    (output / "FACTORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
