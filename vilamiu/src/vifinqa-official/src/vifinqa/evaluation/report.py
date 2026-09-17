
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from vifinqa.evaluation.llm_runner import QuestionAnswerResult
from vifinqa.evaluation.retrieval_metrics import RetrievalMetrics, macro_average, micro_average
from vifinqa.evaluation.retrieval_runner import QuestionRetrievalResult


def _safe_tag(value: str) -> str:
    return value.replace("/", "__").replace(" ", "-")


def model_run_tag(model_name: str) -> str:
    parts = [part for part in model_name.split("/") if part]
    return "/".join(parts[-2:])


def make_run_dir(runs_dir: Path, task: str, tags: list[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parameter_name = "__".join([task, *(_safe_tag(t) for t in tags)])
    parameter_hash = hashlib.sha256(parameter_name.encode("utf-8")).hexdigest()[:12]
    name = f"{timestamp}_{parameter_hash}"
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_config(run_dir: Path, config: dict) -> None:
    payload = {**config, "git_commit": _git_commit_hash(), "timestamp": datetime.now().isoformat()}
    (run_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _metrics_group_summary(results: list[QuestionRetrievalResult]) -> dict[str, dict[str, float]]:
    macro = macro_average([r.metrics for r in results])
    micro = micro_average([r.counts for r in results])
    group: dict[str, dict[str, float]] = {"macro": asdict(macro), "micro": asdict(micro)}
    # Report-level scoring of the same ranking, present whenever the runner recorded it.
    report_metrics = [r.report_metrics for r in results if r.report_metrics is not None]
    report_counts = [r.report_counts for r in results if r.report_counts is not None]
    if report_metrics and report_counts:
        group["report_macro"] = asdict(macro_average(report_metrics))
        group["report_micro"] = asdict(micro_average(report_counts))
    return group


def build_retrieval_summary(
    results: list[QuestionRetrievalResult], *, task: str, extra: dict | None = None
) -> dict:
    by_k: dict[int, list[QuestionRetrievalResult]] = defaultdict(list)
    for r in results:
        by_k[r.k].append(r)

    by_k_difficulty: dict[int, dict[str, dict]] = defaultdict(dict)
    for k, rs in by_k.items():
        by_difficulty: dict[str, list[QuestionRetrievalResult]] = defaultdict(list)
        for r in rs:
            by_difficulty[r.difficulty].append(r)
        for difficulty, rs_d in by_difficulty.items():
            by_k_difficulty[k][difficulty] = _metrics_group_summary(rs_d)

    n_questions = len({r.question_id for r in results})
    summary = {
        "task": task,
        "n_questions": n_questions,
        "by_k": {str(k): _metrics_group_summary(rs) for k, rs in sorted(by_k.items())},
        "by_k_difficulty": {str(k): d for k, d in sorted(by_k_difficulty.items())},
    }
    if extra:
        summary.update(extra)
    return summary


def write_retrieval_report(
    run_dir: Path, results: list[QuestionRetrievalResult], *, task: str = "retrieval", extra: dict | None = None
) -> dict:
    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "id": r.question_id,
                        "original": r.original,
                        "k": r.k,
                        "f2": r.metrics.f2,
                        "recall": r.metrics.recall,
                        "precision": r.metrics.precision,
                        "report_recall": r.report_metrics.recall if r.report_metrics else None,
                        "report_precision": (
                            r.report_metrics.precision if r.report_metrics else None
                        ),
                        "retrieved_tables": list(r.retrieved_tables),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = build_retrieval_summary(results, task=task, extra=extra)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(render_retrieval_summary_md(summary), encoding="utf-8")
    return summary


def load_retrieved_tables(run_dir: Path, *, k: int) -> dict[int, list[str]]:
    path = run_dir / "per_question.jsonl"
    result: dict[int, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("k") != k:
                continue
            if "retrieved_tables" not in row:
                raise ValueError(
                    f"{path} has no 'retrieved_tables' field; rerun eval-retrieval with the current code."
                )
            result[row["id"]] = row["retrieved_tables"]
    return result


def render_retrieval_summary_md(summary: dict) -> str:
    lines = [f"# {summary['task']} — {summary['n_questions']} questions", ""]
    lines.append(
        "| k | macro F2 | macro Recall | macro Precision "
        "| micro F2 | micro Recall | micro Precision "
        "| report Recall | report Precision |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k, group in summary["by_k"].items():
        macro, micro = group["macro"], group["micro"]
        report = group.get("report_macro")
        report_cells = (
            f"| {report['recall']:.3f} | {report['precision']:.3f} |" if report else "| - | - |"
        )
        lines.append(
            f"| {k} | {macro['f2']:.3f} | {macro['recall']:.3f} | {macro['precision']:.3f} "
            f"| {micro['f2']:.3f} | {micro['recall']:.3f} | {micro['precision']:.3f} "
            f"{report_cells}"
        )
    return "\n".join(lines) + "\n"


def _answer_rates(results: list[QuestionAnswerResult], *, splits_crash: bool) -> dict[str, float]:
    """Accuracy plus the paper's Fail/Crash split.

    Program generation separates Crash (execution error or no numeric scalar) from Fail (a
    program that ran and returned a wrong number). Direct answering has no execution step, so
    non-parseable outputs are folded back into Fail, matching the Table 13 note.
    """

    n = len(results)
    if not n:
        return {"accuracy": 0.0, "fail": 0.0, "crash": 0.0}
    accuracy = sum(r.accuracy for r in results) / n
    crash = sum(1 for r in results if r.crash) / n if splits_crash else 0.0
    return {"accuracy": accuracy, "fail": 1.0 - accuracy - crash, "crash": crash}


def _answer_group_summary(results: list[QuestionAnswerResult], *, splits_crash: bool) -> dict:
    by_difficulty: dict[str, list[QuestionAnswerResult]] = defaultdict(list)
    for r in results:
        by_difficulty[r.difficulty].append(r)
    return {
        "n_questions": len(results),
        **_answer_rates(results, splits_crash=splits_crash),
        "by_difficulty": {
            d: {"n_questions": len(rs), **_answer_rates(rs, splits_crash=splits_crash)}
            for d, rs in sorted(by_difficulty.items())
        },
    }


def build_answer_summary(
    results: list[QuestionAnswerResult],
    *,
    task: str,
    extra: dict | None = None,
    strategy: str = "pandas_query",
) -> dict:
    splits_crash = strategy == "pandas_query"
    has_k = any(r.k is not None for r in results)
    if has_k:
        by_k: dict[int, list[QuestionAnswerResult]] = defaultdict(list)
        for r in results:
            by_k[r.k].append(r)  # type: ignore[index]
        summary = {
            "task": task,
            "n_questions": len({r.question_id for r in results}),
            "by_k": {
                str(k): _answer_group_summary(rs, splits_crash=splits_crash)
                for k, rs in sorted(by_k.items())
            },
        }
    else:
        summary = {"task": task, **_answer_group_summary(results, splits_crash=splits_crash)}
    if extra:
        summary.update(extra)
    return summary


def write_answer_report(
    run_dir: Path,
    results: list[QuestionAnswerResult],
    *,
    task: str,
    extra: dict | None = None,
    strategy: str = "pandas_query",
) -> dict:
    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_answer_row(r), ensure_ascii=False, default=str) + "\n")

    summary = build_answer_summary(results, task=task, extra=extra, strategy=strategy)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render_answer_summary_md(summary), encoding="utf-8")
    return summary


def _answer_row(result: QuestionAnswerResult) -> dict:
    row: dict = {"id": result.question_id, "original": result.original}
    if result.k is not None:
        row["k"] = result.k
    row.update(
        {
            "accuracy": result.accuracy,
            "crash": result.crash,
            "expected": result.expected,
            "actual": result.actual,
        }
    )
    if result.error:
        row["error"] = result.error
    if result.raw_output:
        row["raw_output"] = result.raw_output
    return row


def append_answer_checkpoint(run_dir: Path, result: QuestionAnswerResult) -> None:
    with (run_dir / "checkpoint.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(_answer_row(result), ensure_ascii=False, default=str) + "\n")


def render_answer_summary_md(summary: dict) -> str:
    lines = [f"# {summary['task']} — {summary['n_questions']} questions", ""]
    if "by_k" in summary:
        lines.append("| k | n | accuracy | fail | crash |")
        lines.append("|---|---|---|---|---|")
        for k, group in summary["by_k"].items():
            lines.append(
                f"| {k} | {group['n_questions']} | {group['accuracy']:.3f} "
                f"| {group['fail']:.3f} | {group['crash']:.3f} |"
            )
    else:
        lines.append(
            f"**Overall accuracy: {summary['accuracy']:.3f} "
            f"(fail {summary['fail']:.3f}, crash {summary['crash']:.3f})**"
        )
        lines.append("")
        lines.append("| difficulty | n | accuracy | fail | crash |")
        lines.append("|---|---|---|---|---|")
        for d, stat in summary["by_difficulty"].items():
            lines.append(
                f"| {d} | {stat['n_questions']} | {stat['accuracy']:.3f} "
                f"| {stat['fail']:.3f} | {stat['crash']:.3f} |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "RetrievalMetrics",
    "append_answer_checkpoint",
    "build_answer_summary",
    "build_retrieval_summary",
    "load_retrieved_tables",
    "make_run_dir",
    "model_run_tag",
    "render_answer_summary_md",
    "render_retrieval_summary_md",
    "write_answer_report",
    "write_config",
    "write_retrieval_report",
]
