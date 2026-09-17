
from __future__ import annotations

import json
from pathlib import Path

METRICS = ("recall", "f2", "precision")
METRIC_GROUPS = ("macro", "micro")


def load_run_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def infer_label(run_dir: Path) -> str:
    config = load_run_config(run_dir)
    if not config:
        return run_dir.name
    parts = [config.get("retriever", "?")]
    if config.get("embedding_model"):
        parts.append(config["embedding_model"].split("/")[-1])
    if config.get("reranker") and config["reranker"] != "none":
        parts.append(f"+{config['reranker']}")
    return "-".join(parts)


def build_comparison_markdown(runs: list[tuple[str, dict]]) -> str:
    all_ks = sorted({int(k) for _, s in runs for k in s.get("by_k", {})})
    n_questions = {s.get("n_questions") for _, s in runs}
    lines = ["# Retrieval comparison", ""]
    lines.append(f"Question counts: {', '.join(str(n) for n in sorted(n_questions))}")
    lines.append(f"Methods: {', '.join(label for label, _ in runs)}")
    lines.append("")

    for group in METRIC_GROUPS:
        lines.append(f"## {group}")
        for metric in METRICS:
            lines.append(f"### {metric}@k ({group})")
            lines.append("| k | " + " | ".join(label for label, _ in runs) + " |")
            lines.append("|---|" + "---|" * len(runs))
            for k in all_ks:
                row = [str(k)]
                for _, summary in runs:
                    value = summary.get("by_k", {}).get(str(k), {}).get(group, {}).get(metric)
                    row.append(f"{value:.3f}" if value is not None else "-")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    return "\n".join(lines) + "\n"


def compare_run_dirs(run_dirs: list[Path], *, labels: list[str] | None = None) -> str:
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError("labels and run_dirs must have the same length")
    runs = [
        (labels[i] if labels else infer_label(run_dir), load_run_summary(run_dir))
        for i, run_dir in enumerate(run_dirs)
    ]
    return build_comparison_markdown(runs)
