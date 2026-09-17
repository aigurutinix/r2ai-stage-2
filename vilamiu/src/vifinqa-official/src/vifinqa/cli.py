"""Public YAML-first command line interface for ViFinQA."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import click
import typer
import yaml
from typer.main import get_command

from vifinqa.config import env_value, external_to_internal_difficulty, load_config
from vifinqa.config.settings import get_settings
from vifinqa.evaluation.compare import compare_run_dirs
from vifinqa.legacy_cli import app as baseline_app
from vifinqa.legacy_cli import (
    _resolve_reranker_instruction as _resolve_reranker_instruction,
)
from vifinqa.legacy_cli import _search_cache_namespace as _search_cache_namespace
from vifinqa.legacy_cli import (
    _validate_retrieval_representation as _validate_retrieval_representation,
)


app = typer.Typer(no_args_is_help=True, add_completion=False)


def _section(resolved: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = resolved.get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _option(args: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(name)
        return
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    args.extend((name, str(value)))


def _resolved(
    config: Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _, resolved = load_config(config, overrides=overrides)
    return resolved


def _write_config(run_dir: Path, resolved: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(resolved), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_root(resolved: Mapping[str, Any]) -> Path:
    return Path(_section(resolved, "paths").get("runs_dir", "runs"))


def _delegate(
    typer_app: typer.Typer,
    args: list[str],
    *,
    resolved: Mapping[str, Any],
    record_new_runs: bool = False,
) -> None:
    runs_root = _run_root(resolved)
    before = set(runs_root.iterdir()) if record_new_runs and runs_root.is_dir() else set()
    command = get_command(typer_app)
    try:
        command.main(args=args, prog_name="vifinqa", standalone_mode=False)
    except (click.ClickException, click.Abort) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if record_new_runs and runs_root.is_dir():
        for path in set(runs_root.iterdir()) - before:
            if path.is_dir():
                _write_config(path, resolved)


@contextlib.contextmanager
def _generation_environment(resolved: Mapping[str, Any]) -> Iterator[None]:
    paths = _section(resolved, "paths")
    llm = _section(resolved, "llm")
    embedding = _section(resolved, "embedding")
    updates: dict[str, str] = {
        "DATA_ROOT": str(paths.get("data_root", "data/ocr_filter")),
        "COMPANY_META_PATH": str(paths.get("company_meta_path", "data/file_filter.csv")),
        "CACHE_DIR": str(paths.get("cache_dir", ".cache/vifinqa")),
        "EMBEDDING_MODEL": str(embedding.get("model_id", "BAAI/bge-m3")),
        "OPENAI_MODEL": str(llm.get("model_id", "")),
    }
    for target, key in (("OPENAI_URL", "base_url_env"), ("OPENAI_API_KEY", "api_key_env")):
        source = llm.get(key)
        if isinstance(source, str) and source:
            field = key.removesuffix("_env")
            updates[target] = env_value({key: source}, field) or ""
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@app.command()
def catalog(
    config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
    data_root: Path | None = typer.Option(None, "--data-root"),
) -> None:
    """Inspect the local corpus layout without loading models."""

    if config is None:
        args: list[str] = []
        _option(args, "--data-root", data_root)
        command = get_command(baseline_app)
        command.main(args=["catalog", *args], prog_name="vifinqa", standalone_mode=False)
        return
    resolved = _resolved(config, {"paths.data_root": data_root})
    args = ["catalog"]
    _option(args, "--data-root", _section(resolved, "paths").get("data_root"))
    _delegate(baseline_app, args, resolved=resolved)


@app.command()
def generate(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    count: int | None = typer.Option(None, "--count"),
    workers: int | None = typer.Option(None, "--workers"),
    output: Path | None = typer.Option(None, "--output"),
    data_root: Path | None = typer.Option(None, "--data-root"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Generate execution-validated questions for one public difficulty tier."""

    resolved = _resolved(
        config,
        {
            "generation.count": count,
            "generation.workers": workers,
            "generation.output": str(output) if output else None,
            "paths.data_root": str(data_root) if data_root else None,
            "paths.cache_dir": str(cache_dir) if cache_dir else None,
            "paths.runs_dir": str(runs_dir) if runs_dir else None,
        },
    )
    generation = _section(resolved, "generation")
    public_tier = str(generation.get("tier", "easy"))
    internal = external_to_internal_difficulty(public_tier)
    legacy_tier = internal
    run_name = str(_section(resolved, "run").get("name", f"generate-{public_tier}"))
    run_dir = _run_root(resolved) / run_name
    out = Path(generation.get("output") or run_dir / "per_question.jsonl")
    _write_config(run_dir, resolved)
    engine = str(generation.get("engine", "legacy"))
    if engine == "paper":
        if int(generation.get("workers", 1)) != 1:
            raise typer.BadParameter("The paper generation engine currently requires workers=1")
        from vifinqa.factories import build_llm
        from vifinqa.generation.paper_pipeline import generate_from_paper_prompts

        settings = get_settings()
        generated = generate_from_paper_prompts(
            llm=build_llm(_section(resolved, "llm"), hf_token=settings.hf_token),
            prompts=_section(resolved, "prompt"),
            data_root=Path(_section(resolved, "paths").get("data_root", "data/ocr_filter")),
            company_meta_path=Path(
                _section(resolved, "paths").get("company_meta_path", "data/file_filter.csv")
            ),
            tier=public_tier,
            count=int(generation.get("count", 10)),
            out_path=out,
            seed=generation.get("seed"),
            max_candidates=int(generation.get("max_candidates", 100)),
            max_llm_calls=generation.get("max_llm_calls"),
        )
        summary = {
            "task": "generation",
            "tier": public_tier,
            "requested": int(generation.get("count", 10)),
            "generated": generated,
            "output": str(out),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "summary.md").write_text(
            "# Generation summary\n\n"
            f"- Tier: {public_tier}\n- Requested: {summary['requested']}\n"
            f"- Generated: {generated}\n- Output: `{out}`\n",
            encoding="utf-8",
        )
        typer.echo(f"generated: {generated}/{generation.get('count', 10)} -> {out}")
        return
    if engine != "legacy":
        raise typer.BadParameter("generation.engine must be paper or legacy")
    args = ["generate", "--tier", legacy_tier]
    _option(args, "--count", generation.get("count", 10))
    _option(args, "--out", out)
    _option(args, "--seed", generation.get("seed"))
    _option(args, "--max-workers", generation.get("workers", 1))
    _option(args, "--max-candidates", generation.get("max_candidates"))
    _option(args, "--max-llm-calls", generation.get("max_llm_calls"))
    _option(args, "--mode", generation.get("mode"))
    _option(args, "--scenario", generation.get("scenario"))
    _option(args, "--cube-frame", generation.get("cube_frame"))
    with _generation_environment(resolved):
        from vifinqa.generation.legacy_cli import app as generation_app

        _delegate(generation_app, args, resolved=resolved)


def _retrieval_args(
    resolved: Mapping[str, Any], *, command: str = "eval-retrieval"
) -> list[str]:
    if command not in {"build-index", "eval-retrieval", "eval-e2e"}:
        raise ValueError(f"Unsupported retrieval command: {command}")
    retrieval = _section(resolved, "retrieval")
    embedding = _section(resolved, "embedding")
    reranker = _section(resolved, "reranker")
    paths = _section(resolved, "paths")
    args: list[str] = []
    _option(args, "--retriever", retrieval.get("backend", "dense"))
    _option(args, "--embedding-model", embedding.get("model_id"))
    _option(args, "--table-encoding", retrieval.get("table_encoding", "table_retrieval_text"))
    _option(args, "--chunk-rows", retrieval.get("chunk_rows"))
    _option(args, "--chunk-overlap-rows", retrieval.get("chunk_overlap_rows"))
    _option(args, "--context-max-chars", retrieval.get("context_max_chars"))
    _option(args, "--chunk-max-chars", retrieval.get("chunk_max_chars"))
    _option(args, "--table-views", retrieval.get("table_views"))
    _option(args, "--summary-max-chars", retrieval.get("summary_max_chars"))
    _option(args, "--metadata-csv-layout", retrieval.get("metadata_csv_layout"))
    _option(args, "--embed-batch-size", embedding.get("batch_size"))
    _option(args, "--embedding-max-tokens", embedding.get("max_tokens"))
    _option(args, "--data-root", paths.get("data_root"))
    _option(args, "--cache-dir", paths.get("cache_dir"))
    _option(args, "--company-meta-path", paths.get("company_meta_path"))
    if command == "build-index":
        _option(args, "--rebuild", retrieval.get("rebuild"))
        return args

    _option(args, "--active-table-views", retrieval.get("active_table_views"))
    _option(args, "--view-fusion", retrieval.get("view_fusion"))
    _option(args, "--rerank-chunks-per-table", retrieval.get("rerank_chunks_per_table"))
    _option(args, "--multi-chunk-parent-n", retrieval.get("multi_chunk_parent_n"))
    _option(args, "--rrf-fetch-n", retrieval.get("rrf_fetch_n", retrieval.get("fetch_n")))
    _option(args, "--rrf-k", retrieval.get("rrf_k"))
    _option(args, "--metadata-policy", retrieval.get("metadata_policy"))
    _option(args, "--metadata-fetch-n", retrieval.get("metadata_fetch_n"))
    _option(args, "--metadata-global-fallback", retrieval.get("metadata_global_fallback"))
    _option(args, "--entity-balance", retrieval.get("entity_balance"))
    if retrieval.get("search_cache") is False:
        args.append("--no-search-cache")
    elif retrieval.get("search_cache") is True:
        args.append("--search-cache")
    _option(args, "--run-tag", retrieval.get("run_tag"))
    if reranker.get("enabled"):
        adapter = str(reranker.get("adapter", "auto"))
        family = "qwen3" if "qwen" in str(reranker.get("model_id", "")).lower() or adapter == "qwen3" else "bge"
        _option(args, "--reranker", family)
        _option(args, "--reranker-model", reranker.get("model_id"))
        _option(args, "--rerank-top-n", reranker.get("top_n"))
        _option(args, "--reranker-max-tokens", reranker.get("max_tokens"))
        _option(args, "--reranker-instruction", reranker.get("instruction", "default"))
        _option(args, "--rerank-flow", reranker.get("flow"))
        _option(args, "--cascade-top-n", reranker.get("cascade_top_n"))
    return args


@app.command(name="build-index")
def build_index(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    data_root: Path | None = typer.Option(None, "--data-root"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
) -> None:
    """Build and cache a BM25, dense, RRF, row-chunk, or multi-view index."""

    resolved = _resolved(config, {"paths.data_root": data_root, "paths.cache_dir": cache_dir})
    args = ["build-index", *_retrieval_args(resolved, command="build-index")]
    _delegate(baseline_app, args, resolved=resolved)


@app.command(name="eval-retrieval")
def eval_retrieval(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    limit: int | None = typer.Option(None, "--limit"),
    questions_dir: Path | None = typer.Option(None, "--questions-dir"),
    data_root: Path | None = typer.Option(None, "--data-root"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Evaluate table retrieval and write per-question and aggregate metrics."""

    resolved = _resolved(config, {
        "evaluation.limit": limit,
        "paths.questions_dir": questions_dir,
        "paths.data_root": data_root,
        "paths.cache_dir": cache_dir,
        "paths.runs_dir": runs_dir,
    })
    evaluation = _section(resolved, "evaluation")
    paths = _section(resolved, "paths")
    args = ["eval-retrieval", *_retrieval_args(resolved, command="eval-retrieval")]
    _option(args, "--mode", evaluation.get("mode", "full"))
    _option(args, "--ks", evaluation.get("ks"))
    _option(args, "--limit", evaluation.get("limit"))
    _option(args, "--questions-dir", paths.get("questions_dir"))
    _option(args, "--runs-dir", paths.get("runs_dir"))
    _delegate(baseline_app, args, resolved=resolved, record_new_runs=True)


def _answering_args(
    args: list[str],
    *,
    answering: Mapping[str, Any],
    llm: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> None:
    """Forward every inference and scoring knob so the stored config matches the run."""

    _option(args, "--llm-temperature", llm.get("temperature"))
    _option(args, "--llm-max-completion-tokens", llm.get("max_completion_tokens"))
    _option(args, "--llm-max-retries", llm.get("max_retries"))
    _option(args, "--table-max-chars", answering.get("table_max_chars"))
    _option(args, "--answer-abs-tolerance", evaluation.get("answer_abs_tolerance"))
    _option(args, "--prompt-system", prompt.get("system"))
    _option(args, "--prompt-user-template", prompt.get("user_template"))


def _llm_name(llm: Mapping[str, Any]) -> str:
    backend = str(llm.get("backend", "openai_compatible"))
    if backend == "hf_transformers":
        return "hf"
    base_env = str(llm.get("base_url_env", "OPENAI_URL"))
    return "openrouter" if "OPENROUTER" in base_env else "openai"


@contextlib.contextmanager
def _llm_environment(llm: Mapping[str, Any]) -> Iterator[None]:
    provider = _llm_name(llm)
    if provider == "hf":
        yield
        return
    prefix = "OPENROUTER" if provider == "openrouter" else "OPENAI"
    updates: dict[str, str] = {}
    for suffix, field in (("URL", "base_url_env"), ("API_KEY", "api_key_env")):
        source = llm.get(field)
        if isinstance(source, str) and source:
            value_field = field.removesuffix("_env")
            updates[f"{prefix}_{suffix}"] = env_value({field: source}, value_field) or ""
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@app.command(name="eval-answer")
def eval_answer(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    retrieval_run: Path | None = typer.Option(None, "--retrieval-run"),
    limit: int | None = typer.Option(None, "--limit"),
    workers: int | None = typer.Option(None, "--workers"),
    questions_dir: Path | None = typer.Option(None, "--questions-dir"),
    data_root: Path | None = typer.Option(None, "--data-root"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Evaluate program or direct answering with none, gold, or retrieved context."""

    resolved = _resolved(config, {
        "answering.retrieval_run": retrieval_run,
        "evaluation.limit": limit,
        "evaluation.workers": workers,
        "paths.questions_dir": questions_dir,
        "paths.data_root": data_root,
        "paths.runs_dir": runs_dir,
    })
    answering = _section(resolved, "answering")
    llm = _section(resolved, "llm")
    evaluation = _section(resolved, "evaluation")
    paths = _section(resolved, "paths")
    args = ["eval-llm"]
    _option(args, "--llm", _llm_name(llm))
    _option(args, "--llm-model", llm.get("model_id"))
    _option(args, "--llm-reasoning-effort", llm.get("reasoning_effort"))
    _option(args, "--context", answering.get("context", "gold"))
    _option(args, "--strategy", answering.get("strategy", "pandas_query"))
    _option(args, "--retrieval-run", answering.get("retrieval_run"))
    _option(args, "--retrieval-k", answering.get("retrieval_k", 10))
    _option(args, "--max-context-tables", answering.get("max_context_tables"))
    _answering_args(
        args,
        answering=answering,
        llm=llm,
        evaluation=evaluation,
        prompt=_section(resolved, "prompt"),
    )
    _option(args, "--limit", evaluation.get("limit", 5))
    _option(args, "--workers", evaluation.get("workers", 1))
    _option(args, "--data-root", paths.get("data_root"))
    _option(args, "--questions-dir", paths.get("questions_dir"))
    _option(args, "--runs-dir", paths.get("runs_dir"))
    _option(args, "--company-meta-path", paths.get("company_meta_path"))
    with _llm_environment(llm):
        _delegate(baseline_app, args, resolved=resolved, record_new_runs=True)


@app.command(name="eval-llm", hidden=True)
def eval_llm_alias(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Compatibility alias for eval-answer."""

    eval_answer(
        config=config,
        retrieval_run=None,
        limit=None,
        workers=None,
        questions_dir=None,
        data_root=None,
        runs_dir=None,
    )


@app.command(name="eval-e2e")
def eval_e2e(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    limit: int | None = typer.Option(None, "--limit"),
    workers: int | None = typer.Option(None, "--workers"),
    questions_dir: Path | None = typer.Option(None, "--questions-dir"),
    data_root: Path | None = typer.Option(None, "--data-root"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Run retrieval and answer generation as one end-to-end evaluation."""

    resolved = _resolved(config, {
        "evaluation.limit": limit,
        "evaluation.workers": workers,
        "paths.questions_dir": questions_dir,
        "paths.data_root": data_root,
        "paths.cache_dir": cache_dir,
        "paths.runs_dir": runs_dir,
    })
    answering = _section(resolved, "answering")
    llm = _section(resolved, "llm")
    evaluation = _section(resolved, "evaluation")
    paths = _section(resolved, "paths")
    args = ["eval-e2e", *_retrieval_args(resolved, command="eval-e2e")]
    _option(args, "--llm", _llm_name(llm))
    _option(args, "--llm-model", llm.get("model_id"))
    _option(args, "--llm-reasoning-effort", llm.get("reasoning_effort"))
    _option(args, "--strategy", answering.get("strategy", "pandas_query"))
    _option(args, "--ks", evaluation.get("e2e_ks", [10]))
    _option(args, "--max-context-tables", answering.get("max_context_tables", 10))
    _answering_args(
        args,
        answering=answering,
        llm=llm,
        evaluation=evaluation,
        prompt=_section(resolved, "prompt"),
    )
    _option(args, "--limit", evaluation.get("limit", 5))
    _option(args, "--questions-dir", paths.get("questions_dir"))
    _option(args, "--runs-dir", paths.get("runs_dir"))
    with _llm_environment(llm):
        _delegate(baseline_app, args, resolved=resolved, record_new_runs=True)


@app.command(name="compare-runs")
def compare_runs(
    run_dirs: list[Path] = typer.Argument(...),
    out: Path = typer.Option(Path("runs/compare.md"), "--out"),
    config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
) -> None:
    """Create a Markdown comparison table from completed run directories."""

    content = compare_run_dirs(run_dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    if config is not None:
        resolved = _resolved(config)
        snapshot = out.with_suffix(".config.yaml")
        snapshot.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    typer.echo(f"wrote: {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
