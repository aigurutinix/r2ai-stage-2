"""CLI: `uv run vifinqa catalog|generate ...`"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from vifinqa.config import Settings, get_settings
from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.common.corpus.table import load_table
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.hard.recipe.grounded.feasibility import audit_all_grounded
from vifinqa.generation.hard.recipe.grounded.feasibility import (
    build_report_payload as build_grounded_report_payload,
)
from vifinqa.generation.hard.recipe.grounded.feasibility import (
    render_markdown as render_grounded_markdown,
)
from vifinqa.generation.panel.measurements.catalog import MEASUREMENTS
from vifinqa.generation.panel.measurements.evaluator import audit_all
from vifinqa.generation.panel.measurements.report import build_report_payload, render_markdown
from vifinqa.generation.panel.store import JsonCubeStore
from vifinqa.embeddings.base import Embedder
from vifinqa.embeddings.cache import CachedEmbedder
from vifinqa.embeddings.hf_local import HFLocalEmbedder
from vifinqa.generation.table_index.store import NumpyTableIndexStore
from vifinqa.generation.easy import generate_easy
from vifinqa.generation.medium import (
    CrossDocMode,
    generate_medium,
    generate_medium_cross_doc,
    generate_medium_same_doc,
)
from vifinqa.generation.intermediate import (
    generate_intermediate,
    generate_peer_group_longitudinal,
    generate_peer_group_two_periods,
    generate_same_doc_multi_inputs,
    generate_same_doc_multi_role_formula,
    generate_time_series_intermediate,
)
from vifinqa.generation.intermediate_feasibility import write_feasibility_report
from vifinqa.generation.retrieval.peer_group_longitudinal_report import (
    measure_retrieval_feasibility,
    write_report as write_peer_group_longitudinal_report,
)
from vifinqa.generation.hard.p1 import generate_hard_p1
from vifinqa.generation.hard.p3 import generate_hard_p3
from vifinqa.generation.hard.capacity_audit import (
    run_capacity_audit,
    write_capacity_report,
)
from vifinqa.generation.hard.p_cube import generate_hard_cube
from vifinqa.generation.hard.period_filter_select_lookup import (
    generate_period_filter_select_lookup,
)
from vifinqa.generation.hard.template_capacity import (
    build_template_capability_matrix,
    write_template_capability_matrix,
)
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS
from vifinqa.generation.prompts.intermediate import IntermediateMode
from vifinqa.generation.scenarios import ScenarioName, ScenarioSpec, get_scenario
from vifinqa.llm.openai_compatible import OpenAICompatibleLLM
from vifinqa.generation.manual.queue import ManualQueue, QueueError
from vifinqa.generation.manual.service import ManualWorkflow, prepare_manual_queue

app = typer.Typer(help="ViFinQA - generate financial questions from OCR reports")
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def catalog() -> None:
    """Report corpus counts for companies, documents, tables, and eligible tables."""
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    tickers = {d.ticker for d in docs}
    docs_with_text = [d for d in docs if d.has_text]
    total_tables = sum(len(d.table_ids) for d in docs)

    eligible = 0
    for d in docs:
        if d.tables_dir is None:
            continue
        for table_id in d.table_ids:
            table = load_table(
                d.table_csv_path(table_id),
                ticker=d.ticker,
                year=d.year,
                doc_name=d.doc_name,
                table_id=table_id,
            )
            if is_table_eligible(table):
                eligible += 1

    console.print(f"Companies: {len(tickers)}")
    console.print(f"Documents: {len(docs)} (with text: {len(docs_with_text)})")
    console.print(f"Total tables: {total_tables}")
    console.print(f"Eligible tables (>=3 rows, >=6 numeric cells): {eligible}")


panel_app = typer.Typer(help="Financial data cube keyed by Circular 200 line-item codes")
app.add_typer(panel_app, name="panel")

# These are exactly the 12 metrics measured in the plan. Keep the list fixed so actual
# coverage remains comparable with the recorded baseline.
_PANEL_STATS_METRICS = (
    "kqkd:10",
    "kqkd:11",
    "kqkd:20",
    "kqkd:50",
    "kqkd:60",
    "cdkt:100",
    "cdkt:270",
    "cdkt:300",
    "cdkt:310",
    "cdkt:400",
    "lctt:20",
    "lctt:70",
)


@panel_app.command("build")
def panel_build() -> None:
    """Build, or load from cache, a cube for the complete corpus."""
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    store = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir)
    cube = store.load_or_build()
    n_periods = sum(len(cube.years(ticker)) for ticker in cube.tickers())
    console.print(
        f"Cube: {len(cube.tickers())} companies, {n_periods} processed (ticker, year) pairs."
    )


@panel_app.command("stats")
def panel_stats() -> None:
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    store = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir)
    cube = store.load_or_build()
    company_meta = load_company_meta(settings.company_meta_path)
    credit_institution_tickers = {
        ticker
        for ticker, info in company_meta.items()
        if info.industry_l2 == "Tổ chức tín dụng"
    }

    periods = [
        (ticker, year)
        for ticker in cube.tickers()
        if ticker not in credit_institution_tickers
        for year in cube.years(ticker)
    ]
    total = len(periods)
    console.print(
        f"Total processed consolidated (ticker, year) pairs, excluding credit institutions: {total}"
    )
    for metric in _PANEL_STATS_METRICS:
        count = sum(
            1 for ticker, year in periods if metric in cube.metric_keys(ticker, year)
        )
        pct = (count / total * 100) if total else 0.0
        # Disable emoji so Rich does not parse strings such as "cdkt:100:" as emoji shortcodes.
        console.print(f"  {metric}: {count}/{total} ({pct:.1f}%)", emoji=False)


@panel_app.command("measurement-audit")
def panel_measurement_audit(
    json_out: Annotated[
        Path, typer.Option("--json-out", help="File JSON feasibility report")
    ] = Path("data/generated/hard_measurement_feasibility.json"),
    md_out: Annotated[
        Path, typer.Option("--md-out", help="File Markdown feasibility report")
    ] = Path("data/generated/hard_measurement_feasibility.md"),
) -> None:
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    store = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir)
    cube = store.load_or_build()
    company_meta = load_company_meta(settings.company_meta_path)

    reports = audit_all(MEASUREMENTS, cube, company_meta)
    payload = build_report_payload(reports)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(reports), encoding="utf-8")

    console.print(f"Wrote feasibility report: {json_out}, {md_out}")


@panel_app.command("grounded-recipe-feasibility")
def panel_grounded_recipe_feasibility(
    json_out: Annotated[
        Path,
        typer.Option("--json-out", help="File JSON grounded recipe feasibility report"),
    ] = Path("data/generated/hard_grounded_recipe_feasibility.json"),
    md_out: Annotated[
        Path,
        typer.Option(
            "--md-out", help="File Markdown grounded recipe feasibility report"
        ),
    ] = Path("data/generated/hard_grounded_recipe_feasibility.md"),
) -> None:
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    store = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir)
    cube = store.load_or_build()
    company_meta = load_company_meta(settings.company_meta_path)

    reports = audit_all_grounded(cube, company_meta)
    payload = build_grounded_report_payload(reports)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_grounded_markdown(reports), encoding="utf-8")

    console.print(f"Wrote grounded-recipe feasibility report: {json_out}, {md_out}")


@app.command("intermediate-new-scenarios-feasibility")
def intermediate_new_scenarios_feasibility(
    json_out: Annotated[
        Path, typer.Option("--json-out", help="File JSON feasibility report")
    ] = Path("data/generated/intermediate_new_scenarios_cp0_feasibility.json"),
    md_out: Annotated[
        Path, typer.Option("--md-out", help="File Markdown feasibility report")
    ] = Path("data/generated/intermediate_new_scenarios_cp0_report.md"),
) -> None:
    settings = get_settings()
    formula_results, rectangle_results = write_feasibility_report(
        data_root=settings.data_root,
        company_meta_path=settings.company_meta_path,
        json_out=json_out,
        md_out=md_out,
    )
    for r in formula_results:
        console.print(
            f"  formula={r.formula_id}: {r.documents_with_all_roles}/{r.documents_total} documents"
        )
    console.print(
        f"  rectangles (>=3 tickers in one industry_l3, 3 consecutive years): {len(rectangle_results)}"
    )
    console.print(f"Wrote feasibility report: {json_out}, {md_out}")


@app.command("hard-cube-v4-capacity")
def hard_cube_v4_capacity(
    json_out: Annotated[
        Path, typer.Option("--json-out", help="File JSON CP6 capacity report")
    ] = Path("data/generated/hard_cube_v4_cp6_capacity.json"),
    md_out: Annotated[
        Path, typer.Option("--md-out", help="File Markdown CP6 capacity report")
    ] = Path("data/generated/hard_cube_v4_cp6_report.md"),
    semantic_max_calls: Annotated[
        int,
        typer.Option(
            "--semantic-max-calls",
            help="Number of semantic-audit LLM batch calls; 0 runs deterministic checks only",
        ),
    ] = 0,
    semantic_candidate_limit: Annotated[
        int | None,
        typer.Option(
            "--semantic-candidate-limit",
            help="Candidate limit for semantic audit after projected gates",
        ),
    ] = None,
) -> None:
    settings = get_settings()
    llm = None
    if semantic_max_calls > 0:
        if (
            not settings.openai_url
            or not settings.openai_api_key
            or not settings.openai_model
        ):
            raise typer.BadParameter(
                "OPENAI_URL, OPENAI_API_KEY, and OPENAI_MODEL are required in .env for semantic audit."
            )
        llm = OpenAICompatibleLLM(
            base_url=settings.openai_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    result = run_capacity_audit(
        settings=settings,
        llm=llm,
        semantic_max_calls=semantic_max_calls or None,
        semantic_candidate_limit=semantic_candidate_limit,
    )
    write_capacity_report(result, json_out=json_out, md_out=md_out)
    payload = result.to_payload()["summary"]
    console.print(
        "CP6 capacity: "
        f"projected={payload['projected_population_ok']}, "
        f"projected_semantic_accept={payload['semantic_accepted']}, "
        f"dependency_verified={payload['dependency_verified']}, "
        f"families={payload['story_families']} -> {md_out}"
    )


@app.command("hard-template-capacity")
def hard_template_capacity(
    json_out: Annotated[
        Path,
        typer.Option("--json-out", help="Capability-matrix JSON for all 70 templates"),
    ] = Path("data/generated/hard_template_capability.json"),
    md_out: Annotated[
        Path,
        typer.Option("--md-out", help="Capability-matrix Markdown for all 70 templates"),
    ] = Path("data/generated/hard_template_capability.md"),
    max_drafts_per_template: Annotated[
        int, typer.Option("--max-drafts-per-template", min=1)
    ] = 100,
) -> None:
    """Probe typed compilers against current corpus and report all 70 template states."""
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    company_meta = load_company_meta(settings.company_meta_path)
    cube = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir).load_or_build()
    matrix = build_template_capability_matrix(
        cube=cube,
        company_meta=company_meta,
        max_drafts_per_template=max_drafts_per_template,
    )
    write_template_capability_matrix(matrix, json_out=json_out, markdown_out=md_out)
    console.print(
        f"Hard template capability: registry={len(matrix.templates)}/70, "
        f"implemented={sum(row.compiler_frame_id is not None for row in matrix.templates)}, "
        f"deterministic_candidates={matrix.runnable_count} -> {md_out}"
    )


@app.command("peer-group-longitudinal-retrieval-feasibility")
def peer_group_longitudinal_retrieval_feasibility(
    max_groups: Annotated[
        int | None,
        typer.Option(
            "--max-groups", help="Limit the number of sampled industry_l3/scope/window groups"
        ),
    ] = 20,
    max_triples_per_group: Annotated[
        int,
        typer.Option("--max-triples-per-group", help="Ticker triples sampled per group"),
    ] = 3,
    seed: Annotated[int, typer.Option(help="Seed for selecting ticker triples")] = 0,
    json_out: Annotated[Path, typer.Option("--json-out")] = Path(
        "data/generated/intermediate_peer_group_longitudinal_retrieval_report.json"
    ),
    md_out: Annotated[Path, typer.Option("--md-out")] = Path(
        "data/generated/intermediate_peer_group_longitudinal_retrieval_report.md"
    ),
) -> None:
    settings = get_settings()
    docs = scan_catalog(settings.data_root)
    companies = load_company_meta(settings.company_meta_path)
    embedder = _build_embedder(settings)

    results = measure_retrieval_feasibility(
        all_docs=docs,
        companies=companies,
        embedder=embedder,
        cache_dir=settings.cache_dir,
        embedding_model=settings.embedding_model,
        max_groups=max_groups,
        max_triples_per_group=max_triples_per_group,
        seed=seed,
    )
    write_peer_group_longitudinal_report(results, json_out=json_out, md_out=md_out)
    with_bundle = sum(1 for r in results if r.best_tickers is not None)
    console.print(
        f"Measured groups: {len(results)}, groups with a complete bundle: {with_bundle} -> {md_out}"
    )


_CROSS_MODE_MAP: dict[str, CrossDocMode] = {
    "same-company": "same_company_diff_year",
    "same-year": "same_year_diff_company",
}
_INTERMEDIATE_MODE_MAP: dict[str, IntermediateMode] = {
    "same-company": "multi_year_same_company",  # Fixed company across multiple years.
    "same-year": "multi_company_same_year",  # Fixed year across multiple companies.
}
_SCENARIO_MAP: dict[str, ScenarioName] = {
    "same-doc-two-inputs": "same_doc_two_inputs",
    "same-company-two-periods": "same_company_two_periods",
    "two-companies-same-period": "two_companies_same_period",
    "same-doc-multi-inputs": "same_doc_multi_inputs",
    "same-company-time-series": "same_company_time_series",
    "peer-group-same-period": "peer_group_same_period",
    "peer-group-two-periods": "peer_group_two_periods",
    "same-doc-multi-role-formula": "same_doc_multi_role_formula",
    "peer-group-longitudinal": "peer_group_longitudinal",
}
# Keep period handling explicit and deterministic.
_DEPRECATED_SCENARIO_ALIASES: dict[str, str] = {
    "same-doc-multi-inputs": "same-doc-multi-role-formula",
    "peer-group-two-periods": "peer-group-longitudinal",
}


def _build_embedder(settings: Settings) -> Embedder:
    return CachedEmbedder(
        HFLocalEmbedder(settings.embedding_model, hf_token=settings.hf_token),
        cache_dir=settings.cache_dir,
        model_name=settings.embedding_model,
    )


def _print_manual_payload(payload: dict[str, object]) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("manual-prepare")
def manual_prepare(
    count: Annotated[
        int, typer.Option(help="Number of evidence-pack work items to prepare")
    ] = 20,
    seed: Annotated[
        int, typer.Option(help="Deterministic pack-builder seed")
    ] = 20260716,
) -> None:
    """Create manual Hard evidence packs and work items without calling an LLM."""
    if count < 1:
        raise typer.BadParameter("--count must be at least 1")
    settings = get_settings()
    try:
        payload = prepare_manual_queue(
            settings=settings,
            count=count,
            seed=seed,
            repo_root=Path.cwd(),
        )
    except (QueueError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_manual_payload(payload)


@app.command("manual-claim-next")
def manual_claim_next() -> None:
    """Claim the next atomic work item for the current Codex session."""
    settings = get_settings()
    try:
        payload = ManualWorkflow(root=settings.hard_manual_root).claim_next()
    except QueueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_manual_payload(payload or {"claimed": False, "reason": "no_eligible_work"})


@app.command("manual-submit")
def manual_submit(
    work_id: Annotated[
        str, typer.Option("--work-id", help="Work ID claimed by the current session")
    ],
) -> None:
    """Validate and submit the author shard for the current work item."""
    settings = get_settings()
    try:
        payload = ManualWorkflow(root=settings.hard_manual_root).submit_author(work_id)
    except QueueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_manual_payload(payload)


@app.command("manual-status")
def manual_status() -> None:
    """Show the manual Hard queue status."""
    settings = get_settings()
    _print_manual_payload(ManualQueue(settings.hard_manual_root).status())


@app.command("manual-audit-claim-next")
def manual_audit_claim_next() -> None:
    """Claim an unreviewed atomic batch, excluding the current session's own batch."""
    settings = get_settings()
    try:
        payload = ManualWorkflow(root=settings.hard_manual_root).audit_claim_next()
    except QueueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_manual_payload(payload or {"claimed": False, "reason": "no_eligible_audit"})


@app.command("manual-audit-submit")
def manual_audit_submit(
    work_id: Annotated[
        str, typer.Option("--work-id", help="Work ID for the current audit claim")
    ],
) -> None:
    """Rerun validation and submit the batch's separate review shard."""
    settings = get_settings()
    try:
        payload = ManualWorkflow(root=settings.hard_manual_root).submit_audit(work_id)
    except QueueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_manual_payload(payload)


@app.command("manual-merge-accepted")
def manual_merge_accepted() -> None:
    """Curator-only: atomically materialize reviewed accepts into the canonical JSONL ledger."""
    settings = get_settings()
    try:
        payload = ManualQueue(settings.hard_manual_root).merge_accepted()
    except QueueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_manual_payload(payload)


@app.command()
def generate(
    tier: Annotated[
        str, typer.Option(help="easy | medium | intermediate | hard")
    ] = "easy",
    mode: Annotated[str, typer.Option(help="auto | same-doc | cross-doc")] = "auto",
    cross_criterion: Annotated[
        str,
        typer.Option(
            help=(
                "(medium cross-doc / intermediate) same-company (one company across years) | "
                "same-year (one year across companies)"
            )
        ),
    ] = "same-company",
    scenario: Annotated[
        str | None,
        typer.Option(
            help=(
                "tier=intermediate: explicit scenario (overrides legacy --mode/--cross-criterion): "
                + ", ".join(_SCENARIO_MAP)
                + " | tier=hard: 'p1' (default), 'p3', 'depth3', or 'cube'"
            )
        ),
    ] = None,
    count: Annotated[int, typer.Option(help="Number of questions to generate")] = 10,
    out: Annotated[Path, typer.Option(help="File JSONL output")] = Path(
        "data/generated/easy.jsonl"
    ),
    seed: Annotated[
        int | None, typer.Option(help="Random seed for reproducible table selection")
    ] = None,
    top_k: Annotated[
        int, typer.Option(help="(cross-doc) Candidate tables returned by semantic search")
    ] = 5,
    max_workers: Annotated[
        int,
        typer.Option(help="Number of candidates processed concurrently by the thread pool"),
    ] = DEFAULT_MAX_WORKERS,
    max_candidates: Annotated[
        int | None,
        typer.Option(
            help="Maximum candidate attempts; use 1 during review to prevent unintended API calls"
        ),
    ] = None,
    max_llm_calls: Annotated[
        int | None,
        typer.Option(
            help="(hard p3/depth3/cube) Total ChatLLM.complete budget; stop before exceeding it"
        ),
    ] = None,
    cube_frame: Annotated[
        str | None,
        typer.Option(
            "--cube-frame",
            help="Run one enabled analytical frame; intended for Hard Cube checkpoint smoke tests.",
        ),
    ] = None,
    cube_deep: Annotated[
        bool,
        typer.Option(
            "--cube-deep/--cube-standard",
            help=(
                "Hard Cube: require reasoning_depth>=3 and adaptive_edges>=2 (deep), or "
                "retain standard Hard gates while allowing shallower topology (standard)."
            ),
        ),
    ] = True,
    dedup_against: Annotated[
        list[Path] | None,
        typer.Option(
            "--dedup-against",
            help=(
                "(hard cube) Existing JSONL ledger; remove duplicate canonical queries/questions "
                "before downstream LLM audits."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable DEBUG logs with per-stage failure reasons: JSON, scope, answer, check_answer",
        ),
    ] = False,
) -> None:
    """Generate questions for the requested difficulty tier."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Root DEBUG is useful for qagen traces, but OpenAI/httpx/httpcore may log complete
    # request bodies, including very long prompts. Keep third-party libraries at WARNING.
    for noisy_logger in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    hard_scenario = scenario or "p1" if tier == "hard" else None
    if max_llm_calls is not None:
        if max_llm_calls < 1:
            raise typer.BadParameter("--max-llm-calls must be at least 1")
        if tier != "hard" or hard_scenario not in ("p3", "depth3", "cube"):
            raise typer.BadParameter(
                "--max-llm-calls currently applies only to Hard p3/depth3/cube"
            )
    if cube_frame is not None and (tier != "hard" or hard_scenario != "cube"):
        raise typer.BadParameter(
            "--cube-frame applies only to --tier hard --scenario cube"
        )
    if not cube_deep and (tier != "hard" or hard_scenario != "cube"):
        raise typer.BadParameter(
            "--cube-standard applies only to --tier hard --scenario cube"
        )
    if dedup_against is not None:
        if tier != "hard" or hard_scenario != "cube":
            raise typer.BadParameter(
                "--dedup-against applies only to --tier hard --scenario cube"
            )
        missing = [path for path in dedup_against if not path.is_file()]
        if missing:
            raise typer.BadParameter(f"--dedup-against does not exist: {missing[0]}")
    settings = get_settings()

    scenario_spec: ScenarioSpec | None = None
    if scenario is not None and tier in ("medium", "intermediate"):
        if scenario in _DEPRECATED_SCENARIO_ALIASES:
            replacement = _DEPRECATED_SCENARIO_ALIASES[scenario]
            logger.warning(
                "--scenario=%s is deprecated and dispatches to --scenario=%s (new implementation; "
                "see intermediate_v2_rework_plan.md section 8). Update the command to the new name.",
                scenario,
                replacement,
            )
            scenario = replacement
        if scenario not in _SCENARIO_MAP:
            raise typer.BadParameter(
                f"--scenario must be one of {list(_SCENARIO_MAP)}"
            )
        scenario_spec = get_scenario(_SCENARIO_MAP[scenario])
        # Known limitation: Typer cannot distinguish omitted values from explicitly
        # supplied defaults, so explicit ``--mode auto --cross-criterion same-company``
        # is accepted as-is.
        if mode != "auto" or cross_criterion != "same-company":
            raise typer.BadParameter(
                "Do not combine --scenario with non-default --mode/--cross-criterion values."
            )
        expected_tier = (
            "medium"
            if scenario_spec.difficulty == "medium"
            else "intermediate"
        )
        if tier != expected_tier:
            raise typer.BadParameter(
                f"--scenario={scenario} requires --tier {expected_tier}."
            )

    if (
        not settings.openai_url
        or not settings.openai_api_key
        or not settings.openai_model
    ):
        raise typer.BadParameter(
            "OPENAI_URL, OPENAI_API_KEY, and OPENAI_MODEL are missing from .env; see .env.example."
        )

    llm = OpenAICompatibleLLM(
        base_url=settings.openai_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    if scenario_spec is not None:
        embedder = _build_embedder(settings)
        if scenario_spec.name == "same_doc_two_inputs":
            generated = generate_medium_same_doc(
                settings=settings,
                llm=llm,
                count=count,
                out_path=out,
                seed=seed,
                max_workers=max_workers,
                max_candidates=max_candidates,
                scenario=scenario_spec,
            )
        elif scenario_spec.name == "same_company_two_periods":
            generated = generate_medium_cross_doc(
                settings=settings,
                llm=llm,
                embedder=embedder,
                count=count,
                out_path=out,
                seed=seed,
                mode="same_company_diff_year",
                top_k=top_k,
                max_workers=max_workers,
                max_candidates=max_candidates,
                scenario=scenario_spec,
            )
        elif scenario_spec.name == "two_companies_same_period":
            generated = generate_medium_cross_doc(
                settings=settings,
                llm=llm,
                embedder=embedder,
                count=count,
                out_path=out,
                seed=seed,
                mode="same_year_diff_company",
                top_k=top_k,
                max_workers=max_workers,
                max_candidates=max_candidates,
                scenario=scenario_spec,
            )
        elif scenario_spec.name == "same_doc_multi_inputs":
            generated = generate_same_doc_multi_inputs(
                settings=settings,
                llm=llm,
                count=count,
                out_path=out,
                seed=seed,
                scenario=scenario_spec,
                max_workers=max_workers,
                max_candidates=max_candidates,
            )
        elif scenario_spec.name == "same_company_time_series":
            generated = generate_time_series_intermediate(
                settings=settings,
                llm=llm,
                embedder=embedder,
                count=count,
                out_path=out,
                seed=seed,
                scenario=scenario_spec,
                max_workers=max_workers,
                max_candidates=max_candidates,
            )
        elif scenario_spec.name == "peer_group_same_period":
            generated = generate_intermediate(
                settings=settings,
                llm=llm,
                embedder=embedder,
                count=count,
                out_path=out,
                seed=seed,
                mode="multi_company_same_year",
                top_k_search=max(top_k, 30),
                max_workers=max_workers,
                max_candidates=max_candidates,
                scenario=scenario_spec,
            )
        elif scenario_spec.name == "peer_group_two_periods":
            generated = generate_peer_group_two_periods(
                settings=settings,
                llm=llm,
                embedder=embedder,
                count=count,
                out_path=out,
                seed=seed,
                scenario=scenario_spec,
                max_workers=max_workers,
                max_candidates=max_candidates,
            )
        elif scenario_spec.name == "same_doc_multi_role_formula":
            index_store = NumpyTableIndexStore(
                embedder=embedder,
                cache_dir=settings.cache_dir,
                embedding_model=settings.embedding_model,
            )
            generated = generate_same_doc_multi_role_formula(
                settings=settings,
                llm=llm,
                index_store=index_store,
                count=count,
                out_path=out,
                seed=seed,
                scenario=scenario_spec,
                max_workers=max_workers,
                max_candidates=max_candidates,
            )
        else:  # peer_group_longitudinal
            index_store = NumpyTableIndexStore(
                embedder=embedder,
                cache_dir=settings.cache_dir,
                embedding_model=settings.embedding_model,
            )
            generated = generate_peer_group_longitudinal(
                settings=settings,
                llm=llm,
                index_store=index_store,
                count=count,
                out_path=out,
                seed=seed,
                scenario=scenario_spec,
                max_workers=max_workers,
                max_candidates=max_candidates,
            )
    elif tier == "easy":
        generated = generate_easy(
            settings=settings,
            llm=llm,
            count=count,
            out_path=out,
            seed=seed,
            max_workers=max_workers,
            max_candidates=max_candidates,
        )
    elif tier == "medium" and mode == "auto":
        generated = generate_medium(
            settings=settings,
            llm=llm,
            embedder=_build_embedder(settings),
            count=count,
            out_path=out,
            seed=seed,
            top_k=top_k,
            max_workers=max_workers,
            max_candidates=max_candidates,
        )
    elif tier == "medium" and mode == "same-doc":
        generated = generate_medium_same_doc(
            settings=settings,
            llm=llm,
            count=count,
            out_path=out,
            seed=seed,
            max_workers=max_workers,
            max_candidates=max_candidates,
        )
    elif tier == "medium" and mode == "cross-doc":
        if cross_criterion not in _CROSS_MODE_MAP:
            raise typer.BadParameter(f"cross-criterion must be one of {list(_CROSS_MODE_MAP)}")
        generated = generate_medium_cross_doc(
            settings=settings,
            llm=llm,
            embedder=_build_embedder(settings),
            count=count,
            out_path=out,
            seed=seed,
            mode=_CROSS_MODE_MAP[cross_criterion],
            top_k=top_k,
            max_workers=max_workers,
            max_candidates=max_candidates,
        )
    elif tier == "intermediate":
        if mode != "auto" and cross_criterion not in _INTERMEDIATE_MODE_MAP:
            raise typer.BadParameter(
                f"cross-criterion must be one of {list(_INTERMEDIATE_MODE_MAP)}"
            )
        generated = generate_intermediate(
            settings=settings,
            llm=llm,
            embedder=_build_embedder(settings),
            count=count,
            out_path=out,
            seed=seed,
            mode="auto" if mode == "auto" else _INTERMEDIATE_MODE_MAP[cross_criterion],
            top_k_search=max(top_k, 30),
            max_workers=max_workers,
            max_candidates=max_candidates,
        )
    elif tier == "hard":
        # Preserve the default behavior for compatibility.
        # Keep period handling explicit and deterministic.
        hard_scenario = scenario or "p1"
        if hard_scenario == "p1":
            generated = generate_hard_p1(
                settings=settings,
                llm=llm,
                embedder=_build_embedder(settings),
                count=count,
                out_path=out,
                seed=seed,
                top_k_search=max(top_k, 30),
                max_workers=max_workers,
                max_candidates=max_candidates,
            )
        elif hard_scenario == "p3":
            index_store = NumpyTableIndexStore(
                embedder=_build_embedder(settings),
                cache_dir=settings.cache_dir,
                embedding_model=settings.embedding_model,
            )
            generated = generate_hard_p3(
                settings=settings,
                llm=llm,
                index_store=index_store,
                count=count,
                out_path=out,
                seed=seed,
                max_workers=max_workers,
                max_candidates=max_candidates,
                max_llm_calls=max_llm_calls,
            )
        elif hard_scenario == "depth3":
            index_store = NumpyTableIndexStore(
                embedder=_build_embedder(settings),
                cache_dir=settings.cache_dir,
                embedding_model=settings.embedding_model,
            )
            generated = generate_period_filter_select_lookup(
                settings=settings,
                llm=llm,
                index_store=index_store,
                count=count,
                out_path=out,
                seed=seed,
                max_workers=max_workers,
                max_candidates=max_candidates,
                max_llm_calls=max_llm_calls,
            )
        elif hard_scenario == "cube":
            generated = generate_hard_cube(
                settings=settings,
                llm=llm,
                count=count,
                out_path=out,
                seed=seed,
                max_llm_calls=max_llm_calls,
                frame_id=cube_frame,
                dedup_against_paths=tuple(dedup_against or ()),
                require_deep_hard=cube_deep,
            )
        else:
            raise typer.BadParameter(
                "--scenario for tier='hard' must be 'p1', 'p3', 'depth3', or 'cube'."
            )
    else:
        raise typer.BadParameter(
            f"The tier='{tier}' mode='{mode}' combination is not implemented."
        )

    console.print(f"Generated {generated}/{count} questions -> {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
