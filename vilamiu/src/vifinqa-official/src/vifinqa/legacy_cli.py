"""Typer entry point for ``uv run vifinqa <command>``.

Concrete embedder, reranker, LLM, and retriever wiring is centralized here; other
modules depend only on protocols.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from collections import Counter
from functools import partial
from pathlib import Path

import typer

from vifinqa.answering.base import AnswerStrategy
from vifinqa.answering.direct_answer import DirectAnswerStrategy
from vifinqa.answering.pandas_query import PandasQueryStrategy
from vifinqa.answering.prompts import (
    COMMON_USER_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    PROGRAM_SYSTEM_PROMPT,
)
from vifinqa.answering.table_metadata import build_table_metadata_lookup
from vifinqa.config import Settings, get_settings
from vifinqa.constants import (
    ANSWER_ABS_TOL,
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP_ROWS,
    DEFAULT_CHUNK_ROWS,
    DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_MAX_CONTEXT_TABLES,
    DEFAULT_SUMMARY_MAX_CHARS,
    CHUNKED_INDEX_FORMAT_VERSION,
    EMBEDDING_MAX_SEQ_LENGTH,
    E2E_K_SWEEP,
    MULTI_VIEW_INDEX_FORMAT_VERSION,
    RETRIEVAL_K_SWEEP,
)
from vifinqa.common.corpus.catalog import build_doc_name_lookup, scan_catalog
from vifinqa.common.corpus.company_meta import get_company_meta
from vifinqa.embeddings.cache import CachedEmbedder
from vifinqa.embeddings.hf_local import HFLocalEmbedder
from vifinqa.encoding.table_text import METADATA_CSV_LAYOUTS, TABLE_ENCODERS
from vifinqa.encoding.row_chunks import (
    CHUNKED_TABLE_ENCODING,
    MULTI_VIEW_TABLE_ENCODING,
    ROW_CHUNK_VIEW,
    TABLE_VIEWS,
    RowChunkConfig,
)
from vifinqa.encoding.chunk_audit import audit_chunk_lengths
from vifinqa.evaluation.compare import compare_run_dirs
from vifinqa.evaluation.e2e_runner import evaluate_questions as evaluate_e2e_questions
from vifinqa.evaluation.llm_runner import evaluate_questions as evaluate_llm_questions
from vifinqa.evaluation.report import (
    append_answer_checkpoint,
    load_retrieved_tables,
    make_run_dir,
    model_run_tag,
    write_answer_report,
    write_config,
    write_retrieval_report,
)
from vifinqa.evaluation.retrieval_runner import (
    evaluate_questions as evaluate_retrieval_questions,
)
from vifinqa.llm.base import ChatLLM, LLMError
from vifinqa.llm.hf_local import HFLocalLLM
from vifinqa.llm.openai_compatible import OpenAICompatibleLLM
from vifinqa.common.schemas.loader import discover_question_files, load_question_files
from vifinqa.common.schemas.sampling import sample_from_sources, sample_per_source
from vifinqa.rerankers.base import Reranker
from vifinqa.rerankers.cross_encoder import (
    CrossEncoderReranker,
    QWEN3_FINANCIAL_TABLE_INSTRUCTION,
)
from vifinqa.retrieval.base import RetrievalIndex
from vifinqa.retrieval.bm25 import load_or_build_bm25_index
from vifinqa.retrieval.cached import CachedRetrievalIndex
from vifinqa.retrieval.cascade_reranked import FairCascadeRerankedIndex
from vifinqa.retrieval.chunked_dense import load_or_build_chunked_dense_index
from vifinqa.retrieval.entity_balance import MultiEntityBalancedIndex
from vifinqa.retrieval.dense import load_or_build_dense_index
from vifinqa.retrieval.metadata_filter import (
    METADATA_POLICIES,
    FinalScopeGuard,
    MetadataFilteredIndex,
)
from vifinqa.retrieval.reranked import RerankedIndex
from vifinqa.retrieval.rrf import DEFAULT_RRF_K, RRFIndex

app = typer.Typer(no_args_is_help=True, add_completion=False)

RETRIEVERS = ("dense", "bm25", "rrf")
RERANKERS = ("none", "bge", "qwen3")
RERANKER_INSTRUCTIONS = ("default", "financial_table_v1")
_DEFAULT_QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
LLMS = ("openai", "openrouter", "hf")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
STRATEGIES = ("pandas_query", "direct_answer")
CONTEXTS = ("none", "gold", "retrieved")
TABLE_ENCODINGS = (*TABLE_ENCODERS, CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING)


def _validate_retrieval_representation(retriever: str, table_encoding: str) -> None:
    if table_encoding not in TABLE_ENCODINGS:
        raise ValueError(
            f"Invalid table_encoding: {table_encoding!r} (choose from {sorted(TABLE_ENCODINGS)})"
        )
    if (
        table_encoding in (CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING)
        and retriever != "dense"
    ):
        raise ValueError(
            "Chunk and multi-view encodings support only retriever=dense in V1; "
            f"retriever={retriever} is not supported"
        )


def _parse_table_views(
    value: str | None, *, default: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return default
    views = tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )
    if not views:
        raise ValueError("table_views must not be empty")
    unknown = set(views) - set(TABLE_VIEWS)
    if unknown:
        raise ValueError(
            f"Invalid table views: {sorted(unknown)} (choose from {TABLE_VIEWS})"
        )
    return views


def _resolve_table_views(
    table_encoding: str,
    *,
    table_views: str | None,
    active_table_views: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if table_encoding != MULTI_VIEW_TABLE_ENCODING:
        return (ROW_CHUNK_VIEW,), (ROW_CHUNK_VIEW,)
    built = _parse_table_views(table_views, default=TABLE_VIEWS)
    active = _parse_table_views(active_table_views, default=built)
    if set(active) - set(built):
        raise ValueError("active_table_views must be a subset of the built table_views")
    return built, active


def _build_retrieval_index(
    *,
    retriever: str,
    embedding_model: str,
    table_encoding: str,
    data_root: Path,
    cache_dir: Path,
    company_meta_path: Path,
    hf_token: str | None,
    rebuild: bool = False,
    rrf_fetch_n: int = 500,
    rrf_k: int = DEFAULT_RRF_K,
    embed_batch_size: int | None = None,
    chunk_config: RowChunkConfig = RowChunkConfig(),
    table_views: tuple[str, ...] = TABLE_VIEWS,
    active_table_views: tuple[str, ...] | None = None,
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    metadata_csv_layout: str = "head",
    view_fusion: str = "max",
    rerank_chunks_per_table: int = 1,
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
) -> RetrievalIndex:
    _validate_retrieval_representation(retriever, table_encoding)
    if retriever == "dense":
        embedder = CachedEmbedder(
            HFLocalEmbedder(
                embedding_model,
                hf_token=hf_token,
                batch_size=embed_batch_size,
                max_tokens=embedding_max_tokens,
            ),
            cache_dir=cache_dir,
            model_name=embedding_model,
            max_tokens=embedding_max_tokens,
        )
        if table_encoding in (CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING):
            return load_or_build_chunked_dense_index(
                embedder=embedder,
                data_root=data_root,
                cache_dir=cache_dir,
                embedding_model=embedding_model,
                company_meta_path=company_meta_path,
                config=chunk_config,
                table_encoding=table_encoding,
                views=table_views,
                active_views=active_table_views,
                summary_max_chars=summary_max_chars,
                metadata_csv_layout=metadata_csv_layout,
                view_fusion=view_fusion,
                rerank_chunks_per_table=rerank_chunks_per_table,
                embedding_max_tokens=embedding_max_tokens,
                rebuild=rebuild,
            )
        return load_or_build_dense_index(
            embedder=embedder,
            data_root=data_root,
            cache_dir=cache_dir,
            embedding_model=embedding_model,
            table_encoding=table_encoding,
            company_meta_path=company_meta_path,
            embedding_max_tokens=embedding_max_tokens,
            rebuild=rebuild,
        )
    if retriever == "bm25":
        return load_or_build_bm25_index(
            data_root=data_root,
            cache_dir=cache_dir,
            table_encoding=table_encoding,
            company_meta_path=company_meta_path,
            rebuild=rebuild,
        )
    if retriever == "rrf":
        # RRF needs no separate index: it combines ranks from the existing dense and
        # BM25 indexes, each of which has its own cache path.
        dense = _build_retrieval_index(
            retriever="dense",
            embedding_model=embedding_model,
            table_encoding=table_encoding,
            data_root=data_root,
            cache_dir=cache_dir,
            company_meta_path=company_meta_path,
            hf_token=hf_token,
            rebuild=rebuild,
            embed_batch_size=embed_batch_size,
            chunk_config=chunk_config,
            table_views=table_views,
            active_table_views=active_table_views,
            summary_max_chars=summary_max_chars,
            metadata_csv_layout=metadata_csv_layout,
            view_fusion=view_fusion,
            rerank_chunks_per_table=rerank_chunks_per_table,
            embedding_max_tokens=embedding_max_tokens,
        )
        bm25 = _build_retrieval_index(
            retriever="bm25",
            embedding_model=embedding_model,
            table_encoding=table_encoding,
            data_root=data_root,
            cache_dir=cache_dir,
            company_meta_path=company_meta_path,
            hf_token=hf_token,
            rebuild=rebuild,
            chunk_config=chunk_config,
            table_views=table_views,
            active_table_views=active_table_views,
            summary_max_chars=summary_max_chars,
            metadata_csv_layout=metadata_csv_layout,
            view_fusion=view_fusion,
            rerank_chunks_per_table=rerank_chunks_per_table,
            embedding_max_tokens=embedding_max_tokens,
        )
        # bm25s returns a zero-score tail when tokens do not match. Do not let those
        # hits contribute to RRF. Dense has no floor because negative cosine is valid.
        return RRFIndex(
            [dense, bm25], fetch_top_n=rrf_fetch_n, rrf_k=rrf_k, min_scores=[None, 0.0]
        )
    raise ValueError(f"Invalid retriever: {retriever!r} (choose from {RETRIEVERS})")


def _search_cache_namespace(
    *,
    retriever: str,
    table_encoding: str,
    embedding_model: str | None,
    reranker: str,
    reranker_model: str | None,
    rerank_top_n: int,
    reranker_instruction: str = "default",
    rrf_fetch_n: int | None = None,
    rrf_k: int | None = None,
    chunk_config: RowChunkConfig = RowChunkConfig(),
    table_views: tuple[str, ...] = TABLE_VIEWS,
    active_table_views: tuple[str, ...] | None = None,
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    metadata_csv_layout: str = "head",
    view_fusion: str = "max",
    rerank_chunks_per_table: int = 1,
    multi_chunk_parent_n: int = 0,
    metadata_policy: str = "none",
    metadata_fetch_n: int = 5_000,
    metadata_global_fallback: int = 50,
    entity_balance: bool = False,
    rerank_flow: str = "single_stage",
    cascade_top_n: int = 150,
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
    reranker_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
) -> str:
    """Build the ``CachedRetrievalIndex`` namespace from every search-affecting flag.

    Commands with identical configurations can reuse a cache, while any configuration
    difference changes the namespace and prevents stale cache reuse.
    """
    parts = [
        retriever,
        table_encoding,
        embedding_model or "",
        reranker,
        reranker_model or "",
    ]
    if reranker != "none":
        parts.append(str(rerank_top_n))
        # Keep the default suffix-free for compatibility with baseline caches. Custom
        # presets must be versioned so revised instructions do not reuse old scores.
        if reranker_instruction != "default":
            parts.append(f"instruction-{reranker_instruction}")
    if retriever == "rrf":
        parts.append(str(rrf_fetch_n))
        parts.append(str(rrf_k))
    if table_encoding == CHUNKED_TABLE_ENCODING:
        parts.extend([f"chunk-v{CHUNKED_INDEX_FORMAT_VERSION}", chunk_config.namespace])
        # Preserve the baseline namespace for the single best row chunk. Add a suffix
        # only when reranking multiple chunks for the same parent.
        if rerank_chunks_per_table != 1 or multi_chunk_parent_n != 0:
            parts.extend(
                [
                    f"rerank-chunks-{rerank_chunks_per_table}",
                    f"multi-parent-{multi_chunk_parent_n}",
                ]
            )
    elif table_encoding == MULTI_VIEW_TABLE_ENCODING:
        parts.extend(
            [f"multi-view-v{MULTI_VIEW_INDEX_FORMAT_VERSION}", chunk_config.namespace]
        )
        parts.extend(
            [
                f"views-{'+'.join(table_views)}",
                f"active-{'+'.join(active_table_views or table_views)}",
                f"summary-{summary_max_chars}",
                f"csv-layout-{metadata_csv_layout}",
                f"fusion-{view_fusion}",
                f"rerank-chunks-{rerank_chunks_per_table}",
                f"multi-parent-{multi_chunk_parent_n}",
            ]
        )
    if metadata_policy != "none":
        parts.extend(
            [
                f"metadata-{metadata_policy}",
                f"metadata-fetch-{metadata_fetch_n}",
                f"metadata-fallback-{metadata_global_fallback}",
            ]
        )
    if entity_balance:
        parts.append("entity-balance-explicit-v1")
    if rerank_flow != "single_stage":
        parts.extend([f"rerank-flow-{rerank_flow}", f"cascade-top-{cascade_top_n}"])
    if embedding_max_tokens != EMBEDDING_MAX_SEQ_LENGTH:
        parts.append(f"embedding-tokens-{embedding_max_tokens}")
    if reranker_max_tokens != EMBEDDING_MAX_SEQ_LENGTH:
        parts.append(f"reranker-tokens-{reranker_max_tokens}")
    return "|".join(parts)


def _resolve_reranker_instruction(reranker: str, preset: str) -> str | None:
    if preset not in RERANKER_INSTRUCTIONS:
        raise ValueError(
            f"Invalid reranker_instruction={preset!r} (choose from {RERANKER_INSTRUCTIONS})"
        )
    if preset == "default":
        return None
    if reranker != "qwen3":
        raise ValueError("Custom reranker instructions are supported only with reranker=qwen3")
    return QWEN3_FINANCIAL_TABLE_INSTRUCTION


def _build_reranker(
    reranker: str,
    reranker_model: str | None,
    hf_token: str | None,
    *,
    instruction: str | None = None,
    max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
) -> Reranker | None:
    if reranker == "none":
        return None
    if reranker_model is None:
        raise ValueError(f"reranker={reranker!r} requires a model resolved by the CLI wiring")
    if reranker == "bge":
        if instruction is not None:
            raise ValueError("Custom instructions are not supported for reranker=bge")
        return CrossEncoderReranker(
            reranker_model, hf_token=hf_token, max_tokens=max_tokens
        )
    if reranker == "qwen3":
        return CrossEncoderReranker(
            reranker_model,
            hf_token=hf_token,
            instruction=instruction,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Invalid reranker: {reranker!r} (choose from {RERANKERS})")


def _resolved_llm_model(
    llm: str, llm_model: str | None, settings: Settings
) -> str | None:
    if llm == "openai":
        return llm_model or settings.openai_model
    if llm == "openrouter":
        return llm_model or settings.openrouter_model
    return llm_model


def _build_llm(
    llm: str,
    llm_model: str | None,
    settings: Settings,
    *,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
    max_completion_tokens: int = 8192,
    max_retries: int = 0,
) -> ChatLLM:
    if llm == "openai":
        return OpenAICompatibleLLM(
            base_url=settings.openai_url,
            api_key=settings.openai_api_key,
            model=llm_model or settings.openai_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
        )
    if llm == "openrouter":
        return OpenAICompatibleLLM(
            base_url=settings.openrouter_url,
            api_key=settings.openrouter_api_key,
            model=llm_model or settings.openrouter_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort or settings.openrouter_reasoning_effort,
        )
    if llm == "hf":
        if not llm_model:
            raise ValueError(
                "llm=hf requires --llm-model because there is no default model"
            )
        return HFLocalLLM(
            llm_model,
            hf_token=settings.hf_token,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Invalid llm: {llm!r} (choose from {LLMS})")


def _build_answer_strategy(
    strategy: str,
    *,
    table_max_chars: int,
    system_path: str | None = None,
    user_template_path: str | None = None,
) -> AnswerStrategy:
    if strategy == "pandas_query":
        return PandasQueryStrategy(
            table_max_chars=table_max_chars,
            system_path=system_path or PROGRAM_SYSTEM_PROMPT,
            user_template_path=user_template_path or COMMON_USER_PROMPT,
        )
    if strategy == "direct_answer":
        return DirectAnswerStrategy(
            table_max_chars=table_max_chars,
            system_path=system_path or DIRECT_SYSTEM_PROMPT,
            user_template_path=user_template_path or COMMON_USER_PROMPT,
        )
    raise ValueError(f"Invalid strategy: {strategy!r} (choose from {STRATEGIES})")


@app.command()
def version() -> None:
    typer.echo(importlib.metadata.version("vifinqa"))


@app.command()
def catalog(
    data_root: Path = typer.Option(
        None, help="ocr_filter directory (defaults to DATA_ROOT from .env)."
    ),
) -> None:
    """Report corpus counts for companies, documents, and tables."""
    settings = get_settings()
    root = data_root or settings.data_root
    docs = scan_catalog(root)

    tickers = {d.ticker for d in docs}
    n_tables = sum(len(d.table_ids) for d in docs)
    n_with_text = sum(1 for d in docs if d.has_text)

    typer.echo(f"data_root: {root}")
    typer.echo(f"companies: {len(tickers)}")
    typer.echo(f"documents: {len(docs)} ({n_with_text} with extracted.txt)")
    typer.echo(f"tables:    {n_tables}")


@app.command()
def data(
    questions_dir: Path = typer.Option(
        None, help="Directory containing *.jsonl (defaults to QUESTIONS_DIR from .env)."
    ),
    show: int = typer.Option(5, help="Number of sample questions to print."),
) -> None:
    """Load and merge JSONL questions, renumber IDs, and validate CSV paths."""
    settings = get_settings()
    directory = questions_dir or settings.questions_dir
    paths = discover_question_files(directory)
    if not paths:
        typer.echo(f"No .jsonl files found in {directory}")
        raise typer.Exit(code=1)

    questions = load_question_files(paths)
    typer.echo(f"questions_dir: {directory}")
    typer.echo(f"source files: {[p.name for p in paths]}")
    typer.echo(f"total questions: {len(questions)}")

    by_difficulty = Counter(q.difficulty for q in questions)
    for difficulty, count in sorted(by_difficulty.items()):
        typer.echo(f"  {difficulty}: {count}")

    missing = 0
    for q in questions:
        for csv_path in q.csv_paths:
            if not csv_path.exists():
                missing += 1
    typer.echo(
        f"csv_path missing: {missing}/{sum(len(q.csv_paths) for q in questions)}"
    )

    typer.echo(f"\n--- {min(show, len(questions))} sample questions ---")
    for q in questions[:show]:
        typer.echo(
            f"[{q.id}] original={q.original} difficulty={q.difficulty} tables={q.relevant_tables}"
        )
        typer.echo(f"    {q.question}")


@app.command(name="audit-row-chunks")
def audit_row_chunks(
    embedding_model: str = typer.Option(
        None, help="Tokenizer model; defaults to EMBEDDING_MODEL from .env."
    ),
    chunk_rows: int = typer.Option(DEFAULT_CHUNK_ROWS),
    chunk_overlap_rows: int = typer.Option(DEFAULT_CHUNK_OVERLAP_ROWS),
    context_max_chars: int = typer.Option(DEFAULT_CONTEXT_MAX_CHARS),
    chunk_max_chars: int = typer.Option(DEFAULT_CHUNK_MAX_CHARS),
    tokenizer_batch_size: int = typer.Option(256),
    data_root: Path = typer.Option(None),
    company_meta_path: Path = typer.Option(None),
) -> None:
    """Audit row-chunk character and token counts without building a vector index."""
    from transformers import AutoTokenizer

    from vifinqa.constants import EMBEDDING_MAX_SEQ_LENGTH
    from vifinqa.retrieval.chunked_corpus_builder import build_chunked_corpus

    settings = get_settings()
    model_name = embedding_model or settings.embedding_model
    try:
        chunk_config = RowChunkConfig(
            chunk_rows=chunk_rows,
            chunk_overlap_rows=chunk_overlap_rows,
            context_max_chars=context_max_chars,
            chunk_max_chars=chunk_max_chars,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    collected = build_chunked_corpus(
        data_root or settings.data_root,
        company_meta_path=company_meta_path or settings.company_meta_path,
        config=chunk_config,
        log_truncations=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=settings.hf_token)

    def count_tokens(texts: list[str]) -> list[int]:
        encoded = tokenizer(
            texts, add_special_tokens=True, truncation=False, return_length=True
        )
        return [int(length) for length in encoded["length"]]

    audit = audit_chunk_lengths(
        collected.chunks,
        count_tokens=count_tokens,
        max_tokens=EMBEDDING_MAX_SEQ_LENGTH,
        batch_size=tokenizer_batch_size,
    )
    audit.update(
        {
            "table_encoding": CHUNKED_TABLE_ENCODING,
            "chunk_config": chunk_config.as_dict(),
        }
    )
    audit.update(
        {
            "parent_tables": collected.stats.parent_tables,
            "safety_split_tables": collected.stats.safety_split_tables,
            "safety_split_chunks": collected.stats.safety_split_chunks,
            "safety_split_count": collected.stats.safety_split_count,
            "truncated_tables": collected.stats.truncated_tables,
            "truncated_chunks": collected.stats.truncated_chunks,
            "pathological_truncations": collected.stats.pathological_truncations,
        }
    )
    typer.echo(json.dumps(audit, ensure_ascii=False, indent=2))


@app.command(name="build-index")
def build_index(
    retriever: str = typer.Option(
        "dense", help=f"{'|'.join(RETRIEVERS)} (rrf builds both dense and BM25)."
    ),
    embedding_model: str = typer.Option(
        None, help="Defaults to EMBEDDING_MODEL from .env (used by dense/rrf)."
    ),
    table_encoding: str = typer.Option(
        "table_retrieval_text", help=f"{'|'.join(TABLE_ENCODINGS)}."
    ),
    chunk_rows: int = typer.Option(
        DEFAULT_CHUNK_ROWS, help="[row chunks] Number of CSV rows per window."
    ),
    chunk_overlap_rows: int = typer.Option(
        DEFAULT_CHUNK_OVERLAP_ROWS, help="[row chunks] Number of overlapping rows between windows."
    ),
    context_max_chars: int = typer.Option(
        DEFAULT_CONTEXT_MAX_CHARS, help="[row chunks] Character cap cho CONTEXT_BEFORE."
    ),
    chunk_max_chars: int = typer.Option(
        DEFAULT_CHUNK_MAX_CHARS,
        help="[row chunks] Character safety cap for an entire chunk.",
    ),
    table_views: str = typer.Option(
        ",".join(TABLE_VIEWS),
        help="[multi-view] Comma-separated views to build once.",
    ),
    summary_max_chars: int = typer.Option(
        DEFAULT_SUMMARY_MAX_CHARS,
        help="[multi-view] Character cap for each summary view.",
    ),
    metadata_csv_layout: str = typer.Option(
        "head",
        help=f"[multi-view] Layout metadata CSV: {'|'.join(METADATA_CSV_LAYOUTS)}.",
    ),
    embedding_max_tokens: int = typer.Option(
        EMBEDDING_MAX_SEQ_LENGTH, help="Token cap cho embedding documents."
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedder batch size (dense/rrf); sentence-transformers chooses 32 by default.",
    ),
    data_root: Path = typer.Option(None),
    cache_dir: Path = typer.Option(None),
    company_meta_path: Path = typer.Option(
        None, help="Defaults to COMPANY_META_PATH from .env."
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Rebuild the index file; dense may still reuse the existing embedding cache.",
    ),
) -> None:
    """Build and cache a retrieval index from eligible corpus tables."""
    settings = get_settings()
    root = data_root or settings.data_root
    cdir = cache_dir or settings.cache_dir
    cmeta = company_meta_path or settings.company_meta_path
    model_name = embedding_model or settings.embedding_model

    if retriever not in RETRIEVERS:
        typer.echo(f"Invalid retriever={retriever!r} (choose from {RETRIEVERS})")
        raise typer.Exit(code=1)
    try:
        _validate_retrieval_representation(retriever, table_encoding)
        chunk_config = RowChunkConfig(
            chunk_rows=chunk_rows,
            chunk_overlap_rows=chunk_overlap_rows,
            context_max_chars=context_max_chars,
            chunk_max_chars=chunk_max_chars,
        )
        built_views, active_views = _resolve_table_views(
            table_encoding,
            table_views=table_views,
        )
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be greater than 0")
        if metadata_csv_layout not in METADATA_CSV_LAYOUTS:
            raise ValueError(
                f"Invalid metadata_csv_layout: {metadata_csv_layout!r}"
            )
        if embedding_max_tokens <= 0:
            raise ValueError("embedding_max_tokens must be greater than 0")
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    t0 = time.time()
    index = _build_retrieval_index(
        retriever=retriever,
        embedding_model=model_name,
        table_encoding=table_encoding,
        data_root=root,
        cache_dir=cdir,
        company_meta_path=cmeta,
        hf_token=settings.hf_token,
        rebuild=rebuild,
        embed_batch_size=embed_batch_size,
        chunk_config=chunk_config,
        table_views=built_views,
        active_table_views=active_views,
        summary_max_chars=summary_max_chars,
        metadata_csv_layout=metadata_csv_layout,
        embedding_max_tokens=embedding_max_tokens,
    )
    elapsed = time.time() - t0

    typer.echo(f"retriever: {retriever}")
    if retriever in ("dense", "rrf"):
        typer.echo(f"embedding_model: {model_name}")
    typer.echo(f"table_encoding: {table_encoding}")
    if table_encoding in (CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING):
        typer.echo(f"chunk_config: {chunk_config.as_dict()}")
    if table_encoding == MULTI_VIEW_TABLE_ENCODING:
        typer.echo(f"table_views: {built_views}")
        typer.echo(f"summary_max_chars: {summary_max_chars}")
        typer.echo(f"metadata_csv_layout: {metadata_csv_layout}")
    if retriever in ("dense", "rrf"):
        typer.echo(f"embedding_max_tokens: {embedding_max_tokens}")
    typer.echo(f"entries: {index.size}")
    typer.echo(f"elapsed: {elapsed:.1f}s")


@app.command(name="eval-retrieval")
def eval_retrieval(
    retriever: str = typer.Option("dense", help=f"{'|'.join(RETRIEVERS)}."),
    embedding_model: str = typer.Option(
        None, help="Defaults to EMBEDDING_MODEL from .env (dense only)."
    ),
    table_encoding: str = typer.Option(
        "table_retrieval_text", help=f"{'|'.join(TABLE_ENCODINGS)}."
    ),
    chunk_rows: int = typer.Option(
        DEFAULT_CHUNK_ROWS, help="[row chunks] Number of CSV rows per window."
    ),
    chunk_overlap_rows: int = typer.Option(
        DEFAULT_CHUNK_OVERLAP_ROWS, help="[row chunks] Number of overlapping rows between windows."
    ),
    context_max_chars: int = typer.Option(
        DEFAULT_CONTEXT_MAX_CHARS, help="[row chunks] Character cap cho CONTEXT_BEFORE."
    ),
    chunk_max_chars: int = typer.Option(
        DEFAULT_CHUNK_MAX_CHARS,
        help="[row chunks] Character safety cap for an entire chunk.",
    ),
    table_views: str = typer.Option(
        ",".join(TABLE_VIEWS),
        help="[multi-view] Comma-separated views available in the index.",
    ),
    active_table_views: str = typer.Option(
        None,
        help="[multi-view] View subset for this run; defaults to all table_views.",
    ),
    summary_max_chars: int = typer.Option(
        DEFAULT_SUMMARY_MAX_CHARS,
        help="[multi-view] Character cap for each summary view.",
    ),
    metadata_csv_layout: str = typer.Option(
        "head",
        help=f"[multi-view] Layout metadata CSV: {'|'.join(METADATA_CSV_LAYOUTS)}.",
    ),
    view_fusion: str = typer.Option(
        "max", help="[multi-view] max|rrf|quota_union for fusing candidates by parent table."
    ),
    rerank_chunks_per_table: int = typer.Option(
        1, help="Number of representations read per parent within the reranker top-L budget."
    ),
    multi_chunk_parent_n: int = typer.Option(
        0, help="Only the top-L stage-one parents use multiple representations; 0 disables it."
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedder batch size (dense/rrf); sentence-transformers chooses 32 by default.",
    ),
    mode: str = typer.Option(
        "quick", help="quick (specific k and limit) | full (sweep RETRIEVAL_K_SWEEP)."
    ),
    k: int = typer.Option(5, help="[quick] Single k value."),
    limit: int | None = typer.Option(
        None,
        help="Question limit; defaults to 10 in quick mode and all questions in full mode.",
    ),
    ks: str = typer.Option(
        None, help="[full] Comma-separated k values; defaults to RETRIEVAL_K_SWEEP."
    ),
    reranker: str = typer.Option("none", help=f"{'|'.join(RERANKERS)}."),
    reranker_model: str = typer.Option(
        None, help="Defaults to RERANKER_MODEL (bge) or Qwen3-Reranker-4B (qwen3)."
    ),
    reranker_instruction: str = typer.Option(
        "default",
        help=f"Instruction preset cho Qwen3 reranker: {'|'.join(RERANKER_INSTRUCTIONS)}.",
    ),
    rerank_top_n: int = typer.Option(
        100,
        help="Minimum rerank candidate count; rerank at least k candidates when k is larger.",
    ),
    rerank_flow: str = typer.Option(
        "single_stage",
        help="single_stage|cascade; cascade uses summaries followed by balanced row evidence.",
    ),
    cascade_top_n: int = typer.Option(
        150, help="[cascade] Number of stage-one tables reranked with row evidence."
    ),
    embedding_max_tokens: int = typer.Option(
        EMBEDDING_MAX_SEQ_LENGTH, help="Token cap cho embedding documents."
    ),
    reranker_max_tokens: int = typer.Option(
        EMBEDDING_MAX_SEQ_LENGTH,
        help="Token cap for each reranker query-table pair.",
    ),
    rrf_fetch_n: int = typer.Option(
        500, help="[retriever=rrf] Candidates from each retriever before fusion."
    ),
    rrf_k: int = typer.Option(
        DEFAULT_RRF_K, help="[retriever=rrf] The k constant in the RRF formula."
    ),
    metadata_policy: str = typer.Option(
        "none",
        help=f"Candidate policy theo entity/scope: {'|'.join(METADATA_POLICIES)}.",
    ),
    metadata_fetch_n: int = typer.Option(
        5_000, help="Stage-one candidates fetched before applying the metadata policy."
    ),
    metadata_global_fallback: int = typer.Option(
        50, help="Global candidate quota retained to guard against metadata parsing/filtering errors."
    ),
    entity_balance: bool = typer.Option(
        False,
        "--entity-balance/--no-entity-balance",
        help="Use round-robin only when the query explicitly mentions at least two tickers.",
    ),
    search_cache: bool = typer.Option(
        True,
        "--search-cache/--no-search-cache",
        help="Cache rankings by query; disable for large matrices to save disk space.",
    ),
    run_tag: str = typer.Option(None, help="Short label added to run names for readable A/B tests."),
    data_root: Path = typer.Option(None),
    cache_dir: Path = typer.Option(None),
    company_meta_path: Path = typer.Option(
        None, help="Defaults to COMPANY_META_PATH from .env."
    ),
    questions_dir: Path = typer.Option(None),
    runs_dir: Path = typer.Option(None),
) -> None:
    """Evaluate retrieval with per-question and macro/micro F2, recall, and precision at k."""
    settings = get_settings()
    root = data_root or settings.data_root
    cdir = cache_dir or settings.cache_dir
    cmeta = company_meta_path or settings.company_meta_path
    qdir = questions_dir or settings.questions_dir
    rdir = runs_dir or settings.runs_dir
    model_name = embedding_model or settings.embedding_model

    if retriever not in RETRIEVERS:
        typer.echo(f"Invalid retriever={retriever!r} (choose from {RETRIEVERS})")
        raise typer.Exit(code=1)
    try:
        _validate_retrieval_representation(retriever, table_encoding)
        chunk_config = RowChunkConfig(
            chunk_rows=chunk_rows,
            chunk_overlap_rows=chunk_overlap_rows,
            context_max_chars=context_max_chars,
            chunk_max_chars=chunk_max_chars,
        )
        built_views, active_views = _resolve_table_views(
            table_encoding,
            table_views=table_views,
            active_table_views=active_table_views,
        )
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be greater than 0")
        if metadata_csv_layout not in METADATA_CSV_LAYOUTS:
            raise ValueError(
                f"Invalid metadata_csv_layout: {metadata_csv_layout!r}"
            )
        if view_fusion not in ("max", "rrf", "quota_union"):
            raise ValueError("view_fusion must be max|rrf|quota_union")
        if rerank_chunks_per_table <= 0:
            raise ValueError("rerank_chunks_per_table must be greater than 0")
        if multi_chunk_parent_n < 0:
            raise ValueError("multi_chunk_parent_n must be non-negative")
        if metadata_policy not in METADATA_POLICIES:
            raise ValueError(
                f"Invalid metadata_policy (choose from {METADATA_POLICIES})"
            )
        if metadata_fetch_n <= 0:
            raise ValueError("metadata_fetch_n must be greater than 0")
        if metadata_global_fallback < 0:
            raise ValueError("metadata_global_fallback must be non-negative")
        if rerank_flow not in ("single_stage", "cascade"):
            raise ValueError("rerank_flow must be single_stage|cascade")
        if cascade_top_n <= 0:
            raise ValueError("cascade_top_n must be greater than 0")
        if embedding_max_tokens <= 0 or reranker_max_tokens <= 0:
            raise ValueError("embedding_max_tokens and reranker_max_tokens must be greater than 0")
        if rerank_flow == "cascade" and table_encoding != MULTI_VIEW_TABLE_ENCODING:
            raise ValueError(
                "rerank_flow=cascade requires table_encoding=metadata_context_multi_view"
            )
        if rerank_flow == "cascade" and not {ROW_CHUNK_VIEW, "metadata_csv"}.issubset(
            active_views
        ):
            raise ValueError(
                "rerank_flow=cascade requires the row_chunks and metadata_csv active views"
            )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    if reranker not in RERANKERS:
        typer.echo(f"Invalid reranker={reranker!r} (choose from {RERANKERS})")
        raise typer.Exit(code=1)
    if rerank_flow == "cascade" and reranker == "none":
        typer.echo("rerank_flow=cascade requires an enabled reranker")
        raise typer.Exit(code=1)
    try:
        effective_reranker_instruction = _resolve_reranker_instruction(
            reranker, reranker_instruction
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    if mode == "quick":
        k_list = [k]
    elif mode == "full":
        k_list = [int(x) for x in ks.split(",")] if ks else list(RETRIEVAL_K_SWEEP)
    else:
        typer.echo(f"Invalid mode: {mode!r} (quick|full)")
        raise typer.Exit(code=1)

    questions = load_question_files(discover_question_files(qdir))
    effective_limit = 10 if mode == "quick" and limit is None else limit
    if effective_limit is not None:
        questions = questions[:effective_limit]

    index: RetrievalIndex = _build_retrieval_index(
        retriever=retriever,
        embedding_model=model_name,
        table_encoding=table_encoding,
        data_root=root,
        cache_dir=cdir,
        company_meta_path=cmeta,
        hf_token=settings.hf_token,
        rrf_fetch_n=rrf_fetch_n,
        rrf_k=rrf_k,
        embed_batch_size=embed_batch_size,
        chunk_config=chunk_config,
        table_views=built_views,
        active_table_views=active_views,
        summary_max_chars=summary_max_chars,
        metadata_csv_layout=metadata_csv_layout,
        view_fusion=view_fusion,
        rerank_chunks_per_table=rerank_chunks_per_table,
        embedding_max_tokens=embedding_max_tokens,
    )
    if metadata_policy != "none":
        index = MetadataFilteredIndex(
            index,
            companies=get_company_meta(cmeta),
            policy=metadata_policy,
            fetch_n=metadata_fetch_n,
            global_fallback=metadata_global_fallback,
        )
    effective_reranker_model = None
    if reranker == "bge":
        effective_reranker_model = reranker_model or settings.reranker_model
    elif reranker == "qwen3":
        effective_reranker_model = reranker_model or _DEFAULT_QWEN3_RERANKER_MODEL
    reranker_impl = _build_reranker(
        reranker,
        effective_reranker_model,
        settings.hf_token,
        instruction=effective_reranker_instruction,
        max_tokens=reranker_max_tokens,
    )
    if reranker_impl is not None:
        if rerank_flow == "cascade":
            index = FairCascadeRerankedIndex(
                index,
                reranker_impl,
                rerank_top_n=rerank_top_n,
                cascade_top_n=cascade_top_n,
                detail_texts_per_table=rerank_chunks_per_table,
            )
        else:
            index = RerankedIndex(
                index,
                reranker_impl,
                rerank_top_n=rerank_top_n,
                multi_text_top_n=multi_chunk_parent_n,
            )
    companies = get_company_meta(cmeta)
    if entity_balance:
        index = MultiEntityBalancedIndex(index, known_tickers=companies)
    if metadata_policy in ("scope_or_unknown", "scope_hard"):
        index = FinalScopeGuard(index, companies=companies)
    if search_cache:
        index = CachedRetrievalIndex(
            index,
            cache_dir=cdir,
            namespace=_search_cache_namespace(
                retriever=retriever,
                table_encoding=table_encoding,
                embedding_model=model_name if retriever in ("dense", "rrf") else None,
                reranker=reranker,
                reranker_model=effective_reranker_model,
                rerank_top_n=rerank_top_n,
                reranker_instruction=reranker_instruction,
                rrf_fetch_n=rrf_fetch_n,
                rrf_k=rrf_k,
                chunk_config=chunk_config,
                table_views=built_views,
                active_table_views=active_views,
                summary_max_chars=summary_max_chars,
                metadata_csv_layout=metadata_csv_layout,
                view_fusion=view_fusion,
                rerank_chunks_per_table=rerank_chunks_per_table,
                multi_chunk_parent_n=multi_chunk_parent_n,
                metadata_policy=metadata_policy,
                metadata_fetch_n=metadata_fetch_n,
                metadata_global_fallback=metadata_global_fallback,
                entity_balance=entity_balance,
                rerank_flow=rerank_flow,
                cascade_top_n=cascade_top_n,
                embedding_max_tokens=embedding_max_tokens,
                reranker_max_tokens=reranker_max_tokens,
            ),
        )

    t0 = time.time()
    results = evaluate_retrieval_questions(index, questions, ks=k_list)
    elapsed = time.time() - t0

    tags = [
        retriever,
        model_run_tag(model_name) if retriever in ("dense", "rrf") else "",
        table_encoding,
        f"rr-{reranker}",
    ]
    if reranker_instruction != "default":
        tags.append(f"ri-{reranker_instruction}")
    if metadata_policy != "none":
        tags.append(f"meta-{metadata_policy}")
    if table_encoding == MULTI_VIEW_TABLE_ENCODING:
        tags.extend(
            [
                f"views-{'+'.join(active_views)}",
                f"vf-{view_fusion}",
                f"csv-{metadata_csv_layout}",
            ]
        )
    if rerank_chunks_per_table > 1:
        tags.append(f"mc{rerank_chunks_per_table}-l{multi_chunk_parent_n}")
    if rerank_flow != "single_stage":
        tags.append(f"cascade-{cascade_top_n}")
    if entity_balance:
        tags.append("entity-balance")
    tags.extend([f"etok-{embedding_max_tokens}", f"rtok-{reranker_max_tokens}"])
    if run_tag:
        tags.append(run_tag)
    tags.append(mode)
    run_dir = make_run_dir(rdir, "retrieval", [t for t in tags if t])
    summary = write_retrieval_report(run_dir, results, task="retrieval")
    write_config(
        run_dir,
        {
            "retriever": retriever,
            "embedding_model": model_name if retriever in ("dense", "rrf") else None,
            "table_encoding": table_encoding,
            "chunk_config": (
                chunk_config.as_dict()
                if table_encoding in (CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING)
                else None
            ),
            "table_views": list(built_views)
            if table_encoding == MULTI_VIEW_TABLE_ENCODING
            else None,
            "active_table_views": (
                list(active_views)
                if table_encoding == MULTI_VIEW_TABLE_ENCODING
                else None
            ),
            "summary_max_chars": (
                summary_max_chars
                if table_encoding == MULTI_VIEW_TABLE_ENCODING
                else None
            ),
            "metadata_csv_layout": (
                metadata_csv_layout
                if table_encoding == MULTI_VIEW_TABLE_ENCODING
                else None
            ),
            "view_fusion": view_fusion
            if table_encoding == MULTI_VIEW_TABLE_ENCODING
            else None,
            "rerank_chunks_per_table": rerank_chunks_per_table,
            "multi_chunk_parent_n": multi_chunk_parent_n,
            "metadata_policy": metadata_policy,
            "metadata_fetch_n": metadata_fetch_n if metadata_policy != "none" else None,
            "metadata_global_fallback": (
                metadata_global_fallback if metadata_policy != "none" else None
            ),
            "entity_balance": entity_balance,
            "rerank_flow": rerank_flow,
            "cascade_top_n": cascade_top_n if rerank_flow == "cascade" else None,
            "embedding_max_tokens": embedding_max_tokens,
            "reranker_max_tokens": reranker_max_tokens,
            "run_tag": run_tag,
            "search_cache": search_cache,
            "mode": mode,
            "ks": k_list,
            "reranker": reranker,
            "reranker_model": effective_reranker_model,
            "rerank_top_n": rerank_top_n if reranker != "none" else None,
            "reranker_instruction": reranker_instruction
            if reranker == "qwen3"
            else None,
            "reranker_instruction_text": effective_reranker_instruction,
            "rrf_fetch_n": rrf_fetch_n if retriever == "rrf" else None,
            "rrf_k": rrf_k if retriever == "rrf" else None,
            "n_questions": len(questions),
            "index_size": index.size,
            "elapsed_seconds": elapsed,
        },
    )

    typer.echo(f"run_dir: {run_dir}")
    typer.echo(json.dumps(summary["by_k"], ensure_ascii=False, indent=2))


@app.command(name="compare-runs")
def compare_runs(
    run_dirs: list[Path] = typer.Argument(
        ..., help="Run directories from eval-retrieval to compare."
    ),
    out: Path = typer.Option(
        None,
        help="Output Markdown file (defaults to <run_dirs[0].parent>/compare_<ts>.md).",
    ),
    label: list[str] = typer.Option(
        None,
        help="Display name for each run directory in order (inferred by default).",
    ),
) -> None:
    """Combine retrieval evaluations into one Markdown comparison table."""
    for d in run_dirs:
        if not (d / "summary.json").exists():
            typer.echo(f"summary.json was not found in {d}")
            raise typer.Exit(code=1)

    content = compare_run_dirs(run_dirs, labels=label or None)
    out_path = out or (run_dirs[0].parent / f"compare_{int(time.time())}.md")
    out_path.write_text(content, encoding="utf-8")
    typer.echo(f"wrote: {out_path}")


@app.command(name="eval-llm")
def eval_llm(
    llm: str = typer.Option("openai", help=f"{'|'.join(LLMS)}."),
    llm_model: str = typer.Option(
        None, help="Defaults to OPENAI_MODEL from .env for llm=openai; required for llm=hf."
    ),
    llm_reasoning_effort: str = typer.Option(
        None,
        help=f"[openrouter] Reasoning effort; defaults to .env. {'|'.join(REASONING_EFFORTS)}.",
    ),
    strategy: str = typer.Option("pandas_query", help=f"{'|'.join(STRATEGIES)}."),
    context: str = typer.Option(
        "gold",
        help=f"{'|'.join(CONTEXTS)} (none = LLM-only, gold = LLM+gold-table, "
        "retrieved = LLM with tables retrieved by an earlier eval-retrieval run).",
    ),
    retrieval_run: Path = typer.Option(
        None,
        help="[context=retrieved] Previous eval-retrieval run directory containing per_question.jsonl.",
    ),
    retrieval_k: int = typer.Option(
        None, help="[context=retrieved] Read retrieved_tables at this exact k."
    ),
    limit: int = typer.Option(5, help="Question limit."),
    source_files: str = typer.Option(
        None,
        help="[sampling] Comma-separated source file stems without .jsonl.",
    ),
    sample_per_file: int = typer.Option(
        None,
        help="[sampling] Random questions per source file; mutually exclusive with --limit.",
    ),
    sample_size: int = typer.Option(
        None,
        help="[sampling] Total random sample from --source-files; mutually exclusive with --sample-per-file.",
    ),
    sample_seed: int = typer.Option(
        42, help="[sampling] Seed for a reproducible sample."
    ),
    workers: int = typer.Option(
        1, help="Number of concurrent LLM calls; API backends only."
    ),
    exclude_run: Path = typer.Option(
        None,
        help="[sampling] Exclude questions listed in this run's config.json selected_questions.",
    ),
    table_max_chars: int = typer.Option(
        None,
        help="Maximum characters per CSV in the prompt; defaults to TABLE_MAX_CHARS from .env.",
    ),
    llm_temperature: float = typer.Option(
        0.0, help="Sampling temperature; the paper protocol uses 0."
    ),
    llm_max_completion_tokens: int = typer.Option(
        8192, help="Maximum completion tokens per inference call."
    ),
    llm_max_retries: int = typer.Option(
        0, help="SDK-level retries per call; the paper protocol uses 0."
    ),
    answer_abs_tolerance: float = typer.Option(
        ANSWER_ABS_TOL, help="Absolute tolerance when matching a numeric answer."
    ),
    prompt_system: str = typer.Option(
        None, help="Override the strategy system-prompt file."
    ),
    prompt_user_template: str = typer.Option(
        None, help="Override the shared user-message template file."
    ),
    max_context_tables: int = typer.Option(
        None,
        help="Maximum tables per LLM call; unlimited by default.",
    ),
    data_root: Path = typer.Option(
        None, help="[context=retrieved] Required to resolve table_ref to csv_path."
    ),
    company_meta_path: Path = typer.Option(
        None, help="Metadata canonical ticker -> company_name."
    ),
    questions_dir: Path = typer.Option(None),
    runs_dir: Path = typer.Option(None),
) -> None:
    """Evaluate LLM-only, gold-table, or previously retrieved-table accuracy."""
    settings = get_settings()
    root = data_root or settings.data_root
    cmeta = company_meta_path or settings.company_meta_path
    qdir = questions_dir or settings.questions_dir
    rdir = runs_dir or settings.runs_dir
    effective_table_max_chars = (
        table_max_chars if table_max_chars is not None else settings.table_max_chars
    )

    if llm not in LLMS:
        typer.echo(f"Invalid llm={llm!r} (choose from {LLMS})")
        raise typer.Exit(code=1)
    if strategy not in STRATEGIES:
        typer.echo(f"Invalid strategy={strategy!r} (choose from {STRATEGIES})")
        raise typer.Exit(code=1)
    if context not in CONTEXTS:
        typer.echo(f"Invalid context={context!r} (choose from {CONTEXTS})")
        raise typer.Exit(code=1)
    if context == "none" and strategy == "pandas_query":
        typer.echo(
            "strategy=pandas_query requires table data and is incompatible with context=none"
        )
        raise typer.Exit(code=1)
    if context == "retrieved" and (retrieval_run is None or retrieval_k is None):
        typer.echo("context=retrieved requires --retrieval-run and --retrieval-k")
        raise typer.Exit(code=1)
    if context == "retrieved" and not (retrieval_run / "per_question.jsonl").exists():
        typer.echo(f"per_question.jsonl was not found in {retrieval_run}")
        raise typer.Exit(code=1)
    if llm == "hf" and not llm_model:
        typer.echo("llm=hf requires an explicit --llm-model")
        raise typer.Exit(code=1)
    if llm == "openrouter" and not settings.openrouter_api_key:
        typer.echo("llm=openrouter requires OPENROUTER_API_KEY in .env")
        raise typer.Exit(code=1)
    if llm_reasoning_effort is not None and llm != "openrouter":
        typer.echo("--llm-reasoning-effort is supported only with llm=openrouter")
        raise typer.Exit(code=1)
    if (
        llm_reasoning_effort is not None
        and llm_reasoning_effort not in REASONING_EFFORTS
    ):
        typer.echo(f"Invalid llm_reasoning_effort (choose from {REASONING_EFFORTS})")
        raise typer.Exit(code=1)
    if effective_table_max_chars <= 64:
        typer.echo("table_max_chars must be greater than 64")
        raise typer.Exit(code=1)
    if max_context_tables is not None and max_context_tables <= 0:
        typer.echo("max_context_tables must be greater than 0")
        raise typer.Exit(code=1)
    if workers <= 0:
        typer.echo("workers must be greater than 0")
        raise typer.Exit(code=1)
    if llm == "hf" and workers != 1:
        typer.echo("llm=hf supports only --workers 1")
        raise typer.Exit(code=1)
    sampling_modes = int(sample_per_file is not None) + int(sample_size is not None)
    if (source_files is None and sampling_modes) or (
        source_files is not None and sampling_modes != 1
    ):
        typer.echo(
            "sampling requires --source-files and exactly one of --sample-size/--sample-per-file"
        )
        raise typer.Exit(code=1)

    all_questions = load_question_files(discover_question_files(qdir))
    excluded_questions: set[str] = set()
    if exclude_run is not None:
        exclude_config_path = exclude_run / "config.json"
        if not exclude_config_path.exists():
            typer.echo(f"config.json was not found in exclude_run={exclude_run}")
            raise typer.Exit(code=1)
        exclude_config = json.loads(exclude_config_path.read_text(encoding="utf-8"))
        excluded_questions = set(exclude_config.get("selected_questions") or [])
        all_questions = [
            question
            for question in all_questions
            if question.original not in excluded_questions
        ]
    selected_source_files = None
    if sampling_modes:
        selected_source_files = [
            value.strip() for value in source_files.split(",") if value.strip()
        ]
        try:
            if sample_size is not None:
                questions = sample_from_sources(
                    all_questions,
                    source_files=selected_source_files,
                    size=sample_size,
                    seed=sample_seed,
                )
            else:
                questions = sample_per_source(
                    all_questions,
                    source_files=selected_source_files,
                    per_file=sample_per_file,
                    seed=sample_seed,
                )
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
    else:
        questions = all_questions[:limit]

    retrieved_tables = None
    doc_name_lookup = None
    table_metadata_lookup = {}
    if context == "retrieved":
        retrieved_tables = load_retrieved_tables(retrieval_run, k=retrieval_k)
    if context in ("gold", "retrieved"):
        doc_name_lookup = build_doc_name_lookup(root)
        company_names = {
            ticker: info.name for ticker, info in get_company_meta(cmeta).items()
        }
        if context == "gold":
            selected_table_refs = {
                ref for question in questions for ref in question.relevant_tables
            }
        else:
            selected_ids = {question.id for question in questions}
            selected_table_refs = {
                ref
                for question_id, refs in (retrieved_tables or {}).items()
                if question_id in selected_ids
                for ref in refs
            }
        table_metadata_lookup = build_table_metadata_lookup(
            selected_table_refs,
            doc_name_lookup=doc_name_lookup,
            company_names=company_names,
        )

    llm_impl = _build_llm(
        llm,
        llm_model,
        settings,
        reasoning_effort=llm_reasoning_effort,
        temperature=llm_temperature,
        max_completion_tokens=llm_max_completion_tokens,
        max_retries=llm_max_retries,
    )
    strategy_impl = _build_answer_strategy(
        strategy,
        table_max_chars=effective_table_max_chars,
        system_path=prompt_system,
        user_template_path=prompt_user_template,
    )

    run_dir = make_run_dir(rdir, "llm", [llm, strategy, context])
    run_config = {
        "llm": llm,
        "llm_model": _resolved_llm_model(llm, llm_model, settings),
        "llm_reasoning_effort": (
            llm_reasoning_effort or settings.openrouter_reasoning_effort
            if llm == "openrouter"
            else None
        ),
        "strategy": strategy,
        "context": context,
        "retrieval_run": str(retrieval_run) if retrieval_run else None,
        "retrieval_k": retrieval_k,
        "table_max_chars": effective_table_max_chars,
        "max_context_tables": max_context_tables,
        "llm_temperature": llm_temperature,
        "llm_max_completion_tokens": llm_max_completion_tokens,
        "llm_max_retries": llm_max_retries,
        "answer_abs_tolerance": answer_abs_tolerance,
        "prompt_system": prompt_system,
        "prompt_user_template": prompt_user_template,
        "answer_table_metadata": "canonical_anchor_v1"
        if context in ("gold", "retrieved")
        else None,
        "n_questions": len(questions),
        "source_files": selected_source_files,
        "sample_per_file": sample_per_file,
        "sample_size": sample_size,
        "sample_seed": sample_seed if sampling_modes else None,
        "workers": workers,
        "exclude_run": str(exclude_run) if exclude_run else None,
        "excluded_questions": sorted(excluded_questions),
        "selected_questions": [question.original for question in questions],
        "status": "running",
    }
    write_config(run_dir, run_config)
    t0 = time.time()
    try:
        results = evaluate_llm_questions(
            questions,
            strategy=strategy_impl,
            llm=llm_impl,
            context=context,
            retrieved_tables=retrieved_tables,
            doc_name_lookup=doc_name_lookup,
            table_metadata_lookup=table_metadata_lookup,
            max_context_tables=max_context_tables,
            abs_tol=answer_abs_tolerance,
            max_workers=workers,
            on_result=partial(append_answer_checkpoint, run_dir),
        )
    except LLMError as exc:
        run_config.update(
            {"elapsed_seconds": time.time() - t0, "status": "failed", "error": str(exc)}
        )
        write_config(run_dir, run_config)
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    elapsed = time.time() - t0

    summary = write_answer_report(run_dir, results, task="llm", strategy=strategy)
    run_config.update({"elapsed_seconds": elapsed, "status": "completed"})
    write_config(run_dir, run_config)

    typer.echo(f"run_dir: {run_dir}")
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


@app.command(name="eval-e2e")
def eval_e2e(
    retriever: str = typer.Option("dense", help=f"{'|'.join(RETRIEVERS)}."),
    embedding_model: str = typer.Option(
        None, help="Defaults to EMBEDDING_MODEL from .env (used by dense/rrf)."
    ),
    table_encoding: str = typer.Option(
        "table_retrieval_text", help=f"{'|'.join(TABLE_ENCODINGS)}."
    ),
    chunk_rows: int = typer.Option(
        DEFAULT_CHUNK_ROWS, help="[row chunks] Number of CSV rows per window."
    ),
    chunk_overlap_rows: int = typer.Option(
        DEFAULT_CHUNK_OVERLAP_ROWS, help="[row chunks] Number of overlapping rows between windows."
    ),
    context_max_chars: int = typer.Option(
        DEFAULT_CONTEXT_MAX_CHARS, help="[row chunks] Character cap cho CONTEXT_BEFORE."
    ),
    chunk_max_chars: int = typer.Option(
        DEFAULT_CHUNK_MAX_CHARS,
        help="[row chunks] Character safety cap for an entire chunk.",
    ),
    table_views: str = typer.Option(
        ",".join(TABLE_VIEWS),
        help="[multi-view] Comma-separated views available in the index.",
    ),
    active_table_views: str = typer.Option(
        None,
        help="[multi-view] View subset for this run; defaults to all table_views.",
    ),
    summary_max_chars: int = typer.Option(
        DEFAULT_SUMMARY_MAX_CHARS,
        help="[multi-view] Character cap for each summary view.",
    ),
    metadata_csv_layout: str = typer.Option(
        "head",
        help=f"[multi-view] Layout metadata CSV: {'|'.join(METADATA_CSV_LAYOUTS)}.",
    ),
    view_fusion: str = typer.Option(
        "max", help="[multi-view] max|rrf|quota_union for fusing scores by parent table."
    ),
    rerank_chunks_per_table: int = typer.Option(
        1, help="Number of representations read per parent within the reranker top-L budget."
    ),
    multi_chunk_parent_n: int = typer.Option(
        0, help="Only top-L stage-one parents use multiple representations; 0 disables it."
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedder batch size (dense/rrf); sentence-transformers chooses 32 by default.",
    ),
    reranker: str = typer.Option("none", help=f"{'|'.join(RERANKERS)}."),
    reranker_model: str = typer.Option(None),
    reranker_instruction: str = typer.Option(
        "default",
        help=f"Instruction preset cho Qwen3 reranker: {'|'.join(RERANKER_INSTRUCTIONS)}.",
    ),
    rerank_top_n: int = typer.Option(
        100,
        help="Minimum rerank candidate count; rerank at least k when k is larger.",
    ),
    rerank_flow: str = typer.Option(
        "single_stage",
        help="single_stage|cascade; cascade uses summaries followed by balanced row evidence.",
    ),
    cascade_top_n: int = typer.Option(
        150, help="[cascade] Number of stage-one tables reranked with row evidence."
    ),
    embedding_max_tokens: int = typer.Option(
        EMBEDDING_MAX_SEQ_LENGTH, help="Token cap for embedding documents."
    ),
    reranker_max_tokens: int = typer.Option(
        EMBEDDING_MAX_SEQ_LENGTH,
        help="Token cap for each reranker query-table pair.",
    ),
    rrf_fetch_n: int = typer.Option(
        500, help="[retriever=rrf] Candidates from each retriever before fusion."
    ),
    rrf_k: int = typer.Option(
        DEFAULT_RRF_K, help="[retriever=rrf] The k constant in the RRF formula."
    ),
    metadata_policy: str = typer.Option(
        "none",
        help=f"Candidate policy theo entity/scope: {'|'.join(METADATA_POLICIES)}.",
    ),
    metadata_fetch_n: int = typer.Option(
        5_000, help="Stage-one candidates fetched before applying the metadata policy."
    ),
    metadata_global_fallback: int = typer.Option(
        50, help="Global candidate quota retained against metadata parsing/filtering errors."
    ),
    entity_balance: bool = typer.Option(
        False,
        "--entity-balance/--no-entity-balance",
        help="Round-robin results when the query explicitly mentions multiple tickers.",
    ),
    search_cache: bool = typer.Option(
        True,
        "--search-cache/--no-search-cache",
        help="Cache rankings by query; disable for large matrices to save disk space.",
    ),
    llm: str = typer.Option("openai", help=f"{'|'.join(LLMS)}."),
    llm_model: str = typer.Option(
        None, help="Defaults to OPENAI_MODEL for llm=openai; required for llm=hf."
    ),
    llm_reasoning_effort: str = typer.Option(
        None,
        help=f"[openrouter] Reasoning effort; defaults to .env. {'|'.join(REASONING_EFFORTS)}.",
    ),
    strategy: str = typer.Option("pandas_query", help=f"{'|'.join(STRATEGIES)}."),
    ks: str = typer.Option(
        None, help="Comma-separated k values; defaults to E2E_K_SWEEP."
    ),
    max_context_tables: int = typer.Option(
        DEFAULT_MAX_CONTEXT_TABLES,
        help="Maximum tables in the LLM context, regardless of k.",
    ),
    table_max_chars: int = typer.Option(
        None,
        help="Maximum characters per CSV in the prompt; defaults to TABLE_MAX_CHARS from .env.",
    ),
    llm_temperature: float = typer.Option(
        0.0, help="Sampling temperature; the paper protocol uses 0."
    ),
    llm_max_completion_tokens: int = typer.Option(
        8192, help="Maximum completion tokens per inference call."
    ),
    llm_max_retries: int = typer.Option(
        0, help="SDK-level retries per call; the paper protocol uses 0."
    ),
    answer_abs_tolerance: float = typer.Option(
        ANSWER_ABS_TOL, help="Absolute tolerance when matching a numeric answer."
    ),
    prompt_system: str = typer.Option(
        None, help="Override the strategy system-prompt file."
    ),
    prompt_user_template: str = typer.Option(
        None, help="Override the shared user-message template file."
    ),
    limit: int = typer.Option(5, help="Question limit."),
    run_tag: str = typer.Option(None, help="Short run-name label for readable A/B tests."),
    data_root: Path = typer.Option(None),
    cache_dir: Path = typer.Option(None),
    company_meta_path: Path = typer.Option(
        None, help="Defaults to COMPANY_META_PATH from .env."
    ),
    questions_dir: Path = typer.Option(None),
    runs_dir: Path = typer.Option(None),
) -> None:
    """Run retrieval, trim to the top context tables, answer, and compare."""
    settings = get_settings()
    root = data_root or settings.data_root
    cdir = cache_dir or settings.cache_dir
    cmeta = company_meta_path or settings.company_meta_path
    qdir = questions_dir or settings.questions_dir
    rdir = runs_dir or settings.runs_dir
    model_name = embedding_model or settings.embedding_model
    effective_table_max_chars = (
        table_max_chars if table_max_chars is not None else settings.table_max_chars
    )

    if retriever not in RETRIEVERS:
        typer.echo(f"Invalid retriever={retriever!r} (choose from {RETRIEVERS})")
        raise typer.Exit(code=1)
    try:
        _validate_retrieval_representation(retriever, table_encoding)
        chunk_config = RowChunkConfig(
            chunk_rows=chunk_rows,
            chunk_overlap_rows=chunk_overlap_rows,
            context_max_chars=context_max_chars,
            chunk_max_chars=chunk_max_chars,
        )
        built_views, active_views = _resolve_table_views(
            table_encoding,
            table_views=table_views,
            active_table_views=active_table_views,
        )
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be greater than 0")
        if metadata_csv_layout not in METADATA_CSV_LAYOUTS:
            raise ValueError(
                f"Invalid metadata_csv_layout: {metadata_csv_layout!r}"
            )
        if view_fusion not in ("max", "rrf", "quota_union"):
            raise ValueError("view_fusion must be max|rrf|quota_union")
        if rerank_chunks_per_table <= 0:
            raise ValueError("rerank_chunks_per_table must be greater than 0")
        if multi_chunk_parent_n < 0:
            raise ValueError("multi_chunk_parent_n must be non-negative")
        if metadata_policy not in METADATA_POLICIES:
            raise ValueError(
                f"Invalid metadata_policy (choose from {METADATA_POLICIES})"
            )
        if metadata_fetch_n <= 0:
            raise ValueError("metadata_fetch_n must be greater than 0")
        if metadata_global_fallback < 0:
            raise ValueError("metadata_global_fallback must be non-negative")
        if rerank_flow not in ("single_stage", "cascade"):
            raise ValueError("rerank_flow must be single_stage|cascade")
        if cascade_top_n <= 0:
            raise ValueError("cascade_top_n must be greater than 0")
        if embedding_max_tokens <= 0 or reranker_max_tokens <= 0:
            raise ValueError(
                "embedding_max_tokens and reranker_max_tokens must be greater than 0"
            )
        if rrf_fetch_n <= 0 or rrf_k < 0:
            raise ValueError("rrf_fetch_n must be positive and rrf_k must be non-negative")
        if rerank_flow == "cascade" and table_encoding != MULTI_VIEW_TABLE_ENCODING:
            raise ValueError(
                "rerank_flow=cascade requires table_encoding=metadata_context_multi_view"
            )
        if rerank_flow == "cascade" and not {
            ROW_CHUNK_VIEW,
            "metadata_csv",
        }.issubset(active_views):
            raise ValueError(
                "rerank_flow=cascade requires the row_chunks and metadata_csv active views"
            )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    if reranker not in RERANKERS:
        typer.echo(f"Invalid reranker={reranker!r} (choose from {RERANKERS})")
        raise typer.Exit(code=1)
    if rerank_flow == "cascade" and reranker == "none":
        typer.echo("rerank_flow=cascade requires an enabled reranker")
        raise typer.Exit(code=1)
    try:
        effective_reranker_instruction = _resolve_reranker_instruction(
            reranker, reranker_instruction
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    if llm not in LLMS:
        typer.echo(f"Invalid llm={llm!r} (choose from {LLMS})")
        raise typer.Exit(code=1)
    if strategy not in STRATEGIES:
        typer.echo(f"Invalid strategy={strategy!r} (choose from {STRATEGIES})")
        raise typer.Exit(code=1)
    if llm == "hf" and not llm_model:
        typer.echo("llm=hf requires an explicit --llm-model")
        raise typer.Exit(code=1)
    if llm == "openrouter" and not settings.openrouter_api_key:
        typer.echo("llm=openrouter requires OPENROUTER_API_KEY in .env")
        raise typer.Exit(code=1)
    if llm_reasoning_effort is not None and llm != "openrouter":
        typer.echo("--llm-reasoning-effort is supported only with llm=openrouter")
        raise typer.Exit(code=1)
    if (
        llm_reasoning_effort is not None
        and llm_reasoning_effort not in REASONING_EFFORTS
    ):
        typer.echo(f"Invalid llm_reasoning_effort (choose from {REASONING_EFFORTS})")
        raise typer.Exit(code=1)
    if effective_table_max_chars <= 64:
        typer.echo("table_max_chars must be greater than 64")
        raise typer.Exit(code=1)

    k_list = [int(x) for x in ks.split(",")] if ks else list(E2E_K_SWEEP)
    questions = load_question_files(discover_question_files(qdir))[:limit]

    index: RetrievalIndex = _build_retrieval_index(
        retriever=retriever,
        embedding_model=model_name,
        table_encoding=table_encoding,
        data_root=root,
        cache_dir=cdir,
        company_meta_path=cmeta,
        hf_token=settings.hf_token,
        rrf_fetch_n=rrf_fetch_n,
        rrf_k=rrf_k,
        embed_batch_size=embed_batch_size,
        chunk_config=chunk_config,
        table_views=built_views,
        active_table_views=active_views,
        summary_max_chars=summary_max_chars,
        metadata_csv_layout=metadata_csv_layout,
        view_fusion=view_fusion,
        rerank_chunks_per_table=rerank_chunks_per_table,
        embedding_max_tokens=embedding_max_tokens,
    )
    if metadata_policy != "none":
        index = MetadataFilteredIndex(
            index,
            companies=get_company_meta(cmeta),
            policy=metadata_policy,
            fetch_n=metadata_fetch_n,
            global_fallback=metadata_global_fallback,
        )
    effective_reranker_model = None
    if reranker == "bge":
        effective_reranker_model = reranker_model or settings.reranker_model
    elif reranker == "qwen3":
        effective_reranker_model = reranker_model or _DEFAULT_QWEN3_RERANKER_MODEL
    reranker_impl = _build_reranker(
        reranker,
        effective_reranker_model,
        settings.hf_token,
        instruction=effective_reranker_instruction,
        max_tokens=reranker_max_tokens,
    )
    if reranker_impl is not None:
        if rerank_flow == "cascade":
            index = FairCascadeRerankedIndex(
                index,
                reranker_impl,
                rerank_top_n=rerank_top_n,
                cascade_top_n=cascade_top_n,
                detail_texts_per_table=rerank_chunks_per_table,
            )
        else:
            index = RerankedIndex(
                index,
                reranker_impl,
                rerank_top_n=rerank_top_n,
                multi_text_top_n=multi_chunk_parent_n,
            )
    companies = get_company_meta(cmeta)
    if entity_balance:
        index = MultiEntityBalancedIndex(index, known_tickers=companies)
    if metadata_policy in ("scope_or_unknown", "scope_hard"):
        index = FinalScopeGuard(index, companies=companies)
    if search_cache:
        index = CachedRetrievalIndex(
            index,
            cache_dir=cdir,
            namespace=_search_cache_namespace(
                retriever=retriever,
                table_encoding=table_encoding,
                embedding_model=model_name if retriever in ("dense", "rrf") else None,
                reranker=reranker,
                reranker_model=effective_reranker_model,
                rerank_top_n=rerank_top_n,
                reranker_instruction=reranker_instruction,
                rrf_fetch_n=rrf_fetch_n,
                rrf_k=rrf_k,
                chunk_config=chunk_config,
                table_views=built_views,
                active_table_views=active_views,
                summary_max_chars=summary_max_chars,
                metadata_csv_layout=metadata_csv_layout,
                view_fusion=view_fusion,
                rerank_chunks_per_table=rerank_chunks_per_table,
                multi_chunk_parent_n=multi_chunk_parent_n,
                metadata_policy=metadata_policy,
                metadata_fetch_n=metadata_fetch_n,
                metadata_global_fallback=metadata_global_fallback,
                entity_balance=entity_balance,
                rerank_flow=rerank_flow,
                cascade_top_n=cascade_top_n,
                embedding_max_tokens=embedding_max_tokens,
                reranker_max_tokens=reranker_max_tokens,
            ),
        )

    llm_impl = _build_llm(
        llm,
        llm_model,
        settings,
        reasoning_effort=llm_reasoning_effort,
        temperature=llm_temperature,
        max_completion_tokens=llm_max_completion_tokens,
        max_retries=llm_max_retries,
    )
    strategy_impl = _build_answer_strategy(
        strategy,
        table_max_chars=effective_table_max_chars,
        system_path=prompt_system,
        user_template_path=prompt_user_template,
    )

    tags = [
        retriever,
        model_run_tag(model_name) if retriever in ("dense", "rrf") else "",
        f"rr-{reranker}",
    ]
    if reranker_instruction != "default":
        tags.append(f"ri-{reranker_instruction}")
    if metadata_policy != "none":
        tags.append(f"meta-{metadata_policy}")
    if table_encoding == MULTI_VIEW_TABLE_ENCODING:
        tags.extend(
            [
                f"views-{'+'.join(active_views)}",
                f"vf-{view_fusion}",
                f"csv-{metadata_csv_layout}",
            ]
        )
    if rerank_chunks_per_table > 1:
        tags.append(f"mc{rerank_chunks_per_table}-l{multi_chunk_parent_n}")
    if rerank_flow != "single_stage":
        tags.append(f"cascade-{cascade_top_n}")
    if entity_balance:
        tags.append("entity-balance")
    tags.extend([f"etok-{embedding_max_tokens}", f"rtok-{reranker_max_tokens}"])
    if run_tag:
        tags.append(run_tag)
    tags.extend([llm, strategy])
    run_dir = make_run_dir(rdir, "e2e", [t for t in tags if t])
    run_config = {
        "retriever": retriever,
        "embedding_model": model_name if retriever in ("dense", "rrf") else None,
        "table_encoding": table_encoding,
        "chunk_config": (
            chunk_config.as_dict()
            if table_encoding in (CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING)
            else None
        ),
        "table_views": list(built_views)
        if table_encoding == MULTI_VIEW_TABLE_ENCODING
        else None,
        "active_table_views": list(active_views)
        if table_encoding == MULTI_VIEW_TABLE_ENCODING
        else None,
        "summary_max_chars": summary_max_chars
        if table_encoding == MULTI_VIEW_TABLE_ENCODING
        else None,
        "metadata_csv_layout": metadata_csv_layout
        if table_encoding == MULTI_VIEW_TABLE_ENCODING
        else None,
        "view_fusion": view_fusion
        if table_encoding == MULTI_VIEW_TABLE_ENCODING
        else None,
        "rerank_chunks_per_table": rerank_chunks_per_table,
        "multi_chunk_parent_n": multi_chunk_parent_n,
        "metadata_policy": metadata_policy,
        "metadata_fetch_n": metadata_fetch_n if metadata_policy != "none" else None,
        "metadata_global_fallback": metadata_global_fallback
        if metadata_policy != "none"
        else None,
        "entity_balance": entity_balance,
        "rerank_flow": rerank_flow,
        "cascade_top_n": cascade_top_n if rerank_flow == "cascade" else None,
        "embedding_max_tokens": embedding_max_tokens,
        "reranker_max_tokens": reranker_max_tokens,
        "rrf_fetch_n": rrf_fetch_n if retriever == "rrf" else None,
        "rrf_k": rrf_k if retriever == "rrf" else None,
        "run_tag": run_tag,
        "search_cache": search_cache,
        "reranker": reranker,
        "reranker_model": effective_reranker_model,
        "rerank_top_n": rerank_top_n if reranker != "none" else None,
        "reranker_instruction": reranker_instruction if reranker == "qwen3" else None,
        "reranker_instruction_text": effective_reranker_instruction,
        "llm": llm,
        "llm_model": _resolved_llm_model(llm, llm_model, settings),
        "llm_reasoning_effort": (
            llm_reasoning_effort or settings.openrouter_reasoning_effort
            if llm == "openrouter"
            else None
        ),
        "strategy": strategy,
        "ks": k_list,
        "max_context_tables": max_context_tables,
        "table_max_chars": effective_table_max_chars,
        "llm_temperature": llm_temperature,
        "llm_max_completion_tokens": llm_max_completion_tokens,
        "llm_max_retries": llm_max_retries,
        "answer_abs_tolerance": answer_abs_tolerance,
        "prompt_system": prompt_system,
        "prompt_user_template": prompt_user_template,
        "answer_table_metadata": "canonical_anchor_v1",
        "n_questions": len(questions),
        "status": "running",
    }
    write_config(run_dir, run_config)
    t0 = time.time()
    try:
        results = evaluate_e2e_questions(
            index,
            questions,
            ks=k_list,
            max_context_tables=max_context_tables,
            strategy=strategy_impl,
            llm=llm_impl,
            data_root=root,
            company_meta_path=cmeta,
            abs_tol=answer_abs_tolerance,
            on_result=partial(append_answer_checkpoint, run_dir),
        )
    except LLMError as exc:
        run_config.update(
            {"elapsed_seconds": time.time() - t0, "status": "failed", "error": str(exc)}
        )
        write_config(run_dir, run_config)
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    elapsed = time.time() - t0

    summary = write_answer_report(
        run_dir,
        results,
        task="e2e",
        extra={"max_context_tables": max_context_tables},
        strategy=strategy,
    )
    run_config.update({"elapsed_seconds": elapsed, "status": "completed"})
    write_config(run_dir, run_config)

    typer.echo(f"run_dir: {run_dir}")
    typer.echo(json.dumps(summary["by_k"], ensure_ascii=False, indent=2, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
