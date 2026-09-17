"""Four-stage dataset construction pipeline transcribed from paper Appendix C.1."""

from __future__ import annotations

import logging
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.config import external_to_internal_difficulty, resolve_asset_path
from vifinqa.evaluation.answer_match import coerce_number
from vifinqa.generation.common import CandidateTable, build_candidate_table
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.parsing import parse_json_object
from vifinqa.generation.schemas import (
    ConceptMappingResult,
    ConceptSelection,
    PandasQuery,
    QARecord,
    QuestionDraft,
)
from vifinqa.generation.validation.pandas_check import execute_query
from vifinqa.llm.base import ChatLLM


logger = logging.getLogger(__name__)


def _read(path: str) -> str:
    return resolve_asset_path(path).read_text(encoding="utf-8").strip()


def _render(template: str, values: Mapping[str, object]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def _candidate_block(candidate: CandidateTable, template: str) -> str:
    return _render(
        template,
        {
            "TABLE_REF": candidate.table_ref,
            "COMPANY_NAME": candidate.company_name,
            "TICKER": candidate.ticker,
            "DOCUMENT_NAME": candidate.doc_name,
            "YEAR": candidate.year,
            "TABLE_LABELS": candidate.table_labels,
            "SURROUNDING_PAGES": candidate.surrounding_pages,
        },
    )


def _selected_csv_block(candidate: CandidateTable, hint: str, template: str) -> str:
    return _render(
        template,
        {
            "TABLE_REF": candidate.table_ref,
            "COMPANY_NAME": candidate.company_name,
            "TICKER": candidate.ticker,
            "DOCUMENT_NAME": candidate.doc_name,
            "YEAR": candidate.year,
            "ROW_OR_COLUMN_HINT": hint,
            "CSV_TEXT": candidate.csv_text,
            "SURROUNDING_PAGES": candidate.surrounding_pages,
        },
    )


def _complete_json(llm: ChatLLM, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    raw = llm.complete(system=system, user=user)
    return schema.model_validate(parse_json_object(raw))


def _caption_grouped_pool(seed: int | None, cap: int, data_root: Path,
                         company_meta_path: Path) -> list["CandidateTable"]:
    """A candidate pool of tables that are the *same table* at different companies.

    LOCAL ADDITION. The original pool shuffles the whole catalog and takes the
    first `cap` eligible tables, so the twelve reaching stage 1 are unrelated.
    Stage 2 then has to find one concept present in `target_count` of them — two
    for medium, three for intermediate. Across qwen3.7-flash, qwen3.7-plus and
    deepseek-v4-flash the medium tier produced 0, 0 and 1 records from eight seeds
    each, and qwen3.7-plus died at stage 2 on every one. The models were not the
    constraint: two random tables rarely share an indicator. With this pool the
    same eight seeds produced seven records.

    Reads `artifacts/caption_groups.json`, built once by
    `scripts/build_caption_index.py`. It used to read the parquet directly, which
    meant every process in the fleet loaded pandas and several hundred megabytes
    to answer a question whose answer is a few tens of kilobytes — enough to stall
    the machine while the work itself was only waiting on an API.
    """

    index_path = Path(__file__).resolve().parents[4] / "artifacts" / "caption_groups.json"
    try:
        groups = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not groups:
        return []

    rng = random.Random(seed)
    chosen = groups[rng.randrange(len(groups))][: max(cap, 12)]

    docs = {doc.doc_name: doc for doc in scan_catalog(data_root)}
    companies = load_company_meta(company_meta_path)
    candidates: list[CandidateTable] = []
    for doc_name, table_id, _ticker, _year in chosen:
        doc = docs.get(doc_name)
        if doc is None or doc.text_path is None or doc.tables_dir is None:
            continue
        try:
            table = load_table(
                doc.table_csv_path(int(table_id)),
                ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name,
                table_id=int(table_id))
            if not is_table_eligible(table):
                continue
            document = parse_document(doc.text_path)  # type: ignore[arg-type]
            company = companies.get(doc.ticker)
            candidates.append(build_candidate_table(
                doc=doc, table=table, document=document,
                company_name=company.name if company else doc.ticker))
        except Exception:
            continue
    return candidates


def _candidate_pool(
    *,
    data_root: Path,
    company_meta_path: Path,
    seed: int | None,
    cap: int,
) -> list[CandidateTable]:
    if os.environ.get("VIFIN_CAPTION_POOL") == "1":
        grouped = _caption_grouped_pool(seed, cap, data_root, company_meta_path)
        if grouped:
            return grouped

    rng = random.Random(seed)
    docs = scan_catalog(data_root)
    rng.shuffle(docs)
    companies = load_company_meta(company_meta_path)
    candidates: list[CandidateTable] = []
    for doc in docs:
        if doc.text_path is None or doc.tables_dir is None:
            continue
        document = parse_document(doc.text_path)
        table_ids = list(doc.table_ids)
        rng.shuffle(table_ids)
        for table_id in table_ids:
            table = load_table(
                doc.table_csv_path(table_id),
                ticker=doc.ticker,
                year=doc.year,
                doc_name=doc.doc_name,
                table_id=table_id,
            )
            if not is_table_eligible(table):
                continue
            company = companies.get(doc.ticker)
            candidates.append(
                build_candidate_table(
                    doc=doc,
                    table=table,
                    document=document,
                    company_name=company.name if company else doc.ticker,
                )
            )
            if len(candidates) >= cap:
                return candidates
    return candidates


def _scenario(candidates: list[CandidateTable]) -> tuple[str, list[CandidateTable]]:
    by_year: dict[str, list[CandidateTable]] = {}
    by_ticker: dict[str, list[CandidateTable]] = {}
    for candidate in candidates:
        by_year.setdefault(candidate.year, []).append(candidate)
        by_ticker.setdefault(candidate.ticker, []).append(candidate)
    same_period = next(
        (items for items in by_year.values() if len({item.ticker for item in items}) >= 3),
        None,
    )
    if same_period:
        unique: dict[str, CandidateTable] = {}
        for item in same_period:
            unique.setdefault(item.ticker, item)
        return "same_period", list(unique.values())[:12]
    time_series = next(
        (items for items in by_ticker.values() if len({item.year for item in items}) >= 3),
        None,
    )
    if time_series:
        unique = {}
        for item in time_series:
            unique.setdefault(item.year, item)
        return "time_series", list(unique.values())[:12]
    # Small synthetic corpora are useful for offline tests even though the paper pool uses >=3 entities.
    return "same_period", candidates[:12]


def generate_from_paper_prompts(
    *,
    llm: ChatLLM,
    prompts: Mapping[str, Any],
    data_root: Path,
    company_meta_path: Path,
    tier: str,
    count: int,
    out_path: Path,
    seed: int | None = None,
    max_candidates: int = 100,
    max_llm_calls: int | None = None,
) -> int:
    """Generate records using only the English prompt text printed in the PDF."""

    difficulty = external_to_internal_difficulty(tier)
    pool = _candidate_pool(
        data_root=data_root,
        company_meta_path=company_meta_path,
        seed=seed,
        cap=max_candidates,
    )
    if not pool:
        raise ValueError("No eligible tables were found for generation")
    scenario_name, candidates = _scenario(pool)
    candidate_template = _read("prompts/generation/candidate_table.txt")
    selected_template = _read("prompts/generation/selected_table_with_csv.txt")
    candidate_blocks = "\n\n".join(
        _candidate_block(candidate, candidate_template) for candidate in candidates
    )
    scenario_path = (
        "prompts/generation/stage1_same_period.txt"
        if scenario_name == "same_period"
        else "prompts/generation/stage1_time_series.txt"
    )
    stage1_system = _render(
        _read(str(prompts["stage1_system"])),
        {"SCENARIO_INSTRUCTION": _read(scenario_path)},
    )
    stage1_user = _render(
        _read(str(prompts["stage1_user"])),
        {"CANDIDATE_TABLE_BLOCKS": candidate_blocks},
    )
    writer = JsonlWriter(out_path)
    generated = 0
    calls = 0
    for _ in range(count):
        if max_llm_calls is not None and calls + 4 > max_llm_calls:
            break
        try:
            concept = _complete_json(
                llm,
                system=stage1_system,
                user=stage1_user,
                schema=ConceptSelection,
            )
            calls += 1
            assert isinstance(concept, ConceptSelection)
            if not concept.feasible:
                continue
            stage2_user = _render(
                _read(str(prompts["stage2_user"])),
                {
                    "CONCEPT_NAME": concept.concept_name,
                    "CONCEPT_FORMULA": concept.concept_formula,
                    "CANDIDATE_TABLE_BLOCKS": candidate_blocks,
                },
            )
            mapping = _complete_json(
                llm,
                system=_read(str(prompts["stage2_system"])),
                user=stage2_user,
                schema=ConceptMappingResult,
            )
            calls += 1
            assert isinstance(mapping, ConceptMappingResult)
            candidate_by_ref = {candidate.table_ref: candidate for candidate in candidates}
            mapped = [item for item in mapping.mappings if item.has_concept and item.table_ref in candidate_by_ref]
            target_count = {"easy": 1, "medium": 2, "intermediate": 3, "hard": 3}[difficulty]
            mapped = mapped[: max(target_count, 1)]
            if len(mapped) < target_count:
                continue
            selected = [candidate_by_ref[item.table_ref] for item in mapped]
            identities = "\n".join(
                f"- Company: {item.company_name} (ticker: {item.ticker}); document: {item.doc_name} (year: {item.year})"
                for item in selected
            )
            common_values = {
                "CONCEPT_NAME": concept.concept_name,
                "CONCEPT_FORMULA": concept.concept_formula,
                "OPERATION": concept.operation,
                "ANSWER_TYPE": concept.answer_type,
                "UNIT": concept.unit,
                "POPULATION": concept.population,
            }
            question = _complete_json(
                llm,
                system=_read(str(prompts["stage3_system"])),
                user=_render(
                    _read(str(prompts["stage3_user"])),
                    {**common_values, "SELECTED_TABLE_IDENTITIES": identities},
                ),
                schema=QuestionDraft,
            )
            calls += 1
            assert isinstance(question, QuestionDraft)
            selected_blocks = "\n\n".join(
                _selected_csv_block(candidate_by_ref[item.table_ref], item.row_or_column_hint, selected_template)
                for item in mapped
            )
            query = _complete_json(
                llm,
                system=_read(str(prompts["stage4_system"])),
                user=_render(
                    _read(str(prompts["stage4_user"])),
                    {
                        **common_values,
                        "TABLE_TOPIC": concept.table_topic,
                        "FINANCIAL_RATIONALE": concept.financial_rationale,
                        "SELECTED_TABLE_BLOCKS_WITH_CSV": selected_blocks,
                    },
                ),
                schema=PandasQuery,
            )
            calls += 1
            assert isinstance(query, PandasQuery)
            paths = {item.table_ref: item.csv_path for item in selected}
            execution = execute_query(query.pandas_query, paths)
            if not execution.ok or not execution.accessed_refs or not execution.accessed_refs.issubset(paths):
                continue
            # The evaluator scores answers numerically, so a non-numeric result (a boolean or
            # a label) could never be marked correct. Discard the candidate instead.
            if coerce_number(execution.actual) is None:
                logger.warning(
                    "Discarding a candidate whose answer is not numeric: %r", execution.actual
                )
                continue
            writer.append(
                QARecord(
                    id=writer.next_id(),
                    question=question.question,
                    answer=execution.actual,  # type: ignore[arg-type]
                    relevant_docs=list(dict.fromkeys(item.doc_name for item in selected)),
                    relevant_tables=[item.table_ref for item in selected],
                    pandas_query=query.pandas_query,
                    csv_path=[str(item.csv_path) for item in selected],
                    difficulty=difficulty,  # type: ignore[arg-type]
                )
            )
            generated += 1
        except Exception as exc:
            logger.warning("Paper generation candidate failed: %s", exc)
    return generated
