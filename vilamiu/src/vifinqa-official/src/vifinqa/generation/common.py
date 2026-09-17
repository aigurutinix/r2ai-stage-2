"""Shared helpers for table context, candidate pools, indexes, and multi-table records.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.document import Document, parse_document
from vifinqa.common.corpus.table import TableAsset, load_table
from vifinqa.embeddings.base import Embedder
from vifinqa.generation.embedding_index import (
    IndexedTable,
    LexicalTableIndex,
    TableIndex,
    make_table_index,
)
from vifinqa.common.filtering.table_filters import is_numeric_cell, is_table_eligible
from vifinqa.generation.parsing import parse_json_object
from vifinqa.generation.prompts.judge import build_naturalness_judge_prompt
from vifinqa.generation.prompts.plan_lock import build_plan_lock_prompt
from vifinqa.generation.prompts.finance_judge import build_finance_judge_prompt
from vifinqa.generation.prompts.alignment_judge import build_alignment_judge_prompt
from vifinqa.generation.scenarios import ScenarioSpec
from vifinqa.generation.schemas import (
    ConceptMappingResult,
    ConceptSelection,
    Difficulty,
    FinancialValidityJudgment,
    GeneratedQA,
    NaturalnessJudgment,
    PandasQuery,
    QARecord,
    QueryAnswer,
    QuestionDraft,
    TableConceptMapping,
)
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.validation.pandas_check import check_answer, execute_query

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
MAX_REWRITE_ATTEMPTS = 1  # Question rewrites allowed before the attempt is considered failed.
# Do not repeat the entire chain over unchanged tables: semantic and mapping failures
# are usually stable while each loop adds several prompt calls. Stages with actionable
# feedback retain their local retries.
MAX_CHAIN_ATTEMPTS = 1
MAX_QUERY_ATTEMPTS = 2  # Query-only retries before restarting the entire chain.
POOL_DOC_CAP = 200
POOL_TABLE_CAP = 300
MAX_STRING_ANSWER_LEN = 80  # Reject narrative answers instead of one terminal value.

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class CandidateTable:
    ticker: str
    company_name: str
    year: str
    doc_name: str
    table_ref: str  # "{doc_name}|table_{id}"
    csv_path: Path
    csv_text: str
    surrounding_pages: str
    table_labels: str  # Column/row labels for naturalness review; contains no values.
    csv_header: str = ""  # First CSV row; valid evidence for UnitClaim.source="csv_header".
    unit_snippets: tuple[str, ...] = ()  # Unit declarations from the table's source page.


def build_candidate_table(
    *,
    doc: DocumentRef,
    table: TableAsset,
    document: Document,
    company_name: str,
    context_before: int = 1,
    context_after: int = 1,
) -> CandidateTable:
    table_ref = f"{doc.doc_name}|table_{table.table_id}"
    return CandidateTable(
        ticker=doc.ticker,
        company_name=company_name,
        year=doc.year,
        doc_name=doc.doc_name,
        table_ref=table_ref,
        csv_path=table.csv_path,
        csv_text=table.raw_csv_text(),
        surrounding_pages=document.table_context(table.table_id, before=context_before, after=context_after),
        table_labels=table_index_text(table),
        csv_header=",".join(table.header),
        unit_snippets=document.table_unit_snippets(table.table_id),
    )


def candidate_table_from_doc(
    doc: DocumentRef,
    table_id: int,
    *,
    company_name: str,
    context_before: int = 1,
    context_after: int = 1,
) -> CandidateTable:
    """Load a table and document from disk and build a ``CandidateTable``."""
    table = load_table(doc.table_csv_path(table_id), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id)
    document = parse_document(doc.text_path)  # type: ignore[arg-type]
    return build_candidate_table(
        doc=doc, table=table, document=document, company_name=company_name, context_before=context_before, context_after=context_after
    )


def table_id_from_ref(table_ref: str) -> int:
    return int(table_ref.rsplit("_", 1)[-1])


def load_table_for_candidate(candidate: CandidateTable) -> TableAsset:
    """Reload a complete ``TableAsset`` when exact cell locations are required.
    """
    return load_table(
        candidate.csv_path,
        ticker=candidate.ticker,
        year=candidate.year,
        doc_name=candidate.doc_name,
        table_id=table_id_from_ref(candidate.table_ref),
    )


def table_index_text(table: TableAsset, *, max_rows: int = 20) -> str:
    """Build a compact column-and-row-label representation for semantic search."""
    columns = " | ".join(h for h in table.header if h)
    row_labels = " | ".join(
        " / ".join(cell for cell in row[:3] if cell and not is_numeric_cell(cell))
        for row in table.rows[:max_rows]
        if any(cell and not is_numeric_cell(cell) for cell in row[:3])
    )
    return f"Cột: {columns}. Dòng: {row_labels}"


def table_retrieval_text(
    table: TableAsset, surrounding_pages: str, *, max_context_chars: int = 2500
) -> str:
    """Add page context so embeddings can distinguish tables with similar headers."""
    context = " ".join(surrounding_pages.split())[:max_context_chars]
    return f"{table_index_text(table)}. Ngữ cảnh: {context}"


def build_pool_index(
    pool_docs: list[DocumentRef],
    embedder: Embedder,
    *,
    rng: random.Random,
    doc_cap: int = POOL_DOC_CAP,
    table_cap: int = POOL_TABLE_CAP,
) -> tuple[TableIndex | LexicalTableIndex, dict[str, tuple[DocumentRef, TableAsset]]]:
    """Bound the candidate pool, load eligible tables, and build a table index.

    Called once per seed, so the index constructor sits on the generator's inner
    loop — see `embedding_index` for why that made the dense path unusable and what
    `VIFINQA_POOL_INDEX` switches.
    """
    docs = list(pool_docs)
    rng.shuffle(docs)
    docs = docs[:doc_cap]

    pairs = [(d, tid) for d in docs for tid in d.table_ids]
    rng.shuffle(pairs)
    pairs = pairs[:table_cap]

    indexed: list[IndexedTable] = []
    lookup: dict[str, tuple[DocumentRef, TableAsset]] = {}
    document_cache: dict[str, Document] = {}
    for doc, table_id in pairs:
        table = load_table(doc.table_csv_path(table_id), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id)
        if not is_table_eligible(table):
            continue
        document = document_cache.get(doc.doc_name)
        if document is None:
            document = parse_document(doc.text_path)  # type: ignore[arg-type]
            document_cache[doc.doc_name] = document
        page_context = document.table_context(table_id, before=0, after=0)
        ref = f"{doc.doc_name}|table_{table_id}"
        indexed.append(
            IndexedTable(
                ticker=doc.ticker,
                year=doc.year,
                doc_name=doc.doc_name,
                table_id=table_id,
                text=table_retrieval_text(table, page_context),
            )
        )
        lookup[ref] = (doc, table)

    return make_table_index(embedder, indexed), lookup


def judge_and_maybe_rewrite(
    *,
    llm: ChatLLM,
    question: str,
    table_labels: str,
    max_rewrites: int = MAX_REWRITE_ATTEMPTS,
) -> tuple[str, bool, str]:
    """Judge naturalness without allowing changes after the query has been locked."""
    del max_rewrites
    system, user = build_naturalness_judge_prompt(question=question, table_labels=table_labels)
    raw = llm.complete(system=system, user=user)
    try:
        judgment = NaturalnessJudgment.model_validate(parse_json_object(raw))
    except Exception as exc:
        logger.warning("Naturalness judge returned invalid JSON: %s", exc)
        return question, False, f"Judge returned invalid JSON: {exc}"
    return question, judgment.natural, judgment.reason


def report_scope(doc_name: str) -> str:
    normalized = doc_name.casefold().replace("_", "").replace("-", "")
    if any(marker in normalized for marker in ("congtyme", "separate", "parent")):
        return "parent"
    if any(marker in normalized for marker in ("hopnhat", "consol", "consolidated")):
        return "consolidated"
    return "unknown"


def report_scope_error(tables: list[CandidateTable]) -> str:
    scopes = {report_scope(table.doc_name) for table in tables}
    known = scopes - {"unknown"}
    if len(known) > 1 or (known and "unknown" in scopes):
        return f"Tables mix unknown or inconsistent report scopes: {sorted(scopes)}"
    return ""


def measurement_basis_error(
    concept: ConceptSelection, mappings: list[TableConceptMapping]
) -> str:
    if concept.measurement_basis == "unknown":
        return "The plan has not locked measurement_basis (gross/net/not_applicable)."

    unknown_refs = sorted(m.table_ref for m in mappings if m.measurement_basis == "unknown")
    if unknown_refs:
        return f"Could not determine measurement_basis for tables: {unknown_refs}."

    mismatched = sorted(
        m.table_ref
        for m in mappings
        if m.measurement_basis != concept.measurement_basis
    )
    if mismatched:
        actual = {m.table_ref: m.measurement_basis for m in mappings}
        return (
            f"Mapping measurement_basis is inconsistent with the plan "
            f"({concept.measurement_basis}): {actual}."
        )
    return ""


def scenario_error(
    scenario: ScenarioSpec, concept: ConceptSelection, selected_tables: list[CandidateTable]
) -> str:
    entities = {t.ticker for t in selected_tables}
    periods = {t.year for t in selected_tables}
    if len(entities) < scenario.min_entities:
        return f"Scenario {scenario.name}: requires >= {scenario.min_entities} companies; got {len(entities)}."
    if len(periods) < scenario.min_periods:
        return f"Scenario {scenario.name}: requires >= {scenario.min_periods} periods; got {len(periods)}."
    if len(selected_tables) < scenario.min_observations:
        return f"Scenario {scenario.name}: requires >= {scenario.min_observations} tables; got {len(selected_tables)}."
    if concept.operation not in scenario.allowed_operations:
        return f"Scenario {scenario.name}: operation={concept.operation} is not allowed."
    if scenario.require_consecutive_periods:
        years = sorted(int(y) for y in periods)
        if years != list(range(years[0], years[-1] + 1)) or len(years) != scenario.min_periods:
            return f"Scenario {scenario.name}: periods must be consecutive; got {sorted(periods)}."
    # Keep period handling explicit and deterministic.
    if scenario.retrieval_kind == "peer_group" and scenario.min_periods > 1:
        periods_by_entity: dict[str, set[str]] = {}
        for t in selected_tables:
            periods_by_entity.setdefault(t.ticker, set()).add(t.year)
        incomplete = sorted(tk for tk, yrs in periods_by_entity.items() if len(yrs) < scenario.min_periods)
        if incomplete:
            return f"Scenario {scenario.name}: these companies lack {scenario.min_periods} periods: {incomplete}."
    if scenario.normalized_metric_only and concept.answer_type != "percentage":
        return (
            f"Scenario {scenario.name}: requires a normalized metric (answer_type=percentage); "
            f"got answer_type={concept.answer_type}."
        )
    return ""


def question_style_error(question: str) -> str:
    normalized = " ".join(question.casefold().split())
    banned = (
        "tại ngày",
        "tại thời điểm",
        "tại cuối năm",
        "thay đổi như thế nào",
        "diễn biến ra sao",
        "trong báo cáo",
        "theo báo cáo",
        "trong bctc",
        "theo bctc",
    )
    found = [phrase for phrase in banned if phrase in normalized]
    if "vnd" in normalized:
        found.append("VND")
    if found:
        return f"Question uses prohibited wording: {', '.join(found)}"
    return ""


def answer_type_error(answer: object, concept: ConceptSelection) -> str:
    expected = concept.answer_type
    if expected in {"money", "percentage", "number"} and (
        isinstance(answer, bool) or not isinstance(answer, (int, float))
    ):
        return f"answer_type={expected} requires a numeric answer."
    if expected == "year" and not (
        isinstance(answer, int) or (isinstance(answer, str) and answer.strip().isdigit() and len(answer.strip()) == 4)
    ):
        return "answer_type=year requires exactly one year."
    if expected == "company" and (not isinstance(answer, str) or not answer.strip()):
        return "answer_type=company requires one company name or ticker."
    if expected == "boolean" and answer not in (True, False, "Có", "Không"):
        return "answer_type=boolean requires a yes/no answer."
    return ""


def concept_plan_error(concept: ConceptSelection, difficulty: Difficulty) -> str:
    required = {
        "concept_name": concept.concept_name,
        "concept_formula": concept.concept_formula,
        "unit": concept.unit,
        "table_topic": concept.table_topic,
        "population": concept.population,
        "financial_rationale": concept.financial_rationale,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        return f"Plan is missing required fields: {missing}"
    if difficulty == "medium" and concept.operation not in {
        "difference",
        "ratio",
        "growth",
        "share",
    }:
        return "Medium accepts one subtraction, division, growth, or share operation over exactly two facts."
    if difficulty == "intermediate" and concept.answer_type == "company":
        return "Intermediate does not accept a company-name answer; use a number, year, or boolean."
    if difficulty == "hard" and len(concept.calculation_steps) < 2:
        return "A Hard plan must have at least two dependent calculation_steps."
    return ""


def finalize_multi_table_record(
    *,
    llm: ChatLLM,
    system: str,
    user: str,
    candidates: list[CandidateTable],
    record_id: int,
    difficulty: Difficulty,
    min_tables: int = 2,
    diversity_key: Callable[[CandidateTable], str] | None = None,
    min_distinct: int = 1,
    max_attempts: int = MAX_ATTEMPTS,
    run_naturalness_judge: bool = True,
) -> QARecord | None:
    candidate_by_ref = {c.table_ref: c for c in candidates}
    feedback = ""

    for _attempt in range(max_attempts):
        prompt_user = user if not feedback else f"{user}\n\nPrevious error: {feedback}\nCorrect it and return the required JSON."
        raw = llm.complete(system=system, user=prompt_user)
        try:
            gen = GeneratedQA.model_validate(parse_json_object(raw))
        except Exception as exc:
            feedback = f"Could not parse valid JSON: {exc}"
            continue

        if not gen.feasible:
            logger.info(
                "LLM reported that no feasible table combination was found (candidates=%s): %s",
                list(candidate_by_ref),
                gen.reason,
            )
            return None

        used_refs = gen.relevant_tables
        unknown = [ref for ref in used_refs if ref not in candidate_by_ref]
        if unknown:
            feedback = f"relevant_tables contains tables outside the supplied list: {unknown}"
            continue
        if len(set(used_refs)) < min_tables:
            feedback = f"The question must use at least {min_tables} distinct tables and list all of them in relevant_tables."
            continue

        if diversity_key is not None:
            distinct = {diversity_key(candidate_by_ref[ref]) for ref in used_refs}
            if len(distinct) < min_distinct:
                feedback = f"relevant_tables must span at least {min_distinct} distinct values; currently {len(distinct)}."
                continue

        doc_names = sorted({candidate_by_ref[ref].doc_name for ref in used_refs})
        if set(gen.relevant_docs) != set(doc_names):
            feedback = f"relevant_docs ({gen.relevant_docs}) does not match the documents for relevant_tables ({doc_names})."
            continue

        shape_error = answer_shape_error(gen.answer)
        if shape_error:
            logger.info("Answer has the wrong format (narrative instead of one value): %r", gen.answer)
            feedback = shape_error
            continue

        csv_map = {ref: candidate_by_ref[ref].csv_path for ref in used_refs}
        result = check_answer(gen.pandas_query, csv_map, gen.answer)
        if not result.ok:
            feedback = result.detail
            continue

        question = gen.question
        if run_naturalness_judge:
            table_labels = "\n".join(candidate_by_ref[ref].table_labels for ref in used_refs)
            question, natural_ok, judge_detail = judge_and_maybe_rewrite(llm=llm, question=gen.question, table_labels=table_labels)
            if not natural_ok:
                feedback = (
                    f"Question was judged unnatural or copied table labels verbatim: {judge_detail}. "
                    "Rewrite it in natural financial language without copying row or column labels verbatim."
                )
                continue

        return QARecord(
            id=record_id,
            question=question,
            answer=gen.answer,
            relevant_docs=doc_names,
            relevant_tables=used_refs,
            pandas_query=gen.pandas_query,
            csv_path=[str(csv_map[ref]) for ref in used_refs],
            difficulty=difficulty,
        )

    logger.info("Skipping after %d unsuccessful attempts (candidates=%s).", max_attempts, list(candidate_by_ref))
    return None


def call_structured(
    llm: ChatLLM,
    *,
    system: str,
    user: str,
    schema: type[SchemaT],
    max_attempts: int = 2,
    before_call: Callable[[], None] | None = None,
) -> tuple[SchemaT | None, str]:
    feedback = ""
    for _attempt in range(max_attempts):
        prompt_user = user if not feedback else f"{user}\n\nPrevious error: {feedback}\nCorrect it and return the required JSON."
        if before_call is not None:
            before_call()
        raw = llm.complete(system=system, user=prompt_user)
        try:
            return schema.model_validate(parse_json_object(raw)), ""
        except Exception as exc:
            feedback = f"Could not parse valid JSON: {exc}"
    return None, feedback


def _select_by_diversity(
    valid: list[TableConceptMapping],
    candidate_by_ref: dict[str, CandidateTable],
    diversity_key: Callable[[CandidateTable], str],
    min_distinct: int,
    max_total: int,
) -> list[TableConceptMapping] | None:
    selected: list[TableConceptMapping] = []
    seen_keys: set[str] = set()
    for mapping in valid:
        if len(selected) >= max_total:
            break
        key = diversity_key(candidate_by_ref[mapping.table_ref])
        if key in seen_keys:
            continue
        selected.append(mapping)
        seen_keys.add(key)
    if len(seen_keys) < min_distinct:
        return None
    return selected


def answer_shape_error(answer: object) -> str:
    """Reject narrative answers in code instead of relying only on the prompt.

    Return feedback when the answer is not exactly one terminal value; otherwise return
    an empty string.
    """
    if not isinstance(answer, str):
        return ""
    if answer in ("Có", "Không"):
        return ""
    if len(answer) > MAX_STRING_ANSWER_LEN:
        return (
            f"answer is {len(answer)} characters long and appears to describe a trend instead of "
            "one terminal value. Return one number, year, entity name, or boolean. If the current "
            "concept cannot produce one value, change the question to a scalar comparison or selector."
        )
    return ""


@dataclass(frozen=True, slots=True)
class ChainPromptBuilders:

    concept: Callable[[list[CandidateTable]], tuple[str, str]]
    mapping: Callable[[list[CandidateTable], ConceptSelection], tuple[str, str]]
    question: Callable[[ConceptSelection, list[CandidateTable]], tuple[str, str]]
    query: Callable[[ConceptSelection, list[CandidateTable], list[TableConceptMapping]], tuple[str, str]]


def run_prompt_chain(
    *,
    llm: ChatLLM,
    candidates: list[CandidateTable],
    builders: ChainPromptBuilders,
    record_id: int,
    difficulty: Difficulty,
    min_tables: int = 2,
    diversity_key: Callable[[CandidateTable], str] | None = None,
    min_distinct: int = 1,
    max_extra_tables: int | None = None,
    max_chain_attempts: int = MAX_CHAIN_ATTEMPTS,
    max_query_attempts: int = MAX_QUERY_ATTEMPTS,
    scenario: ScenarioSpec | None = None,
) -> QARecord | None:
    candidate_by_ref = {c.table_ref: c for c in candidates}

    for chain_attempt in range(max_chain_attempts):
        system, user = builders.concept(candidates)
        concept, err = call_structured(llm, system=system, user=user, schema=ConceptSelection)
        if concept is None:
            logger.info(
                "Chain (attempt %d): could not parse the concept plan (candidates=%s): %s",
                chain_attempt + 1,
                list(candidate_by_ref),
                err,
            )
            continue
        if not concept.feasible:
            logger.info(
                "Chain: table set has no feasible shared concept (candidates=%s): %s",
                list(candidate_by_ref),
                concept.reason,
            )
            return None
        plan_error = concept_plan_error(concept, difficulty)
        if plan_error:
            logger.info("Chain: %s", plan_error)
            continue

        system, user = builders.mapping(candidates, concept)
        mapping_result, _err = call_structured(llm, system=system, user=user, schema=ConceptMappingResult)
        if mapping_result is None:
            continue
        valid = [m for m in mapping_result.mappings if m.has_concept and m.table_ref in candidate_by_ref]

        if diversity_key is not None:
            max_total = (max_extra_tables + 1) if max_extra_tables is not None else len(valid)
            selected = _select_by_diversity(valid, candidate_by_ref, diversity_key, min_distinct, max_total)
            if selected is None:
                logger.info("Chain: mapping did not reach %d distinct diversity_key values.", min_distinct)
                continue
        else:
            if len(valid) < min_tables:
                logger.info("Chain: only %d/%d tables had valid concepts after mapping.", len(valid), min_tables)
                continue
            selected = valid[:min_tables]

        selected_tables = [candidate_by_ref[m.table_ref] for m in selected]

        system, user = build_plan_lock_prompt(concept, selected_tables, selected)
        locked_concept, _err = call_structured(
            llm, system=system, user=user, schema=ConceptSelection
        )
        if locked_concept is None or not locked_concept.feasible:
            logger.info(
                "Chain: could not lock the plan over the selected tables: %s",
                locked_concept.reason if locked_concept else _err,
            )
            continue
        concept = locked_concept
        plan_error = concept_plan_error(concept, difficulty)
        if plan_error:
            logger.info("Chain sau plan lock: %s", plan_error)
            continue

        scope_error = report_scope_error(selected_tables)
        if scope_error:
            logger.info("Chain: %s", scope_error)
            continue

        basis_error = measurement_basis_error(concept, selected)
        if basis_error:
            logger.info("Chain: %s", basis_error)
            continue

        if scenario is not None:
            scen_error = scenario_error(scenario, concept, selected_tables)
            if scen_error:
                logger.info("Chain: %s", scen_error)
                continue

        csv_map = {t.table_ref: t.csv_path for t in selected_tables}
        feedback = ""
        locked_qa: QueryAnswer | None = None
        for _query_attempt in range(max_query_attempts):
            system, user = builders.query(concept, selected_tables, selected)
            if feedback:
                user = f"{user}\n\nPrevious error: {feedback}\nCorrect it and return the required JSON."
            qa, err = call_structured(llm, system=system, user=user, schema=PandasQuery, max_attempts=1)
            if qa is None:
                feedback = err
                logger.info("Chain: query attempt %d could not be parsed: %s", _query_attempt + 1, feedback)
                continue
            execution = execute_query(qa.pandas_query, csv_map)
            if not execution.ok:
                feedback = execution.detail
                logger.info("Chain: query attempt %d failed during execution: %s", _query_attempt + 1, feedback)
                continue
            shape_error = answer_shape_error(execution.actual)
            if shape_error:
                logger.info(
                    "Chain: result has the wrong format (narrative instead of one value): %r",
                    execution.actual,
                )
                feedback = shape_error
                continue
            type_error = answer_type_error(execution.actual, concept)
            if type_error:
                feedback = type_error
                logger.info("Chain: query attempt %d returned the wrong result type: %s", _query_attempt + 1, feedback)
                continue
            missing_refs = set(csv_map) - set(execution.accessed_refs)
            if missing_refs:
                feedback = f"pandas_query did not actually read the selected tables: {sorted(missing_refs)}"
                logger.info("Chain: query attempt %d omitted tables: %s", _query_attempt + 1, feedback)
                continue

            system, user = build_finance_judge_prompt(
                concept,
                selected_tables,
                selected,
                pandas_query=qa.pandas_query,
                actual_result=execution.actual,
            )
            finance_judgment, err = call_structured(
                llm,
                system=system,
                user=user,
                schema=FinancialValidityJudgment,
                max_attempts=1,
            )
            if finance_judgment is None or not finance_judgment.valid:
                feedback = finance_judgment.reason if finance_judgment else err
                logger.info(
                    "Chain: query attempt %d failed the finance critic: %s",
                    _query_attempt + 1,
                    feedback,
                )
                continue

            locked_qa = QueryAnswer(pandas_query=qa.pandas_query, answer=execution.actual)
            break

        if locked_qa is None:
            logger.info("Chain: pandas_query remained invalid after %d attempts; restarting from step 1.", max_query_attempts)
            continue

        question_feedback = ""
        table_labels = "\n".join(t.table_labels for t in selected_tables)
        for _question_attempt in range(MAX_ATTEMPTS):
            system, user = builders.question(concept, selected_tables)
            if question_feedback:
                user = f"{user}\n\nPrevious error: {question_feedback}\nRewrite it and return the required JSON."
            draft, err = call_structured(llm, system=system, user=user, schema=QuestionDraft, max_attempts=1)
            if draft is None:
                question_feedback = err
                continue
            style_error = question_style_error(draft.question)
            if style_error:
                question_feedback = style_error
                continue
            table_identities = "\n".join(
                f"- {table.company_name} ({table.ticker}), năm {table.year}, "
                f"phạm vi {report_scope(table.doc_name)}, tài liệu {table.doc_name}"
                for table in selected_tables
            )
            system, user = build_alignment_judge_prompt(concept, draft.question, table_identities)
            alignment, err = call_structured(
                llm, system=system, user=user, schema=FinancialValidityJudgment, max_attempts=1
            )
            if alignment is None or not alignment.valid:
                question_feedback = alignment.reason if alignment else err
                continue
            question, natural_ok, judge_detail = judge_and_maybe_rewrite(
                llm=llm, question=draft.question, table_labels=table_labels
            )
            if not natural_ok:
                question_feedback = f"Question is unnatural: {judge_detail}"
                continue

            used_refs = [table.table_ref for table in selected_tables]
            doc_names = sorted({table.doc_name for table in selected_tables})
            return QARecord(
                id=record_id,
                question=question,
                answer=locked_qa.answer,
                relevant_docs=doc_names,
                relevant_tables=used_refs,
                pandas_query=locked_qa.pandas_query,
                csv_path=[str(csv_map[ref]) for ref in used_refs],
                difficulty=difficulty,
            )

        logger.info("Chain: could not write an acceptable question after locking the query: %s", question_feedback)

    logger.info(
        "Skipping after %d unsuccessful full-chain attempts (candidates=%s).", max_chain_attempts, list(candidate_by_ref)
    )
    return None
