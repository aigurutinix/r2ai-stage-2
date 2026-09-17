"""Grounded product pipeline: retrieval -> Pandas -> verification -> citations.

The leaderboard artifact and the live assistant intentionally share the same
table identifiers. This module adds the product-only contract: ask for missing
facets instead of guessing, bind citations to the executed program, re-run the
winning program, and refuse when the retrieval trace is insufficient.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kingpro.answering.llm_client import (
    endpoint_compliance,
    llm_audit_context,
    make_llm_from_env,
)
from kingpro.answering.pandas_answer import requested_unit
from kingpro.answering.program_engine import run_program
from kingpro.answering.sandbox import run_pandas_code
from kingpro.evaluation.metrics import coerce_number
from kingpro.product.cell_lineage import RuntimeCellLineageIndex
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler
from kingpro.retrieval.bm25_index import (
    extract_all_facets,
    fold,
    retrieve_decomposed,
    tables_in_reports,
    tokenize,
)
from kingpro.retrieval.table_reranker import rerank_tables
from kingpro.retrieval.structural_fusion import (
    StructuralSidecarRetriever,
    guarded_structural_rescue,
)
from kingpro.governance.approval import annotate_read_response
from kingpro.governance.audit import get_audit_ledger
from kingpro.governance.validation import validate_financial_result
from kingpro.governance.lifecycle import read_lifecycle_status
from kingpro.governance.crypto import encryption_health
from kingpro.governance.isolation import isolation_health

ROOT = Path(__file__).resolve().parents[3]

_STOPWORDS = {
    "ai", "bao", "bao_nhieu", "bang", "bao_cao", "cac", "cho", "cong",
    "cong_ty", "co", "cua", "duoc", "gia_tri", "hay", "la", "ma", "nam",
    "nao", "nhieu", "nhung", "o", "tai", "theo", "thi", "trong", "tu",
    "va", "ve", "voi", "vnd", "dong", "trieu", "ty", "so",
}

# Registry paraphrase replay is intentionally much stricter than ordinary
# search.  These are folded Vietnamese function words only; financial nouns
# such as ``tai`` (tài sản), ``von`` and ``loi`` must remain discriminative.
_PARAPHRASE_STOPWORDS = {
    "ai", "bao", "bao_nhieu", "bang", "bao_cao", "cac", "cho", "co",
    "cua", "duoc", "hay", "la", "ma", "nam", "nao", "nhieu", "nhung",
    "o", "theo", "thi", "trong", "tu", "va", "ve", "voi", "vnd",
    "dong", "trieu", "ty", "so", "dat", "vao", "cuoi",
}

_ENTITY_LEGAL_TERMS = {
    "cong", "ty", "ct", "ctcp", "tong", "co", "phan", "tap", "doan",
    "ngan", "hang", "tmcp", "tnhh", "mtv", "nha", "nuoc",
}

_INTENT_CUES = {
    "delta": re.compile(r"\b(?:chenh lech|thay doi|muc tang|muc giam|khac nhau)\b"),
    "ratio": re.compile(r"\b(?:ty le|phan tram|diem phan tram|roe|roa|ros)\b"),
    "average": re.compile(r"\b(?:trung binh|binh quan)\b"),
    "extreme": re.compile(r"\b(?:cao nhat|thap nhat|lon nhat|nho nhat)\b"),
    "growth": re.compile(r"\b(?:tang truong|toc do tang|toc do giam)\b"),
    "compare": re.compile(r"\b(?:giua|so voi|hon|kem)\b"),
}

# The competition corpus is a financial-statement warehouse, not a live
# market-data, brand, HR-survey or application-analytics feed.  These topics
# can share entity/year words with a valid BCTC question and therefore receive
# deceptively high BM25 confidence.  Refuse them deterministically before the
# model runs; each code is user-explainable and can be audited independently.
_OUT_OF_CORPUS_CUES = {
    "live_market_data": re.compile(
        r"\b(?:gia co phieu|gia dong cua|thi gia co phieu|von hoa thi truong)\b"
    ),
    "brand_visual": re.compile(r"\b(?:logo|ma hex|mau sac thuong hieu)\b"),
    "application_analytics": re.compile(
        r"\b(?:ung dung(?: di dong)?.{0,40}luot tai|"
        r"luot tai.{0,20}(?:ung dung|app)|nguoi dung hoat dong|analytics ung dung)\b"
    ),
    "employee_survey": re.compile(
        r"\b(?:hai long nhan vien|khao sat noi bo|diem enps)\b"
    ),
    "external_esg_rating": re.compile(
        r"\b(?:xep hang esg|diem esg|msci esg|sustainalytics)\b"
    ),
    "non_financial_absurd": re.compile(
        r"\b(?:thu cung|ky lan|nhiet do van phong)\b"
    ),
}

# Reject instructions that try to turn the financial assistant into a prompt,
# credential, filesystem or network exfiltration tool. This gate runs before
# registry replay, retrieval and model generation, including when a malicious
# request also contains a valid ticker and year.
_UNSAFE_REQUEST_CUES = {
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|forget)\b.{0,50}\b(?:instruction|prompt|system|developer)\b|"
        r"\b(?:bo qua|quen|khong tuan theo)\b.{0,50}\b(?:chi dan|huong dan|lenh|prompt)\b"
    ),
    "prompt_exfiltration": re.compile(
        r"\b(?:system prompt|developer message|hidden prompt|reveal prompt|show prompt|"
        r"tiet lo prompt|hien prompt|chi dan he thong)\b"
    ),
    "credential_exfiltration": re.compile(
        r"\b(?:api[ _-]?key|access[ _-]?token|secret[ _-]?key|password|credential|"
        r"khoa api|ma truy cap|mat khau|bien moi truong|environment variable)\b"
    ),
    "runtime_escape": re.compile(
        r"\b(?:import\s+(?:os|subprocess|socket|urllib|requests)|curl|wget|"
        r"doc file|ghi file|xoa file|read file|write file|delete file|"
        r"truy cap mang|network request)\b"
    ),
}

_DEMO_EVIDENCE_KEYS = (
    "btc_model_size_confirmation",
    "runpod_checkpoint_screenshot",
    "exact_model_card_and_license",
    "team_and_submission_account_eligibility",
    "data_rights_inventory",
    "ip_and_development_tool_disclosure",
    "winning_artifact_handover_inventory",
    "timed_pitch_rehearsal",
)


def _intent_signature(question: str) -> frozenset[str]:
    normalized = fold(question)
    return frozenset(name for name, pattern in _INTENT_CUES.items() if pattern.search(normalized))


def _time_basis(question: str) -> str:
    normalized = fold(question)
    if re.search(r"\b(?:cuoi nam|tai ngay|31\s*/\s*12|31-12)\b", normalized):
        return "year_end"
    if re.search(r"\b(?:trong nam|cho nam|nam tai chinh|ket thuc nam)\b", normalized):
        return "period"
    return ""


def _out_of_corpus_topic(question: str) -> str | None:
    normalized = fold(question)
    return next(
        (
            topic
            for topic, pattern in _OUT_OF_CORPUS_CUES.items()
            if pattern.search(normalized)
        ),
        None,
    )


def _unsafe_request_topic(question: str) -> str | None:
    normalized = fold(question)
    return next(
        (
            topic
            for topic, pattern in _UNSAFE_REQUEST_CUES.items()
            if pattern.search(normalized)
        ),
        None,
    )


def _trace_stage(
    stage: str,
    title: str,
    started: float,
    *,
    detail: str,
    metadata: dict | None = None,
    status: str = "completed",
) -> dict:
    """Build one measured, JSON-safe product trace stage."""

    return {
        "stage": stage,
        "title": title,
        "status": status,
        "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "detail": detail,
        "metadata": metadata or {},
    }


def _notify_progress(callback: Callable[[dict], None] | None, payload: dict) -> None:
    """Progress must never be allowed to break the financial pipeline."""

    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        return


@dataclass(frozen=True)
class RefusalPolicy:
    min_question_chars: int = 8
    max_question_chars: int = 1_500
    max_entity_year_pairs: int = 40
    # Local calibration (32 clean positives + 192 structured out-of-corpus
    # negatives): the full domain-boundary + retrieval stack at 0.90 accepts
    # 90.6% positives and rejects all negatives in that finite development
    # suite.  This is not a hidden-test claim. Recalibrate when the corpus or
    # retriever changes; do not copy this value to another domain.
    min_grounding_confidence: float = 0.90
    max_tables: int = 24
    base_tables: int = 8
    # Retrieve a wider cheap BM25 pool, then rerank by row labels read from the
    # actual extracted CSV. This repairs OCR tables whose first column is only
    # a numeric ``Mã số`` field and therefore absent from legacy search_text.
    table_rerank_pool: int = 40
    table_label_weight: float = 4.0
    vote_count: int = 2
    min_vote_ratio: float = 0.5
    execution_timeout: float = 6.0
    # A paraphrase can replay an audited registry program only when all of
    # these independent gates pass.  The defaults are calibrated on the live
    # 1,012-row registry and deliberately favor refusal over a wrong metric.
    min_registry_paraphrase_score: float = 0.78
    min_registry_candidate_recall: float = 0.75
    min_registry_query_precision: float = 0.65
    min_registry_paraphrase_margin: float = 0.12
    min_registry_shared_terms: int = 4

    @classmethod
    def from_env(cls) -> "RefusalPolicy":
        return cls(
            min_grounding_confidence=float(os.getenv("KINGPRO_MIN_GROUNDING_CONFIDENCE", "0.90")),
            max_tables=int(os.getenv("KINGPRO_MAX_TABLES", "24")),
            table_rerank_pool=int(os.getenv("KINGPRO_TABLE_RERANK_POOL", "40")),
            table_label_weight=float(os.getenv("KINGPRO_TABLE_LABEL_WEIGHT", "4.0")),
            vote_count=int(os.getenv("KINGPRO_VOTE_COUNT", "2")),
            min_vote_ratio=float(os.getenv("KINGPRO_MIN_VOTE_RATIO", "0.5")),
            execution_timeout=float(os.getenv("KINGPRO_EXECUTION_TIMEOUT", "6")),
            min_registry_paraphrase_score=float(
                os.getenv("KINGPRO_REGISTRY_PARAPHRASE_MIN_SCORE", "0.78")
            ),
            min_registry_candidate_recall=float(
                os.getenv("KINGPRO_REGISTRY_PARAPHRASE_MIN_CANDIDATE_RECALL", "0.75")
            ),
            min_registry_query_precision=float(
                os.getenv("KINGPRO_REGISTRY_PARAPHRASE_MIN_QUERY_PRECISION", "0.65")
            ),
            min_registry_paraphrase_margin=float(
                os.getenv("KINGPRO_REGISTRY_PARAPHRASE_MIN_MARGIN", "0.12")
            ),
            min_registry_shared_terms=int(
                os.getenv("KINGPRO_REGISTRY_PARAPHRASE_MIN_SHARED_TERMS", "4")
            ),
        )


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _report_id(table_ref: str) -> str:
    return str(table_ref).split("|", 1)[0]


def _question_terms(question: str, facets: dict) -> set[str]:
    excluded = {str(v).lower() for v in facets.get("tickers", []) + facets.get("years", [])}
    return {
        token.lower()
        for token in tokenize(question)
        if len(token) >= 2 and token.lower() not in _STOPWORDS and token.lower() not in excluded
    }


def _context_terms(rows: list[dict]) -> set[str]:
    terms: set[str] = set()
    for row in rows:
        terms.update(token.lower() for token in tokenize(str(row.get("search_text", ""))) if len(token) >= 2)
    return terms


def _anchor_coverage(question: str, facets: dict, rows: list[dict]) -> float:
    wanted = _question_terms(question, facets)
    if not wanted:
        return 0.0
    return len(wanted & _context_terms(rows)) / len(wanted)


def _registry_key(question: str) -> str:
    """Normalize harmless punctuation/spacing only; never fuzzy-match finance questions."""
    return " ".join(re.findall(r"\w+", str(question).casefold(), flags=re.UNICODE))


# Keep the measured public champion separate from measured non-champion successors.
# Judge View consumes this object through /health, so a leaderboard promotion has
# one backend source of truth instead of another set of hard-coded UI numbers.
PUBLIC_CHAMPION = {
    "version": "v297",
    "submission_id": 3747,
    "artifact": "sub_v297_scope2",
    "artifact_sha256": "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC",
    "submission_json_sha256": "E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85",
    "on_leaderboard": True,
    "release_gate": "PASS",
    "scores": {
        "execution_accuracy": 0.7115,
        "answer_accuracy": 0.7115,
        "tables_f2_macro": 0.6120,
        "tables_precision": 0.5928,
        "tables_recall": 0.6261,
        "tables_mrr5": 0.6514,
        "docs_f2_macro": 0.9618,
        "docs_precision": 0.9587,
        "docs_recall": 0.9678,
        "docs_mrr5": 0.9806,
    },
    "release_proof": {
        "source_correct_answer_ids": [98, 224, 764, 966],
        "q98_measured_retrieval_union": {
            "documents": [
                "HUT_financial_statements_2024_separate",
                "HUT_financial_statements_2024_consolidated",
            ],
            "tables": [
                "HUT_financial_statements_2024_separate|328",
                "HUT_financial_statements_2024_consolidated|325",
            ],
            "physical_answer_table": "HUT_financial_statements_2024_consolidated|325",
        },
        "q764_physical_answer_table": "DPM_financial_statements_2015_consolidated|1909",
        "q224_measured_retrieval_union": {
            "documents": [
                "HUT_financial_statements_2024_consolidated",
                "HUT_financial_statements_2024_separate",
            ],
            "tables": [
                "HUT_financial_statements_2024_consolidated|1243",
                "HUT_financial_statements_2024_separate|1624",
            ],
            "physical_answer_table": "HUT_financial_statements_2024_separate|1624",
        },
        "q966_physical_answer_table": "GEG_financial_statements_2025_consolidated|493",
        "excluded_answer_change_ids": [714],
        "provenance_allowlist_sha256": "7209025C445F78305A26EDC515ADA49AF7A1464B63AD62D46415984675831B73",
    },
    "claim_scope": (
        "BTC public leaderboard row 3747 is Finished and selected. It preserves "
        "Execution/Answer/MRR while improving Tables and Docs retrieval over v290. "
        "Release proof covers inherited q98/q764 plus q224/q966; q714 remains "
        "explicitly excluded. Not a private or final result."
    ),
}

ROLLBACK_FALLBACK = {
    "version": "v290",
    "role": "direct_measured_rollback_fallback",
    "artifact": "sub_v290_scope2",
    "artifact_sha256": "711A3493279387593ECA17C3C4130154E9F860CA891B91013083E7D76E5C04A4",
    "measured": True,
    "submission_id": 3745,
    "complete_score_vector": True,
    "scores": {
        "execution_accuracy": 0.7115,
        "answer_accuracy": 0.7115,
        "tables_f2_macro": 0.6114,
        "tables_precision": 0.5921,
        "tables_recall": 0.6254,
        "tables_mrr5": 0.6514,
        "docs_f2_macro": 0.9611,
        "docs_precision": 0.9580,
        "docs_recall": 0.9672,
        "docs_mrr5": 0.9806,
    },
    "release_gate": "PASS",
    "on_leaderboard": False,
    "fallback_priority": 1,
    "claim_scope": (
        "BTC public submission 3745 is Finished and is the direct measured rollback "
        "fallback behind selected v297."
    ),
}

MEASURED_ROLLBACK_FALLBACK = {
    "version": "v276",
    "role": "secondary_measured_rollback_fallback",
    "artifact": "sub_v276_q638_fix",
    "artifact_sha256": "86A52DA0FE9A9121C6BB08191FDC2FABFDD97620020B9F3F46DAB81B2258478E",
    "measured": True,
    "submission_id": 3742,
    "complete_score_vector": True,
    "scores": dict(ROLLBACK_FALLBACK["scores"]),
    "release_gate": "PASS",
    "on_leaderboard": False,
    "fallback_priority": 2,
    "claim_scope": "Measured V276 rollback behind V290 and ahead of V269.",
}

LOCAL_SUCCESSOR = {
    "version": "v218",
    "artifact": "sub_top123_candidate_v218_existing_table_completeness_batch4",
    "artifact_sha256": "F0C7667D8CB222A120092834DEE553637582E12746BC25BA614E2F3BAA6F4C18",
    "submission_id": 3699,
    "measured": True,
    "complete_score_vector": True,
    "scores": {
        "execution_accuracy": 0.7115,
        "answer_accuracy": 0.7115,
        "tables_f2_macro": 0.6104,
        "tables_precision": 0.5912,
        "tables_recall": 0.6244,
        "tables_mrr5": 0.6514,
        "docs_f2_macro": 0.9611,
        "docs_precision": 0.9580,
        "docs_recall": 0.9672,
        "docs_mrr5": 0.9806,
    },
    "claim_scope": (
        "BTC submission 3699 is Finished and ties selected row 3696 across all "
        "ten measured public metrics. It is not selected on the leaderboard."
    ),
}

AUDITED_CANDIDATE = {
    "version": "v225",
    "artifact": "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9",
    "artifact_sha256": "9E452A3814CB5DA5C62493B5B902ABF5E42F78D4ABC745F270F7E00A6CDF8888",
    "submission_json_sha256": "45BCF57FDB5CF06ACD0E0669CFA88166E376130815FF644AEA88A71154DA64B0",
    "measured": True,
    "submission_id": 3723,
    "complete_score_vector": True,
    "scores": {
        "execution_accuracy": 0.7095,
        "answer_accuracy": 0.7095,
        "tables_f2_macro": 0.6092,
        "tables_precision": 0.5900,
        "tables_recall": 0.6233,
        "tables_mrr5": 0.6504,
        "docs_f2_macro": 0.9608,
        "docs_precision": 0.9570,
        "docs_recall": 0.9672,
        "docs_mrr5": 0.9806,
    },
    "release_gate": "PASS",
    "individual_audit_questions": 1012,
    "claim_scope": (
        "Local audited candidate only: 1,012/1,012 questions have explicit "
        "ledger verdicts and the full release gate passed. BTC public submission "
        "3723 measured 0.7095 execution/answer and is not the champion. Its full "
        "ten-metric vector exactly ties v224 submission 3722, proving the q98 "
        "rollback was public-neutral at displayed precision. Artifact attribution "
        "uses the local upload timeline; no private/final score is claimed."
    ),
}

SOURCE_CLEAN_FALLBACK = {
    "version": "v269",
    "role": "secondary_source_clean_rollback_fallback",
    "artifact": "sub_top123_candidate_v269_source_lineage_control",
    "artifact_sha256": "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
    "submission_json_sha256": "65973279A49DB686F9883411E5202D6FD950CA80A41DC3A0F8451556316C2663",
    "measured": True,
    "submission_id": 3741,
    "complete_score_vector": True,
    "scores": {
        "execution_accuracy": 0.7115,
        "answer_accuracy": 0.7115,
        "tables_f2_macro": 0.6104,
        "tables_precision": 0.5912,
        "tables_recall": 0.6244,
        "tables_mrr5": 0.6514,
        "docs_f2_macro": 0.9611,
        "docs_precision": 0.9580,
        "docs_recall": 0.9672,
        "docs_mrr5": 0.9806,
    },
    "release_gate": "PASS",
    "source_lineage_repairs": [24, 61, 709, 826, 861],
    "on_leaderboard": False,
    "fallback_priority": 3,
    "claim_scope": (
        "BTC public submission 3741 is Finished and ties former selected v217 "
        "across all ten metrics. It keeps five source-lineage repairs but is now "
        "the tertiary fallback behind selected v297, measured v290 and v276, not a "
        "private/final score claim."
    ),
}

DEFAULT_REPLAY_CANDIDATES = (
    PUBLIC_CHAMPION["artifact"],
    ROLLBACK_FALLBACK["artifact"],
    MEASURED_ROLLBACK_FALLBACK["artifact"],
    "sub_top123_candidate_v217_missing_panel_operand_batch3",
    "sub_top123_candidate_v207_semantic_batch6_final",
    "sub_top123_candidate_v206_semantic_batch11",
    "sub_top123_candidate_v205_q118_vcb_general_provision",
    "sub_top123_candidate_v203_q24_q607_double_unit",
    "sub_top123_candidate_v202_q915_hag_finance_expense_unit",
    "sub_top123_candidate_v192_data_derived_table_order",
    "sub_top123_candidate_v190_pdr_growth_count",
    "sub_top123_candidate_v184_mbb_credit_provision_ratio",
    "sub_top123_candidate_v165_hag_long_receivables",
    "sub_top123_candidate_v161_legacy_source43",
)


class ProductService:
    """One process-safe facade used by CLI, HTTP API and future UI."""

    def __init__(
        self,
        *,
        root: str | Path = ROOT,
        llm_fn: Callable[[str, str], str] | None = None,
        policy: RefusalPolicy | None = None,
        replay_submission: str | Path | None = None,
        structural_sidecar: StructuralSidecarRetriever | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.policy = policy or RefusalPolicy.from_env()
        self.enforce_model_policy = llm_fn is None
        self.llm_fn = llm_fn or make_llm_from_env()
        self._catalog: dict[str, dict] | None = None
        self._catalog_summary: dict | None = None
        self._company_names: dict[str, str] | None = None
        structural_catalog = os.getenv("KINGPRO_STRUCTURAL_CATALOG")
        structural_index = os.getenv("KINGPRO_STRUCTURAL_INDEX")
        if structural_sidecar is None and bool(structural_catalog) != bool(structural_index):
            raise ValueError(
                "KINGPRO_STRUCTURAL_CATALOG and KINGPRO_STRUCTURAL_INDEX must be set together"
            )
        self._structural_sidecar = structural_sidecar
        if self._structural_sidecar is None and structural_catalog and structural_index:
            self._structural_sidecar = StructuralSidecarRetriever(
                structural_catalog,
                structural_index,
            )
        replay_candidates = [
            self.root / name / "submission.json" for name in DEFAULT_REPLAY_CANDIDATES
        ]
        default_replay = next(
            (candidate for candidate in replay_candidates if candidate.is_file()),
            replay_candidates[-1],
        )
        configured_replay = replay_submission or os.getenv("KINGPRO_REPLAY_SUBMISSION")
        self.replay_submission = (
            Path(configured_replay).resolve()
            if configured_replay
            else default_replay.resolve()
        )
        self._replay_registry: dict[str, dict] | None = None
        self._replay_paraphrase_index: list[dict] | None = None
        self._cell_lineage = RuntimeCellLineageIndex(
            self.root, self.replay_submission.parent
        )
        self._deterministic_compiler = DeterministicFinancialCompiler(self.root)
        self._cache_lock = threading.Lock()
        self._retrieval_lock = threading.Lock()

    def _load_catalog(self) -> dict[str, dict]:
        if self._catalog is None:
            with self._cache_lock:
                if self._catalog is None:
                    path = self.root / "build" / "catalog.jsonl"
                    rows: dict[str, dict] = {}
                    with path.open(encoding="utf-8") as handle:
                        for line in handle:
                            row = json.loads(line)
                            rows[row["table_ref"]] = row
                    self._catalog = rows
        return self._catalog

    def _load_company_names(self) -> dict[str, str]:
        if self._company_names is None:
            with self._cache_lock:
                if self._company_names is None:
                    import csv

                    names: dict[str, str] = {}
                    path = self.root / "data" / "code_stock.csv"
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        for row in csv.DictReader(handle):
                            ticker = (row.get("Mã CK") or row.get("Ma CK") or "").strip()
                            name = (row.get("Tên công ty") or row.get("Ten cong ty") or ticker).strip()
                            if ticker:
                                names[ticker] = name
                    self._company_names = names
        return self._company_names

    def _load_replay_registry(self) -> dict[str, dict]:
        if self._replay_registry is None:
            with self._cache_lock:
                if self._replay_registry is None:
                    registry: dict[str, dict] = {}
                    if self.replay_submission.is_file():
                        rows = json.loads(self.replay_submission.read_text(encoding="utf-8"))
                        for row in rows:
                            key = _registry_key(str(row.get("question", "")))
                            # Duplicate normalized questions would make replay
                            # ambiguous. Remove them rather than choose one.
                            if not key:
                                continue
                            if key in registry:
                                registry[key] = {}
                            else:
                                registry[key] = row
                        registry = {key: row for key, row in registry.items() if row}
                    self._replay_registry = registry
        return self._replay_registry

    def _verified_registry_replay(
        self, question: str, trace_id: str, started: float,
    ) -> dict | None:
        row = self._load_replay_registry().get(_registry_key(question))
        if not row:
            return None
        return self._replay_registry_row(
            row,
            question,
            trace_id,
            started,
            mode="verified_registry",
            match={"exact": True, "score": 1.0, "margin": 1.0},
        )

    def _semantic_terms(self, question: str, facets: dict) -> set[str]:
        excluded = {fold(str(value)) for value in facets.get("tickers", []) + facets.get("years", [])}
        names = self._load_company_names()
        for ticker in facets.get("tickers", []):
            for token in tokenize(names.get(str(ticker), "")):
                normalized = fold(token)
                if normalized not in _ENTITY_LEGAL_TERMS:
                    excluded.add(normalized)
        terms: set[str] = set()
        for token in tokenize(question):
            normalized = fold(token)
            if (
                len(normalized) >= 2
                and normalized not in _PARAPHRASE_STOPWORDS
                and normalized not in excluded
            ):
                terms.add(normalized)
        return terms

    def _load_replay_paraphrase_index(self) -> list[dict]:
        if self._replay_paraphrase_index is None:
            # Prime dependent caches before taking the shared cache lock;
            # threading.Lock is intentionally non-reentrant.
            registry_rows = list(self._load_replay_registry().values())
            self._load_company_names()
            with self._cache_lock:
                if self._replay_paraphrase_index is None:
                    index: list[dict] = []
                    for row in registry_rows:
                        question = str(row.get("question", ""))
                        facets = extract_all_facets(question)
                        terms = self._semantic_terms(question, facets)
                        unit = requested_unit(question)
                        if facets["tickers"] and facets["years"] and terms:
                            index.append(
                                {
                                    "row": row,
                                    "facets": facets,
                                    "terms": terms,
                                    "unit": unit,
                                    "intent": _intent_signature(question),
                                    "time_basis": _time_basis(question),
                                }
                            )
                    self._replay_paraphrase_index = index
        return self._replay_paraphrase_index

    def _match_registry_paraphrase(self, question: str, facets: dict) -> tuple[dict, dict] | None:
        """Return one uniquely safe registry row, otherwise refuse to guess.

        Matching is permitted only after exact entity/year/scope/unit/operator
        agreement.  Lexical similarity is then used solely to recognize a
        reordered or lightly rephrased version of the same financial metric.
        """
        query_terms = self._semantic_terms(question, facets)
        if len(query_terms) < self.policy.min_registry_shared_terms:
            return None
        query_unit = requested_unit(question)
        query_intent = _intent_signature(question)
        query_time = _time_basis(question)
        query_tickers = set(facets["tickers"])
        query_years = set(facets["years"])
        scored: list[tuple[float, float, float, int, dict]] = []
        for item in self._load_replay_paraphrase_index():
            candidate_facets = item["facets"]
            if query_tickers != set(candidate_facets["tickers"]):
                continue
            if query_years != set(candidate_facets["years"]):
                continue
            if facets["scope"] != candidate_facets["scope"]:
                continue
            if query_unit != item["unit"] or query_intent != item["intent"]:
                continue
            candidate_time = str(item["time_basis"])
            if query_time and candidate_time and query_time != candidate_time:
                continue
            candidate_terms = set(item["terms"])
            shared = len(query_terms & candidate_terms)
            if shared < self.policy.min_registry_shared_terms:
                continue
            query_precision = shared / len(query_terms)
            candidate_recall = shared / len(candidate_terms)
            score = 2 * query_precision * candidate_recall / (query_precision + candidate_recall)
            scored.append((score, query_precision, candidate_recall, shared, item["row"]))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], int(item[4].get("id", 0))))
        best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        margin = best[0] - runner_up
        if (
            best[0] < self.policy.min_registry_paraphrase_score
            or best[1] < self.policy.min_registry_query_precision
            or best[2] < self.policy.min_registry_candidate_recall
            or margin < self.policy.min_registry_paraphrase_margin
        ):
            return None
        return best[4], {
            "exact": False,
            "score": round(best[0], 4),
            "query_precision": round(best[1], 4),
            "candidate_recall": round(best[2], 4),
            "shared_terms": best[3],
            "margin": round(margin, 4),
            "source_question_id": best[4].get("id"),
        }

    def _verified_registry_paraphrase(
        self, question: str, facets: dict, trace_id: str, started: float,
    ) -> dict | None:
        matched = self._match_registry_paraphrase(question, facets)
        if matched is None:
            return None
        row, match = matched
        return self._replay_registry_row(
            row,
            question,
            trace_id,
            started,
            mode="verified_registry_paraphrase",
            match=match,
        )

    def _replay_registry_row(
        self,
        row: dict,
        question: str,
        trace_id: str,
        started: float,
        *,
        mode: str,
        match: dict,
    ) -> dict | None:
        base = self.replay_submission.parent
        csv_paths = {
            str(item["variable"]): str((base / str(item["csv_path"])).resolve())
            for item in row.get("evidence", [])
            if item.get("variable") and item.get("csv_path")
        }
        if not csv_paths or any(not Path(path).is_file() for path in csv_paths.values()):
            return None
        replay = run_pandas_code(
            str(row.get("pandas_query", "")),
            csv_paths,
            timeout=self.policy.execution_timeout,
        )
        replay_number = coerce_number(replay.get("result")) if replay.get("ok") else None
        expected = coerce_number(row.get("answer"))
        if replay_number is None or expected is None:
            return None
        if not math.isfinite(replay_number) or not math.isfinite(expected):
            return None
        if abs(expected - replay_number) > 1e-6 + 1e-9 * max(abs(expected), abs(replay_number), 1.0):
            return None
        table_refs = _unique([str(ref) for ref in row.get("relevant_tables", []) if ref])
        citations = self._citations([{"table_ref": ref} for ref in table_refs])
        if not citations or len(citations) != len(table_refs):
            return None
        facets = extract_all_facets(question)
        unit_name, _unit_multiplier = requested_unit(question)
        try:
            source_artifact = str(base.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            source_artifact = str(base)
        source_cells = self._cell_lineage.for_question(row.get("id"))
        causal_cells = sum(
            cell.get("verification")
            == "counterfactual_result_dependency_and_coordinate_raw_match"
            for cell in source_cells
        )
        response = {
            "trace_id": trace_id,
            "status": "answered",
            "grounded": True,
            "answer": replay_number,
            "unit": unit_name,
            "pandas_query": str(row["pandas_query"]),
            "citations": citations,
            "retrieval": {
                "facets": facets,
                "checks": {
                    "requested_pairs": [],
                    "found_pairs": [],
                    "pair_coverage": 1.0,
                    "anchor_coverage": 1.0,
                    "confidence": 1.0,
                    "threshold": self.policy.min_grounding_confidence,
                },
                "documents": _unique([_report_id(ref) for ref in table_refs]),
                "tables": table_refs,
            },
            "verification": {
                "citation_bound": True,
                "replay_match": True,
                "registry_match": True,
                "registry_exact": bool(match.get("exact")),
                "mode": mode,
                "paraphrase_match": match if not match.get("exact") else None,
                "source_cells": source_cells,
                "cell_lineage": {
                    "status": "verified" if source_cells else "unavailable",
                    "verified_cells": len(source_cells),
                    "causal_cells": causal_cells,
                    "policy": (
                        "hash_bound_counterfactual_dependency_and_coordinate_raw_match"
                        if causal_cells
                        else "coordinate_and_raw_match"
                    ),
                },
                "vote_ratio": 1.0,
            },
            "submission_export": {
                "source_artifact": source_artifact,
                "answer": replay_number,
                "relevant_docs": list(row.get("relevant_docs", [])),
                "relevant_tables": list(row.get("relevant_tables", [])),
                "evidence": [dict(item) for item in row.get("evidence", [])],
                "pandas_query": str(row["pandas_query"]),
            },
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        response["business_validation"] = validate_financial_result(
            question, replay_number, unit_name
        )
        return annotate_read_response(
            response,
            authority="verified-registry-pandas-replay",
            llm_role="none",
        )

    @staticmethod
    def _refusal(trace_id: str, code: str, message: str, *, details: dict | None = None) -> dict:
        return annotate_read_response({
            "trace_id": trace_id,
            "status": "refused",
            "grounded": False,
            "answer": None,
            "pandas_query": "",
            "citations": [],
            "refusal": {"code": code, "message": message, "details": details or {}},
        }, authority="deterministic-policy-gate", llm_role="none")

    def _retrieve(self, question: str, facets: dict | None = None) -> tuple[dict, list[dict], list[dict], dict]:
        facets = facets or extract_all_facets(question)
        with self._retrieval_lock:
            doc_hits = retrieve_decomposed(
                question,
                per_pair=1,
                cap=40,
                backfill_unique_preferred_scope=True,
                backfill_unique_scope_neutral=True,
                backfill_explicit_opening_year=True,
                backfill_financial_formula_year=True,
                backfill_growth_scan_year=True,
                backfill_comparative_selection_year=True,
                backfill_comparative_series_year=True,
                retain_ambiguous_series_reports=True,
                implicit_ownership_scope_minimum_ratio=1.2,
                retain_count_scope_fallback=True,
                expand_catalog_universe=True,
                universe_cap=200,
            )
            report_ids = _unique([_report_id(hit["table_ref"]) for hit in doc_hits])
            pair_count = max(1, len(facets["tickers"]) * len(facets["years"]))
            table_limit = min(self.policy.max_tables, max(self.policy.base_tables, pair_count * 4))
        catalog = self._load_catalog()
        candidate_limit = max(table_limit, self.policy.table_rerank_pool)
        with self._retrieval_lock:
            raw_tables = tables_in_reports(question, report_ids, n=candidate_limit)
        try:
            raw_tables = rerank_tables(
                question,
                raw_tables,
                catalog,
                self.root / "build" / "tables",
                label_weight=self.policy.table_label_weight,
            )[:table_limit]
        except Exception:
            # Retrieval remains available if a malformed OCR CSV defeats the
            # optional reranker. The original BM25 order is the safe fallback.
            raw_tables = raw_tables[:table_limit]
        structural_trace = {"enabled": False, "rescues": [], "rejections": []}
        if self._structural_sidecar is not None:
            try:
                with self._retrieval_lock:
                    structural_raw = self._structural_sidecar.tables_in_reports(
                        question,
                        report_ids,
                        n=candidate_limit,
                        pool=2000,
                    )
                structural_ranked = rerank_tables(
                    question,
                    structural_raw,
                    self._structural_sidecar.catalog,
                    self.root / "build" / "tables",
                    label_weight=self.policy.table_label_weight,
                )
                raw_tables, structural_trace = guarded_structural_rescue(
                    question,
                    raw_tables,
                    structural_ranked,
                    self._structural_sidecar.catalog,
                    limit=table_limit,
                )
                structural_trace = {"enabled": True, **structural_trace}
            except Exception as error:
                # The sidecar is an optional rescue channel. Any malformed or
                # stale sidecar fails closed to the untouched baseline order.
                structural_trace = {
                    "enabled": True,
                    "rescues": [],
                    "rejections": [],
                    "error": f"{type(error).__name__}: {str(error)[:240]}",
                }
        tables: list[dict] = []
        for hit in raw_tables:
            row = catalog.get(hit["table_ref"])
            if not row:
                continue
            merged = {**row, "score": float(hit.get("score", 0.0))}
            if hit.get("retrieval_label_text"):
                merged["search_text"] = (
                    str(row.get("search_text", ""))
                    + ". Nhãn dòng đã xác minh từ CSV: "
                    + str(hit["retrieval_label_text"])
                )
            for field in ("bm25_score", "label_match_score", "rerank_score"):
                if field in hit:
                    merged[field] = float(hit[field])
            merged["csv_path"] = str((self.root / "build" / "tables" / row["csv_path"]).resolve())
            tables.append(merged)

        requested_pairs = {(ticker, year) for ticker in facets["tickers"] for year in facets["years"]}
        found_pairs = {(hit.get("ticker"), hit.get("year")) for hit in doc_hits}
        pair_coverage = (
            len(requested_pairs & found_pairs) / len(requested_pairs) if requested_pairs else 0.0
        )
        anchors = _anchor_coverage(question, facets, tables[: min(8, len(tables))])
        facet_score = (float(bool(facets["tickers"])) + float(bool(facets["years"]))) / 2
        positive_score = float(bool(tables) and max(t.get("score", 0.0) for t in tables) > 0)
        confidence = 0.25 * facet_score + 0.40 * pair_coverage + 0.30 * anchors + 0.05 * positive_score
        checks = {
            "requested_pairs": sorted([list(pair) for pair in requested_pairs]),
            "found_pairs": sorted([list(pair) for pair in (requested_pairs & found_pairs)]),
            "pair_coverage": round(pair_coverage, 4),
            "anchor_coverage": round(anchors, 4),
            "confidence": round(confidence, 4),
            "threshold": self.policy.min_grounding_confidence,
            "structural_rescue": structural_trace,
        }
        return facets, doc_hits, tables, checks

    def _citations(self, evidence: list[dict]) -> list[dict]:
        catalog = self._load_catalog()
        names = self._load_company_names()
        citations = []
        for item in evidence:
            row = catalog.get(item["table_ref"])
            if not row:
                continue
            labels = str(row.get("search_text", "")).split("Chỉ tiêu:", 1)[-1].strip()
            citations.append(
                {
                    "table_ref": row["table_ref"],
                    "document": row["report_id"],
                    "company": names.get(row["ticker"], row["ticker"]),
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "scope": row["scope"],
                    "page": row.get("page"),
                    "table_line": row.get("line"),
                    "section": row.get("section_title") or labels[:240],
                    "verified": True,
                }
            )
        return citations

    def _compiled_financial_answer(
        self,
        question: str,
        facets: dict,
        trace_id: str,
        started: float,
    ) -> dict | None:
        """Answer a constrained standard metric without depending on model uptime.

        The compiler emits code from verified source coordinates. The code is
        executed in the sandbox and must agree with the independently parsed
        normalized-cube value before an answer or citation is returned.
        """
        compiled = self._deterministic_compiler.compile(question, facets)
        if compiled is None:
            return None
        first = run_pandas_code(
            compiled.pandas_query,
            compiled.csv_paths,
            timeout=self.policy.execution_timeout,
        )
        first_number = coerce_number(first.get("result")) if first.get("ok") else None
        expected = coerce_number(compiled.answer)
        if first_number is None or expected is None:
            return None
        if not all(math.isfinite(value) for value in (first_number, expected)):
            return None

        def agrees(left: float, right: float) -> bool:
            return abs(left - right) <= 1e-6 + 1e-9 * max(abs(left), abs(right), 1.0)

        if not agrees(first_number, expected):
            return None
        citations = self._citations([{"table_ref": ref} for ref in compiled.table_refs])
        if not citations or len(citations) != len(compiled.table_refs):
            return None
        labels_by_table: dict[str, list[str]] = {}
        for cell in compiled.source_cells:
            labels_by_table.setdefault(str(cell["table_ref"]), []).append(str(cell["label"]))
        for citation in citations:
            labels = _unique(labels_by_table.get(str(citation["table_ref"]), []))
            if labels:
                citation["section"] = " · ".join(labels)
        pairs = [
            list(pair)
            for pair in dict.fromkeys(
                (str(cell["ticker"]), str(cell["year"]))
                for cell in compiled.source_cells
            )
        ]
        compiler_evidence = []
        for variable, source_path in compiled.csv_paths.items():
            path = Path(source_path).resolve()
            try:
                relative = str(path.relative_to(self.root)).replace("\\", "/")
            except ValueError:
                relative = str(path)
            compiler_evidence.append({"variable": variable, "source_path": relative})
        response = {
            "trace_id": trace_id,
            "status": "answered",
            "grounded": True,
            "answer": first_number,
            "unit": compiled.unit,
            "pandas_query": compiled.pandas_query,
            "citations": citations,
            "retrieval": {
                "facets": facets,
                "checks": {
                    "requested_pairs": pairs,
                    "found_pairs": pairs,
                    "pair_coverage": 1.0,
                    "anchor_coverage": 1.0,
                    "confidence": 1.0,
                    "threshold": self.policy.min_grounding_confidence,
                },
                "documents": _unique([_report_id(ref) for ref in compiled.table_refs]),
                "tables": compiled.table_refs,
            },
            "verification": {
                "citation_bound": True,
                "replay_match": True,
                "registry_match": False,
                "registry_exact": False,
                "mode": "deterministic_compiler",
                "compiler_metric": compiled.metric,
                "source_cells": compiled.source_cells,
                "vote_ratio": 1.0,
            },
            "submission_export": {
                "source_artifact": ".",
                "answer": first_number,
                "relevant_docs": _unique([_report_id(ref) for ref in compiled.table_refs]),
                "relevant_tables": list(compiled.table_refs),
                "evidence": compiler_evidence,
                "pandas_query": compiled.pandas_query,
            },
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        response["business_validation"] = validate_financial_result(
            question, first_number, compiled.unit
        )
        return annotate_read_response(
            response,
            authority="typed-compiler-plus-independent-replay",
            llm_role="none",
        )

    def health(self) -> dict:
        base_url = os.getenv("KINGPRO_LLM_BASE_URL", "")
        configured = bool(base_url and os.getenv("KINGPRO_LLM_MODEL"))
        model = os.getenv("KINGPRO_LLM_MODEL", "")
        allowed = self._allowed_models()
        endpoint = endpoint_compliance(base_url)
        model_attested = os.getenv("KINGPRO_LLM_ATTESTED", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        paraphrase_entries = len(self._load_replay_paraphrase_index())
        replay_entries = len(self._load_replay_registry())
        catalog_ready = (self.root / "build" / "catalog.jsonl").is_file()
        bm25_ready = (self.root / "build" / "bm25").is_dir()
        replay_ready = self.replay_submission.is_file()
        dynamic_generation_available = bool(
            configured
            and model
            and model in allowed
            and endpoint["allowed"]
            and model_attested
        )
        demo_readiness = self._demo_readiness_summary(
            stage_components_ready=bool(
                catalog_ready
                and bm25_ready
                and replay_ready
                and self._deterministic_compiler.available
            ),
            dynamic_generation_available=dynamic_generation_available,
        )
        audit_integrity = get_audit_ledger().verify()
        isolation = isolation_health()
        lifecycle_path = os.getenv(
            "KINGPRO_SOURCE_LIFECYCLE_REPORT",
            str(self.root / "build" / "source_lifecycle" / "latest.json"),
        )
        return {
            "status": "ok",
            "catalog": catalog_ready,
            "bm25_index": bm25_ready,
            "llm_configured": configured,
            "model": model,
            "model_allowed": bool(model and model in allowed),
            "allowed_models": sorted(allowed),
            "endpoint_allowed": endpoint["allowed"],
            "endpoint_host": endpoint["host"],
            "endpoint_provider": endpoint["provider"],
            "endpoint_secure_transport": endpoint["secure_transport"],
            "model_attested": model_attested,
            "dynamic_generation_available": dynamic_generation_available,
            "governance": {
                "authority_policy": "code-decides-llm-proposes-language-or-bounded-plan",
                "read_write_policy": "human-approval-before-side-effect",
                "llm_gateway_audit": audit_integrity,
                "source_lifecycle": read_lifecycle_status(lifecycle_path),
                "tenant_data_encryption": encryption_health(),
                "os_code_isolation": isolation,
            },
            "demo_readiness": demo_readiness,
            "deterministic_compiler": self._deterministic_compiler.available,
            "compiler_metrics": self._deterministic_compiler.supported_metrics(),
            "table_reranker": {
                "enabled": bool(
                    self.policy.table_rerank_pool > 0
                    and self.policy.table_label_weight > 0
                ),
                "candidate_pool": self.policy.table_rerank_pool,
                "label_weight": self.policy.table_label_weight,
                "source": "extracted_csv_label_column",
                "fail_safe": "preserve_bm25_order",
            },
            "document_retrieval": {
                "policy_version": "v22",
                "per_pair": 1,
                "cap": 40,
                "unique_preferred_scope_backfill": True,
                "unique_scope_neutral_backfill": True,
                "explicit_opening_year_backfill": True,
                "financial_formula_year_backfill": True,
                "period_growth_scan_year_backfill": True,
                "implicit_comparative_selection_year_backfill": True,
                "bounded_multi_entity_comparative_series_year_backfill": True,
                "ambiguous_long_series_max_reports_per_facet": 2,
                "implicit_ownership_dual_scope_minimum_ratio": 1.2,
                "multi_entity_count_scope_fallback": True,
                "catalog_universe_scan": True,
                "catalog_universe_cap": 200,
                "broad_sparse_series_backfill": False,
                "omitted_year_guessing": False,
                "mixed_scope_year_binding": True,
                "shadowed_alias_suppression": True,
                "source_bound_regression": {
                    "questions": 1012,
                    "macro_precision": 0.975283,
                    "macro_recall": 0.997908,
                    "macro_f2": 0.990582,
                    "missed_questions": 4,
                    "promotion_gate": "PASS",
                    "claim": "local_source_bound_not_btc_hidden_gold",
                },
                "source": "question_semantics_and_catalog_ticker_year_scope_only",
                "ambiguity_policy": "bounded_dual_filing_only_for_multi_stage_long_series",
            },
            "replay_registry": replay_ready,
            "replay_entries": replay_entries,
            "replay_artifact": self.replay_submission.parent.name,
            "cell_lineage": {
                **self._cell_lineage.coverage(),
                "policy": (
                    "fail_closed_manifest_or_hash_bound_counterfactual_"
                    "dependency_with_coordinate_raw_match"
                ),
            },
            "public_champion": PUBLIC_CHAMPION,
            "rollback_fallback": ROLLBACK_FALLBACK,
            "measured_rollback_fallback": MEASURED_ROLLBACK_FALLBACK,
            "local_successor": LOCAL_SUCCESSOR,
            "audited_candidate": AUDITED_CANDIDATE,
            "source_clean_fallback": SOURCE_CLEAN_FALLBACK,
            "paraphrase_replay": bool(paraphrase_entries),
            "paraphrase_entries": paraphrase_entries,
            # Exact replay remains available for every verified row. Rows
            # without explicit entity/year facets are deliberately excluded
            # from fuzzy replay so a paraphrase cannot invent missing scope.
            "paraphrase_exact_only_entries": max(0, replay_entries - paraphrase_entries),
            "paraphrase_policy": {
                "min_score": self.policy.min_registry_paraphrase_score,
                "min_candidate_recall": self.policy.min_registry_candidate_recall,
                "min_query_precision": self.policy.min_registry_query_precision,
                "min_margin": self.policy.min_registry_paraphrase_margin,
                "min_shared_terms": self.policy.min_registry_shared_terms,
                "exact_facets": ["tickers", "years", "scope", "unit", "intent"],
            },
        }

    def _demo_readiness_summary(
        self,
        *,
        stage_components_ready: bool,
        dynamic_generation_available: bool,
    ) -> dict:
        """Expose a sanitized, fail-closed summary of the latest stage audit.

        The UI must never infer human eligibility or a live model identity from
        configuration alone.  The detailed report stays on disk; health only
        exposes boolean gates, counts and non-sensitive evidence key names.
        """
        configured_path = os.getenv(
            "KINGPRO_DEMO_READINESS_REPORT",
            "build/demo_compliance/demo_readiness_hanoi_v297_v52.json",
        )
        report_path = Path(configured_path)
        if not report_path.is_absolute():
            report_path = self.root / report_path
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

        technical = payload.get("technical") if isinstance(payload, dict) else {}
        if not isinstance(technical, dict):
            technical = {}
        checks = technical.get("technical_checks")
        if not isinstance(checks, dict):
            checks = {}
        technical_total = len(checks)
        technical_passed = sum(value is True for value in checks.values())

        smoke = payload.get("stage_smoke") if isinstance(payload, dict) else {}
        if not isinstance(smoke, dict):
            smoke = {}
        smoke_ok = smoke.get("ok") is True

        manual = payload.get("manual_evidence") if isinstance(payload, dict) else {}
        if not isinstance(manual, dict):
            manual = {}
        entries = manual.get("entries")
        if not isinstance(entries, dict):
            entries = {}
        confirmed_keys = []
        pending_keys = []
        for key in _DEMO_EVIDENCE_KEYS:
            entry = entries.get(key)
            confirmed = bool(
                isinstance(entry, dict)
                and entry.get("confirmed") is True
                and str(entry.get("evidence_path") or "").strip()
            )
            (confirmed_keys if confirmed else pending_keys).append(key)

        runtime = payload.get("runtime") if isinstance(payload, dict) else {}
        if not isinstance(runtime, dict):
            runtime = {}
        technical_ok = bool(
            payload
            and technical.get("technical_pass") is True
            and technical_total > 0
            and technical_passed == technical_total
        )
        stage_safe_now = bool(
            payload.get("stage_safe_local") is True
            and stage_components_ready
            and technical_ok
            and smoke_ok
        )
        runtime_operational_reported = runtime.get("operational_ready") is True
        manual_complete = not pending_keys
        full_demo_ready_now = bool(
            stage_safe_now
            and dynamic_generation_available
            and runtime_operational_reported
            and manual_complete
        )
        return {
            "report_present": bool(payload),
            "report_name": report_path.name if payload else "",
            "stage_safe_now": stage_safe_now,
            "full_demo_ready_now": full_demo_ready_now,
            "technical_passed": technical_passed,
            "technical_total": technical_total,
            "smoke_passed": int(smoke.get("passed") or 0),
            "smoke_total": int(smoke.get("total") or 0),
            "smoke_elapsed_ms": int(smoke.get("total_elapsed_ms") or 0),
            "runtime_operational_reported": runtime_operational_reported,
            "manual_confirmed": len(confirmed_keys),
            "manual_total": len(_DEMO_EVIDENCE_KEYS),
            "manual_complete": manual_complete,
            "manual_confirmed_keys": confirmed_keys,
            "manual_pending_keys": pending_keys,
            "claim_limit": (
                "Stage-safe proves the audited replay/compiler path only. "
                "Full Demo-ready additionally requires an operational attested "
                "model and all team-supplied evidence."
            ),
        }

    def catalog_summary(self) -> dict:
        """Return a compact, UI-safe overview of the local financial corpus."""
        if self._catalog_summary is None:
            catalog = self._load_catalog()
            names = self._load_company_names()
            with self._cache_lock:
                if self._catalog_summary is None:
                    reports: set[str] = set()
                    years: set[str] = set()
                    scope_counts: dict[str, int] = {}
                    companies: dict[str, dict] = {}
                    for row in catalog.values():
                        ticker = str(row.get("ticker", "")).strip()
                        year = str(row.get("year", "")).strip()
                        scope = str(row.get("scope", "không xác định")).strip()
                        report_id = str(row.get("report_id", "")).strip()
                        if report_id:
                            reports.add(report_id)
                        if year:
                            years.add(year)
                        scope_counts[scope] = scope_counts.get(scope, 0) + 1
                        if not ticker:
                            continue
                        company = companies.setdefault(
                            ticker,
                            {
                                "ticker": ticker,
                                "company": names.get(ticker, ticker),
                                "years": set(),
                                "reports": set(),
                                "tables": 0,
                                "scopes": set(),
                            },
                        )
                        if year:
                            company["years"].add(year)
                        if report_id:
                            company["reports"].add(report_id)
                        company["scopes"].add(scope)
                        company["tables"] += 1
                    rows = []
                    for company in companies.values():
                        company_years = sorted(company["years"])
                        rows.append(
                            {
                                "ticker": company["ticker"],
                                "company": company["company"],
                                "year_from": company_years[0] if company_years else "",
                                "year_to": company_years[-1] if company_years else "",
                                "year_count": len(company_years),
                                "report_count": len(company["reports"]),
                                "table_count": company["tables"],
                                "scopes": sorted(company["scopes"]),
                            }
                        )
                    rows.sort(key=lambda item: item["ticker"])
                    self._catalog_summary = {
                        "status": "ok",
                        "company_count": len(rows),
                        "report_count": len(reports),
                        "table_count": len(catalog),
                        "years": sorted(years),
                        "scope_counts": scope_counts,
                        "companies": rows,
                    }
        return self._catalog_summary

    @staticmethod
    def _allowed_models() -> set[str]:
        configured = os.getenv(
            "KINGPRO_ALLOWED_MODELS",
            "Qwen/Qwen2.5-Coder-14B-Instruct,Qwen/Qwen3-14B",
        )
        return {item.strip() for item in configured.split(",") if item.strip()}

    def ask(
        self,
        question: str,
        progress_fn: Callable[[dict], None] | None = None,
    ) -> dict:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        trace: list[dict] = []

        def stage_started(stage: str, title: str, detail: str, metadata: dict | None = None) -> None:
            _notify_progress(
                progress_fn,
                {
                    "stage": stage,
                    "title": title,
                    "status": "started",
                    "elapsed_ms": 0,
                    "detail": detail,
                    "metadata": metadata or {},
                },
            )

        def stage_finished(
            stage: str,
            title: str,
            stage_clock: float,
            detail: str,
            metadata: dict | None = None,
        ) -> dict:
            payload = _trace_stage(
                stage,
                title,
                stage_clock,
                detail=detail,
                metadata=metadata,
            )
            trace.append(payload)
            _notify_progress(progress_fn, payload)
            return payload

        def refuse(code: str, message: str, *, details: dict | None = None) -> dict:
            payload = _trace_stage(
                "refusal",
                "Fail-closed refusal",
                started,
                detail=message,
                metadata={"code": code, **(details or {})},
                status="error",
            )
            trace.append(payload)
            _notify_progress(progress_fn, payload)
            response = self._refusal(trace_id, code, message, details=details)
            response["trace"] = trace
            response["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            return response

        question = " ".join(str(question or "").split())
        if len(question) < self.policy.min_question_chars:
            return refuse("question_too_short", "Câu hỏi chưa đủ thông tin để truy hồi.")
        if len(question) > self.policy.max_question_chars:
            return refuse("question_too_long", "Câu hỏi vượt giới hạn an toàn.")
        unsafe_topic = _unsafe_request_topic(question)
        if unsafe_topic is not None:
            return refuse(
                                "unsafe_instruction",
                "Yêu cầu cố thay đổi chỉ dẫn hoặc truy xuất dữ liệu hệ thống nên đã bị chặn trước khi chạy model.",
                details={"topic": unsafe_topic, "blocked_before_model": True},
            )
        registry_started = time.perf_counter()
        stage_started(
            "verified_registry_replay",
            "Verified registry replay",
            "Đang tìm chương trình đã kiểm toán và chạy lại Pandas trên CSV nguồn.",
        )
        registry_answer = self._verified_registry_replay(question, trace_id, started)
        if registry_answer is not None:
            registry_answer["trace"] = [
                stage_finished(
                    "verified_registry_replay",
                    "Verified registry replay",
                    registry_started,
                    "Khớp câu hỏi với chương trình đã kiểm toán, chạy lại Pandas và khóa citation.",
                    {
                        "mode": registry_answer.get("verification", {}).get("mode"),
                        "table_count": len(registry_answer.get("retrieval", {}).get("tables", [])),
                        "citation_count": len(registry_answer.get("citations", [])),
                    },
                )
            ]
            return registry_answer
        facets = extract_all_facets(question)
        # A deliberately narrow deterministic grammar can safely answer a
        # broad all-company query by scanning the normalized statement cube.
        # Give it a chance before the ordinary API guard rejects questions
        # without an explicit ticker.  Unknown broad questions still return
        # ``None`` here and continue to the fail-closed missing-company path.
        compiler_started = time.perf_counter()
        stage_started(
            "deterministic_compiler",
            "Deterministic financial compiler",
            "Đang thử typed formula trên semantic statement cube.",
        )
        compiled_answer = self._compiled_financial_answer(
            question, facets, trace_id, started
        )
        if compiled_answer is not None:
            compiled_answer["trace"] = [
                stage_finished(
                    "deterministic_compiler",
                    "Deterministic financial compiler",
                    compiler_started,
                    "Lập typed formula từ tọa độ nguồn, chạy Pandas và đối chiếu độc lập với statement cube.",
                    {
                        "metric": compiled_answer.get("verification", {}).get("compiler_metric"),
                        "source_cell_count": len(compiled_answer.get("verification", {}).get("source_cells", [])),
                        "citation_count": len(compiled_answer.get("citations", [])),
                    },
                )
            ]
            return compiled_answer
        if not facets["tickers"]:
            return refuse(
                "missing_company", "Anh hãy nêu tên công ty hoặc mã chứng khoán cần tra cứu.",
                details={"facets": facets},
            )
        if not facets["years"]:
            return refuse(
                "missing_year", "Anh hãy nêu năm hoặc giai đoạn tài chính cần tra cứu.",
                details={"facets": facets},
            )
        outside_topic = _out_of_corpus_topic(question)
        if outside_topic is not None:
            return refuse(
                                "outside_financial_statements",
                "Câu hỏi cần nguồn ngoài kho báo cáo tài chính nên hệ thống không suy đoán.",
                details={
                    "topic": outside_topic,
                    "available_scope": "BCTC doanh nghiệp niêm yết do BTC cung cấp",
                },
            )
        pair_count = len(facets["tickers"]) * len(facets["years"])
        if pair_count > self.policy.max_entity_year_pairs:
            return refuse(
                "query_too_broad", "Phạm vi quá rộng; hãy chia nhỏ số công ty hoặc giai đoạn.",
                details={"pairs": pair_count, "limit": self.policy.max_entity_year_pairs},
            )
        paraphrase_started = time.perf_counter()
        stage_started(
            "guarded_paraphrase_replay",
            "Guarded paraphrase replay",
            "Đang kiểm tra facet, overlap và margin với registry đã kiểm toán.",
        )
        registry_paraphrase = self._verified_registry_paraphrase(
            question, facets, trace_id, started
        )
        if registry_paraphrase is not None:
            match = registry_paraphrase.get("verification", {}).get("paraphrase_match") or {}
            registry_paraphrase["trace"] = [
                stage_finished(
                    "guarded_paraphrase_replay",
                    "Guarded paraphrase replay",
                    paraphrase_started,
                    "Paraphrase vượt các cổng facet, overlap và margin; chương trình nguồn được chạy lại trước khi trả lời.",
                    {
                        "score": match.get("score"),
                        "margin": match.get("margin"),
                        "table_count": len(registry_paraphrase.get("retrieval", {}).get("tables", [])),
                    },
                )
            ]
            return registry_paraphrase
        if self.enforce_model_policy:
            base_url = os.getenv("KINGPRO_LLM_BASE_URL", "")
            model = os.getenv("KINGPRO_LLM_MODEL", "")
            if not base_url or not model:
                return refuse(
                    "llm_not_configured", "Endpoint model mở chưa được cấu hình cho phiên demo.",
                )
            if model not in self._allowed_models():
                return refuse(
                    "model_not_allowed",
                    "Model runtime không nằm trong allowlist đã duyệt theo giới hạn tham số của BTC.",
                    details={"model": model, "allowed_models": sorted(self._allowed_models())},
                )
            endpoint = endpoint_compliance(base_url)
            if not endpoint["allowed"]:
                return refuse(
                                        "endpoint_not_allowed",
                    "Endpoint model khong nam trong allowlist host/transport da duyet.",
                    details={
                        "host": endpoint["host"],
                        "provider": endpoint["provider"],
                        "secure_transport": endpoint["secure_transport"],
                    },
                )
            model_attested = os.getenv("KINGPRO_LLM_ATTESTED", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
            if not model_attested:
                return refuse(
                                        "model_not_attested",
                    "Sinh truy vấn tự do đang khóa vì deployment model chưa được operator xác nhận cho phiên demo.",
                    details={
                        "registry_replay_available": self.replay_submission.is_file(),
                        "action": "Verify checkpoint/revision and set KINGPRO_LLM_ATTESTED=true.",
                    },
                )
        retrieval_started = time.perf_counter()
        stage_started(
            "table_retrieval",
            "Table retrieval",
            "Đang khóa entity/year/scope, lấy báo cáo và rerank bảng nguồn.",
        )
        try:
            facets, doc_hits, tables, checks = self._retrieve(question, facets)
        except Exception as exc:
            return refuse(
                "retrieval_error", "Không thể kiểm chứng nguồn dữ liệu lúc này.",
                details={"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
            )
        stage_finished(
            "table_retrieval",
            "Table retrieval",
            retrieval_started,
            "Khóa entity/year/scope, lấy báo cáo rồi rerank bảng bằng nhãn dòng vật lý.",
            {
                "document_count": len(doc_hits),
                "table_count": len(tables),
                "confidence": checks.get("confidence"),
                "pair_coverage": checks.get("pair_coverage"),
            },
        )
        if not doc_hits or not tables:
            return refuse(
                "no_source", "Kho BCTC chưa có bảng đủ liên quan để trả lời câu này.",
                details={"facets": facets},
            )
        if checks["pair_coverage"] < 1.0:
            return refuse(
                "incomplete_coverage", "Chưa tìm đủ báo cáo cho mọi cặp công ty–năm được hỏi.",
                details=checks,
            )
        if checks["confidence"] < self.policy.min_grounding_confidence:
            return refuse(
                "weak_retrieval", "Các bảng tìm được chưa bám đủ sát nội dung câu hỏi nên hệ thống không đoán.",
                details=checks,
            )

        generation_started = time.perf_counter()
        isolation = isolation_health()
        if not isolation["dynamic_execution_allowed"]:
            return refuse(
                "os_sandbox_unavailable",
                "Sinh và chạy code động đang khóa vì sandbox cấp hệ điều hành chưa sẵn sàng.",
                details=isolation,
            )
        stage_started(
            "pandas_generation_execution",
            "Pandas generation & sandbox execution",
            "Đang sinh chương trình bằng model mở và chạy trong sandbox.",
        )
        try:
            with llm_audit_context(
                trace_id=trace_id,
                purpose="bounded-pandas-program-proposal",
                authority="deterministic-replay-verifier",
            ):
                result = run_program(
                    question,
                    tables,
                    self.llm_fn,
                    max_fix=3,
                    n_vote=self.policy.vote_count,
                    timeout=self.policy.execution_timeout,
                )
        except Exception as exc:
            return refuse(
                "generation_error", "Model sinh truy vấn hiện không sẵn sàng; hệ thống không đoán đáp án.",
                details={"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
            )
        if not result.get("ok"):
            return refuse(
                "execution_failed", "Không tạo được phép tính Pandas có thể kiểm chứng.",
                details={"error": result.get("error", ""), "retrieval": checks},
            )
        stage_finished(
            "pandas_generation_execution",
            "Pandas generation & sandbox execution",
            generation_started,
            "Model mở đề xuất chương trình; sandbox chạy và nhóm kết quả theo đồng thuận số học.",
            {
                "attempts": result.get("attempts"),
                "agree": result.get("agree"),
                "evidence_count": len(result.get("evidence") or []),
            },
        )

        evidence = result.get("evidence") or []
        selected_refs = {table["table_ref"] for table in tables}
        evidence_refs = {item.get("table_ref") for item in evidence}
        if not evidence_refs or not evidence_refs <= selected_refs:
            return refuse(
                "citation_mismatch", "Kết quả không gắn được với đúng bảng đã truy hồi.",
            )
        vote_ratio = result.get("agree", 1) / max(1, result.get("attempts", 1))
        if vote_ratio < self.policy.min_vote_ratio:
            return refuse(
                "models_disagree", "Các lần sinh phép tính chưa đạt đồng thuận tối thiểu.",
                details={"vote_ratio": vote_ratio, "threshold": self.policy.min_vote_ratio},
            )

        verification_started = time.perf_counter()
        stage_started(
            "deterministic_verification",
            "Deterministic replay & citation binding",
            "Đang chạy lại code và khóa citation vào DataFrame đã sử dụng.",
        )
        csv_paths = {f"df{i + 1}": table["csv_path"] for i, table in enumerate(tables)}
        replay = run_pandas_code(
            result["pandas_query"], csv_paths, timeout=self.policy.execution_timeout
        )
        replay_number = coerce_number(replay.get("result")) if replay.get("ok") else None
        answer = coerce_number(result.get("answer"))
        if answer is None or replay_number is None or not math.isfinite(answer) or not math.isfinite(replay_number):
            return refuse("non_numeric_result", "Kết quả không phải một số tài chính hữu hạn.")
        if abs(answer - replay_number) > 1e-6 + 1e-9 * max(abs(answer), abs(replay_number), 1.0):
            return refuse(
                "non_deterministic", "Phép tính chạy lại không tái lập cùng kết quả.",
                details={"first": answer, "replay": replay_number},
            )

        citations = self._citations(evidence)
        if not citations:
            return refuse("citation_missing", "Không dựng được trích dẫn kiểm chứng.")
        unit_name, _unit_multiplier = requested_unit(question)
        business_validation = validate_financial_result(question, answer, unit_name)
        if not business_validation["passed"]:
            return refuse(
                "business_validation_failed",
                "Kết quả vi phạm miền giá trị tất định của loại chỉ tiêu được hỏi.",
                details=business_validation,
            )
        stage_finished(
            "deterministic_verification",
            "Deterministic replay & citation binding",
            verification_started,
            "Chạy lại cùng code trên CSV nguồn, so số với lần đầu và chỉ giữ citation thực sự được chương trình dùng.",
            {
                "replay_match": True,
                "citation_count": len(citations),
                "vote_ratio": round(vote_ratio, 4),
            },
        )
        generated_export_evidence = []
        for index, table in enumerate(tables, start=1):
            path = Path(str(table["csv_path"])).resolve()
            try:
                relative = str(path.relative_to(self.root)).replace("\\", "/")
            except ValueError:
                relative = str(path)
            generated_export_evidence.append(
                {"variable": f"df{index}", "source_path": relative}
            )
        used_table_refs = [str(item["table_ref"]) for item in evidence]
        response = {
            "trace_id": trace_id,
            "status": "answered",
            "grounded": True,
            "answer": answer,
            "unit": unit_name,
            "pandas_query": result["pandas_query"],
            "citations": citations,
            "retrieval": {
                "facets": facets,
                "checks": checks,
                "documents": _unique([_report_id(hit["table_ref"]) for hit in doc_hits]),
                "tables": [table["table_ref"] for table in tables],
            },
            "verification": {
                "citation_bound": True,
                "replay_match": True,
                "vote_ratio": round(vote_ratio, 4),
            },
            "submission_export": {
                "source_artifact": ".",
                "answer": answer,
                "relevant_docs": _unique([_report_id(ref) for ref in used_table_refs]),
                "relevant_tables": _unique(used_table_refs),
                "evidence": generated_export_evidence,
                "pandas_query": result["pandas_query"],
            },
            "trace": trace,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        response["business_validation"] = business_validation
        return annotate_read_response(
            response,
            authority="sandbox-replay-plus-citation-verifier",
            llm_role="bounded-pandas-program-proposer",
        )
