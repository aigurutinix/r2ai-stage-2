"""Small streaming coordinator for the interactive financial demo."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, AsyncIterator

import pandas as pd

LOGGER = logging.getLogger(__name__)

from docs import load_companies, load_reports, parse_question, required_report_years, retrieve_docs
from .answers import strict_number
from .generated_query import (
    GeneratedQueryError,
    compile_lookup_query,
    execute_generated_query,
    is_direct_lookup_question,
    validate_generated_query,
)
from .retrieval import (
    load_reports as load_table_reports,
    report_tables,
    retrieve_rows,
    table_budget,
)
from .rerank import INVENTORY, table_representation
# The scoring prompt lives under demo/ so this demo changes no module the
# submission methodology itself uses.
from demo.rerank_prompt import rerank_prompt
from .fusion import fuse
from .graph import NOTE_REF, evidence_slots, select_paths, table_paths
from .notes import linked_note_lines
from .notes import normalize as normalize_note_text
from .slots import (
    append_confidence_item_tables,
    load_line_items,
    named_line_items,
    select_slot_tables,
)


try:  # Keep fixture tests usable before optional serving dependencies are installed.
    from pydantic import BaseModel, Field, model_validator
except ImportError:  # pragma: no cover - live deployments install pydantic
    BaseModel = None


if BaseModel is not None:
    class PlanFact(BaseModel):
        id: str = Field(pattern=r"^f\d+$")
        query: str = Field(min_length=3)
        entity: str
        period: int | None = None


    class PlanStep(BaseModel):
        id: str = Field(pattern=r"^s\d+$")
        op: str
        inputs: list[str]
        instruction: str = Field(min_length=3)


    class Plan(BaseModel):
        facts: list[PlanFact] = Field(min_length=1)
        steps: list[PlanStep]

        @model_validator(mode="after")
        def references_are_forward_safe(self) -> "Plan":
            allowed = {fact.id for fact in self.facts}
            for step in self.steps:
                if any(ref not in allowed for ref in step.inputs):
                    raise ValueError(f"{step.id} references an unknown or future input")
                allowed.add(step.id)
            return self
else:
    @dataclass(frozen=True)
    class PlanFact:
        id: str
        query: str
        entity: str
        period: int | None = None

    @dataclass(frozen=True)
    class PlanStep:
        id: str
        op: str
        inputs: list[str]
        instruction: str

    @dataclass(frozen=True)
    class Plan:
        facts: list[PlanFact]
        steps: list[PlanStep]


ALLOWED_OPS = {
    "select", "add", "subtract", "multiply", "divide", "sum", "mean",
    "change", "argmax", "argmin", "compare", "lookup",
}
EVIDENCE_ROW_LIMIT = 40
EVIDENCE_TABLE_ROWS = 200
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\(?\d[\d., ]*\)?%?(?![A-Za-z0-9])")


@dataclass(frozen=True)
class DemoConfig:
    data_root: Path
    mode: str = "live"
    planner_url: str = ""
    rerank_url: str = ""
    embedding_url: str = ""
    dense_root: Path | None = None
    dense_depth: int = 40
    api_key: str = "local"
    planner_model: str = "Qwen/Qwen3.5-9B"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    candidate_depth: int = 50        # export_rerank_pairs.py --depth
    depth_per_report: int = 0        # 0 = the graded build's flat depth; >0 scales with the gate
    line_items_path: Path | None = None
    graph_depth: int = 30
    graph_max_added: int = 3
    note_max_links: int = 0      # the frozen best applied no note links; >0 opts in
    rerank_depth: int = 130      # run.py --rerank-depth
    rerank_concurrency: int = 16  # in-flight scored pairs; matches the server's max-num-seqs
    fusion_weight: float = 0.5       # apply_rerank_scores.py --weight
    hierarchy: bool = True           # build_final_candidates.sh --hierarchy
    family_prior: bool = False       # --table-family-prior is not passed
    plan_queries: bool = False       # the graded build retrieves on the question alone
    max_candidates: int = 200
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "DemoConfig":
        return cls(
            data_root=Path(os.getenv("VIFINQA_DATA_ROOT", "data/raw/vifinqa")),
            mode=os.getenv("DEMO_MODE", "live"),
            planner_url=os.getenv("PLANNER_URL", os.getenv("OPENAI_BASE_URL", "")),
            rerank_url=os.getenv("RERANK_URL", ""),
            embedding_url=os.getenv("EMBEDDING_URL", ""),
            dense_root=Path(os.environ["DENSE_ROOT"]) if os.getenv("DENSE_ROOT") else None,
            dense_depth=int(os.getenv("DENSE_DEPTH", "40")),
            api_key=os.getenv("OPENAI_API_KEY", "local"),
            planner_model=os.getenv("PLANNER_MODEL", cls.planner_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            candidate_depth=int(os.getenv("CANDIDATE_DEPTH", "50")),
            depth_per_report=int(os.getenv("DEPTH_PER_REPORT", "0")),
            hierarchy=os.getenv("HIERARCHY", "1") != "0",
            family_prior=os.getenv("FAMILY_PRIOR", "0") != "0",
            plan_queries=os.getenv("PLAN_QUERIES", "0") != "0",
            line_items_path=Path(os.environ["LINE_ITEMS"]) if os.getenv("LINE_ITEMS") else None,
            max_candidates=int(os.getenv("MAX_CANDIDATES", "200")),
            rerank_depth=int(os.getenv("RERANK_DEPTH", "130")),
            rerank_concurrency=int(os.getenv("RERANK_CONCURRENCY", "16")),
            note_max_links=int(os.getenv("NOTE_MAX_LINKS", "0")),
            timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")),
        )


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _validate_plan(payload: dict, metadata: dict) -> Plan:
    if BaseModel is not None:
        plan = Plan.model_validate(payload)
        facts = plan.facts
        steps = plan.steps
    else:
        facts = [PlanFact(**fact) for fact in payload.get("facts", [])]
        steps = [PlanStep(**step) for step in payload.get("steps", [])]
        if not facts:
            raise ValueError("plan requires at least one fact")
        available = {fact.id for fact in facts}
        for step in steps:
            if step.id in available or step.op not in ALLOWED_OPS:
                raise ValueError("invalid plan step")
            if any(ref not in available for ref in step.inputs):
                raise ValueError(f"{step.id} references an unknown or future input")
            available.add(step.id)
        plan = Plan(facts, steps)
    entities = set(str(x) for x in metadata.get("tickers", ()))
    years = set(int(x) for x in metadata.get("slot_years", metadata.get("years", ())))
    # The question names the company in words, so the planner writes "Tổng Công ty
    # Cảng Hàng không Việt Nam" where the gate resolved "ACV". Both name the same
    # approved filing, and rejecting the wording threw away the whole step list.
    names = {str(x).casefold() for x in metadata.get("entity_names", ())}
    approved = {x.casefold() for x in entities} | names
    for fact in facts:
        if approved and fact.entity.casefold() not in approved:
            raise ValueError(f"fact {fact.id} uses an unapproved entity")
        if fact.period is not None and years and fact.period not in years:
            raise ValueError(f"fact {fact.id} uses an unapproved period")
    if any(step.op not in ALLOWED_OPS for step in steps):
        raise ValueError("unknown operation")
    return plan


def _plan_prompt(question: str, metadata: dict, draft: dict | None = None) -> str:
    task = "Plan" if draft is None else "Correct the draft plan"
    return f"""{task} a Vietnamese financial question using source evidence only.
Question: {question}
Allowed metadata: {json.dumps(metadata, ensure_ascii=False)}
A fact is one value read straight from a filing: one accounting item, one
entity, one period. Keep accounting concepts whole and never split a line item.
A step is one arithmetic move over earlier ids, using op from {sorted(ALLOWED_OPS)}.
Every step references only facts or steps defined before it, and the last step
answers the question. Write each instruction in Vietnamese.
Return JSON only, in this shape:
{{"facts":[{{"id":"f1","query":"Tổng tài sản của X cuối năm 2023","entity":"X","period":2023}},
{{"id":"f2","query":"Tổng tài sản của X cuối năm 2022","entity":"X","period":2022}}],
"steps":[{{"id":"s1","op":"subtract","inputs":["f1","f2"],"instruction":"Lấy f1 trừ f2 để ra mức thay đổi"}},
{{"id":"s2","op":"divide","inputs":["s1","f2"],"instruction":"Chia s1 cho f2 rồi nhân 100 để ra phần trăm"}}]}}
""" + (f"Draft: {json.dumps(draft, ensure_ascii=False)}" if draft else "")


def _mask(value: object) -> str:
    text = str(value)
    return text if re.fullmatch(r"(?:19|20)\d{2}", text.strip()) else NUMBER_RE.sub("<NUM>", text)


def _frame_context(row: dict, frames: dict[str, pd.DataFrame]) -> str:
    blocks = []
    for evidence in row.get("evidence", []):
        variable = evidence["variable"]
        frame = frames[variable]
        lines = [f"TABLE {variable}", "COLUMNS " + " | ".join(f"{i}:{_mask(c)}" for i, c in enumerate(frame.columns))]
        for index, values in frame.head(EVIDENCE_ROW_LIMIT).iterrows():
            lines.append("ROW " + str(index) + " | " + " | ".join(_mask(value) for value in values.tolist()))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _citation(table: dict, frame: pd.DataFrame, row: int, column: int, citation_id: str) -> dict:
    raw = str(frame.iloc[row, column])
    return {
        "id": citation_id,
        "label": f"{table['report_id']} · row {row} · column {column}",
        "report_id": table["report_id"],
        "table_id": table["table_id"],
        "row": row,
        "column": column,
        "raw_value": raw,
        "unit": table.get("unit", ""),
    }


DENSE_DIMS = 1024
DENSE_INSTRUCT = ("Given a Vietnamese financial question, retrieve the financial "
                  "statement table that contains the answer")


def canonical_scope(scope: str) -> str:
    """Match the document gate's treatment of aggregated reports."""
    return "consolidated" if scope == "aggregated" else scope


def same_triple_reports(reports: dict, selected_docs: list[str]) -> set[str]:
    """Reports a dense candidate may come from: the gate's own ticker/year/scope triples."""
    triples = {
        (report.identity.ticker, report.identity.year, canonical_scope(report.identity.scope))
        for report_id in selected_docs
        if (report := reports.get(report_id)) is not None
    }
    return {
        report_id for report_id, report in reports.items()
        if (report.identity.ticker, report.identity.year, canonical_scope(report.identity.scope)) in triples
    }


@lru_cache(maxsize=64)
def report_text(path: str) -> str:
    """One report's OCR text, cached because the hops re-read the same filings."""
    return Path(path).read_text(encoding="utf-8")


def _local_execution_ok() -> bool:
    """Prove the sandbox still validates, compiles, and runs a query on this host."""
    frame = pd.DataFrame([["x", "1.234"]], columns=["label", "2024"])
    try:
        return execute_generated_query(
            "result = float(str(df0.iloc[0, 1]).replace('.', ''))", {"df0": frame}
        ) == 1234.0
    except Exception:
        return False


def _as_assignment(expression: object) -> str:
    """Accept a bare expression and give it the `result =` the validator requires.

    Models routinely answer with the expression alone. That is a formatting
    difference, not a different computation, so it is normalised here rather
    than spent on a repair attempt. Anything already assigning is left alone.
    """
    text = str(expression or "").strip().strip("`").removeprefix("python").strip()
    if not text or "\n" in text or text.startswith("result"):
        return text
    return f"result = {text}"


YEAR_RE = re.compile(r"(?<!\d)20\d{2}(?!\d)")


def _high_arity(question: str, docs: list[str], line_items: list[str]) -> bool:
    """The gate select_slot_tables.py puts in front of the adaptive append.

    A single-filing, single-item, single-year question keeps the slot selection
    untouched; only a question with more than one of something gets extra item
    carriers appended.
    """
    return len(docs) >= 2 or len(line_items) >= 2 or len(YEAR_RE.findall(question)) >= 2


def _salvage_facts(payload: dict | None, metadata: dict) -> list:
    """Keep the facts a rejected plan got right, dropping everything else."""
    facts = []
    for entry in (payload or {}).get("facts", []):
        if not isinstance(entry, dict):
            continue
        try:
            # The gate already decided the entity and the years. A planner guess
            # that contradicts them is wrong by construction, so it is replaced
            # rather than used to reject an otherwise usable fact.
            tickers = [str(t) for t in metadata.get("tickers", ())]
            entity = str(entry.get("entity") or "")
            if tickers and entity not in tickers:
                entity = tickers[0]
            years = {int(y) for y in metadata.get("slot_years", metadata.get("years", ()))}
            period = int(entry["period"]) if str(entry.get("period", "")).strip().isdigit() else None
            if period is not None and years and period not in years:
                period = None
            candidate = {
                "id": f"f{len(facts) + 1}",
                "query": str(entry["query"]),
                "entity": entity,
                "period": period,
            }
            _validate_plan({"facts": [candidate], "steps": []}, metadata)
        except Exception:
            continue
        facts.append(PlanFact(**candidate) if BaseModel is None else PlanFact.model_validate(candidate))
    return facts


def _answer_prompt(
    question: str, plan: Plan, frames: dict[str, pd.DataFrame], context: str
) -> str:
    bounds = "; ".join(
        f"{name}: rows 0-{min(len(frame.index), EVIDENCE_ROW_LIMIT) - 1}, columns 0-{len(frame.columns) - 1}"
        for name, frame in sorted(frames.items())
    )
    return f"""Return JSON only with one executable expression: {{"expression":"result = ..."}}.
Read values only with <df>.iloc[row, column] using the dataframes {sorted(frames)} and the
row/column numbers shown below. Do not invent numbers or use column labels.
Valid coordinates: {bounds}. Any other row or column number is rejected.
Every cell is a string, never a number. A figure is written 1.234.567.890, where
the dots are thousands separators and there is no decimal part, and a negative
one is written in parentheses. So read a cell as
float(str(<df>.iloc[r, c]).replace('.', '').replace('(', '-').replace(')', ''))
and never subtract or divide the cells directly.
Answer in the unit the question asks for: multiply by 1e-09 for tỷ đồng, by
1e-06 for triệu đồng, and for a phần trăm growth use (a / b - 1) * 100.
Example: result = round((float(str(df0.iloc[4, 3]).replace('.', '')) - float(str(df1.iloc[4, 3]).replace('.', ''))) * 1e-09, 2)
Question: {question}
Plan: {json.dumps(_json(plan), ensure_ascii=False)}
Tables:
{context}"""


def _cells_used(expression: str, frames: dict[str, pd.DataFrame]) -> list[tuple[str, int, int]]:
    """Every <df>.iloc[row, column] literal the validated expression reads, in order."""
    cells: list[tuple[str, int, int]] = []
    for node in ast.walk(ast.parse(expression)):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Attribute):
            continue
        target = node.value
        if target.attr != "iloc" or not isinstance(target.value, ast.Name):
            continue
        variable = target.value.id
        index = node.slice
        if variable not in frames or not isinstance(index, ast.Tuple) or len(index.elts) != 2:
            continue
        if not all(isinstance(part, ast.Constant) and isinstance(part.value, int) for part in index.elts):
            continue
        cell = (variable, index.elts[0].value, index.elts[1].value)
        if cell not in cells:
            cells.append(cell)
    return cells


def _apply_unit_scale(
    expression: str, question: str, frames: dict[str, pd.DataFrame]
) -> str:
    """Recompile a single-cell lookup so the table's own unit reaches the asked-for unit."""
    validate_generated_query(expression, set(frames))
    cells = _cells_used(expression, frames)
    for variable, row, column in cells:
        frame = frames[variable]
        if not (0 <= row < len(frame.index) and 0 <= column < len(frame.columns)):
            # Say which coordinate is wrong and what the real range is; the one
            # repair attempt gets this text, and a raw IndexError wastes it.
            raise GeneratedQueryError(
                f"{variable}.iloc[{row}, {column}] does not exist; "
                f"{variable} has rows 0-{len(frame.index) - 1} and columns 0-{len(frame.columns) - 1}"
            )
    if len(cells) != 1 or not is_direct_lookup_question(question):
        return expression
    variable, row, column = cells[0]
    return compile_lookup_query(question, variable, frames[variable], row, column)


def _citations_for(
    expression: str, frames: dict[str, pd.DataFrame], sources: dict[str, dict]
) -> tuple[list[dict], dict[str, dict]]:
    """Cite the cells the executed query actually read - never any other cell."""
    citations, evidence = [], {}
    for position, (variable, row, column) in enumerate(_cells_used(expression, frames), 1):
        frame = frames[variable]
        if not (0 <= row < len(frame.index) and 0 <= column < len(frame.columns)):
            continue
        citation = _citation(sources[variable], frame, row, column, f"c{position}")
        citation["row_label"] = str(frame.iloc[row, 0])
        citation["column_label"] = str(frame.columns[column])
        citations.append(citation)
        # The drawer renders the whole source table and highlights the cited cell,
        # so the presenter can see the number in its own row and period column.
        evidence[citation["id"]] = {
            **citation,
            "headers": [str(name) for name in frame.columns],
            "rows": [[str(value) for value in values] for values in frame.head(EVIDENCE_TABLE_ROWS).values.tolist()],
            "highlight": {"row": row, "column": column},
            "truncated": len(frame.index) > EVIDENCE_TABLE_ROWS,
        }
    return citations, evidence


def _yes_probability(top_logprobs: dict) -> float:
    """The reranker's score: how much likelier "yes" is than "no" at one position."""
    yes = no = None
    for token, logprob in (top_logprobs or {}).items():
        stripped = token.strip().lower()
        if stripped == "yes":
            yes = logprob if yes is None else max(yes, logprob)
        elif stripped == "no":
            no = logprob if no is None else max(no, logprob)
    if yes is None and no is None:
        return 0.0
    # One of the pair can fall outside the returned top-k when the other is
    # overwhelming, which is a confident answer rather than a missing one.
    if yes is None:
        return 0.0
    if no is None:
        return 1.0
    return math.exp(yes) / (math.exp(yes) + math.exp(no))


class DemoPipeline:
    """One-question facade used by the HTML demo and nothing else."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self._evidence: dict[str, dict[str, dict]] = {}
        self._request_tables: dict[str, dict[str, dict]] = {}
        self._companies = None
        self._documents = None
        self._tables = None
        self._report_meta = None
        self._dense = None
        self._last_plan_payload: dict | None = None
        self._line_items: tuple[str, ...] | None = None
        if config.mode != "fixture":
            if not config.planner_url or not config.rerank_url:
                raise RuntimeError("Live mode requires PLANNER_URL and RERANK_URL")
            if config.data_root is None or not config.data_root.exists():
                raise FileNotFoundError(f"Missing VIFINQA_DATA_ROOT: {config.data_root}")

    @classmethod
    def from_env(cls) -> "DemoPipeline":
        return cls(DemoConfig.from_env())

    def _load_data(self) -> None:
        if self._tables is not None:
            return
        root = self.config.data_root
        self._companies = load_companies(root / "code_stock.csv")
        self._documents = load_reports(root / "financial_statements")
        self._tables = load_table_reports(root)
        self._report_meta = {report.identity.report_id: report for report in self._tables}

    async def health(self) -> dict:
        if self.config.mode == "fixture":
            return {"ready": True, "mode": "fixture", "models": {"fixture": True}}
        self._load_data()
        endpoints = await self._probe_endpoints()
        execution_ok = _local_execution_ok()
        return {
            "ready": bool(self._tables) and execution_ok and all(v != "down" for v in endpoints.values()),
            "mode": self.config.mode,
            "models": {
                "planner": self.config.planner_model,
                "embedding": self.config.embedding_model,
                "reranker": self.config.reranker_model,
            },
            "endpoints": endpoints,
            "local_execution": execution_ok,
            "dense": self._dense_coverage(),
            "reports": len(self._documents or {}),
            "tables": len(self._tables or []),
        }

    def _dense_coverage(self) -> dict:
        """How much of the corpus the stored embedding index actually covers."""
        index = self._load_dense()
        if index is None:
            return {"enabled": False}
        _, ids = index
        indexed = {report_id for report_id, _ in ids}
        total = len(self._report_meta or {})
        return {
            "enabled": True,
            "indexed_tables": len(ids),
            "indexed_reports": len(indexed),
            "uncovered_reports": sorted(set(self._report_meta or {}) - indexed)[:5],
            "report_coverage": round(len(indexed & set(self._report_meta or {})) / total, 4) if total else 0.0,
        }

    async def _probe_endpoints(self) -> dict[str, str]:
        """One cheap GET per serving endpoint, distinguishing down from busy.

        A single-threaded model server mid-inference cannot answer a probe, so a
        timeout means "working", not "gone". Reporting both as an outage sends a
        presenter chasing a problem that does not exist.
        """
        import httpx

        probes = {
            "planner": self.config.planner_url.rstrip("/") + "/v1/models",
            "reranker": self.config.rerank_url.rstrip("/") + "/v1/models",
        }
        if self.config.embedding_url:
            probes["embedding"] = self.config.embedding_url.rstrip("/") + "/v1/models"
        results: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in probes.items():
                try:
                    results[name] = "up" if (await client.get(url)).status_code < 500 else "down"
                except httpx.TimeoutException:
                    results[name] = "busy"
                except Exception:
                    results[name] = "down"
        return results

    async def _chat(self, url: str, prompt: str, model: str) -> dict:
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise RuntimeError("Install openai and pydantic for live demo mode") from error
        client = AsyncOpenAI(base_url=url.rstrip("/") + "/v1", api_key=self.config.api_key, timeout=self.config.timeout_seconds)
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("model returned no JSON")
        payload = json.loads(content)
        if isinstance(payload, dict) and "facts" in payload:
            self._last_plan_payload = payload
        return payload

    async def _plan(self, question: str, metadata: dict) -> Plan:
        draft = await self._chat(self.config.planner_url, _plan_prompt(question, metadata), self.config.planner_model)
        try:
            parsed = _validate_plan(draft, metadata)
        except Exception:
            corrected = await self._chat(self.config.planner_url, _plan_prompt(question, metadata, draft), self.config.planner_model)
            parsed = _validate_plan(corrected, metadata)
        verified = await self._chat(self.config.planner_url, _plan_prompt(question, metadata, _json(parsed)), self.config.planner_model)
        return _validate_plan(verified, metadata)

    async def _plan_or_facts(self, question: str, metadata: dict) -> tuple[Plan, bool]:
        """Fall back to a facts-only plan when the model cannot produce valid steps.

        Only `facts` drives retrieval, reranking and slot selection; `steps` is
        narration for the prompt and the trace. Losing the steps is a worse
        answer, losing the facts is no answer, so a malformed step list must not
        take the question down with it.
        """
        try:
            return await self._plan(question, metadata), False
        except Exception as error:
            payload = getattr(error, "payload", None) or self._last_plan_payload
            LOGGER.warning("plan rejected (%s); payload=%r", error, payload)
            facts = _salvage_facts(payload, metadata)
            if not facts:
                raise
            return Plan(facts=facts, steps=[]), True

    def _retrieve(
        self, question: str, plan: Plan, parsed: Any, extra_table_ids: list[str] | None = None
    ) -> tuple[dict, dict]:
        self._load_data()
        docs, decisions = retrieve_docs(parsed, self._documents)
        metadata = {
            "tickers": parsed.tickers,
            "years": parsed.years,
            "slot_years": required_report_years(parsed),
            "scope": parsed.scope,
        }
        if not docs:
            # `report_ids=None` means "search every report", which is a
            # corpus-wide scan (~100s) and never the right answer for a question
            # whose company could not be identified.
            raise ValueError("No filing matches this company, year, and scope")
        depth = self._pool_depth(len(docs))
        queries = list(dict.fromkeys(
            [question, *[fact.query for fact in plan.facts]] if self.config.plan_queries else [question]
        ))
        merged: dict[str, dict] = {}
        for query in queries:
            result = retrieve_rows(query, metadata, self._tables, top_k=depth, report_ids=docs or None, mode="report-coverage",
                                   hierarchy=self.config.hierarchy,
                                   family_prior=self.config.family_prior)
            for rank, table in enumerate(result["tables"], 1):
                item = merged.setdefault(table["table_id"], {
                    **table,
                    "retrieval_rank": rank,
                    "sparse_rank": rank,
                    "text": " ".join((table.get("title", ""), *(
                        " ".join(row) for row in table.get("rows", [])
                    ))),
                })
                item["retrieval_score"] = item.get("retrieval_score", 0.0) + 1 / (60 + rank)
        for rank, table_id in enumerate(extra_table_ids or (), 1):
            if table_id in merged:
                merged[table_id]["retrieval_score"] += 1 / (60 + rank)
                merged[table_id].setdefault("dense_rank", rank)
                continue
            table = self._table_by_id(table_id)
            if table is not None:
                merged[table_id] = {**table, "retrieval_rank": rank, "dense_rank": rank,
                                    "retrieval_score": 1 / (60 + rank)}
        candidates = sorted(merged.values(), key=lambda item: (-item["retrieval_score"], item["table_id"]))[:depth]
        record = {"id": 0, "question": question, "candidates": candidates, "selected_docs": docs}
        return record, {
            "metadata": metadata,
            "docs": docs,
            "decisions": decisions,
            "candidate_depth": depth,
            "candidate_count": len(candidates),
            "table_budget": table_budget(len(docs)),
            "dense_enabled": bool(self.config.embedding_url and self.config.dense_root),
            "dense_added": sum(1 for table_id in (extra_table_ids or ()) if table_id in merged),
        }

    def _pool_depth(self, report_count: int) -> int:
        """Candidates to rerank, scaled to the gate.

        The graded build used a flat `--depth 50`, which is the default here so
        the demo reproduces it. Setting `depth_per_report` opts into scaling the
        pool with the gate instead, bounded by `max_candidates`.
        """
        if self.config.depth_per_report <= 0:
            return self.config.candidate_depth
        scaled = self.config.depth_per_report * max(report_count, 1)
        return min(self.config.max_candidates, max(self.config.candidate_depth, scaled))

    def _table_by_id(self, table_id: str) -> dict | None:
        """Expand a dense-only hit into the same candidate shape the sparse arm returns."""
        report_id, _, _ = table_id.partition("|")
        report = self._report_meta.get(report_id)
        if report is None:
            return None
        for table in report_tables(str(report.path), report.identity):
            if table.table_id != table_id:
                continue
            return {
                "table_id": table.table_id,
                "report_id": table.report_id,
                "page": table.page,
                "start_line": table.start_line,
                "score": 0.0,
                "rows": [list(row) for row in table.rows],
                "title": table.title,
                "periods": list(table.periods),
                "unit": table.unit,
            }
        return None

    def _labels(self) -> tuple[str, ...]:
        """The corpus line-item lexicon the graded selector matched against."""
        if self._line_items is None:
            path = self.config.line_items_path or (self.config.data_root.parent.parent / "derived" / "line_items.json")
            # load_line_items normalises and sorts longest-first, which is what
            # makes named_line_items drop a label that is a substring of one it
            # already kept. Reading the JSON directly loses that ordering and
            # returns redundant items.
            try:
                self._line_items = load_line_items(Path(path))
            except (OSError, ValueError):
                self._line_items = ()
        return self._line_items

    def _report_text(self, report_id: str) -> str | None:
        report = (self._report_meta or {}).get(report_id)
        return None if report is None else report_text(str(report.path))

    def _graph_tables(self, record: dict, docs: list[str], items: list[str]) -> list[str]:
        """The graded graph hop: follow a named row to the table its note points at."""
        if not items or not docs:
            return []
        order = [candidate["table_id"] for candidate in record["candidates"]][: self.config.graph_depth]
        by_id = {candidate["table_id"]: candidate for candidate in record["candidates"]}
        slots = evidence_slots(docs, items)
        covered = {
            slot for slot in slots
            if any(
                table_id.rsplit("|", 1)[0] == slot.report_id
                and f" {slot.line_item} " in f" {normalize_note_text(by_id[table_id].get('text', ''))} "
                for table_id in order
            )
        }
        paths = []
        for table_id in order:
            candidate = by_id[table_id]
            text = self._report_text(candidate["report_id"])
            if text is None:
                continue
            grid = [list(cells) for cells in candidate.get("rows", [])]
            for slot in slots:
                found = table_paths(slot, table_id, grid, text, candidate["start_line"])
                if slot in covered:
                    found = [p for p in found if p.edge.kind == NOTE_REF and p.anchor.table_id in order]
                paths.extend(found)
        added = select_paths(paths, set(order), self.config.graph_max_added)
        return [path.edge.table_id for path in added]

    def _materialize_order(
        self, question: str, parsed: Any, docs: list[str], order: list[str], record: dict,
    ) -> list[str]:
        """Reproduce `run.py --table-top-k ranking` when turning a ranking into tables.

        Packaging is not a pass-through. The frozen best re-retrieves at
        `--rerank-depth`, keeps that deeper set, promotes only ranking tables the
        gate missed that a sidecar can resolve, sorts by ranking position with
        unranked tables last, and cuts to the ranking's own length. A ranking
        table the deep retrieval never returned is dropped, and a deep-retrieval
        table the ranking never mentions takes the vacated slot.
        """
        if not order:
            return order
        top_k = len(order)
        metadata = {"tickers": parsed.tickers, "years": parsed.years,
                    "slot_years": required_report_years(parsed), "scope": parsed.scope}
        deep = retrieve_rows(
            question, metadata, self._tables, top_k=self.config.rerank_depth,
            report_ids=docs or None, mode="report-coverage",
            hierarchy=self.config.hierarchy, family_prior=self.config.family_prior,
        )
        tables = list(deep["tables"])
        held = {table["table_id"] for table in tables}
        known = {candidate["table_id"] for candidate in record["candidates"]}
        for table_id in order[:top_k]:
            if table_id in held or table_id not in known:
                continue
            resolved = self._table_by_id(table_id)      # the sidecar's role here
            if resolved is not None:
                tables.append(resolved)
                held.add(table_id)
        rank_of = {table_id: rank for rank, table_id in enumerate(order)}
        tables.sort(key=lambda table: rank_of.get(table["table_id"], len(rank_of)))
        chosen = tables[:top_k]
        by_id = {candidate["table_id"]: candidate for candidate in record["candidates"]}
        for table in chosen:                            # keep frames buildable
            by_id.setdefault(table["table_id"], table)
        record["candidates"] = list(by_id.values())
        return [table["table_id"] for table in chosen]

    def _note_links(self, record: dict, prefix: list[str], items: list[str]) -> list[str]:
        """The graded note-link hop, followed only from the tables actually submitted."""
        if not items or self.config.note_max_links <= 0:
            return []
        by_id = {candidate["table_id"]: candidate for candidate in record["candidates"]}
        links, seen = [], set(prefix)
        for table_id in prefix:
            if len(links) >= self.config.note_max_links:
                break
            candidate = by_id.get(table_id)
            if candidate is None:
                continue
            text = self._report_text(candidate["report_id"])
            if text is None:
                continue
            grid = [list(cells) for cells in candidate.get("rows", [])]
            for _note, line in linked_note_lines(text, grid, items, candidate["start_line"]):
                linked = f"{candidate['report_id']}|{line}"
                if linked in seen:
                    continue
                seen.add(linked)
                links.append(linked)
                if len(links) >= self.config.note_max_links:
                    break
        return links

    def _load_dense(self) -> tuple[Any, list[list[str]]] | None:
        """Memory-map the stored 4B table index once; None when dense is switched off."""
        if not (self.config.embedding_url and self.config.dense_root):
            return None
        if self._dense is None:
            import numpy as np

            root = self.config.dense_root
            vectors = np.load(root / "table_vectors.npy", mmap_mode="r")
            ids = json.loads((root / "table_ids.json").read_text())
            if vectors.shape != (len(ids), DENSE_DIMS):
                raise RuntimeError(
                    f"dense index is {vectors.shape}, expected ({len(ids)}, {DENSE_DIMS})"
                )
            self._dense = (vectors, ids)
        return self._dense

    async def _embed_query(self, question: str) -> Any:
        """Embed one fresh question the same way the stored corpus index was built."""
        import numpy as np
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=self.config.embedding_url.rstrip("/") + "/v1",
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )
        response = await client.embeddings.create(
            model=self.config.embedding_model,
            input=[f"Instruct: {DENSE_INSTRUCT}\nQuery: {question}"],
        )
        if response.model and self.config.embedding_model.split("/")[-1] not in response.model:
            raise RuntimeError(
                f"embedding endpoint served {response.model}, not {self.config.embedding_model}; "
                "the stored index is only valid for its own model"
            )
        vector = np.asarray(response.data[0].embedding, dtype="float32")
        if vector.shape[0] < DENSE_DIMS:
            raise RuntimeError(f"embedding has {vector.shape[0]} dims, need at least {DENSE_DIMS}")
        # Truncate the Matryoshka prefix first, then renormalise - a prefix of a
        # unit vector is not itself a unit vector.
        vector = vector[:DENSE_DIMS]
        return vector / (float(np.linalg.norm(vector)) or 1.0)

    async def _dense_candidates(self, question: str, docs: list[str]) -> list[str]:
        """Table ids the embedding index ranks highest inside the gate's own triples."""
        index = self._load_dense()
        if index is None or not docs:
            return []
        import numpy as np

        vectors, ids = index
        allowed = same_triple_reports(self._report_meta, docs)
        rows = [position for position, (report_id, _) in enumerate(ids) if report_id in allowed]
        if not rows:
            return []
        query = await self._embed_query(question)
        scores = np.asarray(vectors[rows], dtype="float32") @ query
        best = np.argsort(-scores)[: self.config.dense_depth]
        return [ids[rows[int(position)]][1] for position in best]

    async def _rerank(self, question: str, plan: Plan, record: dict) -> dict[str, float]:
        """Score each candidate the way the frozen run scored it.

        Not vLLM's /rerank route: that applies its own generic instruction and
        offers no way to pass ours, which ranks a filing's own note below
        unrelated tables. The frozen scorer reads the yes and no logits under
        the instruction in rerank_prompt, so this asks for one token with
        logprobs and takes the same ratio.
        """
        import httpx

        scores: dict[str, float] = {}
        documents = [table_representation(
            self._table_object(table), 0, inventory=INVENTORY,
            identity=self._report_meta[table["report_id"]].identity,
        ) for table in record["candidates"]]
        limit = asyncio.Semaphore(self.config.rerank_concurrency)

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async def judge(query: str, document: str) -> float:
                async with limit:
                    response = await client.post(
                        self.config.rerank_url.rstrip("/") + "/v1/completions",
                        json={
                            "model": self.config.reranker_model,
                            "prompt": rerank_prompt(query, document),
                            "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
                        },
                    )
                response.raise_for_status()
                choices = response.json().get("choices") or [{}]
                top = (choices[0].get("logprobs") or {}).get("top_logprobs") or [{}]
                return _yes_probability(top[0])

            for fact in plan.facts:
                values = await asyncio.gather(*(judge(fact.query, document) for document in documents))
                for table, value in zip(record["candidates"], values):
                    table_id = table["table_id"]
                    scores[table_id] = max(scores.get(table_id, 0.0), value)
        return scores

    def _table_object(self, table: dict) -> Any:
        report = self._report_meta.get(table["report_id"])
        if report is not None:
            for parsed in report_tables(str(report.path), report.identity):
                if parsed.table_id == table["table_id"]:
                    return parsed
        from .retrieval import Table
        return Table(
            table["table_id"], table["report_id"], table.get("page"),
            table["start_line"], tuple(tuple(row) for row in table["rows"]),
            table.get("title", ""), tuple(), tuple(),
            tuple(table.get("periods", ())), table.get("unit", ""),
        )

    def _answer_frames(self, row: dict, order: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        """Materialise the selected tables as frames, keeping each frame's source table."""
        candidates = {table["table_id"]: table for table in row["candidates"]}
        frames: dict[str, pd.DataFrame] = {}
        sources: dict[str, dict] = {}
        for table_id in order:
            table = candidates.get(table_id)
            rows = (table or {}).get("rows") or []
            if len(rows) < 2:
                continue
            variable = f"df{len(frames)}"
            frames[variable] = pd.DataFrame(rows[1:], columns=[str(x) for x in rows[0]])
            sources[variable] = table
        return frames, sources

    async def _answer(self, question: str, plan: Plan, row: dict, order: list[str]) -> dict:
        frames, sources = self._answer_frames(row, order)
        if not frames:
            raise ValueError("selected tables have no usable evidence")
        context = _frame_context({"evidence": [{"variable": name} for name in frames]}, frames)
        prompt = _answer_prompt(question, plan, frames, context)
        expression, answer, repaired = await self._solve(prompt, question, frames)
        citations, evidence = _citations_for(expression, frames, sources)
        if not citations:
            raise GeneratedQueryError("query cites no source cell")
        return {
            "answer": str(round(answer, 2)),
            "code": expression,
            "citations": citations,
            "checks": {
                "plan_valid": True,
                "query_valid": True,
                "executed": True,
                "repaired": repaired,
                "cells_cited": len(citations),
            },
            "_evidence": evidence,
        }


    async def _solve(
        self, prompt: str, question: str, frames: dict[str, pd.DataFrame]
    ) -> tuple[str, float, bool]:
        """Ask for an expression, then allow exactly one repair attempt."""
        attempt = prompt
        for repaired in (False, True):
            payload = await self._chat(self.config.planner_url, attempt, self.config.planner_model)
            expression = _as_assignment(payload.get("expression", ""))
            try:
                expression = _apply_unit_scale(expression, question, frames)
                return expression, execute_generated_query(expression, frames), repaired
            except (GeneratedQueryError, TypeError, ValueError, IndexError, KeyError, ZeroDivisionError) as error:
                if repaired:
                    raise GeneratedQueryError(f"query failed after one repair: {error}") from error
                attempt = (
                    f"{prompt}\n\nThe previous attempt {expression!r} failed with: {error}.\n"
                    "Return corrected JSON that reads only cells shown above."
                )
        raise GeneratedQueryError("no attempt produced a result")

    async def run(self, question: str, request_id: str = "request") -> AsyncIterator[dict]:
        started = time.perf_counter()
        if not question.strip():
            yield {"type": "error", "stage": "input", "status": "failed", "message": "Question is empty", "data": {}}
            return
        if self.config.mode == "fixture":
            citation = {"id": "c1", "label": "fixture report · cash · 2024", "report_id": "fixture_2024", "table_id": "fixture_2024|1", "row": 0, "column": 1, "raw_value": "42", "unit": "tỷ đồng", "row_label": "Tiền và tương đương tiền", "column_label": "2024", "headers": ["Chỉ tiêu", "2024"], "rows": [["Tiền và tương đương tiền", "42"]], "highlight": {"row": 0, "column": 1}}
            result = {"answer": "42.00", "code": "result = 42.0", "citations": [citation], "checks": {"plan_valid": True, "query_valid": True, "executed": True}}
            self._evidence[request_id] = {"c1": citation}
            yield {"type": "stage", "stage": "plan", "status": "done", "message": "Fixture plan ready", "data": {"facts": [{"id": "f1", "query": question, "entity": "fixture", "period": 2024}]}}
            yield {"type": "result", "stage": "answer", "status": "done", "message": "Fixture answer", "data": result}
            return
        try:
            self._load_data()
            parsed = parse_question(question, self._companies)
            if not parsed.tickers:
                raise ValueError("Could not identify a unique company")
            metadata = {"tickers": parsed.tickers, "years": parsed.years, "slot_years": required_report_years(parsed), "scope": parsed.scope}
            metadata["entity_names"] = [
                self._companies[ticker].name for ticker in parsed.tickers if ticker in self._companies
            ]
            yield {"type": "stage", "stage": "metadata", "status": "done", "message": "Company, years, and scope identified", "data": metadata}
            plan, degraded = await self._plan_or_facts(question, metadata)
            yield {"type": "stage", "stage": "plan", "status": "degraded" if degraded else "done",
                   "message": "Facts kept, steps rejected" if degraded else "Plan verified",
                   "data": {**_json(plan), "steps_rejected": degraded}}
            docs, _ = retrieve_docs(parsed, self._documents)
            line_items = named_line_items(question, self._labels())
            dense_ids = await self._dense_candidates(question, docs)
            record, retrieval = self._retrieve(question, plan, parsed, dense_ids)
            for table_id in self._graph_tables(record, docs, line_items):
                table = self._table_by_id(table_id)
                if table is not None:
                    record["candidates"].append({**table, "sparse_rank": len(record["candidates"]) + 1})
            retrieval["line_items"] = line_items
            retrieval["graph_added"] = len(record["candidates"]) - retrieval["candidate_count"]
            self._remember_tables(request_id, record["candidates"])
            candidate_tables = [self._table_card(c) for c in record["candidates"]]
            yield {"type": "stage", "stage": "retrieve", "status": "done",
                   "message": f"Retrieved {len(record['candidates'])} candidate tables",
                   "data": {**retrieval, "candidate_count": len(record["candidates"]),
                            "candidate_tables": candidate_tables}}
            scores = await self._rerank(question, plan, record)
            arm = "hybrid" if retrieval["dense_enabled"] else "sparse"
            fused = fuse(record["candidates"], scores, mode="fuse",
                         weight=self.config.fusion_weight, arm=arm)
            budget = table_budget(len(retrieval["docs"]))
            # select_slot_tables.py runs the slot selection first and only then
            # appends, and it appends on complex questions alone. Starting from a
            # raw fused prefix and appending unconditionally submits ~1.4 tables
            # per question more than the frozen best did.
            base, _slots = select_slot_tables(record, fused, line_items, budget)
            selected = (
                append_confidence_item_tables(
                    record, base, fused, line_items, scores, 3.0, 3,
                ) if _high_arity(question, retrieval["docs"], line_items) else list(base)
            )
            links = self._note_links(record, selected, line_items)
            ranking = (selected + links)[:30]
            order = self._materialize_order(question, parsed, retrieval["docs"], ranking, record)
            self._remember_tables(request_id, record["candidates"])
            by_id = {candidate["table_id"]: candidate for candidate in record["candidates"]}
            top_tables = [self._table_card(by_id[table_id], scores.get(table_id))
                          for table_id in order if table_id in by_id]
            yield {"type": "stage", "stage": "rerank", "status": "done", "message": "Candidates reranked", "data": {"scored": len(scores), "budget": budget, "note_links": len(links),
                            "ranking": len(ranking),
                            "selected": len(order), "top": order[:10], "top_tables": top_tables}}
            result = await self._answer(question, plan, record, order)
            self._evidence[request_id] = result.pop("_evidence")
            result["latency_seconds"] = round(time.perf_counter() - started, 2)
            yield {"type": "result", "stage": "answer", "status": "done", "message": "Answer executed from source evidence", "data": result}
        except Exception as error:
            # Timeouts stringify to "", and an empty message leaves the presenter
            # with a generic "pipeline error"; the log gets the traceback.
            logging.getLogger(__name__).exception("demo pipeline failed for %r", request_id)
            yield {
                "type": "error", "stage": "pipeline", "status": "failed",
                "message": str(error) or type(error).__name__,
                "data": {"error_type": type(error).__name__},
            }

    def evidence(self, request_id: str, citation_id: str) -> dict | None:
        return self._evidence.get(request_id, {}).get(citation_id)

    def _remember_tables(self, request_id: str, candidates: list[dict]) -> None:
        """Keep every candidate a question has touched, so a trace-panel click
        can show the real table instead of only the final cited cell."""
        store = self._request_tables.setdefault(request_id, {})
        for candidate in candidates:
            store[candidate["table_id"]] = candidate

    def _table_card(self, table: dict, score: float | None = None) -> dict:
        """The compact descriptor the trace renders as an evidence card."""
        card = {"table_id": table["table_id"], "report_id": table["report_id"], "title": table.get("title", "")}
        if score is not None:
            card["score"] = round(float(score), 4)
        return card

    def table_preview(self, request_id: str, table_id: str) -> dict | None:
        """The full table behind one pipeline-trace chip: headers, rows, source."""
        table = self._request_tables.get(request_id, {}).get(table_id)
        if table is None:
            return None
        rows = table.get("rows") or []
        headers = [str(cell) for cell in rows[0]] if rows else []
        body = rows[1 : 1 + EVIDENCE_TABLE_ROWS]
        return {
            "table_id": table_id,
            "report_id": table["report_id"],
            "title": table.get("title", ""),
            "unit": table.get("unit", ""),
            "headers": headers,
            "rows": [[str(value) for value in row] for row in body],
            "truncated": len(rows) - 1 > EVIDENCE_TABLE_ROWS if rows else False,
        }
