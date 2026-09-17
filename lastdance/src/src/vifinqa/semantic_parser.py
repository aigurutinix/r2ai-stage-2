"""Semantic Parser module for ViFinQA.

This module parses natural language questions into structured QuestionSpec objects,
separating entities, periods, metrics, scope, operation, and units.
"""

import csv
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class EntityRef:
    ticker: Optional[str]
    name: Optional[str]
    alias: Optional[str]

@dataclass(frozen=True)
class PeriodRef:
    year: Optional[int]
    is_comparative: bool = False
    is_flow: bool = False

@dataclass(frozen=True)
class Constraint:
    type: str
    value: str

@dataclass(frozen=True)
class QuestionSpec:
    entities: tuple[EntityRef, ...]
    periods: tuple[PeriodRef, ...]
    scope: str
    requested_metrics: tuple[str, ...]
    operation: str
    output_unit: str
    constraints: tuple[Constraint, ...]
    ambiguities: tuple[str, ...]

def load_stock_codes() -> dict[str, str]:
    mapping = {}
    path = Path("ViFinQA/code_stock.csv")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) >= 2:
                    mapping[row[1].strip().lower()] = row[0].strip()
    return mapping

STOCK_CODES = load_stock_codes()

def parse_entities(question: str) -> tuple[EntityRef, ...]:
    entities = []
    tickers = set(re.findall(r'\b[A-Z]{3}\b', question))
    for t in tickers:
        entities.append(EntityRef(ticker=t, name=None, alias=None))
    
    lower_q = question.lower()
    for name, ticker in STOCK_CODES.items():
        if name in lower_q:
            if ticker not in tickers:
                entities.append(EntityRef(ticker=ticker, name=name, alias=None))
                tickers.add(ticker)
    return tuple(entities)

def parse_periods(question: str) -> tuple[PeriodRef, ...]:
    years = sorted(list(set(int(y) for y in re.findall(r'\b(20[1-2][0-9])\b', question))))
    return tuple(PeriodRef(year=y) for y in years)

def parse_scope(question: str) -> str:
    lower_q = question.lower()
    if "công ty mẹ" in lower_q or "riêng lẻ" in lower_q:
        return "separate"
    if "hợp nhất" in lower_q:
        return "consolidated"
    return "any"

def parse_unit(question: str) -> str:
    lower_q = question.lower()
    if "tỷ đồng" in lower_q:
        return "VND_1e9"
    if "triệu đồng" in lower_q:
        return "VND_1e6"
    if "nghìn đồng" in lower_q or "ngàn đồng" in lower_q:
        return "VND_1e3"
    if "đồng" in lower_q and "tỷ đồng" not in lower_q and "triệu đồng" not in lower_q and "nghìn đồng" not in lower_q and "ngàn đồng" not in lower_q:
        return "VND_1"
    if "phần trăm" in lower_q or "%" in lower_q:
        return "percent"
    if "lần" in lower_q:
        return "ratio"
    return "number"

def parse_question(question: str) -> QuestionSpec:
    """Parse a question into a QuestionSpec."""
    entities = parse_entities(question)
    periods = parse_periods(question)
    scope = parse_scope(question)
    unit = parse_unit(question)
    return QuestionSpec(
        entities=entities,
        periods=periods,
        scope=scope,
        requested_metrics=(),
        operation="lookup",
        output_unit=unit,
        constraints=(),
        ambiguities=()
    )

