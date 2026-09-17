
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompanyInfo:
    ticker: str
    name: str
    industry_l1: str
    industry_l2: str
    industry_l3: str
    exchange: str


def load_company_meta(path: Path) -> dict[str, CompanyInfo]:
    companies: dict[str, CompanyInfo] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("Mã CK") or "").strip()
            if not ticker:
                continue
            companies[ticker] = CompanyInfo(
                ticker=ticker,
                name=(row.get("Tên công ty") or "").strip(),
                industry_l1=(row.get("Ngành cấp 1") or "").strip(),
                industry_l2=(row.get("Ngành cấp 2") or "").strip(),
                industry_l3=(row.get("Ngành cấp 3") or "").strip(),
                exchange=(row.get("Sàn") or "").strip(),
            )
    return companies


@lru_cache
def get_company_meta(path: Path) -> dict[str, CompanyInfo]:
    return load_company_meta(path)
